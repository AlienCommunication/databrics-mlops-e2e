"""Databricks task implementations; invoked by the thin notebook task runner."""

import json
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.pyfunc import load_model
from mlflow.exceptions import MlflowException
from databricks.sdk import WorkspaceClient
from modeling import (
    FEATURES,
    NUMERIC,
    TARGET,
    synthetic,
    validate_features,
    split_data,
    build_model,
    metrics,
    promotion_decision,
    psi,
)


class Workflow:
    def __init__(self, spark, dbutils, catalog, schema, endpoint, experiment, as_of, release):
        for identifier in (catalog, schema):
            if not identifier.replace("_", "").isalnum():
                raise ValueError("SQL identifier must be alphanumeric/underscore")
        self.spark, self.dbutils = spark, dbutils
        self.ns, self.endpoint = f"{catalog}.{schema}", endpoint
        self.catalog, self.schema = catalog, schema
        self.model_name = f"{self.ns}.used_car_price"
        self.as_of = date.fromisoformat(as_of) if as_of else datetime.now(timezone.utc).date()
        self.release = release
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(experiment)
        self.client = MlflowClient()
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.ns}")

    def table(self, name):
        return f"{self.ns}.{name}"

    def read(self, name):
        return self.spark.table(self.table(name)).toPandas()

    def merge(self, name, pdf, keys):
        from delta.tables import DeltaTable

        sdf = self.spark.createDataFrame(pdf)
        name = self.table(name)
        if not self.spark.catalog.tableExists(name):
            sdf.write.format("delta").saveAsTable(name)
        else:
            condition = " AND ".join(f"t.`{k}` = s.`{k}`" for k in keys)
            (
                DeltaTable.forName(self.spark, name)
                .alias("t")
                .merge(sdf.alias("s"), condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )

    def put(self, key, value):
        self.dbutils.jobs.taskValues.set(key=key, value=value)

    def get(self, task, key):
        return self.dbutils.jobs.taskValues.get(taskKey=task, key=key)

    def alias_version(self, alias):
        try:
            return self.client.get_model_version_by_alias(self.model_name, alias).version
        except MlflowException as exc:
            if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise

    def prepare(self):
        # Immutable reference dataset. Real ingestion replaces only this source adapter.
        raw = synthetic()
        validate_features(raw)
        self.merge("bronze_sales", raw, ["car_id"])
        self.merge("features", raw, ["car_id"])
        version = self.spark.sql(f"DESCRIBE HISTORY {self.table('features')} LIMIT 1").first()[
            "version"
        ]
        self.put("data_version", int(version))
        print(json.dumps({"rows": len(raw), "delta_version": int(version)}))

    def training_data(self, version):
        return (
            self.spark.read.option("versionAsOf", version).table(self.table("features")).toPandas()
        )

    def train(self):
        data_version = self.get("prepare", "data_version")
        train, val, test = split_data(self.training_data(data_version))
        model = build_model().fit(validate_features(train), train[TARGET])
        with mlflow.start_run(run_name=f"train-{self.release}") as run:
            mlflow.log_params(
                {
                    "release": self.release,
                    "synthetic": True,
                    "currency": "INR",
                    "data_version": data_version,
                    "data_table": self.table("features"),
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "test_rows": len(test),
                }
            )
            mlflow.log_input(
                mlflow.data.from_pandas(train, source=self.table("features"), targets=TARGET),
                context="training",
            )
            mlflow.log_metrics(
                {
                    f"validation_{k}": v
                    for k, v in metrics(val[TARGET], model.predict(val[FEATURES])).items()
                }
            )
            info = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=mlflow.models.infer_signature(
                    train[FEATURES], model.predict(train[FEATURES])
                ),
                input_example=train[FEATURES].head(3),
                pip_requirements=[
                    "mlflow==2.22.2",
                    "scikit-learn==1.5.2",
                    "pandas==2.2.3",
                    "numpy==1.26.4",
                ],
            )
            registered = mlflow.register_model(info.model_uri, self.model_name)
            for key, value in {
                "synthetic": "true",
                "release": self.release,
                "data_version": str(data_version),
            }.items():
                self.client.set_model_version_tag(self.model_name, registered.version, key, value)
            self.put("version", registered.version)
            self.put("run_id", run.info.run_id)
            self.put("data_version", data_version)

    def validate(self):
        version, run_id = self.get("train", "version"), self.get("train", "run_id")
        train, _, test = split_data(self.training_data(self.get("train", "data_version")))
        model = load_model(f"models:/{self.model_name}/{version}")
        scores = metrics(test[TARGET], model.predict(test[FEATURES]))
        previous = self.alias_version("Champion")
        champion = None
        if previous:
            champion = metrics(
                test[TARGET],
                load_model(f"models:/{self.model_name}/{previous}").predict(test[FEATURES]),
            )
        baseline = metrics(test[TARGET], np.repeat(train[TARGET].median(), len(test)))["mae"]
        decision = promotion_decision(scores, baseline, champion)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"test_{k}": v for k, v in scores.items()})
            mlflow.log_dict(
                dict(decision, candidate=scores, champion=champion, baseline_mae=baseline),
                "validation.json",
            )
        self.client.set_model_version_tag(
            self.model_name, version, "validation", "passed" if decision["passed"] else "failed"
        )
        if not decision["passed"]:
            raise ValueError(f"Promotion rejected: {decision}")
        self.client.set_registered_model_alias(self.model_name, "Challenger", version)
        self.put("version", version)
        self.put("previous", previous or "")
        print(json.dumps(dict(decision, metrics=scores)))

    def deploy(self):
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput,
            ServedEntityInput,
            AiGatewayInferenceTableConfig,
        )
        from databricks.sdk.errors import NotFound

        version, previous = str(self.get("validate", "version")), self.get("validate", "previous")
        # Fail closed if task values are spoofed or candidate was invalidated.
        mv = self.client.get_model_version(self.model_name, version)
        if mv.tags.get("validation") != "passed":
            raise ValueError("Candidate does not have a passed validation gate")
        if (self.alias_version("Champion") or "") != previous:
            raise ValueError("Champion changed after validation; revalidate before deploying")
        w = WorkspaceClient()
        entity = ServedEntityInput(
            entity_name=self.model_name,
            entity_version=version,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
        try:
            existing = w.serving_endpoints.get(self.endpoint)
        except NotFound:
            existing = None
        try:
            if existing:
                w.serving_endpoints.update_config_and_wait(
                    self.endpoint, served_entities=[entity], timeout=timedelta(minutes=40)
                )
            else:
                w.serving_endpoints.create_and_wait(
                    self.endpoint,
                    config=EndpointCoreConfigInput(name=self.endpoint, served_entities=[entity]),
                    timeout=timedelta(minutes=40),
                )
            w.serving_endpoints.put_ai_gateway(
                self.endpoint,
                inference_table_config=AiGatewayInferenceTableConfig(
                    catalog_name=self.catalog,
                    schema_name=self.schema,
                    table_name_prefix="online",
                    enabled=True,
                ),
            )
            example = synthetic(2)[FEATURES]
            actual = w.serving_endpoints.query(
                self.endpoint, dataframe_records=example.to_dict(orient="records")
            ).predictions
            expected = load_model(f"models:/{self.model_name}/{version}").predict(example)
            np.testing.assert_allclose(actual, expected, rtol=1e-5)
        except Exception:
            # Compensate an online change; never promote the batch alias on failure.
            if existing and previous:
                rollback = ServedEntityInput(
                    entity_name=self.model_name,
                    entity_version=previous,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
                w.serving_endpoints.update_config_and_wait(
                    self.endpoint, served_entities=[rollback], timeout=timedelta(minutes=40)
                )
            raise
        # Endpoint is explicitly updated; aliases do not update endpoints automatically.
        if previous and previous != version:
            self.client.set_registered_model_alias(self.model_name, "PreviousChampion", previous)
        self.client.set_registered_model_alias(self.model_name, "Champion", version)
        self.merge(
            "deployments",
            pd.DataFrame(
                [
                    dict(
                        version=version,
                        previous=previous,
                        endpoint=self.endpoint,
                        release=self.release,
                        deployed_at=datetime.now(timezone.utc).isoformat(),
                    )
                ]
            ),
            ["version"],
        )
        print(
            json.dumps({"endpoint": self.endpoint, "version": version, "smoke_predictions": actual})
        )

    def inputs(self):
        # Replay seven days so delayed labels can be demonstrated on the first run.
        inputs, outcomes = [], []
        for offset in range(7):
            day = self.as_of - timedelta(days=offset)
            data = synthetic(100, int(day.strftime("%Y%m%d")), str(day), 1)
            labels = data[["car_id", TARGET]].copy()
            labels["available_date"] = str(day + timedelta(days=2))
            outcomes.append(labels)
            inputs.append(data.drop(columns=TARGET))
        self.merge("outcomes", pd.concat(outcomes, ignore_index=True), ["car_id"])
        self.merge("scoring_inputs", pd.concat(inputs, ignore_index=True), ["car_id"])

    def batch(self):
        version = self.alias_version("Champion")
        if not version:
            raise ValueError("No approved Champion; train/validate/deploy first")
        # Resolve once per run; every row carries exact model lineage.
        model = load_model(f"models:/{self.model_name}/{version}")
        data = self.read("scoring_inputs")
        data = data[pd.to_datetime(data.event_date).dt.date <= self.as_of]
        validate_features(data)
        data["prediction_inr"] = model.predict(data[FEATURES])
        data["model_version"], data["scoring_date"] = str(version), str(self.as_of)
        self.merge("predictions", data, ["car_id", "scoring_date", "model_version"])
        from pyspark.sql import functions as F

        written = self.spark.table(self.table("predictions")).where(
            (F.col("scoring_date") == str(self.as_of)) & (F.col("model_version") == str(version))
        )
        if (
            written.count() != len(data)
            or written.groupBy("car_id", "scoring_date", "model_version")
            .count()
            .where("count > 1")
            .count()
        ):
            raise AssertionError("Batch count/idempotency invariant failed")
        self.put("version", version)
        print(json.dumps({"rows": len(data), "version": version}))
        return {
            "rows": len(data),
            "version": version,
            "duplicate_keys": 0,
            "as_of": str(self.as_of),
        }

    def monitor(self):
        reference = self.read("features")
        predictions = self.read("predictions")
        current = predictions[predictions.scoring_date == str(self.as_of)]
        if current.empty:
            raise ValueError("No predictions for monitoring date")
        labels = self.read("outcomes")
        labels = labels[labels.available_date <= str(self.as_of)]
        rows = []
        for version, group in current.groupby("model_version"):
            drift = {col: psi(reference[col], group[col]) for col in NUMERIC}
            labeled = group.merge(labels, on="car_id")
            quality = metrics(labeled[TARGET], labeled.prediction_inr) if len(labeled) > 1 else {}
            alerts = ["drift:" + col for col, val in drift.items() if val > 0.2]
            if quality.get("mape", 0) > 0.2:
                alerts.append("mape_above_20pct")
            rows.append(
                dict(
                    monitor_date=str(self.as_of),
                    model_version=str(version),
                    prediction_count=len(group),
                    label_count=len(labeled),
                    metrics_json=json.dumps(quality),
                    drift_json=json.dumps(drift),
                    alerts_json=json.dumps(alerts),
                )
            )
        self.merge("monitoring", pd.DataFrame(rows), ["monitor_date", "model_version"])
        # A newly enabled inference table can initially contain only a request-ID column.
        # Record unavailable telemetry explicitly; do not mistake it for zero-error traffic.
        online_result = dict(
            monitor_date=str(self.as_of),
            request_count=0,
            error_count=0,
            mean_execution_ms=0.0,
            log_status="TABLE_NOT_YET_AVAILABLE",
        )
        if self.spark.catalog.tableExists(self.table("online_payload")):
            from pyspark.sql import functions as F

            online = self.spark.table(self.table("online_payload"))
            cols = set(online.columns)
            date_col = next((c for c in ["request_date", "date"] if c in cols), None)
            duration = next(
                (c for c in ["execution_duration_ms", "execution_time_ms"] if c in cols), None
            )
            if date_col and duration and "status_code" in cols:
                summary = (
                    online.where(F.col(date_col) == str(self.as_of))
                    .agg(
                        F.count("*").alias("request_count"),
                        F.sum(F.when(F.col("status_code") >= 400, 1).otherwise(0)).alias(
                            "error_count"
                        ),
                        F.avg(duration).alias("mean_execution_ms"),
                    )
                    .first()
                )
                online_result.update(
                    request_count=int(summary["request_count"]),
                    error_count=int(summary["error_count"] or 0),
                    mean_execution_ms=float(summary["mean_execution_ms"] or 0.0),
                    log_status="OBSERVED" if summary["request_count"] else "NO_LOGGED_REQUESTS",
                )
            else:
                online_result["log_status"] = "PENDING_OR_UNSUPPORTED_SCHEMA"
        if self.spark.catalog.tableExists(self.table("online_monitoring")):
            if "log_status" not in self.spark.table(self.table("online_monitoring")).columns:
                self.spark.sql(
                    f"ALTER TABLE {self.table('online_monitoring')} ADD COLUMNS (log_status STRING)"
                )
        self.merge("online_monitoring", pd.DataFrame([online_result]), ["monitor_date"])
        print(json.dumps({"online_monitoring": online_result}))
        print(json.dumps(rows))
        if any(json.loads(row["alerts_json"]) for row in rows):
            raise ValueError(
                "Monitoring threshold exceeded; review monitoring table before retraining"
            )
        return rows
