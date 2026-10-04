# Databricks notebook source
# MAGIC %md
# MAGIC # Monitor production: performance, data drift, prediction drift → retrain decision
# MAGIC
# MAGIC 1. **Label backfill**: joins ground-truth sale prices from `raw_listings` onto
# MAGIC    `predictions` as they become available (labels lag predictions).
# MAGIC 2. **Lakehouse Monitoring**: creates (first run) or refreshes an inference-log monitor on
# MAGIC    `predictions`, with the champion's `drift_baseline` as baseline, for dashboards.
# MAGIC 3. **Performance (concept drift)**: MAE/MAPE of the champion over the last `window_days`
# MAGIC    of *labelled* predictions.
# MAGIC 4. **Data & prediction drift**: PSI of key inputs and of the predictions vs
# MAGIC    `drift_baseline`. Available immediately, before labels arrive.
# MAGIC 5. **Decision**: `decide_retraining` combines the signals with a cooldown guard rail.
# MAGIC    Results go to `model_performance_history` / `feature_drift_history` (a SQL alert
# MAGIC    watches `alert_triggered`), and the task value `retrain` drives the job's
# MAGIC    `condition_task` → `run_job_task` that launches the model training job.

# COMMAND ----------

# MAGIC %pip install -q -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pyspark.sql import functions as F

from used_car_price.deployment import CHAMPION_ALIAS
from used_car_price.drift import compute_drift
from used_car_price.monitoring import RetrainPolicy, decide_retraining
from used_car_price.schema import (
    DATE_COL,
    DRIFT_CATEGORICAL_COLS,
    DRIFT_NUMERIC_COLS,
    ID_COL,
    LABEL_COL,
    MODEL_VERSION_COL,
    PREDICTION_COL,
    SCORED_AT_COL,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "used_car_price_dev")
dbutils.widgets.text("model_name", "workspace.used_car_price_dev.used_car_price_model")
dbutils.widgets.text("run_date", "")
dbutils.widgets.text("window_days", "7")
dbutils.widgets.text("max_mape", "0.12")
dbutils.widgets.text("min_labeled_rows", "200")
dbutils.widgets.text("max_feature_psi", "0.25")
dbutils.widgets.text("min_drifted_features", "1")
dbutils.widgets.text("max_prediction_psi", "0.25")
dbutils.widgets.text("cooldown_days", "3")
dbutils.widgets.dropdown("enable_lakehouse_monitor", "true", ["true", "false"])
dbutils.widgets.dropdown("auto_retrain", "false", ["true", "false"])

w = dbutils.widgets.get
catalog, schema, model_name = w("catalog"), w("schema"), w("model_name")
raw_table = f"{catalog}.{schema}.raw_listings"
predictions_table = f"{catalog}.{schema}.predictions"
baseline_table = f"{catalog}.{schema}.drift_baseline"
history_table = f"{catalog}.{schema}.model_performance_history"
drift_history_table = f"{catalog}.{schema}.feature_drift_history"
run_date = dt.date.fromisoformat(w("run_date")) if w("run_date") else dt.date.today()
window_start = run_date - dt.timedelta(days=int(w("window_days")) - 1)
policy = RetrainPolicy(
    max_mape=float(w("max_mape")),
    min_labeled_rows=int(w("min_labeled_rows")),
    max_feature_psi=float(w("max_feature_psi")),
    min_drifted_features=int(w("min_drifted_features")),
    max_prediction_psi=float(w("max_prediction_psi")),
    min_rows_for_drift=int(w("min_labeled_rows")),
    cooldown_days=int(w("cooldown_days")),
)
ws = WorkspaceClient()
client = MlflowClient(registry_uri="databricks-uc")
try:
    champion_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS).version
except MlflowException as e:
    raise RuntimeError(f"{model_name} has no @{CHAMPION_ALIAS}; run model training first") from e

# COMMAND ----------

# MAGIC %md ## 1. Backfill labels

# COMMAND ----------

display(spark.sql(f"""
    MERGE INTO {predictions_table} p
    USING (SELECT {ID_COL}, {LABEL_COL} FROM {raw_table} WHERE {LABEL_COL} IS NOT NULL) r
    ON p.{ID_COL} = r.{ID_COL}
    WHEN MATCHED AND p.{LABEL_COL} IS NULL THEN UPDATE SET p.{LABEL_COL} = r.{LABEL_COL}
"""))

# COMMAND ----------

# MAGIC %md ## 2. Data profiling (Lakehouse Monitoring)

# COMMAND ----------

if w("enable_lakehouse_monitor") == "true":
    from databricks.sdk.service.catalog import (
        MonitorInferenceLog,
        MonitorInferenceLogProblemType,
    )

    try:
        ws.quality_monitors.get(table_name=predictions_table)
        ws.quality_monitors.run_refresh(table_name=predictions_table)
        print(f"Refreshing monitor on {predictions_table}")
    except NotFound:
        me = ws.current_user.me().user_name
        has_baseline = spark.catalog.tableExists(baseline_table)
        ws.quality_monitors.create(
            table_name=predictions_table,
            assets_dir=f"/Workspace/Users/{me}/lakehouse_monitoring/{predictions_table}",
            output_schema_name=f"{catalog}.{schema}",
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
                prediction_col=PREDICTION_COL,
                label_col=LABEL_COL,
                model_id_col=MODEL_VERSION_COL,
                timestamp_col=SCORED_AT_COL,
                granularities=["1 day"],
            ),
            slicing_exprs=["make", "body_type", "region"],
            # Drift metrics vs the champion's validation data, not just vs previous windows.
            baseline_table_name=baseline_table if has_baseline else None,
        )
        print(f"Created inference monitor on {predictions_table}")
    except DatabricksError as e:
        # Monitoring is not available on every workspace tier; don't block the custom check.
        print(f"WARNING: Lakehouse Monitoring unavailable, continuing: {e}")

# COMMAND ----------

# MAGIC %md ## 3. Performance of the champion (concept drift)

# COMMAND ----------

window = spark.table(predictions_table).where(
    F.col(DATE_COL).between(F.lit(window_start), F.lit(run_date))
    & (F.col(MODEL_VERSION_COL) == champion_version)
)
labeled = window.where(F.col(LABEL_COL).isNotNull())
agg = labeled.agg(
    F.count("*").alias("labeled_rows"),
    F.avg(F.abs(F.col(PREDICTION_COL) - F.col(LABEL_COL))).alias("mae"),
    F.sqrt(F.avg(F.pow(F.col(PREDICTION_COL) - F.col(LABEL_COL), 2))).alias("rmse"),
    F.avg(F.abs((F.col(PREDICTION_COL) - F.col(LABEL_COL)) / F.col(LABEL_COL))).alias("mape"),
).first()
labeled_rows = agg["labeled_rows"]
metrics = {"mae": agg["mae"], "rmse": agg["rmse"], "mape": agg["mape"]} if labeled_rows else None
print(f"v{champion_version} window {window_start}..{run_date}: rows={labeled_rows} {metrics}")

# COMMAND ----------

# MAGIC %md ## 4. Data drift and prediction drift (PSI vs the champion's baseline)

# COMMAND ----------

drift_cols = [*DRIFT_NUMERIC_COLS, *DRIFT_CATEGORICAL_COLS, PREDICTION_COL]
current = window.select(*drift_cols).toPandas()
drift = None
if spark.catalog.tableExists(baseline_table):
    reference = spark.table(baseline_table).select(*drift_cols).toPandas()
    drift = compute_drift(
        reference,
        current,
        numeric_cols=DRIFT_NUMERIC_COLS,
        categorical_cols=DRIFT_CATEGORICAL_COLS,
        prediction_col=PREDICTION_COL,
        threshold=policy.max_feature_psi,
    )
    print(f"feature PSI: { {k: round(v, 3) for k, v in drift.feature_psi.items()} }")
    print(f"prediction PSI: {drift.prediction_psi:.3f}; drifted: {drift.drifted_features}")
else:
    print(f"No {baseline_table} yet (written by ModelDeployment); skipping drift check")

# COMMAND ----------

# MAGIC %md ## 5. Retraining decision

# COMMAND ----------

versions = client.search_model_versions(f"name='{model_name}'")
last_trained_ms = max((v.creation_timestamp for v in versions), default=None)
days_since_training = (
    (dt.datetime.now(dt.timezone.utc).timestamp() * 1000 - last_trained_ms) / 86_400_000
    if last_trained_ms
    else None
)
decision = decide_retraining(
    metrics, labeled_rows, drift, len(current), days_since_training, policy
)
retrain = decision.retrain and w("auto_retrain") == "true"
print("\n".join(decision.reasons))
print(f"triggers={decision.triggers} retrain_recommended={decision.retrain} retrain={retrain}")
if decision.retrain and not retrain:
    print("Retraining recommended but auto_retrain is disabled for this target")

# COMMAND ----------

m = metrics or {}
history_row = spark.createDataFrame(
    [
        (
            run_date,
            window_start,
            champion_version,
            labeled_rows,
            len(current),
            m.get("mae"),
            m.get("rmse"),
            m.get("mape"),
            drift.max_feature_psi if drift else None,
            drift.prediction_psi if drift else None,
            ",".join(drift.drifted_features) if drift else "",
            ",".join(decision.triggers),
            decision.alert,
            decision.retrain,
            retrain,
            "; ".join(decision.reasons),
        )
    ],
    "run_date DATE, window_start DATE, model_version STRING, labeled_rows BIGINT, "
    "scored_rows BIGINT, mae DOUBLE, rmse DOUBLE, mape DOUBLE, max_feature_psi DOUBLE, "
    "prediction_psi DOUBLE, drifted_features STRING, triggers STRING, alert_triggered BOOLEAN, "
    "retrain_recommended BOOLEAN, retrain_triggered BOOLEAN, reason STRING",
)
history_row.createOrReplaceTempView("history_row")
spark.sql(f"CREATE TABLE IF NOT EXISTS {history_table} AS SELECT * FROM history_row LIMIT 0")
spark.sql(f"""MERGE INTO {history_table} t USING history_row s ON t.run_date = s.run_date
              WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")

if drift:
    drift_rows = spark.createDataFrame(
        [
            (run_date, champion_version, col, float(psi), col in drift.drifted_features)
            for col, psi in [*drift.feature_psi.items(), (PREDICTION_COL, drift.prediction_psi)]
        ],
        "run_date DATE, model_version STRING, feature STRING, psi DOUBLE, drifted BOOLEAN",
    )
    drift_rows.createOrReplaceTempView("drift_rows")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {drift_history_table} AS SELECT * FROM drift_rows LIMIT 0")
    spark.sql(f"""MERGE INTO {drift_history_table} t USING drift_rows s
                  ON t.run_date = s.run_date AND t.feature = s.feature
                  WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")

# COMMAND ----------

# Consumed by the job's condition_task (retrain == "true") and passed to the training job.
dbutils.jobs.taskValues.set("retrain", "true" if retrain else "false")
dbutils.jobs.taskValues.set("trigger_reason", ",".join(decision.triggers) or "none")
