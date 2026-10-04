# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy: champion vs challenger
# MAGIC
# MAGIC Compares the `challenger` model version with the current `champion` on the same, most
# MAGIC recent holdout window. The challenger is promoted to `champion` when there is no champion
# MAGIC yet or it beats the champion on `comparison_metric` by `min_relative_improvement`.
# MAGIC
# MAGIC Batch inference always loads `models:/<name>@champion`, so promotion is a single,
# MAGIC atomic alias update (and rollback is re-pointing the alias to a previous version).
# MAGIC
# MAGIC If `serving_endpoint_name` is set, a Model Serving endpoint is created/updated to serve the
# MAGIC new champion (zero-downtime config update). Because the model was logged with the Feature
# MAGIC Store, the endpoint looks up market features from the **online store** automatically, so
# MAGIC `online_store_name` must be set on the feature job before enabling serving.

# COMMAND ----------

# MAGIC %pip install -q -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../src")))

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pyspark.sql import functions as F

from used_car_price.data_quality import filter_valid
from used_car_price.deployment import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    decide_promotion,
)
from used_car_price.metrics import regression_metrics
from used_car_price.schema import (
    DATE_COL,
    LABEL_COL,
    LISTING_MONTH_COL,
    LISTING_YEAR_COL,
    MODEL_VERSION_COL,
    PREDICTION_COL,
    PREDICTION_TABLE_COLS,
    RAW_COLS,
    SCORED_AT_COL,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "used_car_price_dev")
dbutils.widgets.text("model_name", "workspace.used_car_price_dev.used_car_price_model")
dbutils.widgets.text("holdout_days", "30")
dbutils.widgets.dropdown("comparison_metric", "mape", ["mape", "mae", "rmse", "r2"])
dbutils.widgets.text("min_relative_improvement", "0.0")
dbutils.widgets.text("serving_endpoint_name", "")

w = dbutils.widgets.get
raw_table = f"{w('catalog')}.{w('schema')}.raw_listings"
baseline_table = f"{w('catalog')}.{w('schema')}.drift_baseline"
model_name = w("model_name")

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe = FeatureEngineeringClient()


def version_for(alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(model_name, alias).version
    except MlflowException:
        return None


challenger_version = version_for(CHALLENGER_ALIAS)
champion_version = version_for(CHAMPION_ALIAS)
if challenger_version is None:
    raise RuntimeError(f"{model_name} has no @{CHALLENGER_ALIAS}; run ModelValidation first")

expected = dbutils.jobs.taskValues.get(
    taskKey="Train", key="model_version", default="", debugValue=""
)
if expected and expected != challenger_version:
    raise RuntimeError(
        f"@{CHALLENGER_ALIAS} is v{challenger_version} but this run trained v{expected}"
    )
print(f"challenger=v{challenger_version} champion={'v' + champion_version if champion_version else None}")

# COMMAND ----------

raw_sdf = spark.table(raw_table).select(*RAW_COLS)
end = raw_sdf.agg(F.max(DATE_COL)).first()[0]
window = raw_sdf.where(F.col(DATE_COL) > F.date_sub(F.lit(end), int(w("holdout_days"))))
eval_sdf = spark.createDataFrame(filter_valid(window.toPandas()), schema=window.schema)
eval_sdf = eval_sdf.withColumns(
    {
        LISTING_YEAR_COL: F.year(DATE_COL).cast("long"),
        LISTING_MONTH_COL: F.month(DATE_COL).cast("long"),
    }
)


def score(version: str):
    # Feature lookups are resolved by the Feature Store from the model's packaged metadata.
    return fe.score_batch(
        model_uri=f"models:/{model_name}/{version}", df=eval_sdf, result_type="double"
    )


def evaluate(version: str) -> dict[str, float]:
    pdf = score(version).select(LABEL_COL, "prediction").toPandas()
    return regression_metrics(pdf[LABEL_COL], pdf["prediction"])


challenger_metrics = evaluate(challenger_version)
champion_metrics = (
    evaluate(champion_version)
    if champion_version and champion_version != challenger_version
    else None
)
decision = decide_promotion(
    challenger_metrics,
    champion_metrics,
    metric=w("comparison_metric"),
    min_relative_improvement=float(w("min_relative_improvement")),
)
print(f"challenger: {challenger_metrics}\nchampion:   {champion_metrics}\n{decision.reason}")

# COMMAND ----------

client.set_model_version_tag(
    model_name, challenger_version, "deployment_decision",
    "PROMOTED" if decision.promote else "REJECTED",
)
client.set_model_version_tag(model_name, challenger_version, "deployment_reason", decision.reason)

if decision.promote:
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, challenger_version)
    if champion_version:
        client.set_model_version_tag(model_name, champion_version, "replaced_by", challenger_version)
    print(f"Promoted v{challenger_version} to @{CHAMPION_ALIAS}")
else:
    print(f"Kept champion v{champion_version}")

current_champion = version_for(CHAMPION_ALIAS)
dbutils.jobs.taskValues.set("champion_version", current_champion or "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drift baseline
# MAGIC
# MAGIC The champion's predictions on its evaluation window become the **reference distribution**
# MAGIC for data/prediction drift (PSI in the Monitoring task and the Lakehouse Monitor baseline).
# MAGIC Same schema as the `predictions` table. Rewritten whenever the champion changes.

# COMMAND ----------

baseline_exists = spark.catalog.tableExists(baseline_table)
baseline_version = (
    spark.table(baseline_table).select(MODEL_VERSION_COL).first()[0] if baseline_exists else None
)
if current_champion and str(baseline_version) != current_champion:
    baseline = (
        score(current_champion)
        .withColumnRenamed("prediction", PREDICTION_COL)
        .withColumn(MODEL_VERSION_COL, F.lit(current_champion))
        .withColumn(SCORED_AT_COL, F.current_timestamp())
        .select(*PREDICTION_TABLE_COLS)
    )
    (
        baseline.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(baseline_table)
    )
    print(f"Wrote drift baseline for v{current_champion} to {baseline_table}")
else:
    print(f"Drift baseline already matches champion v{current_champion}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: real-time Model Serving endpoint

# COMMAND ----------

endpoint_name = w("serving_endpoint_name")
champion = current_champion
if endpoint_name and champion:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    ws = WorkspaceClient()
    served = [
        ServedEntityInput(
            entity_name=model_name,
            entity_version=champion,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ]
    try:
        current = ws.serving_endpoints.get(endpoint_name)
        served_versions = {e.entity_version for e in (current.config.served_entities or [])}
        if champion in served_versions:
            print(f"{endpoint_name} already serves v{champion}")
        else:
            ws.serving_endpoints.update_config_and_wait(endpoint_name, served_entities=served)
            print(f"Updated {endpoint_name} -> v{champion}")
    except NotFound:
        ws.serving_endpoints.create_and_wait(
            name=endpoint_name, config=EndpointCoreConfigInput(served_entities=served)
        )
        print(f"Created {endpoint_name} serving v{champion}")
else:
    print("Model Serving disabled (serving_endpoint_name is empty)")
