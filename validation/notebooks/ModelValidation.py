# Databricks notebook source
# MAGIC %md
# MAGIC # Validate the newly trained model version
# MAGIC
# MAGIC Gates a model version before it can be considered for production:
# MAGIC
# MAGIC 1. **Performance thresholds** on the untouched time-based holdout (MAPE, MAE, R²), scored
# MAGIC    through `FeatureEngineeringClient.score_batch` exactly like production batch inference.
# MAGIC 2. **Business sanity rules**: predictions must be finite and positive.
# MAGIC 3. **Contract checks**: the model has an input signature, and a cold-start listing whose
# MAGIC    make/model/region has no market features still gets a valid prediction.
# MAGIC
# MAGIC On success the version gets the `challenger` alias. Results are recorded as model version
# MAGIC tags for auditability.
# MAGIC
# MAGIC `run_mode`: `enabled` fails the job on validation failure; `dry_run` records failures but
# MAGIC still assigns the `challenger` alias (useful while bootstrapping a new environment).

# COMMAND ----------

# MAGIC %pip install -q -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../src")))

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow import MlflowClient
from pyspark.sql import functions as F

from used_car_price.data_quality import filter_valid
from used_car_price.deployment import CHALLENGER_ALIAS
from used_car_price.metrics import regression_metrics
from used_car_price.schema import (
    DATE_COL,
    LABEL_COL,
    LISTING_MONTH_COL,
    LISTING_YEAR_COL,
    RAW_COLS,
)
from used_car_price.validation import (
    ValidationResult,
    ValidationThresholds,
    check_prediction_sanity,
    check_thresholds,
    combine,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "used_car_price_dev")
dbutils.widgets.text("model_name", "workspace.used_car_price_dev.used_car_price_model")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("holdout_days", "30")
dbutils.widgets.text("max_mape", "0.12")
dbutils.widgets.text("max_mae", "2500")
dbutils.widgets.text("min_r2", "0.90")
dbutils.widgets.dropdown("run_mode", "enabled", ["enabled", "dry_run"])

w = dbutils.widgets.get
tv = dbutils.jobs.taskValues
raw_table = f"{w('catalog')}.{w('schema')}.raw_listings"
model_name = w("model_name")
model_version = w("model_version") or tv.get(taskKey="Train", key="model_version")
thresholds = ValidationThresholds(
    max_mape=float(w("max_mape")), max_mae=float(w("max_mae")), min_r2=float(w("min_r2"))
)

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe = FeatureEngineeringClient()


def with_calendar(sdf):
    return sdf.withColumns(
        {
            LISTING_YEAR_COL: F.year(DATE_COL).cast("long"),
            LISTING_MONTH_COL: F.month(DATE_COL).cast("long"),
        }
    )


# COMMAND ----------

# Evaluate on exactly the holdout window/snapshot the Train task held out. When run standalone,
# fall back to the latest `holdout_days` of data.
holdout_start = tv.get(taskKey="Train", key="holdout_start", default="", debugValue="")
holdout_end = tv.get(taskKey="Train", key="holdout_end", default="", debugValue="")
table_version = tv.get(taskKey="Train", key="source_table_version", default="", debugValue="")

reader = spark.read
if table_version:
    reader = reader.option("versionAsOf", int(table_version))
raw_sdf = reader.table(raw_table).select(*RAW_COLS)
if not holdout_start:
    holdout_end = str(raw_sdf.agg(F.max(DATE_COL)).first()[0])
    holdout_start = str(
        raw_sdf.select(F.date_sub(F.lit(holdout_end), int(w("holdout_days")) - 1)).first()[0]
    )
window = raw_sdf.where(F.col(DATE_COL).between(holdout_start, holdout_end))
holdout_sdf = with_calendar(
    spark.createDataFrame(filter_valid(window.toPandas()), schema=window.schema)
)
print(f"Validating {model_name} v{model_version} [{holdout_start} .. {holdout_end}]")

# COMMAND ----------

model_uri = f"models:/{model_name}/{model_version}"
scored = fe.score_batch(model_uri=model_uri, df=holdout_sdf, result_type="double").toPandas()
preds = scored["prediction"]
metrics = regression_metrics(scored[LABEL_COL], preds)
print(f"{len(scored):,} rows: {metrics}")

contract_failures = []
if mlflow.models.get_model_info(model_uri).signature is None:
    contract_failures.append("model has no input signature")
# Cold start: a make/model/region with no market history gets null features (NaN) and must
# still score.
cold_start = holdout_sdf.limit(5).withColumn("region", F.lit("__unseen_region__"))
cold_preds = fe.score_batch(model_uri=model_uri, df=cold_start, result_type="double")
cold_preds = cold_preds.select("prediction").toPandas()["prediction"]
if len(cold_preds) != 5 or not check_prediction_sanity(cold_preds).passed:
    contract_failures.append("cold-start listing (no market features) did not score correctly")

result = combine(
    check_thresholds(metrics, thresholds),
    check_prediction_sanity(preds),
    ValidationResult(passed=not contract_failures, failures=contract_failures),
)

# COMMAND ----------

status = "PASSED" if result.passed else "FAILED"
client.set_model_version_tag(model_name, model_version, "validation_status", status)
client.set_model_version_tag(
    model_name, model_version, "validation_window", f"{holdout_start}..{holdout_end}"
)
for k, v in metrics.items():
    client.set_model_version_tag(model_name, model_version, f"validation_{k}", f"{v:.6f}")
if result.failures:
    client.set_model_version_tag(
        model_name, model_version, "validation_failures", "; ".join(result.failures)[:5000]
    )

print(f"Validation {status}: {result.failures or 'all checks passed'}")
if not result.passed and w("run_mode") == "enabled":
    raise RuntimeError(f"Model validation failed for {model_uri}: {result.failures}")

client.set_registered_model_alias(model_name, CHALLENGER_ALIAS, model_version)
print(f"Assigned @{CHALLENGER_ALIAS} -> v{model_version}")
