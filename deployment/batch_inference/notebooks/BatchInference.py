# Databricks notebook source
# MAGIC %md
# MAGIC # Batch inference with the champion model
# MAGIC
# MAGIC Scores listings from the last `lookback_days` up to `run_date` with
# MAGIC `models:/<name>@champion` using `FeatureEngineeringClient.score_batch`, which looks up the
# MAGIC point-in-time market features the model was trained with, and `MERGE`s the
# MAGIC results into the `predictions` table keyed on `(listing_id, model_version)`, so re-runs and
# MAGIC backfills are idempotent.
# MAGIC
# MAGIC The `price` column is left empty here: labels are joined later by the monitoring task as
# MAGIC they arrive, mirroring how ground truth lags predictions in production.
# MAGIC
# MAGIC Compute: serverless in dev/test; an autoscaling **job cluster** (`batch_inference_cluster`)
# MAGIC sized for the daily volume in staging/prod.

# COMMAND ----------

# MAGIC %pip install -q -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../src")))

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pyspark.sql import functions as F

from used_car_price.deployment import CHAMPION_ALIAS
from used_car_price.schema import (
    DATE_COL,
    ID_COL,
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
dbutils.widgets.text("run_date", "")
dbutils.widgets.text("lookback_days", "1")

w = dbutils.widgets.get
catalog, schema, model_name = w("catalog"), w("schema"), w("model_name")
raw_table = f"{catalog}.{schema}.raw_listings"
predictions_table = f"{catalog}.{schema}.predictions"
run_date = dt.date.fromisoformat(w("run_date")) if w("run_date") else dt.date.today()
start_date = run_date - dt.timedelta(days=int(w("lookback_days")) - 1)

mlflow.set_registry_uri("databricks-uc")
try:
    champion_version = MlflowClient().get_model_version_by_alias(model_name, CHAMPION_ALIAS).version
except MlflowException as e:
    raise RuntimeError(
        f"{model_name} has no @{CHAMPION_ALIAS} yet — run the model training job first"
    ) from e
# Pin the resolved version so a concurrent promotion cannot mix versions within one run.
model_uri = f"models:/{model_name}/{champion_version}"
print(f"Scoring {start_date}..{run_date} with @{CHAMPION_ALIAS} = {model_uri}")

# COMMAND ----------

listings = (
    spark.table(raw_table)
    .where(F.col(DATE_COL).between(F.lit(start_date), F.lit(run_date)))
    .select(*RAW_COLS)
    .drop(LABEL_COL)
    .withColumns(
        {
            LISTING_YEAR_COL: F.year(DATE_COL).cast("long"),
            LISTING_MONTH_COL: F.month(DATE_COL).cast("long"),
        }
    )
)
scored = (
    FeatureEngineeringClient()
    .score_batch(model_uri=model_uri, df=listings, result_type="double")
    .withColumnRenamed("prediction", PREDICTION_COL)
    .withColumn(MODEL_VERSION_COL, F.lit(str(champion_version)))
    .withColumn(SCORED_AT_COL, F.current_timestamp())
    .withColumn(LABEL_COL, F.lit(None).cast("double"))
    .select(*PREDICTION_TABLE_COLS)
)

# COMMAND ----------

scored.createOrReplaceTempView("scored")
spark.sql(
    f"""CREATE TABLE IF NOT EXISTS {predictions_table}
        CLUSTER BY ({DATE_COL})
        COMMENT 'Used car price predictions from the champion model; price is backfilled when known'
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS SELECT * FROM scored LIMIT 0"""
)

result = spark.sql(
    f"""MERGE INTO {predictions_table} t
        USING scored s
        ON t.{ID_COL} = s.{ID_COL} AND t.{MODEL_VERSION_COL} = s.{MODEL_VERSION_COL}
        WHEN MATCHED THEN UPDATE SET t.{PREDICTION_COL} = s.{PREDICTION_COL},
                                     t.{SCORED_AT_COL} = s.{SCORED_AT_COL}
        WHEN NOT MATCHED THEN INSERT *"""
)
display(result)
