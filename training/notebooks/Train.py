# Databricks notebook source
# MAGIC %md
# MAGIC # Train used car price model (Feature Store backed)
# MAGIC
# MAGIC Trains a gradient-boosted regressor on `raw_listings` joined point-in-time with the
# MAGIC `market_price_features` feature table, tracks the run in MLflow and registers a new
# MAGIC version of the Unity Catalog model with `FeatureEngineeringClient.log_model`.
# MAGIC
# MAGIC * **Feature Store lineage**: the model records which feature table/columns it uses, so
# MAGIC   `score_batch` and Model Serving look features up automatically — callers only send the
# MAGIC   listing attributes (no training/serving skew).
# MAGIC * **Point-in-time correctness**: `timestamp_lookup_key=listing_date` joins the latest
# MAGIC   feature row with `index_date <= listing_date`; the window itself excludes that day.
# MAGIC * **Time-based holdout**: the last `holdout_days` days are never seen in training.
# MAGIC * Compute: serverless in dev/test; a dedicated single-node **job cluster** in staging/prod
# MAGIC   (CPU by default; set `training_node_type` / `training_spark_version` to a GPU node and a
# MAGIC   `-gpu-ml-` runtime for deep learning models).

# COMMAND ----------

# MAGIC %pip install -q -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

SRC = os.path.abspath(os.path.join(os.getcwd(), "../../src"))
sys.path.insert(0, SRC)

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow import MlflowClient
from pyspark.sql import functions as F

from used_car_price import __version__
from used_car_price.data_quality import filter_valid
from used_car_price.schema import (
    DATE_COL,
    ID_COL,
    LABEL_COL,
    LISTING_MONTH_COL,
    LISTING_YEAR_COL,
    MARKET_FEATURE_COLS,
    MARKET_FEATURE_TABLE,
    MARKET_KEY_COLS,
    RAW_COLS,
)
from used_car_price.training import (
    SKOPS_TRUSTED_TYPES,
    TrainingParams,
    evaluate_model,
    holdout_cutoff,
    train_model,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "used_car_price_dev")
dbutils.widgets.text("model_name", "workspace.used_car_price_dev.used_car_price_model")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("training_lookback_days", "365")
dbutils.widgets.text("holdout_days", "30")
dbutils.widgets.text("learning_rate", "0.08")
dbutils.widgets.text("max_iter", "400")
dbutils.widgets.text("max_leaf_nodes", "31")
dbutils.widgets.text("trigger_reason", "scheduled")

w = dbutils.widgets.get
catalog, schema = w("catalog"), w("schema")
raw_table = f"{catalog}.{schema}.raw_listings"
feature_table = f"{catalog}.{schema}.{MARKET_FEATURE_TABLE}"
model_name = w("model_name")
holdout_days = int(w("holdout_days"))
params = TrainingParams(
    learning_rate=float(w("learning_rate")),
    max_iter=int(w("max_iter")),
    max_leaf_nodes=int(w("max_leaf_nodes")),
)

mlflow.set_registry_uri("databricks-uc")
if w("experiment_name"):
    mlflow.set_experiment(w("experiment_name"))
fe = FeatureEngineeringClient()

# COMMAND ----------

# Pin the Delta version so the run is reproducible and lineage points at an exact snapshot.
table_version = spark.sql(f"DESCRIBE HISTORY {raw_table} LIMIT 1").select("version").first()[0]
raw_sdf = spark.read.option("versionAsOf", table_version).table(raw_table)
max_date = raw_sdf.agg(F.max(DATE_COL)).first()[0]
holdout_start = holdout_cutoff(max_date, holdout_days)

window_sdf = raw_sdf.where(
    F.col(DATE_COL) > F.date_sub(F.lit(max_date), int(w("training_lookback_days")))
).select(*RAW_COLS)
# Data quality gate (same rules as ingestion) applied to the labelled training window.
labelled = spark.createDataFrame(filter_valid(window_sdf.toPandas()), schema=window_sdf.schema)
labelled = labelled.withColumns(
    {
        LISTING_YEAR_COL: F.year(DATE_COL).cast("long"),
        LISTING_MONTH_COL: F.month(DATE_COL).cast("long"),
    }
)

feature_lookups = [
    FeatureLookup(
        table_name=feature_table,
        lookup_key=MARKET_KEY_COLS,
        timestamp_lookup_key=DATE_COL,
        feature_names=MARKET_FEATURE_COLS,
    )
]


def training_set_for(sdf):
    return fe.create_training_set(
        df=sdf,
        feature_lookups=feature_lookups,
        label=LABEL_COL,
        # listing_date is only the point-in-time key; the model uses listing_year/month.
        exclude_columns=[ID_COL, DATE_COL],
    )


train_set = training_set_for(labelled.where(F.col(DATE_COL) < F.lit(holdout_start)))
holdout_set = training_set_for(labelled.where(F.col(DATE_COL) >= F.lit(holdout_start)))
train_df = train_set.load_df().toPandas()
holdout_df = holdout_set.load_df().toPandas()
if train_df.empty or holdout_df.empty:
    raise ValueError(f"train={len(train_df)} holdout={len(holdout_df)}: ingest more history")
print(f"train={len(train_df):,} holdout={len(holdout_df):,} holdout_start={holdout_start}")

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.log_input(
        mlflow.data.from_spark(window_sdf, table_name=raw_table, version=str(table_version)),
        context="training",
    )
    mlflow.log_params(
        {
            **params.to_dict(),
            "holdout_days": holdout_days,
            "holdout_start": str(holdout_start),
            "train_rows": len(train_df),
            "holdout_rows": len(holdout_df),
            "source_table_version": table_version,
            "feature_table": feature_table,
            "code_version": __version__,
        }
    )
    mlflow.set_tag("trigger_reason", w("trigger_reason"))

    model = train_model(train_df, params)
    metrics = evaluate_model(model, holdout_df)
    mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()})
    print(metrics)

    # Packages the model with its feature lookups: scoring needs only the request columns.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=train_set,
        registered_model_name=model_name,
        infer_input_example=True,
        code_paths=[os.path.join(SRC, "used_car_price")],
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

# The UC registry only supports filtering by name; match the run id client-side.
versions = MlflowClient().search_model_versions(f"name='{model_name}'")
model_version = max(int(v.version) for v in versions if v.run_id == run.info.run_id)
print(f"Registered {model_name} version {model_version} (run {run.info.run_id})")

# COMMAND ----------

dbutils.jobs.taskValues.set("model_version", str(model_version))
dbutils.jobs.taskValues.set("run_id", run.info.run_id)
dbutils.jobs.taskValues.set("holdout_start", str(holdout_start))
dbutils.jobs.taskValues.set("holdout_end", str(max_date))
dbutils.jobs.taskValues.set("source_table_version", str(table_version))
