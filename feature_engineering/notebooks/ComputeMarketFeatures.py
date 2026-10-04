# Databricks notebook source
# MAGIC %md
# MAGIC # Compute market price features (Feature Engineering in Unity Catalog)
# MAGIC
# MAGIC Maintains the `market_price_features` **time-series feature table**, keyed by
# MAGIC `(make, model, region, index_date)`. Each row summarises the trailing 30 days of listings
# MAGIC *before* `index_date`, so point-in-time lookups during training never leak labels.
# MAGIC
# MAGIC * First run (table missing): backfills every date covered by `raw_listings`.
# MAGIC * Daily runs: recompute the last `backfill_days` index dates and `MERGE` (idempotent).
# MAGIC * If `online_store_name` is set, the table is published to a Databricks Online Feature
# MAGIC   Store so Model Serving can look features up in real time.

# COMMAND ----------

# MAGIC %pip install -q -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
from pyspark.sql import types as T

from used_car_price.market_features import WINDOW_DAYS, compute_market_index
from used_car_price.schema import DATE_COL, LABEL_COL, MARKET_FEATURE_TABLE, MARKET_KEY_COLS, MARKET_TS_COL

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "used_car_price_dev")
dbutils.widgets.text("run_date", "")
dbutils.widgets.text("backfill_days", "3")
dbutils.widgets.text("online_store_name", "")

w = dbutils.widgets.get
catalog, schema = w("catalog"), w("schema")
raw_table = f"{catalog}.{schema}.raw_listings"
feature_table = f"{catalog}.{schema}.{MARKET_FEATURE_TABLE}"
run_date = dt.date.fromisoformat(w("run_date")) if w("run_date") else dt.date.today()
fe = FeatureEngineeringClient()

# COMMAND ----------

table_exists = spark.catalog.tableExists(feature_table)
if table_exists:
    start_date = run_date - dt.timedelta(days=int(w("backfill_days")) - 1)
else:
    start_date = spark.table(raw_table).agg(F.min(DATE_COL)).first()[0] + dt.timedelta(days=1)
print(f"Computing {feature_table} for index dates {start_date}..{run_date}")

OUTPUT_SCHEMA = T.StructType(
    [
        T.StructField("make", T.StringType(), False),
        T.StructField("model", T.StringType(), False),
        T.StructField("region", T.StringType(), False),
        T.StructField(MARKET_TS_COL, T.DateType(), False),
        T.StructField("market_listing_count_30d", T.LongType()),
        T.StructField("market_avg_log_price_30d", T.DoubleType()),
        T.StructField("market_avg_vehicle_age_30d", T.DoubleType()),
        T.StructField("market_avg_log_mileage_30d", T.DoubleType()),
    ]
)


def per_group(pdf):
    # Same tested pandas implementation, distributed per (make, model, region) group.
    return compute_market_index(pdf, start_date, run_date, WINDOW_DAYS)


features = (
    spark.table(raw_table)
    .where(
        F.col(DATE_COL).between(start_date - dt.timedelta(days=WINDOW_DAYS), run_date)
        & F.col(LABEL_COL).isNotNull()
    )
    .select(*MARKET_KEY_COLS, DATE_COL, LABEL_COL, "year", "mileage")
    .groupBy(*MARKET_KEY_COLS)
    .applyInPandas(per_group, schema=OUTPUT_SCHEMA)
)

# COMMAND ----------

if not table_exists:
    fe.create_table(
        name=feature_table,
        primary_keys=[*MARKET_KEY_COLS, MARKET_TS_COL],
        timeseries_columns=[MARKET_TS_COL],
        schema=OUTPUT_SCHEMA,
        description=(
            "Trailing 30-day market statistics per make/model/region, excluding index_date "
            "itself. Used via point-in-time lookup on listing_date."
        ),
    )
    # Change Data Feed is required to publish to the online store.
    spark.sql(f"ALTER TABLE {feature_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

fe.write_table(name=feature_table, df=features, mode="merge")
print(f"Merged features into {feature_table}")

# COMMAND ----------

# MAGIC %md ## Optional: publish to the Online Feature Store (needed for real-time serving)

# COMMAND ----------

store_name = w("online_store_name")
if store_name:
    store = fe.get_online_store(name=store_name)
    if store is None:
        print(f"Creating online store {store_name} (Lakebase, CU_1)")
        fe.create_online_store(name=store_name, capacity="CU_1")
        store = fe.get_online_store(name=store_name)
    fe.publish_table(
        online_store=store,
        source_table_name=feature_table,
        online_table_name=f"{feature_table}_online",
    )
    print(f"Published {feature_table} -> {feature_table}_online")
else:
    print("Online store disabled (online_store_name is empty)")
