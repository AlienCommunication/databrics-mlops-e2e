# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest used car listings
# MAGIC
# MAGIC Lands listings into the `raw_listings` Delta table in Unity Catalog.
# MAGIC
# MAGIC * **Source**: deterministic synthetic marketplace feed (default) or a CSV in a UC Volume
# MAGIC   (`source_path`) matching the `raw_listings` schema.
# MAGIC * **Idempotent / backfill-safe**: rows are `MERGE`d on `listing_id`, and synthetic data is
# MAGIC   seeded by date, so re-running a date never duplicates rows.
# MAGIC * **Bootstrap**: if the table does not exist yet, `initial_backfill_days` of history is loaded.
# MAGIC * **Data quality gate**: the run fails if more than `max_invalid_fraction` of rows break
# MAGIC   validation rules; remaining invalid rows are written to `raw_listings_quarantine`.

# COMMAND ----------

# MAGIC %pip install -q -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from used_car_price.data_generation import generate_date_range
from used_car_price.data_quality import filter_valid, validate_listings
from used_car_price.schema import ID_COL, RAW_COLS

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "used_car_price_dev")
dbutils.widgets.text("run_date", "")
dbutils.widgets.text("backfill_days", "1")
dbutils.widgets.text("initial_backfill_days", "365")
dbutils.widgets.text("rows_per_day", "200")
dbutils.widgets.text("market_drift", "0.0")
dbutils.widgets.text("fleet_age_shift", "0.0")
dbutils.widgets.text("source_path", "")
dbutils.widgets.text("max_invalid_fraction", "0.02")

w = dbutils.widgets.get
catalog, schema = w("catalog"), w("schema")
run_date = dt.date.fromisoformat(w("run_date")) if w("run_date") else dt.date.today()
raw_table = f"{catalog}.{schema}.raw_listings"
quarantine_table = f"{catalog}.{schema}.raw_listings_quarantine"

# COMMAND ----------

RAW_DDL = """
    listing_id STRING NOT NULL COMMENT 'Unique marketplace listing identifier',
    listing_date DATE NOT NULL COMMENT 'Date the vehicle was listed/sold',
    make STRING, model STRING, body_type STRING, fuel_type STRING,
    transmission STRING, drivetrain STRING, condition STRING, region STRING,
    year BIGINT, mileage BIGINT, engine_size_l DOUBLE, num_owners BIGINT,
    accident_history BOOLEAN,
    price DOUBLE COMMENT 'Sale price in USD (label)'
"""

table_exists = spark.catalog.tableExists(raw_table)
spark.sql(
    f"""CREATE TABLE IF NOT EXISTS {raw_table} ({RAW_DDL})
        CLUSTER BY (listing_date)
        COMMENT 'Used car listings with sale price — source for training and scoring'"""
)

# COMMAND ----------

if w("source_path"):
    pdf = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(w("source_path"))
        .toPandas()
    )
else:
    num_days = int(w("backfill_days")) if table_exists else int(w("initial_backfill_days"))
    pdf = generate_date_range(
        run_date,
        num_days=num_days,
        rows_per_day=int(w("rows_per_day")),
        market_drift=float(w("market_drift")),
        fleet_age_shift=float(w("fleet_age_shift")),
    )
print(f"Loaded {len(pdf):,} candidate rows (table existed: {table_exists})")

# COMMAND ----------

report = validate_listings(pdf)
print(f"Data quality: {report.invalid_rows}/{report.total_rows} invalid; {report.violations}")
if not report.passed(float(w("max_invalid_fraction"))):
    raise ValueError(f"Data quality gate failed: {report}")

valid = filter_valid(pdf)[RAW_COLS]
invalid = pdf.loc[~pdf[ID_COL].isin(valid[ID_COL])]

# COMMAND ----------

target_schema = spark.table(raw_table).schema
spark.createDataFrame(valid, schema=target_schema).createOrReplaceTempView("incoming")
result = spark.sql(
    f"""MERGE INTO {raw_table} t USING incoming s ON t.listing_id = s.listing_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *"""
)
display(result)

if len(invalid):
    (
        spark.createDataFrame(invalid.astype(str))
        .write.mode("append").option("mergeSchema", True)
        .saveAsTable(quarantine_table)
    )
    print(f"Quarantined {len(invalid)} rows to {quarantine_table}")
