# Databricks notebook source
# MAGIC %md
# MAGIC # Used-car MLOps: inspect a completed run
# MAGIC Select an environment below. Run after its training and daily jobs succeed.
# MAGIC Prices and outcomes are synthetic; this notebook explains operational evidence.

# COMMAND ----------
import json

dbutils.widgets.dropdown("environment", "dev", ["dev", "staging", "prod"])
env = dbutils.widgets.get("environment")
ns = f"db_azure_workspace.used_car_{env}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data and model inputs
# MAGIC Check time ranges, numeric types, and the absence of sale price in scoring inputs.

# COMMAND ----------
display(spark.table(f"{ns}.features").limit(20))
display(spark.table(f"{ns}.scoring_inputs").limit(20))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Predictions with exact model lineage
# MAGIC Rerun the daily job for the same date. The duplicate query must still return no rows.

# COMMAND ----------
display(spark.table(f"{ns}.predictions").orderBy("scoring_date", ascending=False).limit(50))
display(
    spark.sql(f"""SELECT car_id, scoring_date, model_version, count(*) AS n
FROM {ns}.predictions GROUP BY car_id, scoring_date, model_version HAVING count(*) > 1""")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Drift versus accuracy
# MAGIC `drift_json` compares input distributions. `metrics_json` uses only outcomes available
# MAGIC by the scoring date. A two-day label delay explains why label_count is below prediction_count.

# COMMAND ----------
display(spark.table(f"{ns}.monitoring").orderBy("monitor_date", ascending=False))
if spark.catalog.tableExists(f"{ns}.online_monitoring"):
    display(spark.table(f"{ns}.online_monitoring"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Online request evidence
# MAGIC Send a request with scripts/predict.py, then inspect asynchronous endpoint logs.

# COMMAND ----------
if spark.catalog.tableExists(f"{ns}.online_payload"):
    display(spark.table(f"{ns}.online_payload").limit(20))

# COMMAND ----------
# A compact, machine-readable record when this notebook is run as a verification job.
dbutils.notebook.exit(
    json.dumps(
        {
            "environment": env,
            "monitoring": [r.asDict() for r in spark.table(f"{ns}.monitoring").collect()],
            "online_monitoring": [
                r.asDict() for r in spark.table(f"{ns}.online_monitoring").collect()
            ],
            "logged_requests": spark.table(f"{ns}.online_payload").count(),
            "duplicate_keys": spark.sql(
                f"SELECT car_id, scoring_date, model_version, count(*) AS n FROM {ns}.predictions "
                "GROUP BY car_id, scoring_date, model_version HAVING count(*) > 1"
            ).count(),
        },
        default=str,
    )
)
