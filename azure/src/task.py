# Databricks notebook source
# MAGIC %pip install mlflow==2.22.2 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 databricks-sdk==0.74.0

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
import sys

sys.path.insert(0, os.getcwd())
from workflow import Workflow

for name, default in {
    "task": "prepare",
    "catalog": "db_azure_workspace",
    "schema": "used_car_dev",
    "endpoint": "used-car-dev",
    "experiment": "/Shared/used-car-mlops/dev",
    "as_of": "",
    "release": "local",
}.items():
    dbutils.widgets.text(name, default)
config = {
    name: dbutils.widgets.get(name)
    for name in ["catalog", "schema", "endpoint", "experiment", "as_of", "release"]
}
task = dbutils.widgets.get("task")
if task not in {"prepare", "train", "validate", "deploy", "inputs", "batch", "monitor"}:
    raise ValueError(f"Unknown task {task}")
workflow = Workflow(spark, dbutils, **config)
result = getattr(workflow, task)()
import json

dbutils.notebook.exit(json.dumps({"task": task, "result": result}))
