"""Used car price prediction — shared, unit-tested logic for the Databricks MLOps pipeline.

Notebooks under ``*/notebooks`` are thin orchestration layers; all business logic lives here
so it can be tested locally and packaged with the model via MLflow ``code_paths``.
"""

__version__ = "0.1.0"
