"""Logs the model exactly as Train.py does and reloads it as MLflow pyfunc (as Serving does).

Runs locally against a file-based MLflow tracking store; skipped when mlflow isn't installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from used_car_price.schema import LISTING_YEAR_COL
from used_car_price.training import (
    SKOPS_TRUSTED_TYPES,
    TrainingParams,
    to_model_input,
    train_model,
)

mlflow = pytest.importorskip("mlflow")

SRC_PKG = Path(__file__).resolve().parents[2] / "src" / "used_car_price"


def test_logged_model_scores_serving_payload(
    listings_with_market: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    model = train_model(listings_with_market, TrainingParams(max_iter=100))
    sample = to_model_input(listings_with_market.tail(50))

    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=mlflow.models.infer_signature(sample, model.predict(sample)),
            input_example=sample.head(3),
            code_paths=[os.fspath(SRC_PKG)],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    assert loaded.metadata.get_input_schema() is not None

    # Same records a client would POST to a serving endpoint as dataframe_records.
    records = sample.head(5).to_dict(orient="records")
    preds = loaded.predict(pd.DataFrame(records))
    assert len(preds) == 5
    assert (preds > 0).all()
    assert isinstance(records[0][LISTING_YEAR_COL], int)
