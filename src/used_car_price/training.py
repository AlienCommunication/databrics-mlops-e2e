"""Model definition and time-aware train/holdout split."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from used_car_price.features import (
    ENGINEERED_NUMERIC_COLS,
    FEATURE_CATEGORICAL_COLS,
    add_features,
)
from used_car_price.metrics import regression_metrics
from used_car_price.schema import (
    DATE_COL,
    LABEL_COL,
    LISTING_MONTH_COL,
    LISTING_YEAR_COL,
    MARKET_FEATURE_COLS,
    MODEL_INPUT_COLS,
)

# Types MLflow's skops serializer must be told to trust when loading this model. Keep the list
# minimal and explicit rather than disabling secure deserialization.
SKOPS_TRUSTED_TYPES: list[str] = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "used_car_price.features.add_features",
]


@dataclass(frozen=True)
class TrainingParams:
    learning_rate: float = 0.08
    max_iter: int = 400
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1
    early_stopping: bool = True
    random_state: int = 42

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline(params: TrainingParams) -> TransformedTargetRegressor:
    """Raw listing columns in, price out. Trains on log(price) for multiplicative errors."""
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", ENGINEERED_NUMERIC_COLS),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
                FEATURE_CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=params.learning_rate,
        max_iter=params.max_iter,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        early_stopping=params.early_stopping,
        random_state=params.random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocess", preprocess),
            ("regressor", regressor),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def holdout_cutoff(max_date: dt.date, holdout_days: int) -> dt.date:
    """First date of the holdout window: the last ``holdout_days`` days up to ``max_date``."""
    if holdout_days < 1:
        raise ValueError("holdout_days must be >= 1")
    return max_date - dt.timedelta(days=holdout_days - 1)


def time_based_split(
    df: pd.DataFrame, holdout_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, dt.date]:
    """Split so the model is always evaluated on listings newer than anything it trained on."""
    dates = pd.to_datetime(df[DATE_COL]).dt.date
    cutoff = holdout_cutoff(dates.max(), holdout_days)
    train, holdout = df.loc[dates < cutoff], df.loc[dates >= cutoff]
    if train.empty or holdout.empty:
        raise ValueError(
            f"time split at {cutoff} produced train={len(train)} holdout={len(holdout)} rows; "
            "ingest more history or reduce holdout_days"
        )
    return train.reset_index(drop=True), holdout.reset_index(drop=True), cutoff


def add_calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Derive listing_year / listing_month from listing_date (pandas twin of the Spark logic)."""
    dates = pd.to_datetime(df[DATE_COL])
    return df.assign(**{LISTING_YEAR_COL: dates.dt.year, LISTING_MONTH_COL: dates.dt.month})


def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame to the model's input contract (64-bit numerics, float market features).

    Market features may be absent (e.g. a new make/model/region); they become NaN, which the
    gradient-boosted model handles natively.
    """
    if LISTING_YEAR_COL not in df.columns:
        df = add_calendar_cols(df)
    out = df.reindex(columns=MODEL_INPUT_COLS)
    for col in (LISTING_YEAR_COL, LISTING_MONTH_COL, "year", "mileage", "num_owners"):
        out[col] = out[col].astype("int64")
    for col in ("engine_size_l", *MARKET_FEATURE_COLS):
        out[col] = out[col].astype("float64")
    out["accident_history"] = out["accident_history"].astype(bool)
    return out


def train_model(train_df: pd.DataFrame, params: TrainingParams) -> TransformedTargetRegressor:
    model = build_pipeline(params)
    model.fit(to_model_input(train_df), train_df[LABEL_COL].astype(float))
    return model


def evaluate_model(model: object, df: pd.DataFrame) -> dict[str, float]:
    preds = model.predict(to_model_input(df))  # type: ignore[attr-defined]
    return regression_metrics(df[LABEL_COL].astype(float), preds)
