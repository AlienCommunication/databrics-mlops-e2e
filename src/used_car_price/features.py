"""Feature engineering for used car listings.

``add_features`` is the single source of truth for request-time derived features. It runs inside
the sklearn pipeline (via ``FunctionTransformer``), so it is applied identically during
training, batch inference and Model Serving. Market features looked up from the feature store
are passed through (NaN when a make/model/region has no recent history; the model handles
missing values natively).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from used_car_price.schema import (
    CATEGORICAL_RAW_COLS,
    CONDITIONS,
    LISTING_MONTH_COL,
    LISTING_YEAR_COL,
    MARKET_FEATURE_COLS,
)

LUXURY_MAKES: frozenset[str] = frozenset({"BMW", "Mercedes-Benz", "Audi", "Lexus", "Tesla"})

CONDITION_SCORE: dict[str, int] = {c: len(CONDITIONS) - i for i, c in enumerate(CONDITIONS)}

HIGH_MILEAGE_PER_YEAR = 15_000

ENGINEERED_NUMERIC_COLS: list[str] = [
    "vehicle_age",
    "mileage",
    "log_mileage",
    "miles_per_year",
    "engine_size_l",
    "num_owners",
    "accident_history",
    "condition_score",
    "is_luxury",
    "is_high_mileage",
    "listing_month",
    *MARKET_FEATURE_COLS,
]

FEATURE_CATEGORICAL_COLS: list[str] = list(CATEGORICAL_RAW_COLS)

FEATURE_COLS: list[str] = [*ENGINEERED_NUMERIC_COLS, *FEATURE_CATEGORICAL_COLS]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame containing exactly ``FEATURE_COLS`` derived from model input columns."""
    listing_year = pd.to_numeric(df[LISTING_YEAR_COL], errors="coerce")
    year = pd.to_numeric(df["year"], errors="coerce")
    mileage = pd.to_numeric(df["mileage"], errors="coerce").clip(lower=0)

    # A car listed in the same model year it was built is treated as half a year old so
    # miles_per_year stays finite and realistic.
    vehicle_age = (listing_year - year).clip(lower=0).astype(float)
    effective_age = vehicle_age.where(vehicle_age > 0, 0.5)
    miles_per_year = mileage / effective_age

    out = pd.DataFrame(index=df.index)
    out["vehicle_age"] = vehicle_age
    out["mileage"] = mileage.astype(float)
    out["log_mileage"] = np.log1p(mileage.astype(float))
    out["miles_per_year"] = miles_per_year.astype(float)
    out["engine_size_l"] = pd.to_numeric(df["engine_size_l"], errors="coerce").astype(float)
    out["num_owners"] = pd.to_numeric(df["num_owners"], errors="coerce").astype(float)
    out["accident_history"] = df["accident_history"].astype(float)
    out["condition_score"] = (
        df["condition"].astype(str).str.lower().map(CONDITION_SCORE).astype(float)
    )
    out["is_luxury"] = df["make"].isin(LUXURY_MAKES).astype(float)
    out["is_high_mileage"] = (miles_per_year > HIGH_MILEAGE_PER_YEAR).astype(float)
    out["listing_month"] = pd.to_numeric(df[LISTING_MONTH_COL], errors="coerce").astype(float)
    for col in MARKET_FEATURE_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        else:
            out[col] = np.nan
    for col in FEATURE_CATEGORICAL_COLS:
        out[col] = df[col].astype(str)
    return out[FEATURE_COLS]
