"""Column contracts shared by ingestion, features, training, inference and monitoring."""

from __future__ import annotations

ID_COL = "listing_id"
DATE_COL = "listing_date"
LABEL_COL = "price"
PREDICTION_COL = "predicted_price"
MODEL_VERSION_COL = "model_version"
SCORED_AT_COL = "scored_at"

# Calendar columns derived from listing_date. The model consumes these (not the date itself) so
# real-time Serving payloads are plain JSON numbers.
LISTING_YEAR_COL = "listing_year"
LISTING_MONTH_COL = "listing_month"

CATEGORICAL_RAW_COLS: list[str] = [
    "make",
    "model",
    "body_type",
    "fuel_type",
    "transmission",
    "drivetrain",
    "condition",
    "region",
]

NUMERIC_RAW_COLS: list[str] = [
    "year",
    "mileage",
    "engine_size_l",
    "num_owners",
    "accident_history",
]

RAW_COLS: list[str] = [ID_COL, DATE_COL, *CATEGORICAL_RAW_COLS, *NUMERIC_RAW_COLS, LABEL_COL]

# --- Feature store: market price index ------------------------------------------------------
MARKET_FEATURE_TABLE = "market_price_features"
MARKET_KEY_COLS: list[str] = ["make", "model", "region"]
MARKET_TS_COL = "index_date"
MARKET_FEATURE_COLS: list[str] = [
    "market_listing_count_30d",
    "market_avg_log_price_30d",
    "market_avg_vehicle_age_30d",
    "market_avg_log_mileage_30d",
]

# Columns the caller supplies (batch or Serving); market features are looked up by the
# feature store using MARKET_KEY_COLS.
REQUEST_COLS: list[str] = [
    LISTING_YEAR_COL,
    LISTING_MONTH_COL,
    *CATEGORICAL_RAW_COLS,
    *NUMERIC_RAW_COLS,
]

# Full model input after feature lookup.
MODEL_INPUT_COLS: list[str] = [*REQUEST_COLS, *MARKET_FEATURE_COLS]

# Columns whose distributions are monitored for data drift.
DRIFT_NUMERIC_COLS: list[str] = ["year", "mileage", "engine_size_l", "num_owners"]
DRIFT_CATEGORICAL_COLS: list[str] = ["make", "body_type", "fuel_type", "condition", "region"]

CONDITIONS: list[str] = ["excellent", "good", "fair", "poor"]

# Shared layout of the `predictions` table and the champion's `drift_baseline` table (identical
# schemas so the baseline can be used directly by Lakehouse Monitoring).
PREDICTION_TABLE_COLS: list[str] = [
    ID_COL,
    DATE_COL,
    *REQUEST_COLS,
    PREDICTION_COL,
    LABEL_COL,
    MODEL_VERSION_COL,
    SCORED_AT_COL,
]
