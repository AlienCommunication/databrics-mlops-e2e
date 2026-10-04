from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from used_car_price.features import FEATURE_COLS, add_features
from used_car_price.training import add_calendar_cols


def _row(**overrides: object) -> pd.DataFrame:
    base = {
        "listing_year": 2026,
        "listing_month": 5,
        "make": "BMW",
        "model": "X5",
        "body_type": "suv",
        "fuel_type": "gasoline",
        "transmission": "automatic",
        "drivetrain": "awd",
        "condition": "Good",
        "region": "west",
        "year": 2022,
        "mileage": 80_000,
        "engine_size_l": 3.0,
        "num_owners": 2,
        "accident_history": True,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_feature_columns_and_values() -> None:
    feats = add_features(_row())
    assert list(feats.columns) == FEATURE_COLS
    row = feats.iloc[0]
    assert row["vehicle_age"] == 4
    assert row["miles_per_year"] == pytest.approx(20_000)
    assert row["is_high_mileage"] == 1.0
    assert row["is_luxury"] == 1.0
    assert row["condition_score"] == 3.0
    assert row["accident_history"] == 1.0
    assert row["listing_month"] == 5.0


def test_brand_new_car_has_finite_mileage_rate() -> None:
    row = add_features(_row(year=2026, mileage=3_000)).iloc[0]
    assert row["vehicle_age"] == 0
    assert row["miles_per_year"] == pytest.approx(6_000)


def test_future_model_year_clipped_to_zero_age() -> None:
    assert add_features(_row(year=2027)).iloc[0]["vehicle_age"] == 0


def test_market_features_pass_through_or_default_to_nan() -> None:
    assert np.isnan(add_features(_row()).iloc[0]["market_avg_log_price_30d"])
    feats = add_features(_row(market_avg_log_price_30d=10.5, market_listing_count_30d=12))
    assert feats.iloc[0]["market_avg_log_price_30d"] == 10.5
    assert feats.iloc[0]["market_listing_count_30d"] == 12.0


def test_does_not_mutate_input(listings_with_market: pd.DataFrame) -> None:
    df = add_calendar_cols(listings_with_market)
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)
