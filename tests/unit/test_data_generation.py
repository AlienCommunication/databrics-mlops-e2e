from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from used_car_price.data_generation import generate_date_range, generate_listings
from used_car_price.schema import RAW_COLS


def test_generation_is_deterministic_per_date() -> None:
    day = dt.date(2026, 1, 15)
    pd.testing.assert_frame_equal(generate_listings(day, 50), generate_listings(day, 50))


def test_different_dates_produce_different_data() -> None:
    a = generate_listings(dt.date(2026, 1, 15), 50)
    b = generate_listings(dt.date(2026, 1, 16), 50)
    assert not a["price"].equals(b["price"])
    assert set(a["listing_id"]).isdisjoint(b["listing_id"])


def test_schema_and_ranges() -> None:
    df = generate_listings(dt.date(2026, 3, 1), 500)
    assert list(df.columns) == RAW_COLS
    assert df["listing_id"].is_unique
    assert (df["price"] >= 1_000).all()
    assert (df["year"] <= 2026).all()
    assert (df["mileage"] >= 0).all()


def test_market_drift_raises_prices() -> None:
    day = dt.date(2026, 3, 1)
    base = generate_listings(day, 300)["price"].mean()
    drifted = generate_listings(day, 300, market_drift=0.2)["price"].mean()
    assert drifted == pytest.approx(base * 1.2, rel=0.02)


def test_date_range_covers_each_day() -> None:
    df = generate_date_range(dt.date(2026, 3, 10), num_days=10, rows_per_day=5)
    assert len(df) == 50
    assert df["listing_date"].nunique() == 10
    assert df["listing_date"].max() == dt.date(2026, 3, 10)


def test_negative_rows_rejected() -> None:
    with pytest.raises(ValueError):
        generate_listings(dt.date(2026, 3, 1), -1)


def test_fleet_age_shift_makes_cars_older() -> None:
    day = dt.date(2026, 3, 1)
    base = generate_listings(day, 500)["year"].mean()
    older = generate_listings(day, 500, fleet_age_shift=4)["year"].mean()
    assert base - older == pytest.approx(4, abs=0.5)
