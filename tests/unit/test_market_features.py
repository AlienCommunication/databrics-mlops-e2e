from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from used_car_price.market_features import (
    OUTPUT_COLS,
    compute_market_index,
)


def _listing(day: dt.date, price: float, make: str = "Toyota") -> dict[str, object]:
    return {
        "listing_id": f"{make}-{day}-{price}",
        "listing_date": day,
        "make": make,
        "model": "RAV4",
        "region": "west",
        "year": day.year - 2,
        "mileage": 20_000,
        "price": price,
    }


D = dt.date(2026, 5, 10)


def test_window_excludes_index_date_itself() -> None:
    listings = pd.DataFrame([_listing(D - dt.timedelta(days=1), 10_000), _listing(D, 99_999)])
    out = compute_market_index(listings, D, D, window_days=30)
    row = out.iloc[0]
    assert row["market_listing_count_30d"] == 1
    assert row["market_avg_log_price_30d"] == pytest.approx(np.log(10_000))


def test_window_drops_old_listings() -> None:
    listings = pd.DataFrame(
        [_listing(D - dt.timedelta(days=31), 50_000), _listing(D - dt.timedelta(days=2), 20_000)]
    )
    row = compute_market_index(listings, D, D, window_days=30).iloc[0]
    assert row["market_listing_count_30d"] == 1
    assert row["market_avg_log_price_30d"] == pytest.approx(np.log(20_000))


def test_empty_window_gives_null_features() -> None:
    listings = pd.DataFrame([_listing(D - dt.timedelta(days=60), 10_000)])
    row = compute_market_index(listings, D, D, window_days=30).iloc[0]
    assert row["market_listing_count_30d"] == 0
    assert np.isnan(row["market_avg_log_price_30d"])


def test_one_row_per_group_and_date() -> None:
    listings = pd.DataFrame(
        [_listing(D - dt.timedelta(days=1), 10_000), _listing(D, 30_000, make="Honda")]
    )
    out = compute_market_index(listings, D, D + dt.timedelta(days=4))
    assert list(out.columns) == OUTPUT_COLS
    assert len(out) == 2 * 5
    assert not out.duplicated(["make", "model", "region", "index_date"]).any()


def test_invalid_range() -> None:
    with pytest.raises(ValueError):
        compute_market_index(pd.DataFrame([_listing(D, 1)]), D, D - dt.timedelta(days=1))


def test_point_in_time_join_never_uses_future(listings_with_market: pd.DataFrame) -> None:
    # Every listing's market window ended strictly before its own date, so no listing's own
    # price can be included: verify on a listing whose group had no prior data.
    first_day = listings_with_market["listing_date"].min()
    first = listings_with_market[listings_with_market["listing_date"] == first_day]
    assert first["market_avg_log_price_30d"].isna().all()
    later = listings_with_market[listings_with_market["listing_date"] > first_day]
    assert later["market_avg_log_price_30d"].notna().mean() > 0.9
