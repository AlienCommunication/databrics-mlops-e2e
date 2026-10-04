from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from used_car_price.data_generation import generate_date_range
from used_car_price.market_features import compute_market_index, point_in_time_join

END_DATE = dt.date(2026, 6, 30)


@pytest.fixture(scope="session")
def listings() -> pd.DataFrame:
    return generate_date_range(END_DATE, num_days=120, rows_per_day=60)


@pytest.fixture(scope="session")
def listings_with_market(listings: pd.DataFrame) -> pd.DataFrame:
    """Listings joined point-in-time with market features, as the Feature Store would."""
    start = listings["listing_date"].min()
    index = compute_market_index(listings, start, END_DATE)
    return point_in_time_join(listings, index)
