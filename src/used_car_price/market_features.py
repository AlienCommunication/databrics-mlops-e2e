"""Market price index features published to the Databricks Feature Store.

For every (make, model, region) and every ``index_date`` D, statistics are computed over listings
in the trailing window [D - window_days, D - 1]. The window excludes D itself, so a point-in-time
lookup for a listing dated D never sees that day's prices (no label leakage).

These features carry information the request alone cannot: how the market for a given vehicle
is currently pricing, which lets the model track market-wide price shifts between retrains.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from used_car_price.schema import DATE_COL, LABEL_COL, MARKET_KEY_COLS, MARKET_TS_COL

WINDOW_DAYS = 30

OUTPUT_COLS: list[str] = [
    *MARKET_KEY_COLS,
    MARKET_TS_COL,
    "market_listing_count_30d",
    "market_avg_log_price_30d",
    "market_avg_vehicle_age_30d",
    "market_avg_log_mileage_30d",
]


def compute_market_index(
    listings: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Return one row per (make, model, region, index_date) for index dates in [start, end].

    ``listings`` must cover at least [start_date - window_days, end_date - 1] for full windows.
    Groups with no listings in a window get a count of 0 and null averages.
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if listings.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = listings.assign(
        _date=pd.to_datetime(listings[DATE_COL]),
        _log_price=np.log(listings[LABEL_COL].astype(float)),
        _age=(pd.to_datetime(listings[DATE_COL]).dt.year - listings["year"]).clip(lower=0),
        _log_mileage=np.log1p(listings["mileage"].astype(float)),
    )
    daily = (
        df.groupby([*MARKET_KEY_COLS, "_date"])
        .agg(
            n=("_log_price", "size"),
            s_price=("_log_price", "sum"),
            s_age=("_age", "sum"),
            s_mileage=("_log_mileage", "sum"),
        )
        .reset_index()
    )

    index_dates = pd.date_range(start_date, end_date, freq="D")
    calendar = pd.date_range(
        pd.Timestamp(start_date) - pd.Timedelta(days=window_days), pd.Timestamp(end_date)
    )
    frames = []
    for keys, group in daily.groupby(MARKET_KEY_COLS):
        sums = (
            group.set_index("_date")[["n", "s_price", "s_age", "s_mileage"]]
            .reindex(calendar, fill_value=0)
            # Trailing window ending the day *before* each index date.
            .rolling(window_days, min_periods=1)
            .sum()
            .shift(1)
            .reindex(index_dates)
            .fillna(0)
        )
        n = sums["n"]
        has_data = n > 0
        out = pd.DataFrame(
            {
                MARKET_TS_COL: index_dates.date,
                "market_listing_count_30d": n.astype("int64").to_numpy(),
                "market_avg_log_price_30d": (sums["s_price"] / n).where(has_data).to_numpy(),
                "market_avg_vehicle_age_30d": (sums["s_age"] / n).where(has_data).to_numpy(),
                "market_avg_log_mileage_30d": (sums["s_mileage"] / n).where(has_data).to_numpy(),
            }
        )
        for col, value in zip(MARKET_KEY_COLS, keys, strict=True):
            out[col] = value
        frames.append(out)
    return pd.concat(frames, ignore_index=True)[OUTPUT_COLS]


def point_in_time_join(listings: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest market features with ``index_date <= listing_date`` to each listing.

    Local (pandas) equivalent of the Feature Store ``FeatureLookup(timestamp_lookup_key=...)``
    join, used in tests and for offline analysis.
    """
    left = listings.assign(_ts=pd.to_datetime(listings[DATE_COL])).sort_values("_ts")
    right = (
        index.assign(_ts=pd.to_datetime(index[MARKET_TS_COL]))
        .drop(columns=[MARKET_TS_COL])
        .sort_values("_ts")
    )
    joined = pd.merge_asof(left, right, on="_ts", by=MARKET_KEY_COLS, direction="backward")
    return joined.drop(columns=["_ts"]).sort_index()
