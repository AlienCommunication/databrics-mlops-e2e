from __future__ import annotations

import datetime as dt
import pickle

import pandas as pd
import pytest

from used_car_price.schema import MARKET_FEATURE_COLS, MODEL_INPUT_COLS
from used_car_price.training import (
    TrainingParams,
    evaluate_model,
    holdout_cutoff,
    time_based_split,
    to_model_input,
    train_model,
)

FAST_PARAMS = TrainingParams(max_iter=150)


def test_holdout_cutoff() -> None:
    assert holdout_cutoff(dt.date(2026, 6, 30), 1) == dt.date(2026, 6, 30)
    assert holdout_cutoff(dt.date(2026, 6, 30), 30) == dt.date(2026, 6, 1)
    with pytest.raises(ValueError):
        holdout_cutoff(dt.date(2026, 6, 30), 0)


def test_time_split_has_no_leakage(listings: pd.DataFrame) -> None:
    train, holdout, cutoff = time_based_split(listings, holdout_days=21)
    assert train["listing_date"].max() < cutoff <= holdout["listing_date"].min()
    assert len(train) + len(holdout) == len(listings)
    assert holdout["listing_date"].nunique() == 21


def test_time_split_rejects_empty_partition(listings: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        time_based_split(listings, holdout_days=1_000)


def test_model_learns_signal(listings_with_market: pd.DataFrame) -> None:
    train, holdout, _ = time_based_split(listings_with_market, holdout_days=21)
    model = train_model(train, FAST_PARAMS)
    metrics = evaluate_model(model, holdout)
    assert metrics["r2"] > 0.85
    assert metrics["mape"] < 0.15


def test_model_handles_unseen_categories_and_round_trips(
    listings_with_market: pd.DataFrame,
) -> None:
    model = train_model(listings_with_market, FAST_PARAMS)
    unseen = listings_with_market.head(3).copy()
    unseen["make"] = "Rivian"
    unseen["model"] = "R1S"
    unseen[MARKET_FEATURE_COLS] = float("nan")  # no market history for a new vehicle
    restored = pickle.loads(pickle.dumps(model))
    preds = restored.predict(to_model_input(unseen))
    assert (preds > 0).all()


def test_to_model_input_contract(listings: pd.DataFrame) -> None:
    out = to_model_input(listings.head(10))
    assert list(out.columns) == MODEL_INPUT_COLS
    assert out["listing_year"].iloc[0] == listings["listing_date"].iloc[0].year
    assert str(out["year"].dtype) == "int64"
    assert out["accident_history"].dtype == bool
    # Market features absent from the input become NaN instead of failing.
    assert out[MARKET_FEATURE_COLS].isna().all().all()


def test_market_features_improve_accuracy_under_market_shift(listings: pd.DataFrame) -> None:
    """With a market-wide price jump, market features let the model adapt without retraining."""
    import datetime as dt_

    from used_car_price.data_generation import generate_date_range
    from used_car_price.market_features import compute_market_index, point_in_time_join

    history = listings
    shifted = generate_date_range(dt_.date(2026, 8, 31), 62, 60, market_drift=0.2)
    everything = pd.concat([history, shifted], ignore_index=True)
    index = compute_market_index(everything, history["listing_date"].min(), dt_.date(2026, 8, 31))
    joined = point_in_time_join(everything, index)

    train = joined[pd.to_datetime(joined["listing_date"]) <= "2026-06-30"]
    test = joined[pd.to_datetime(joined["listing_date"]) >= "2026-08-01"]
    with_market = evaluate_model(train_model(train, FAST_PARAMS), test)["mape"]
    without = evaluate_model(
        train_model(train.drop(columns=MARKET_FEATURE_COLS), FAST_PARAMS),
        test.drop(columns=MARKET_FEATURE_COLS),
    )["mape"]
    assert with_market < without
