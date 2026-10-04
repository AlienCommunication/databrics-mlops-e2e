from __future__ import annotations

import numpy as np
import pytest

from used_car_price.metrics import regression_metrics
from used_car_price.validation import (
    ValidationThresholds,
    check_prediction_sanity,
    check_thresholds,
    combine,
)


def test_regression_metrics() -> None:
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == pytest.approx(15)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mape"] == pytest.approx((0.1 + 0.1) / 2)


def test_regression_metrics_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        regression_metrics([], [])
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def test_thresholds_pass_and_fail() -> None:
    thresholds = ValidationThresholds(max_mape=0.1, max_mae=1000, min_r2=0.9)
    assert check_thresholds({"mape": 0.05, "mae": 500, "r2": 0.95}, thresholds).passed
    result = check_thresholds({"mape": 0.2, "mae": 500, "r2": 0.5}, thresholds)
    assert not result.passed
    assert len(result.failures) == 2


def test_nan_metrics_fail() -> None:
    result = check_thresholds({"mape": float("nan"), "mae": 1, "r2": 1}, ValidationThresholds())
    assert not result.passed


def test_prediction_sanity() -> None:
    assert check_prediction_sanity([1.0, 2.0]).passed
    assert not check_prediction_sanity([]).passed
    assert not check_prediction_sanity([1.0, float("inf")]).passed
    assert not check_prediction_sanity([1.0, -3.0]).passed


def test_combine() -> None:
    result = combine(check_prediction_sanity([1.0]), check_prediction_sanity([-1.0]))
    assert not result.passed
    assert result.failures == ["1 non-positive predictions"]
