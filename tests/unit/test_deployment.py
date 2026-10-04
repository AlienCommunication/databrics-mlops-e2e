from __future__ import annotations

import pytest

from used_car_price.deployment import decide_promotion


def test_promotes_without_champion() -> None:
    assert decide_promotion({"mape": 0.1}, None).promote


def test_lower_is_better_metric() -> None:
    assert decide_promotion({"mape": 0.09}, {"mape": 0.10}).promote
    assert not decide_promotion({"mape": 0.11}, {"mape": 0.10}).promote


def test_required_improvement_margin() -> None:
    champion = {"mape": 0.100}
    assert not decide_promotion({"mape": 0.0995}, champion, min_relative_improvement=0.01).promote
    assert decide_promotion({"mape": 0.098}, champion, min_relative_improvement=0.01).promote


def test_negative_margin_tolerates_small_regression() -> None:
    assert decide_promotion(
        {"mape": 0.101}, {"mape": 0.100}, min_relative_improvement=-0.02
    ).promote


def test_higher_is_better_metric() -> None:
    assert decide_promotion({"r2": 0.95}, {"r2": 0.90}, metric="r2").promote
    assert not decide_promotion({"r2": 0.85}, {"r2": 0.90}, metric="r2").promote


def test_unknown_metric() -> None:
    with pytest.raises(ValueError):
        decide_promotion({"foo": 1}, {"foo": 1}, metric="foo")
