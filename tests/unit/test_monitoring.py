from __future__ import annotations

from used_car_price.drift import DriftReport
from used_car_price.monitoring import RetrainPolicy, decide_retraining

POLICY = RetrainPolicy(max_mape=0.12, min_labeled_rows=100, min_rows_for_drift=100, cooldown_days=3)
NO_DRIFT = DriftReport({"mileage": 0.02, "make": 0.01}, prediction_psi=0.03, threshold=0.25)
DATA_DRIFT = DriftReport({"mileage": 0.6, "make": 0.01}, prediction_psi=0.03, threshold=0.25)
PRED_DRIFT = DriftReport({"mileage": 0.05}, prediction_psi=0.4, threshold=0.25)
HEALTHY = {"mape": 0.06}


def decide(metrics=HEALTHY, labeled=500, drift=NO_DRIFT, scored=500, days=10.0):  # type: ignore[no-untyped-def]
    return decide_retraining(metrics, labeled, drift, scored, days, POLICY)


def test_healthy_model_does_not_retrain() -> None:
    d = decide()
    assert not d.retrain and not d.alert


def test_performance_breach_triggers() -> None:
    d = decide(metrics={"mape": 0.2})
    assert d.retrain and d.triggers == ["performance"]


def test_data_drift_triggers_without_labels() -> None:
    d = decide(metrics=None, labeled=0, drift=DATA_DRIFT)
    assert d.retrain and d.triggers == ["data_drift"]


def test_prediction_drift_triggers() -> None:
    d = decide(drift=PRED_DRIFT)
    assert d.retrain and d.triggers == ["prediction_drift"]


def test_cooldown_blocks_retrain_but_still_alerts() -> None:
    d = decide(metrics={"mape": 0.2}, days=1.0)
    assert not d.retrain
    assert d.alert
    assert any("cooldown" in r for r in d.reasons)


def test_too_few_rows_ignores_signals() -> None:
    d = decide(metrics={"mape": 0.9}, labeled=10, drift=DATA_DRIFT, scored=10)
    assert not d.retrain and not d.alert


def test_never_trained_is_not_in_cooldown() -> None:
    assert decide(metrics={"mape": 0.2}, days=None).retrain
