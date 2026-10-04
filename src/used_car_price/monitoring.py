"""Retraining policy combining performance (concept drift) and data/prediction drift signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from used_car_price.drift import DriftReport


@dataclass(frozen=True)
class RetrainPolicy:
    # Performance: requires ground-truth labels (sale prices) — detects concept drift.
    max_mape: float = 0.12
    min_labeled_rows: int = 200
    # Data drift: input distributions vs the champion's validation data — an early warning
    # available before labels arrive.
    max_feature_psi: float = 0.25
    min_drifted_features: int = 1
    # Prediction drift: output distribution shift.
    max_prediction_psi: float = 0.25
    min_rows_for_drift: int = 200
    # Guard rail against retraining loops when drift persists.
    cooldown_days: int = 3


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    triggers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def alert(self) -> bool:
        """Something breached, regardless of whether retraining was allowed."""
        return bool(self.triggers)


def decide_retraining(
    metrics: dict[str, float] | None,
    labeled_rows: int,
    drift: DriftReport | None,
    scored_rows: int,
    days_since_last_training: float | None,
    policy: RetrainPolicy,
) -> RetrainDecision:
    triggers: list[str] = []
    reasons: list[str] = []

    if metrics is None or labeled_rows < policy.min_labeled_rows:
        reasons.append(f"performance: {labeled_rows} labeled rows < {policy.min_labeled_rows}")
    elif metrics["mape"] > policy.max_mape:
        triggers.append("performance")
        reasons.append(f"performance: mape {metrics['mape']:.4f} > {policy.max_mape}")
    else:
        reasons.append(f"performance: mape {metrics['mape']:.4f} ok")

    if drift is None or scored_rows < policy.min_rows_for_drift:
        reasons.append(f"drift: {scored_rows} scored rows < {policy.min_rows_for_drift}")
    else:
        drifted = [c for c, v in drift.feature_psi.items() if v == v and v > policy.max_feature_psi]
        if len(drifted) >= policy.min_drifted_features:
            triggers.append("data_drift")
            reasons.append(f"data drift: PSI > {policy.max_feature_psi} for {drifted}")
        else:
            reasons.append(f"data drift: max PSI {drift.max_feature_psi:.3f} ok")
        pred_psi = drift.prediction_psi
        if pred_psi is not None and pred_psi == pred_psi and pred_psi > policy.max_prediction_psi:
            triggers.append("prediction_drift")
            reasons.append(f"prediction drift: PSI {pred_psi:.3f} > {policy.max_prediction_psi}")

    retrain = bool(triggers)
    in_cooldown = (
        days_since_last_training is not None and days_since_last_training < policy.cooldown_days
    )
    if retrain and in_cooldown:
        retrain = False
        reasons.append(
            f"cooldown: last training {days_since_last_training:.1f}d ago "
            f"< {policy.cooldown_days}d; not retraining"
        )
    return RetrainDecision(retrain, triggers, reasons)
