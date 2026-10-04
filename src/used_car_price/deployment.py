"""Champion/challenger promotion logic using Unity Catalog model aliases."""

from __future__ import annotations

from dataclasses import dataclass

from used_car_price.metrics import HIGHER_IS_BETTER

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    *,
    metric: str = "mape",
    min_relative_improvement: float = 0.0,
) -> PromotionDecision:
    """Promote when there is no champion, or the challenger beats it by the required margin.

    ``min_relative_improvement`` of 0.01 means the challenger must be at least 1% better on
    ``metric``; a negative value tolerates a small regression (e.g. to pick up fresher data).
    """
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"unsupported comparison metric: {metric}")
    if champion_metrics is None:
        return PromotionDecision(True, "no existing champion")

    new, old = challenger_metrics[metric], champion_metrics[metric]
    if HIGHER_IS_BETTER[metric]:
        target = old + abs(old) * min_relative_improvement
        better = new >= target
    else:
        target = old - abs(old) * min_relative_improvement
        better = new <= target
    verdict = "beats" if better else "does not beat"
    return PromotionDecision(
        better,
        f"challenger {metric}={new:.4f} {verdict} required {target:.4f} (champion {old:.4f})",
    )
