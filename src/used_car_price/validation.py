"""Model validation gates run before a new model version can become the ``challenger``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ValidationThresholds:
    max_mape: float = 0.12
    max_mae: float = 2_500.0
    min_r2: float = 0.90


@dataclass
class ValidationResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def check_thresholds(
    metrics: dict[str, float], thresholds: ValidationThresholds
) -> ValidationResult:
    failures = []
    if not metrics["mape"] <= thresholds.max_mape:
        failures.append(f"mape {metrics['mape']:.4f} > {thresholds.max_mape}")
    if not metrics["mae"] <= thresholds.max_mae:
        failures.append(f"mae {metrics['mae']:.2f} > {thresholds.max_mae}")
    if not metrics["r2"] >= thresholds.min_r2:
        failures.append(f"r2 {metrics['r2']:.4f} < {thresholds.min_r2}")
    return ValidationResult(passed=not failures, failures=failures)


def check_prediction_sanity(predictions: ArrayLike) -> ValidationResult:
    """Business rules: predictions must be finite and strictly positive."""
    preds = np.asarray(predictions, dtype=float)
    failures = []
    if preds.size == 0:
        failures.append("model produced no predictions")
    if not np.isfinite(preds).all():
        failures.append(f"{int((~np.isfinite(preds)).sum())} non-finite predictions")
    if (preds <= 0).any():
        failures.append(f"{int((preds <= 0).sum())} non-positive predictions")
    return ValidationResult(passed=not failures, failures=failures)


def combine(*results: ValidationResult) -> ValidationResult:
    failures = [f for r in results for f in r.failures]
    return ValidationResult(passed=not failures, failures=failures)
