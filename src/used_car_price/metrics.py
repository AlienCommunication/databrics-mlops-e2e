"""Regression metrics used consistently across training, validation, deployment and monitoring."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return mae, rmse, mape and r2. MAPE ignores rows where the true value is zero."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot compute metrics on empty input")
    nonzero = true != 0
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mape": float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero]))),
        "r2": float(r2_score(true, pred)) if true.size > 1 else float("nan"),
    }


# Direction of improvement for each metric; used when comparing models.
HIGHER_IS_BETTER: dict[str, bool] = {"mae": False, "rmse": False, "mape": False, "r2": True}
