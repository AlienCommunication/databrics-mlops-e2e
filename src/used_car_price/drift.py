"""Data and prediction drift detection using the Population Stability Index (PSI).

PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
Reference data is the champion's validation window (written at promotion time), so drift is
always measured against what the current model was validated on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPSILON = 1e-4


def _psi_from_proportions(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(ref, EPSILON, None)
    cur = np.clip(cur, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI over quantile bins of the reference distribution."""
    ref = reference.dropna().astype(float).to_numpy()
    cur = current.dropna().astype(float).to_numpy()
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        # Constant reference: any deviation from that constant is drift.
        return _psi_from_proportions(
            np.array([1.0, 0.0]), np.array([np.mean(cur == edges[0]), np.mean(cur != edges[0])])
        )
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, edges)[0] / ref.size
    cur_counts = np.histogram(cur, edges)[0] / cur.size
    return _psi_from_proportions(ref_counts, cur_counts)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category frequencies; unseen categories in ``current`` count as drift."""
    ref = reference.dropna().astype(str).value_counts(normalize=True)
    cur = current.dropna().astype(str).value_counts(normalize=True)
    if ref.empty or cur.empty:
        return float("nan")
    categories = ref.index.union(cur.index)
    return _psi_from_proportions(
        ref.reindex(categories, fill_value=0).to_numpy(),
        cur.reindex(categories, fill_value=0).to_numpy(),
    )


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    prediction_psi: float | None
    threshold: float
    drifted_features: list[str] = field(default_factory=list)

    @property
    def max_feature_psi(self) -> float:
        values = [v for v in self.feature_psi.values() if not np.isnan(v)]
        return max(values) if values else 0.0


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prediction_col: str | None = None,
    threshold: float = 0.25,
) -> DriftReport:
    psi = {c: numeric_psi(reference[c], current[c]) for c in numeric_cols}
    psi.update({c: categorical_psi(reference[c], current[c]) for c in categorical_cols})
    prediction_psi = (
        numeric_psi(reference[prediction_col], current[prediction_col]) if prediction_col else None
    )
    drifted = sorted(c for c, v in psi.items() if not np.isnan(v) and v > threshold)
    return DriftReport(psi, prediction_psi, threshold, drifted)
