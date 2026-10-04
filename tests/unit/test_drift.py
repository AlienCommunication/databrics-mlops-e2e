from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from used_car_price.drift import categorical_psi, compute_drift, numeric_psi

rng = np.random.default_rng(0)


def test_numeric_psi_stable_vs_shifted() -> None:
    ref = pd.Series(rng.normal(0, 1, 5_000))
    same = pd.Series(rng.normal(0, 1, 5_000))
    shifted = pd.Series(rng.normal(1, 1, 5_000))
    assert numeric_psi(ref, same) < 0.02
    assert numeric_psi(ref, shifted) > 0.25


def test_numeric_psi_constant_reference() -> None:
    ref = pd.Series([1.0] * 100)
    assert numeric_psi(ref, pd.Series([1.0] * 100)) == pytest.approx(0, abs=1e-6)
    assert numeric_psi(ref, pd.Series([2.0] * 100)) > 0.25


def test_numeric_psi_empty_is_nan() -> None:
    assert np.isnan(numeric_psi(pd.Series([], dtype=float), pd.Series([1.0])))


def test_categorical_psi() -> None:
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    assert categorical_psi(ref, pd.Series(["a"] * 50 + ["b"] * 50)) == pytest.approx(0)
    assert categorical_psi(ref, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.25
    # A brand-new category is a strong drift signal.
    assert categorical_psi(ref, pd.Series(["c"] * 100)) > 1


def test_compute_drift_flags_only_shifted_features() -> None:
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2_000), "c": ["a", "b"] * 1_000})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 2_000), "c": ["a", "b"] * 1_000})
    ref["pred"] = ref["x"] * 2
    cur["pred"] = cur["x"] * 2
    report = compute_drift(
        ref, cur, numeric_cols=["x"], categorical_cols=["c"], prediction_col="pred"
    )
    assert report.drifted_features == ["x"]
    assert report.prediction_psi is not None and report.prediction_psi > 0.25
    assert report.max_feature_psi == report.feature_psi["x"]
