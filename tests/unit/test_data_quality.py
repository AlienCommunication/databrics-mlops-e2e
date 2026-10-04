from __future__ import annotations

import pandas as pd

from used_car_price.data_quality import filter_valid, validate_listings


def test_clean_data_passes(listings: pd.DataFrame) -> None:
    report = validate_listings(listings)
    assert report.invalid_rows == 0
    assert report.passed(max_invalid_fraction=0.0)


def test_detects_bad_rows(listings: pd.DataFrame) -> None:
    bad = listings.head(10).copy()
    bad.loc[0, "price"] = -5
    bad.loc[1, "mileage"] = 10_000_000
    bad.loc[2, "condition"] = "mint"
    report = validate_listings(bad)
    assert report.invalid_rows == 3
    assert report.violations == {
        "price_out_of_range": 1,
        "mileage_out_of_range": 1,
        "unknown_condition": 1,
    }
    assert not report.passed(max_invalid_fraction=0.1)
    assert len(filter_valid(bad)) == 7


def test_missing_columns_fail(listings: pd.DataFrame) -> None:
    report = validate_listings(listings.drop(columns=["mileage"]))
    assert report.missing_columns == ["mileage"]
    assert not report.passed(max_invalid_fraction=1.0)


def test_label_optional_for_scoring(listings: pd.DataFrame) -> None:
    unlabeled = listings.drop(columns=["price"])
    assert validate_listings(unlabeled, require_label=False).passed(0.0)
