"""Data quality checks applied at ingestion and before training."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from used_car_price.schema import CONDITIONS, ID_COL, LABEL_COL, RAW_COLS

MIN_YEAR = 1990
MAX_MILEAGE = 500_000
MIN_PRICE = 500.0
MAX_PRICE = 500_000.0


@dataclass
class DataQualityReport:
    total_rows: int
    invalid_rows: int
    missing_columns: list[str] = field(default_factory=list)
    violations: dict[str, int] = field(default_factory=dict)

    @property
    def invalid_fraction(self) -> float:
        return self.invalid_rows / self.total_rows if self.total_rows else 0.0

    def passed(self, max_invalid_fraction: float) -> bool:
        return not self.missing_columns and self.invalid_fraction <= max_invalid_fraction


def _row_rules(df: pd.DataFrame, require_label: bool) -> dict[str, pd.Series]:
    """Row-level rules. Rules whose columns are absent are skipped, so the same checks apply to
    raw listings and to feature-store training sets (which drop the id/date columns)."""
    next_year = dt.date.today().year + 1
    rules = {}
    if ID_COL in df.columns:
        rules["null_listing_id"] = df[ID_COL].isna()
    rules["year_out_of_range"] = ~df["year"].between(MIN_YEAR, next_year)
    rules["mileage_out_of_range"] = ~df["mileage"].between(0, MAX_MILEAGE)
    rules["num_owners_out_of_range"] = ~df["num_owners"].between(1, 20)
    rules["unknown_condition"] = ~df["condition"].astype(str).str.lower().isin(CONDITIONS)
    rules["null_categorical"] = df[["make", "model", "body_type", "fuel_type"]].isna().any(axis=1)
    if require_label:
        rules["price_out_of_range"] = ~df[LABEL_COL].between(MIN_PRICE, MAX_PRICE)
    return rules


def validate_listings(df: pd.DataFrame, *, require_label: bool = True) -> DataQualityReport:
    """Return a report describing schema and row-level rule violations."""
    required = RAW_COLS if require_label else [c for c in RAW_COLS if c != LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return DataQualityReport(total_rows=len(df), invalid_rows=len(df), missing_columns=missing)

    rules = _row_rules(df, require_label)
    invalid = pd.concat(rules.values(), axis=1).any(axis=1)
    return DataQualityReport(
        total_rows=len(df),
        invalid_rows=int(invalid.sum()),
        violations={name: int(mask.sum()) for name, mask in rules.items() if mask.any()},
    )


def filter_valid(df: pd.DataFrame, *, require_label: bool = True) -> pd.DataFrame:
    """Drop rows that violate any row-level rule."""
    rules = _row_rules(df, require_label)
    invalid = pd.concat(rules.values(), axis=1).any(axis=1)
    return df.loc[~invalid].reset_index(drop=True)
