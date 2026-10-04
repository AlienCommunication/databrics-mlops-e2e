"""Deterministic synthetic used-car listings.

Stands in for an upstream marketplace feed so the project runs end-to-end in any workspace
without external network access. Each ``listing_date`` produces the same rows on every run
(seeded by date), which keeps ingestion idempotent and backfill-safe.

Two knobs simulate drift so you can exercise monitoring and retraining:

* ``market_drift`` — concept drift: a market-wide price shift (0.15 = prices 15% higher for the
  same car). Inputs look unchanged; only error metrics (once labels arrive) reveal it.
* ``fleet_age_shift`` — data (covariate) drift: listed cars are on average this many years
  older/newer, shifting year/mileage distributions.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from used_car_price.schema import CONDITIONS, RAW_COLS


@dataclass(frozen=True)
class VehicleSpec:
    make: str
    model: str
    body_type: str
    base_msrp: float
    fuel_types: tuple[str, ...]
    engine_sizes: tuple[float, ...]


CATALOG: tuple[VehicleSpec, ...] = (
    VehicleSpec("Toyota", "Corolla", "sedan", 24_000, ("gasoline", "hybrid"), (1.8, 2.0)),
    VehicleSpec("Toyota", "RAV4", "suv", 31_000, ("gasoline", "hybrid"), (2.5,)),
    VehicleSpec("Honda", "Civic", "sedan", 25_000, ("gasoline",), (1.5, 2.0)),
    VehicleSpec("Honda", "CR-V", "suv", 32_000, ("gasoline", "hybrid"), (1.5, 2.0)),
    VehicleSpec("Ford", "F-150", "truck", 45_000, ("gasoline", "diesel"), (2.7, 3.5, 5.0)),
    VehicleSpec("Ford", "Escape", "suv", 29_000, ("gasoline", "hybrid"), (1.5, 2.0)),
    VehicleSpec("Chevrolet", "Silverado", "truck", 44_000, ("gasoline", "diesel"), (2.7, 5.3)),
    VehicleSpec("Chevrolet", "Malibu", "sedan", 25_000, ("gasoline",), (1.5,)),
    VehicleSpec("Nissan", "Altima", "sedan", 26_000, ("gasoline",), (2.0, 2.5)),
    VehicleSpec("Hyundai", "Tucson", "suv", 28_000, ("gasoline", "hybrid"), (2.5,)),
    VehicleSpec("Subaru", "Outback", "wagon", 30_000, ("gasoline",), (2.4, 2.5)),
    VehicleSpec("BMW", "3 Series", "sedan", 45_000, ("gasoline",), (2.0, 3.0)),
    VehicleSpec("BMW", "X5", "suv", 65_000, ("gasoline", "hybrid"), (3.0, 4.4)),
    VehicleSpec("Mercedes-Benz", "C-Class", "sedan", 47_000, ("gasoline",), (2.0, 3.0)),
    VehicleSpec("Audi", "Q5", "suv", 46_000, ("gasoline", "hybrid"), (2.0,)),
    VehicleSpec("Lexus", "RX", "suv", 50_000, ("gasoline", "hybrid"), (2.4, 3.5)),
    VehicleSpec("Tesla", "Model 3", "sedan", 42_000, ("electric",), (0.0,)),
    VehicleSpec("Tesla", "Model Y", "suv", 48_000, ("electric",), (0.0,)),
)

REGIONS: dict[str, float] = {
    "northeast": 1.03,
    "southeast": 0.98,
    "midwest": 0.96,
    "southwest": 1.00,
    "west": 1.05,
}
CONDITION_MULTIPLIER: dict[str, float] = {
    "excellent": 1.08,
    "good": 1.00,
    "fair": 0.88,
    "poor": 0.72,
}
CONDITION_PROBS: tuple[float, ...] = (0.2, 0.5, 0.22, 0.08)
TRANSMISSIONS: tuple[str, ...] = ("automatic", "manual")
DRIVETRAINS: dict[str, tuple[str, ...]] = {
    "sedan": ("fwd", "awd", "rwd"),
    "suv": ("awd", "fwd"),
    "truck": ("4wd", "rwd"),
    "wagon": ("awd",),
}
DRIVETRAIN_MULTIPLIER: dict[str, float] = {"fwd": 1.0, "rwd": 1.02, "awd": 1.05, "4wd": 1.07}
FUEL_MULTIPLIER: dict[str, float] = {
    "gasoline": 1.0,
    "hybrid": 1.06,
    "diesel": 1.08,
    "electric": 1.0,
}
ANNUAL_DEPRECIATION = 0.13
MAX_VEHICLE_AGE = 20
MIN_PRICE = 1_000.0


def _seed_for(listing_date: dt.date, salt: int) -> int:
    digest = hashlib.sha256(f"{listing_date.isoformat()}:{salt}".encode()).hexdigest()
    return int(digest[:12], 16)


def generate_listings(
    listing_date: dt.date,
    n: int,
    *,
    market_drift: float = 0.0,
    fleet_age_shift: float = 0.0,
    salt: int = 0,
) -> pd.DataFrame:
    """Generate ``n`` reproducible listings (with sale price) for ``listing_date``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = np.random.default_rng(_seed_for(listing_date, salt))

    spec_idx = rng.integers(0, len(CATALOG), size=n)
    specs = [CATALOG[i] for i in spec_idx]
    age = np.clip(
        (rng.gamma(shape=2.0, scale=2.5, size=n) + fleet_age_shift).round(), 0, MAX_VEHICLE_AGE
    )
    year = (listing_date.year - age).astype(int)
    eff_age = np.where(age > 0, age, 0.5)
    mileage = np.maximum(rng.normal(12_000, 4_000, size=n), 2_000) * eff_age
    mileage = np.round(mileage, -1).astype(int)
    condition = rng.choice(CONDITIONS, size=n, p=CONDITION_PROBS)
    num_owners = np.clip(1 + rng.poisson(age / 5), 1, 6).astype(int)
    accident = rng.random(n) < np.clip(0.05 + 0.02 * age, 0, 0.5)
    region = rng.choice(list(REGIONS), size=n)
    transmission = rng.choice(TRANSMISSIONS, size=n, p=(0.92, 0.08))

    fuel = np.array([s.fuel_types[rng.integers(len(s.fuel_types))] for s in specs])
    engine = np.array([s.engine_sizes[rng.integers(len(s.engine_sizes))] for s in specs])
    drivetrain = np.array(
        [DRIVETRAINS[s.body_type][rng.integers(len(DRIVETRAINS[s.body_type]))] for s in specs]
    )
    base = np.array([s.base_msrp for s in specs])

    price = (
        base
        * np.exp(-ANNUAL_DEPRECIATION * age)
        * np.exp(-0.0000025 * np.maximum(mileage - 12_000 * eff_age, -50_000))
        * np.array([CONDITION_MULTIPLIER[c] for c in condition])
        * np.array([REGIONS[r] for r in region])
        * np.array([DRIVETRAIN_MULTIPLIER[d] for d in drivetrain])
        * np.array([FUEL_MULTIPLIER[f] for f in fuel])
        * np.where(accident, 0.85, 1.0)
        * (1 - 0.03 * (num_owners - 1))
        * np.where(transmission == "manual", 0.97, 1.0)
        * (1 + market_drift)
        * rng.lognormal(mean=0.0, sigma=0.07, size=n)
    )
    price = np.maximum(np.round(price, -1), MIN_PRICE)

    ids = [f"{listing_date:%Y%m%d}-{salt:02d}-{i:06d}" for i in range(n)]
    df = pd.DataFrame(
        {
            "listing_id": ids,
            "listing_date": pd.Timestamp(listing_date).date(),
            "make": [s.make for s in specs],
            "model": [s.model for s in specs],
            "body_type": [s.body_type for s in specs],
            "fuel_type": fuel,
            "transmission": transmission,
            "drivetrain": drivetrain,
            "condition": condition,
            "region": region,
            "year": year,
            "mileage": mileage,
            "engine_size_l": engine.astype(float),
            "num_owners": num_owners,
            "accident_history": accident,
            "price": price.astype(float),
        }
    )
    return df[RAW_COLS]


def generate_date_range(
    end_date: dt.date,
    num_days: int,
    rows_per_day: int,
    *,
    market_drift: float = 0.0,
    fleet_age_shift: float = 0.0,
) -> pd.DataFrame:
    """Generate listings for the ``num_days`` days ending on (and including) ``end_date``."""
    frames = [
        generate_listings(
            end_date - dt.timedelta(days=offset),
            rows_per_day,
            market_drift=market_drift,
            fleet_age_shift=fleet_age_shift,
        )
        for offset in range(num_days - 1, -1, -1)
    ]
    return pd.concat(frames, ignore_index=True)
