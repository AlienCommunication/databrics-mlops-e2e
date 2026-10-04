"""Pure, locally testable synthetic-data and modeling functions. Prices are INR."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ['age_years', 'mileage_km', 'engine_cc', 'owners']
CATEGORICAL = ['brand', 'fuel', 'transmission']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'price_inr'

def synthetic(n=5000, seed=42, start='2024-01-01', days=600):
    """Repeatable fictitious transactions; the price rule is NOT a market valuation."""
    rng = np.random.default_rng(seed)
    brand = rng.choice(['Maruti', 'Hyundai', 'Honda', 'Toyota', 'BMW'], n)
    age = rng.uniform(0.5, 15, n).round(1)
    engine = rng.choice([1000, 1200, 1500, 2000, 3000], n).astype(float)
    mileage = (age * rng.uniform(5000, 22000, n)).round()
    owners = rng.choice([1., 2., 3.], n, p=[.65, .28, .07])
    fuel = rng.choice(['Petrol', 'Diesel', 'Electric'], n, p=[.65, .3, .05])
    trans = rng.choice(['Manual', 'Automatic'], n, p=[.7, .3])
    base = pd.Series(brand).map(dict(Maruti=650000, Hyundai=850000, Honda=1100000,
                                   Toyota=1500000, BMW=3500000)).to_numpy()
    price = (base * (engine / 1500) ** .35 * np.exp(-.10 * age - mileage / 600000)
             * (1 - .07 * (owners - 1)) * np.where(trans == 'Automatic', 1.10, 1)
             * np.where(fuel == 'Electric', 1.15, 1) * rng.lognormal(0, .07, n))
    return pd.DataFrame(dict(car_id=[f'{seed}-{i:06d}' for i in range(n)],
        event_date=pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, days, n), unit='D'),
        age_years=age, mileage_km=mileage, engine_cc=engine, owners=owners,
        brand=brand, fuel=fuel, transmission=trans, price_inr=np.maximum(40000, price).round(2)))

def validate_features(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    if frame.empty or frame[FEATURES].isna().any().any():
        raise ValueError('Empty input or null features')
    bounds = {'age_years': (0, 40), 'mileage_km': (0, 1000000),
              'engine_cc': (500, 8000), 'owners': (1, 10)}
    for col, (low, high) in bounds.items():
        if not np.isfinite(frame[col]).all() or not frame[col].between(low, high).all():
            raise ValueError(f'Invalid {col}')
    for col in CATEGORICAL:
        if not frame[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'Invalid {col}')
    return frame[FEATURES]

def split_data(frame):
    """Chronological train/validation/test; one unique vehicle per synthetic row."""
    frame = frame.sort_values(['event_date', 'car_id']).reset_index(drop=True)
    n = len(frame)
    if n < 100 or frame.car_id.duplicated().any():
        raise ValueError('Need at least 100 unique vehicles')
    return frame.iloc[:int(n*.7)], frame.iloc[int(n*.7):int(n*.85)], frame.iloc[int(n*.85):]

def build_model():
    preprocess = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    regressor = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=20,
            learning_rate=.09, l2_regularization=2, random_state=42),
        func=np.log1p, inverse_func=np.expm1)
    return Pipeline([('features', preprocess), ('regressor', regressor)])

def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if len(y) == 0 or not np.isfinite(prediction).all():
        raise ValueError('Empty evaluation or nonfinite predictions')
    return dict(mae=float(mean_absolute_error(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mape=float(np.mean(np.abs(y-prediction)/np.maximum(np.abs(y), 1))),
                r2=float(r2_score(y, prediction)))

def promotion_decision(candidate, baseline_mae, champion=None):
    checks = {'mae_below_150k': candidate['mae'] <= 150000,
              'mape_below_20pct': candidate['mape'] <= .20,
              'beats_median_baseline': candidate['mae'] < baseline_mae,
              'no_material_regression': champion is None or candidate['mae'] <= champion['mae']*1.02}
    return {'passed': all(checks.values()), 'checks': checks}

def psi(reference, current, bins=10):
    """Reference quantile bins with infinite tails; finite even for constant columns."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins+1)))
    if len(edges) < 2:
        return float(np.mean(np.asarray(current) != np.asarray(reference)[0]))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, edges)[0] / len(reference)
    b = np.histogram(current, edges)[0] / len(current)
    a, b = np.maximum(a, 1e-6), np.maximum(b, 1e-6)
    return float(np.sum((b-a)*np.log(b/a)))
