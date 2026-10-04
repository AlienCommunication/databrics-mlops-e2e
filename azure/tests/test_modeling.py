import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
import numpy as np
import pandas as pd
import pytest
from modeling import *

def test_reproducibility_and_split_no_leakage():
    data = synthetic(1000)
    pd.testing.assert_frame_equal(data, synthetic(1000))
    train, val, test = split_data(data)
    assert set(train.car_id).isdisjoint(test.car_id)
    assert train.event_date.max() <= val.event_date.min() <= test.event_date.min()
    assert TARGET not in FEATURES and 'car_id' not in FEATURES

def test_model_quality_unseen_category_and_serialization(tmp_path):
    import joblib
    train, val, test = split_data(synthetic())
    model = build_model().fit(validate_features(train), train[TARGET])
    p = model.predict(validate_features(test))
    score = metrics(test[TARGET], p)
    assert promotion_decision(score, metrics(test[TARGET], np.repeat(train[TARGET].median(),len(test)))['mae'])['passed']
    joblib.dump(model, tmp_path/'model.pkl')
    np.testing.assert_allclose(p, joblib.load(tmp_path/'model.pkl').predict(test[FEATURES]))
    unseen = test[FEATURES].iloc[:1].copy(); unseen['brand'] = 'NewBrand'
    assert np.isfinite(model.predict(unseen)).all()

def test_rejected_candidate_and_contract():
    assert not promotion_decision({'mae':160000,'mape':.1},200000)['passed']
    assert not promotion_decision({'mae':100000,'mape':.1},200000,{'mae':80000})['passed']
    invalid = synthetic(2); invalid.loc[0,'mileage_km'] = -1
    with pytest.raises(ValueError): validate_features(invalid)
    with pytest.raises(ValueError): validate_features(invalid.drop(columns='brand'))

def test_drift():
    rng = np.random.default_rng(3); reference = rng.normal(size=5000)
    assert psi(reference, reference) == 0
    assert psi(reference, rng.normal(3,1,5000)) > .2
    assert np.isfinite(psi(np.ones(100), np.ones(100)))
