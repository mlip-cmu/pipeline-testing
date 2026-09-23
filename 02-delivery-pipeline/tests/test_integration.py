from pathlib import Path

import pandas as pd
import pytest

from delivery.features import clean_data, encode_day_of_week, encode_hour, encode_month, encode_weather
from delivery.pipeline import prepare_data

DATA = Path(__file__).parent / "data"


@pytest.fixture
def raw_data():
    return pd.read_csv(DATA / "pipelinetest_training.csv")


def test_cleaning_with_feature_engineering(raw_data):
    encoded = encode_weather(encode_hour(encode_month(encode_day_of_week(clean_data(raw_data)))))
    dummies = encoded.filter(regex="^(dayofweek|month|hour|weather)_")
    assert not encoded.isna().any().any()
    assert dummies.isin([0, 1]).all().all()
    assert (dummies.filter(like="dayofweek_").sum(axis=1) == 1).all()


def test_prepared_data_is_aligned_and_numeric(raw_data):
    X, y = prepare_data(raw_data)
    assert len(X) == len(y) > 0
    assert (X.index == y.index).all()
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    assert "delivery_count" not in X.columns


def test_training_and_test_data_have_same_features(raw_data):
    X_train, _ = prepare_data(raw_data)
    X_test, _ = prepare_data(pd.read_csv(DATA / "pipelinetest_test.csv"))
    assert list(X_train.columns) == list(X_test.columns)
