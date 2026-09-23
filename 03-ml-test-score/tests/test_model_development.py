"""ML Test Score: tests for model development."""

import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score

from covid.model import ModelSpec, auc, build_estimator, recall, train
from covid.features import LABEL


def test_simpler_model_is_not_better(train_data, test_data, model):
    baseline = train(train_data, ModelSpec(features=("age", "fever")))
    assert auc(model, test_data) > auc(baseline, test_data) + 0.05


def test_hyperparameters_are_tuned(train_data):
    def cv_auc(C):
        spec = ModelSpec(C=C)
        estimator = build_estimator(spec)
        return cross_val_score(estimator, train_data[list(spec.features)], train_data[LABEL],
                               cv=5, scoring="roc_auc").mean()

    best = max(cv_auc(C) for C in [0.001, 0.01, 0.1, 1, 10, 100])
    assert cv_auc(ModelSpec.C) >= best - 0.005


@pytest.mark.parametrize("phone_model", ["pixel-8", "iphone-15", "galaxy-a14"])
def test_quality_on_phone_model_slices(phone_model, model, test_data):
    data_slice = test_data[test_data.phone_model == phone_model]
    assert auc(model, data_slice) > 0.75
    assert recall(model, data_slice) > 0.55


def test_recall_similar_across_sexes(model, test_data):
    recalls = [recall(model, group) for _, group in test_data.groupby("sex")]
    assert max(recalls) - min(recalls) < 0.15


def test_recall_similar_across_age_groups(model, test_data):
    age_group = pd.cut(test_data.age, [0, 40, 65, 120])
    recalls = [recall(model, group) for _, group in test_data.groupby(age_group, observed=True)]
    assert min(recalls) > 0.55
    assert max(recalls) - min(recalls) < 0.2
