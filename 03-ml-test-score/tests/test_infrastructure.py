"""ML Test Score: tests for ML infrastructure."""

import numpy as np
import pytest

from covid.features import LABEL, build_feature_table
from covid.model import ModelSpec, TrainedModel, auc, recall, train
from covid.recordings import simulate_recordings
from covid.registry import ModelRegistry, ModelRejected
from covid.serving import CovidDetectionService, canary


def metrics(model, data):
    return {"auc": auc(model, data), "recall": recall(model, data)}


def test_training_is_reproducible(train_data, test_data):
    first = train(train_data, ModelSpec(seed=3))
    second = train(train_data, ModelSpec(seed=3))
    np.testing.assert_array_equal(first.predict_proba(test_data), second.predict_proba(test_data))


@pytest.mark.parametrize("spec", [
    ModelSpec(C=0),
    ModelSpec(threshold=1.5),
    ModelSpec(features=()),
    ModelSpec(features=("age", "blood_type")),
])
def test_invalid_model_specs_are_rejected(spec, train_data):
    with pytest.raises(ValueError):
        train(train_data, spec)


def test_training_requires_both_classes(train_data):
    with pytest.raises(ValueError, match="positive and negative"):
        train(train_data[train_data[LABEL] == 0])


def test_pipeline_integration(tmp_path):
    recordings = simulate_recordings(400, seed=10)
    train_df, val_df = build_feature_table(recordings[:300]), build_feature_table(recordings[300:])
    model = train(train_df)
    registry = ModelRegistry(tmp_path, min_auc=0.7, min_recall=0.3)
    version = registry.publish(model, metrics(model, val_df), canary=canary(recordings[300:310]))

    service = CovidDetectionService.from_registry(registry)
    prediction = service.predict(simulate_recordings(1, seed=11)[0])
    assert 0 <= prediction.probability <= 1
    assert prediction.model_version == version


def test_model_quality_is_validated_before_serving(tmp_path, train_data, test_data, model):
    registry = ModelRegistry(tmp_path)
    shuffled = train_data.assign(**{LABEL: train_data[LABEL].sample(frac=1, random_state=0).to_numpy()})
    bad_model = train(shuffled)
    with pytest.raises(ModelRejected):
        registry.publish(bad_model, metrics(bad_model, test_data))
    assert registry.current_version() is None

    registry.publish(model, metrics(model, test_data))
    weaker = train(train_data, ModelSpec(features=tuple(f for f in model.spec.features if f != "fever")))
    with pytest.raises(ModelRejected, match="worse than deployed"):
        registry.publish(weaker, metrics(weaker, test_data))
    assert registry.current_version() == 1


def test_model_is_debuggable(model, test_recordings):
    feverish = next(r for r in test_recordings if r.fever)
    top = dict(CovidDetectionService(model).explain(feverish, top=20))
    assert top["numeric__fever"] > 0
    assert len(top) >= 5


class BrokenModel(TrainedModel):
    def predict_proba(self, features):
        return np.full(len(features), np.nan)


def test_canary_prevents_deploying_broken_model(tmp_path, model, test_data, test_recordings):
    registry = ModelRegistry(tmp_path)
    broken = BrokenModel(model.spec, model.estimator)
    with pytest.raises(ArithmeticError):
        registry.publish(broken, metrics(model, test_data), canary=canary(test_recordings[:5]))
    assert registry.versions() == []


def test_serving_model_can_be_rolled_back(tmp_path, model, train_data, test_data, test_recordings):
    registry = ModelRegistry(tmp_path)
    registry.publish(model, metrics(model, test_data))
    newer = train(train_data, ModelSpec(C=1.0))
    registry.publish(newer, metrics(newer, test_data))
    assert registry.current_version() == 2

    assert registry.rollback() == 1
    service = CovidDetectionService.from_registry(registry)
    assert service.model_version == 1
    expected = model.predict_proba(build_feature_table(test_recordings[:1]))[0]
    assert service.predict(test_recordings[0]).probability == pytest.approx(expected)


def test_rollback_without_earlier_version_fails(tmp_path, model, test_data):
    registry = ModelRegistry(tmp_path)
    registry.publish(model, metrics(model, test_data))
    with pytest.raises(LookupError):
        registry.rollback()
