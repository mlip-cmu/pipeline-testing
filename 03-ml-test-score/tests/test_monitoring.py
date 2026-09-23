"""ML Test Score: monitoring tests."""

import dataclasses
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from covid.features import AUDIO_FEATURES, build_feature_table
from covid.monitoring import (
    InputMonitor,
    QualityMonitor,
    check_dependencies,
    check_input_invariants,
    check_staleness,
    check_training_serving_skew,
)
from covid.recordings import PHONE_MODELS, SAMPLE_RATE, simulate_recordings
from covid.registry import ModelRegistry, dependency_versions
from covid.serving import CovidDetectionService, InvalidRequest


@pytest.fixture
def registry(tmp_path, model):
    registry = ModelRegistry(tmp_path)
    registry.publish(model, {"auc": 0.9, "recall": 0.7}, trained_at=datetime(2026, 3, 1, tzinfo=timezone.utc))
    return registry


def test_dependency_changes_result_in_notification(registry, notifier):
    assert check_dependencies(registry.metadata(), dependency_versions(), notifier) == {}
    assert notifier.messages == []

    upgraded = {**dependency_versions(), "scikit-learn": "99.0"}
    changed = check_dependencies(registry.metadata(), upgraded, notifier)
    assert "scikit-learn" in changed
    assert "scikit-learn" in notifier.messages[0]


@pytest.mark.parametrize("change, message", [
    ({"sample_rate": 44_100}, "sample rate"),
    ({"audio": np.zeros(SAMPLE_RATE, dtype=np.float32)}, "silent"),
    ({"audio": np.full(SAMPLE_RATE, np.nan, dtype=np.float32)}, "NaN"),
    ({"audio": np.ones(100, dtype=np.float32)}, "duration"),
    ({"age": 150}, "age"),
    ({"phone_model": "nokia-3310"}, "phone model"),
])
def test_data_invariants_hold_for_inputs(change, message, model, test_recordings):
    recording = dataclasses.replace(test_recordings[0], **change)
    assert any(message in v for v in check_input_invariants(recording))
    with pytest.raises(InvalidRequest):
        CovidDetectionService(model).predict(recording)


def test_input_monitor_alerts_on_many_invalid_requests(model, test_recordings, notifier):
    service = CovidDetectionService(model, input_monitor=InputMonitor(notifier, window=50))
    for i, recording in enumerate(test_recordings[:50]):
        if i % 10 == 0:
            recording = dataclasses.replace(recording, sample_rate=8_000)
        try:
            service.predict(recording)
        except InvalidRequest:
            pass
    assert len(notifier.messages) == 1


def test_training_and_serving_features_are_identical(model, test_recordings):
    service = CovidDetectionService(model)
    for recording in test_recordings[:20]:
        service.predict(recording)
    served = pd.DataFrame(service.served_features)
    offline = build_feature_table(test_recordings[:20])[served.columns]
    pd.testing.assert_frame_equal(served, offline, check_dtype=False)


def test_training_serving_skew_is_detected(train_data, notifier, monkeypatch):
    same_distribution = build_feature_table(simulate_recordings(200, seed=5))
    assert check_training_serving_skew(train_data, same_distribution, AUDIO_FEATURES, notifier) == []

    monkeypatch.setitem(PHONE_MODELS, "pixel-8", {**PHONE_MODELS["pixel-8"], "gain": 3.0, "cutoff_hz": 3_000})
    after_app_update = build_feature_table(simulate_recordings(200, seed=6, phone_model="pixel-8"))
    skewed = check_training_serving_skew(train_data, after_app_update, AUDIO_FEATURES, notifier)
    assert "rms" in skewed
    assert len(notifier.messages) == 1


def test_models_are_not_too_stale(registry, notifier):
    trained = datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert not check_staleness(registry.metadata(), notifier, now=trained + timedelta(days=10))
    assert check_staleness(registry.metadata(), notifier, now=trained + timedelta(days=45))
    assert "45 days" in notifier.messages[0]


def test_model_is_numerically_stable(model, test_recordings):
    service = CovidDetectionService(model)
    base = test_recordings[0]
    rng = np.random.default_rng(0)
    extremes = [
        np.sign(rng.normal(size=SAMPLE_RATE)).astype(np.float32),
        (rng.normal(size=SAMPLE_RATE) * 0.002).astype(np.float32),
        np.clip(rng.normal(size=10 * SAMPLE_RATE), -1, 1).astype(np.float32),
    ]
    for audio in extremes:
        for age in (1, 119):
            prediction = service.predict(dataclasses.replace(base, audio=audio, age=age))
            assert 0 <= prediction.probability <= 1


def test_computing_performance_has_not_regressed(model, test_recordings):
    service = CovidDetectionService(model)
    latencies = [service.predict(r).latency_ms for r in test_recordings[:100]]
    assert np.percentile(latencies, 95) < 50


def test_prediction_quality_regression_is_detected(model, test_recordings, notifier):
    service = CovidDetectionService(model)
    monitor = QualityMonitor(notifier, window=100, min_accuracy=0.7)
    for recording in test_recordings[:100]:
        monitor.record(service.predict(recording).positive, recording.covid_positive)
    assert notifier.messages == []

    for recording in test_recordings[100:200]:
        monitor.record(service.predict(recording).positive, not recording.covid_positive)
    assert len(notifier.messages) == 1
