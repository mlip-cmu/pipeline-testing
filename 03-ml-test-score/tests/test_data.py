"""ML Test Score: data tests."""

import time

import numpy as np
import pytest

from covid.features import FEATURES, PII_FIELDS, count_coughs, extract_audio_features, extract_features
from covid.model import ModelSpec, auc, train
from covid.recordings import SAMPLE_RATE
from covid.schema import validate_feature_table


def test_feature_expectations_are_captured_in_schema(train_data):
    assert validate_feature_table(train_data) == []


def test_schema_detects_problems(train_data):
    broken = train_data.head(10).copy()
    broken.loc[0, "age"] = 250
    broken.loc[1, "phone_model"] = "nokia-3310"
    broken["rms"] = broken["rms"].astype(str)
    problems = validate_feature_table(broken.drop(columns=["fever"]))
    assert any(p.startswith("age:") for p in problems)
    assert any(p.startswith("phone_model:") for p in problems)
    assert any(p.startswith("rms:") for p in problems)
    assert "fever: missing" in problems


@pytest.mark.parametrize("feature", FEATURES)
def test_no_feature_harms_model_quality(feature, train_data, test_data, model):
    without = train(train_data, ModelSpec(features=tuple(f for f in FEATURES if f != feature)))
    assert auc(without, test_data) <= auc(model, test_data) + 0.005


def test_feature_cost_is_acceptable(train_recordings):
    start = time.perf_counter()
    for recording in train_recordings[:100]:
        extract_features(recording)
    assert (time.perf_counter() - start) / 100 < 0.02


def test_no_personal_data_in_features(train_data):
    assert PII_FIELDS.isdisjoint(train_data.columns)
    assert PII_FIELDS.isdisjoint(FEATURES)


def test_personal_data_cannot_be_used_as_feature():
    with pytest.raises(ValueError, match="identifiable"):
        ModelSpec(features=("age", "phone_number")).validate()


def _tone(frequency, seconds=1.0, amplitude=0.5):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * frequency * t)


def test_spectral_centroid_of_pure_tone():
    features = extract_audio_features(_tone(1000), SAMPLE_RATE)
    assert features["spectral_centroid_hz"] == pytest.approx(1000, rel=0.02)
    assert features["zero_crossing_rate"] == pytest.approx(2000 / SAMPLE_RATE, rel=0.02)
    assert features["rms"] == pytest.approx(0.5 / np.sqrt(2), rel=0.01)
    assert features["duration_s"] == pytest.approx(1.0)


def test_count_coughs_counts_separate_bursts():
    audio = np.zeros(2 * SAMPLE_RATE)
    for start in (0.1, 0.8, 1.5):
        i = int(start * SAMPLE_RATE)
        audio[i:i + 3200] = _tone(500, 0.2)
    assert count_coughs(audio, SAMPLE_RATE) == 3


@pytest.mark.parametrize("audio", [np.zeros(SAMPLE_RATE), np.zeros(0), np.zeros(10)])
def test_feature_extraction_handles_degenerate_audio(audio):
    features = extract_audio_features(audio, SAMPLE_RATE)
    assert all(np.isfinite(v) for v in features.values())
    assert features["cough_count"] == 0
