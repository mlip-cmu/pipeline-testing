"""Prediction service used by the cloud backend of the app."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from covid.features import extract_features
from covid.model import TrainedModel
from covid.monitoring import check_input_invariants
from covid.recordings import Recording


class InvalidRequest(ValueError):
    pass


@dataclass
class Prediction:
    probability: float
    positive: bool
    model_version: int | None
    latency_ms: float


class CovidDetectionService:
    def __init__(self, model: TrainedModel, model_version: int | None = None, input_monitor=None):
        self.model = model
        self.model_version = model_version
        self.input_monitor = input_monitor
        self.served_features: list[dict] = []

    @classmethod
    def from_registry(cls, registry, **kwargs):
        return cls(registry.load(), registry.current_version(), **kwargs)

    def features(self, recording: Recording) -> pd.DataFrame:
        return pd.DataFrame([extract_features(recording)])

    def predict(self, recording: Recording) -> Prediction:
        start = time.perf_counter()
        violations = check_input_invariants(recording)
        if self.input_monitor is not None:
            self.input_monitor.record(violations)
        if violations:
            raise InvalidRequest("; ".join(violations))
        features = self.features(recording)
        probability = float(self.model.predict_proba(features)[0])
        if not np.isfinite(probability):
            raise ArithmeticError("Model returned a non-finite probability")
        self.served_features.append(features.iloc[0].to_dict())
        return Prediction(probability, probability >= self.model.spec.threshold, self.model_version,
                          (time.perf_counter() - start) * 1000)

    def explain(self, recording: Recording, top: int = 3) -> list[tuple[str, float]]:
        contributions = self.model.contributions(self.features(recording)).iloc[0]
        return sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top]


def canary(sample: list[Recording]):
    def check(model: TrainedModel) -> None:
        service = CovidDetectionService(model)
        for recording in sample:
            prediction = service.predict(recording)
            if not 0 <= prediction.probability <= 1:
                raise ValueError(f"Canary failed: probability {prediction.probability}")
    return check
