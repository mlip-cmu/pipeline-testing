"""Checks that run in production and notify the on-call team."""

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from covid.recordings import PHONE_MODELS, SAMPLE_RATE, Recording


class Notifier(Protocol):
    def notify(self, message: str) -> None: ...


class PrintNotifier:
    def notify(self, message: str) -> None:
        print(f"[ALERT] {message}")


def check_input_invariants(recording: Recording) -> list[str]:
    violations = []
    if recording.sample_rate != SAMPLE_RATE:
        violations.append(f"sample rate {recording.sample_rate} != {SAMPLE_RATE}")
    audio = np.asarray(recording.audio)
    duration = len(audio) / max(recording.sample_rate, 1)
    if not 0.5 <= duration <= 10:
        violations.append(f"duration {duration:.2f}s outside 0.5-10s")
    if len(audio) and not np.isfinite(audio).all():
        violations.append("audio contains NaN or infinite values")
    elif len(audio) and np.abs(audio).max() < 1e-3:
        violations.append("audio is silent")
    if not 0 < recording.age < 120:
        violations.append(f"age {recording.age} implausible")
    if recording.phone_model not in PHONE_MODELS:
        violations.append(f"unknown phone model {recording.phone_model}")
    return violations


class InputMonitor:
    def __init__(self, notifier: Notifier, window: int = 100, max_violation_rate: float = 0.05):
        self.notifier = notifier
        self.results = deque(maxlen=window)
        self.max_violation_rate = max_violation_rate

    def record(self, violations: list[str]) -> None:
        self.results.append(bool(violations))
        rate = sum(self.results) / len(self.results)
        if len(self.results) == self.results.maxlen and rate > self.max_violation_rate:
            self.notifier.notify(f"{rate:.0%} of recent requests violate input invariants")
            self.results.clear()


def feature_skew(training: pd.DataFrame, serving: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    std = training[columns].std().replace(0, 1)
    return ((serving[columns].mean() - training[columns].mean()).abs() / std).to_dict()


def check_training_serving_skew(training, serving, columns, notifier: Notifier, threshold: float = 0.5) -> list[str]:
    skewed = [c for c, d in feature_skew(training, serving, columns).items() if d > threshold]
    if skewed:
        notifier.notify(f"Training/serving skew in features: {', '.join(skewed)}")
    return skewed


def check_staleness(metadata: dict, notifier: Notifier, max_age: timedelta = timedelta(days=30),
                    now: datetime | None = None) -> bool:
    age = (now or datetime.now(timezone.utc)) - datetime.fromisoformat(metadata["trained_at"])
    if age > max_age:
        notifier.notify(f"Model v{metadata['version']} is {age.days} days old")
        return True
    return False


def check_dependencies(metadata: dict, installed: dict[str, str], notifier: Notifier) -> dict:
    changed = {name: (v, installed.get(name)) for name, v in metadata["dependencies"].items()
               if installed.get(name) != v}
    if changed:
        notifier.notify("Dependencies changed since training: " +
                        ", ".join(f"{n} {old} -> {new}" for n, (old, new) in changed.items()))
    return changed


class QualityMonitor:
    """Tracks confirmed test results (reported later by users) against predictions."""

    def __init__(self, notifier: Notifier, window: int = 200, min_accuracy: float = 0.7):
        self.notifier = notifier
        self.outcomes = deque(maxlen=window)
        self.min_accuracy = min_accuracy
        self.alerted = False

    def record(self, predicted_positive: bool, confirmed_positive: bool) -> None:
        self.outcomes.append(predicted_positive == confirmed_positive)
        if len(self.outcomes) == self.outcomes.maxlen:
            accuracy = sum(self.outcomes) / len(self.outcomes)
            if accuracy < self.min_accuracy and not self.alerted:
                self.notifier.notify(f"Prediction accuracy dropped to {accuracy:.0%}")
                self.alerted = True
