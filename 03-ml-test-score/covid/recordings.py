"""Simulated recordings as uploaded by the smartphone app: cough audio plus metadata."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

SAMPLE_RATE = 16_000
PHONE_MODELS = {
    "pixel-8": {"cutoff_hz": 7_500, "gain": 1.0, "noise": 0.004},
    "iphone-15": {"cutoff_hz": 7_800, "gain": 0.8, "noise": 0.003},
    "galaxy-a14": {"cutoff_hz": 4_500, "gain": 1.4, "noise": 0.012},
}
SEXES = ["female", "male"]


@dataclass
class Recording:
    audio: np.ndarray
    sample_rate: int
    phone_model: str
    age: int
    sex: str
    fever: bool
    recorded_at: datetime
    user_id: str = ""
    phone_number: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    covid_positive: bool | None = None
    extra: dict = field(default_factory=dict)


def _cough(rng: np.random.Generator, tilt: float, length: int) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(size=length))
    freqs = np.fft.rfftfreq(length, 1 / SAMPLE_RATE)
    spectrum *= (1 + freqs / 300) ** -tilt
    burst = np.fft.irfft(spectrum, n=length)
    envelope = np.exp(-np.linspace(0, 6, length)) * (1 - np.exp(-np.linspace(0, 60, length)))
    return burst / (np.abs(burst).max() + 1e-12) * envelope


def _microphone(audio: np.ndarray, rng: np.random.Generator, phone: dict) -> np.ndarray:
    spectrum = np.fft.rfft(audio)
    spectrum[np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE) > phone["cutoff_hz"]] = 0
    audio = np.fft.irfft(spectrum, n=len(audio)) * phone["gain"]
    return np.clip(audio + rng.normal(0, phone["noise"], len(audio)), -1, 1).astype(np.float32)


def simulate_recording(rng: np.random.Generator, positive: bool, phone_model: str | None = None,
                       recorded_at: datetime | None = None) -> Recording:
    phone_model = phone_model or rng.choice(list(PHONE_MODELS), p=[0.4, 0.35, 0.25])
    audio = np.zeros(1024 * int(rng.integers(24, 48)))
    tilt = rng.normal(0.95 if positive else 0.75, 0.15)
    n_coughs = 1 + rng.poisson(2.6 if positive else 1.8)
    for _ in range(n_coughs):
        length = 256 * int(rng.integers(10, 22))
        start = rng.integers(0, len(audio) - length)
        audio[start:start + length] += _cough(rng, tilt, length) * rng.uniform(0.3, 0.9)
    age = int(np.clip(rng.normal(50 if positive else 44, 16), 18, 95))
    return Recording(
        audio=_microphone(audio, rng, PHONE_MODELS[phone_model]),
        sample_rate=SAMPLE_RATE,
        phone_model=str(phone_model),
        age=age,
        sex=str(rng.choice(SEXES)),
        fever=bool(rng.random() < (0.45 if positive else 0.2)),
        recorded_at=recorded_at or datetime(2026, 3, 1) + timedelta(minutes=int(rng.integers(0, 60 * 24 * 60))),
        user_id=f"user-{rng.integers(1e9):09d}",
        phone_number=f"+1-412-{rng.integers(1000):03d}-{rng.integers(10000):04d}",
        latitude=float(rng.uniform(40.3, 40.6)),
        longitude=float(rng.uniform(-80.1, -79.8)),
        covid_positive=bool(positive),
    )


def simulate_recordings(n: int, seed: int = 0, prevalence: float = 0.3, **kwargs) -> list[Recording]:
    rng = np.random.default_rng(seed)
    return [simulate_recording(rng, bool(rng.random() < prevalence), **kwargs) for _ in range(n)]
