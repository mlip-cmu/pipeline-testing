"""Feature extraction shared by training and serving."""

import numpy as np
import pandas as pd

from covid.recordings import Recording

AUDIO_FEATURES = ["duration_s", "rms", "zero_crossing_rate", "spectral_centroid_hz",
                  "spectral_rolloff_hz", "cough_count"]
METADATA_FEATURES = ["age", "fever", "phone_model", "sex"]
FEATURES = AUDIO_FEATURES + METADATA_FEATURES
PII_FIELDS = {"user_id", "phone_number", "latitude", "longitude"}
LABEL = "covid_positive"


def count_coughs(audio: np.ndarray, sample_rate: int, threshold: float = 0.2) -> int:
    frame = sample_rate // 50
    n_frames = len(audio) // frame
    if n_frames == 0:
        return 0
    energy = np.sqrt((audio[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1))
    energy = np.convolve(energy, np.ones(5) / 5, mode="same")
    if energy.max() < 1e-3:
        return 0
    active = energy > threshold * energy.max()
    return int(np.sum(active[1:] & ~active[:-1]) + active[0])


def extract_audio_features(audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float64)
    n_fft = 1 << max(len(audio) - 1, 1).bit_length()
    power = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    total = power.sum()
    if total <= 0:
        centroid = rolloff = 0.0
    else:
        centroid = float((freqs * power).sum() / total)
        rolloff = float(freqs[np.searchsorted(np.cumsum(power), 0.85 * total)])
    signs = np.signbit(audio)
    return {
        "duration_s": len(audio) / sample_rate,
        "rms": float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0,
        "zero_crossing_rate": float(np.mean(signs[1:] != signs[:-1])) if len(audio) > 1 else 0.0,
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": rolloff,
        "cough_count": count_coughs(audio, sample_rate),
    }


def extract_features(recording: Recording) -> dict:
    return {
        **extract_audio_features(recording.audio, recording.sample_rate),
        "age": recording.age,
        "fever": int(recording.fever),
        "phone_model": recording.phone_model,
        "sex": recording.sex,
    }


def build_feature_table(recordings: list[Recording]) -> pd.DataFrame:
    rows = []
    for r in recordings:
        row = extract_features(r)
        if r.covid_positive is not None:
            row[LABEL] = int(r.covid_positive)
        row["recorded_at"] = r.recorded_at
        rows.append(row)
    return pd.DataFrame(rows)
