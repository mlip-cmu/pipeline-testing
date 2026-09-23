"""Expectations for the feature table, reviewed together with the feature code."""

import pandas as pd

from covid.recordings import PHONE_MODELS, SEXES

SCHEMA = {
    "duration_s": {"type": "float", "min": 0.5, "max": 10},
    "rms": {"type": "float", "min": 0.001, "max": 1},
    "zero_crossing_rate": {"type": "float", "min": 0, "max": 1},
    "spectral_centroid_hz": {"type": "float", "min": 50, "max": 8_000},
    "spectral_rolloff_hz": {"type": "float", "min": 50, "max": 8_000},
    "cough_count": {"type": "int", "min": 1, "max": 20},
    "age": {"type": "int", "min": 18, "max": 120},
    "fever": {"type": "int", "min": 0, "max": 1},
    "phone_model": {"type": "category", "values": set(PHONE_MODELS)},
    "sex": {"type": "category", "values": set(SEXES)},
}


def validate_feature_table(df: pd.DataFrame) -> list[str]:
    problems = []
    for column, rule in SCHEMA.items():
        if column not in df.columns:
            problems.append(f"{column}: missing")
            continue
        values = df[column]
        if values.isna().any():
            problems.append(f"{column}: {values.isna().sum()} missing values")
        if rule["type"] == "category":
            unexpected = set(values.dropna()) - rule["values"]
            if unexpected:
                problems.append(f"{column}: unexpected values {sorted(unexpected)}")
            continue
        if not pd.api.types.is_numeric_dtype(values):
            problems.append(f"{column}: not numeric")
            continue
        if rule["type"] == "int" and not pd.api.types.is_integer_dtype(values):
            problems.append(f"{column}: not integer")
        out_of_range = ~values.dropna().between(rule["min"], rule["max"])
        if out_of_range.any():
            problems.append(f"{column}: {out_of_range.sum()} values outside [{rule['min']}, {rule['max']}]")
    return problems
