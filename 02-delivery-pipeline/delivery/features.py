import numpy as np
import pandas as pd
from scipy.special import boxcox1p, inv_boxcox1p

BOXCOX_LAMBDA = 0.4
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = list(range(1, 13))
HOURS = list(range(24))
WEATHER = {1: "clear", 2: "mist", 3: "light_rain", 4: "heavy_rain"}
NUMERIC_RANGES = {"temp": (-40, 50), "humidity": (0, 100), "windspeed": (0, 150)}


def _check_datetime_column(df: pd.DataFrame) -> None:
    if "datetime" not in df.columns:
        raise ValueError("Column datetime missing")
    if not pd.api.types.is_string_dtype(df["datetime"]):
        raise ValueError("Invalid type for column datetime")


def _one_hot(df: pd.DataFrame, column: str, values, categories) -> pd.DataFrame:
    df[column] = pd.Categorical(values, categories=categories)
    return pd.get_dummies(df, columns=[column], dtype=int)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["datetime", "weather"])
    if "delivery_count" in df.columns:
        df = df[df["delivery_count"].notna() & (df["delivery_count"] >= 0)]
    for column, (low, high) in NUMERIC_RANGES.items():
        df = df[df[column].between(low, high)]
    return df.reset_index(drop=True)


def encode_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    _check_datetime_column(df)
    return _one_hot(df.copy(), "dayofweek", pd.to_datetime(df["datetime"]).dt.day_name(), DAYS)


def encode_month(df: pd.DataFrame) -> pd.DataFrame:
    _check_datetime_column(df)
    return _one_hot(df.copy(), "month", pd.to_datetime(df["datetime"]).dt.month, MONTHS)


def encode_hour(df: pd.DataFrame) -> pd.DataFrame:
    _check_datetime_column(df)
    return _one_hot(df.copy(), "hour", pd.to_datetime(df["datetime"]).dt.hour, HOURS)


def encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    unknown = set(df["weather"].unique()) - set(WEATHER)
    if unknown:
        raise ValueError(f"Unknown weather codes: {sorted(unknown)}")
    return _one_hot(df.copy(), "weather", df["weather"].map(WEATHER), list(WEATHER.values()))


def encode_count(counts: pd.Series) -> pd.Series:
    if (counts < 0).any():
        raise ValueError("Delivery counts must not be negative")
    return pd.Series(boxcox1p(counts.astype(float), BOXCOX_LAMBDA), index=counts.index, name=counts.name)


def decode_count(transformed) -> np.ndarray:
    return np.asarray(inv_boxcox1p(transformed, BOXCOX_LAMBDA))
