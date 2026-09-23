import numpy as np
import pandas as pd
import pytest

from delivery.features import (
    DAYS,
    decode_count,
    encode_count,
    encode_day_of_week,
    encode_hour,
    encode_month,
    encode_weather,
)


def test_day_of_week_encoding():
    df = pd.DataFrame({"datetime": ["2020-01-01", "2020-01-02", "2020-01-08"], "delivery_count": [1, 2, 3]})
    encoded = encode_day_of_week(df)
    assert "dayofweek_Wednesday" in encoded.columns
    assert (encoded["dayofweek_Wednesday"] == [1, 0, 1]).all()


def test_day_of_week_encoding_has_column_for_every_day():
    df = pd.DataFrame({"datetime": ["2020-01-01"]})
    encoded = encode_day_of_week(df)
    assert [c for c in encoded.columns if c.startswith("dayofweek_")] == [f"dayofweek_{d}" for d in DAYS]
    assert encoded.filter(like="dayofweek_").sum(axis=1).tolist() == [1]


def test_day_of_week_encoding_does_not_modify_input():
    df = pd.DataFrame({"datetime": ["2020-01-01"]})
    encode_day_of_week(df)
    assert list(df.columns) == ["datetime"]


def test_day_of_week_encoding_requires_datetime_column():
    with pytest.raises(ValueError, match="datetime missing"):
        encode_day_of_week(pd.DataFrame({"date": ["2020-01-01"]}))


def test_day_of_week_encoding_rejects_non_string_datetime():
    with pytest.raises(ValueError, match="Invalid type"):
        encode_day_of_week(pd.DataFrame({"datetime": [20200101]}))


def test_invalid_day_of_week_data():
    with pytest.raises(ValueError):
        encode_day_of_week(pd.DataFrame({"datetime": ["2020-13-45"]}))


@pytest.mark.parametrize("datetime, column", [
    ("2020-01-01 00:00:00", "month_1"),
    ("2020-12-31 23:59:59", "month_12"),
    ("2020-02-29 12:00:00", "month_2"),
])
def test_month_encoding_boundaries(datetime, column):
    encoded = encode_month(pd.DataFrame({"datetime": [datetime]}))
    assert encoded[column].tolist() == [1]
    assert encoded.filter(like="month_").shape[1] == 12


@pytest.mark.parametrize("datetime, column", [("2020-01-01 00:00:00", "hour_0"), ("2020-01-01 23:30:00", "hour_23")])
def test_hour_encoding_boundaries(datetime, column):
    encoded = encode_hour(pd.DataFrame({"datetime": [datetime]}))
    assert encoded[column].tolist() == [1]


def test_weather_encoding():
    encoded = encode_weather(pd.DataFrame({"weather": [1, 4, 4]}))
    assert encoded["weather_clear"].tolist() == [1, 0, 0]
    assert encoded["weather_heavy_rain"].tolist() == [0, 1, 1]
    assert encoded.filter(like="weather_").shape[1] == 4


def test_weather_encoding_rejects_unknown_codes():
    with pytest.raises(ValueError, match="Unknown weather"):
        encode_weather(pd.DataFrame({"weather": [1, 5]}))


def test_count_encoding_handles_zero_deliveries():
    encoded = encode_count(pd.Series([0, 1, 100]))
    assert np.isfinite(encoded).all()
    assert encoded[0] == 0


def test_count_encoding_rejects_negative_counts():
    with pytest.raises(ValueError):
        encode_count(pd.Series([3, -1]))


def test_count_encoding_is_monotonic_and_invertible():
    counts = pd.Series([0, 1, 5, 50, 500])
    encoded = encode_count(counts)
    assert encoded.is_monotonic_increasing
    assert decode_count(encoded) == pytest.approx(counts.to_numpy())
