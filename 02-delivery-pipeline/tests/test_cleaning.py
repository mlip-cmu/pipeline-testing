import numpy as np
import pandas as pd

from delivery.features import clean_data


def make_rows(**overrides):
    row = {"datetime": "2023-05-01 12:00:00", "weather": 1, "temp": 20.0, "humidity": 50.0,
           "windspeed": 10.0, "delivery_count": 80.0}
    return {**row, **overrides}


def test_valid_rows_are_kept_unchanged():
    df = pd.DataFrame([make_rows(), make_rows(datetime="2023-05-01 13:00:00", delivery_count=0.0)])
    pd.testing.assert_frame_equal(clean_data(df), df)


def test_negative_and_missing_counts_are_removed():
    df = pd.DataFrame([make_rows(), make_rows(delivery_count=-1.0), make_rows(delivery_count=np.nan)])
    assert clean_data(df)["delivery_count"].tolist() == [80.0]


def test_implausible_measurements_are_removed():
    df = pd.DataFrame([make_rows(), make_rows(humidity=250.0), make_rows(temp=-80.0), make_rows(windspeed=-3.0)])
    assert len(clean_data(df)) == 1


def test_duplicate_rows_are_removed():
    df = pd.DataFrame([make_rows(), make_rows()])
    assert len(clean_data(df)) == 1


def test_rows_without_label_are_kept_for_prediction():
    df = pd.DataFrame([make_rows()]).drop(columns=["delivery_count"])
    assert len(clean_data(df)) == 1
