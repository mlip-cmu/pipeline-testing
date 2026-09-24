"""Tests written from the first rows of the data, the rows seen in the notebook output."""

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parents[2] / "data" / "googleplaystore.csv"


@pytest.fixture
def first_rows():
    return pd.read_csv(DATA).head()


def test_size_in_bytes(w, first_rows):
    out = w.convert_size(first_rows)
    assert out['Size'][0] == 19_000_000
    assert out['Size'][4] == 41_000_000
    assert pd.isna(out['Size'][3])


def test_missing_size_is_filled(w, first_rows):
    out = w.fill_missing_size(w.convert_size(first_rows))
    assert out['Size'].notna().all()


def test_update_year(w, first_rows):
    out = w.add_update_year(first_rows)
    assert out['Update_year'].tolist() == [2015, 2015, 2017, 2017, 2016]


def test_known_ratings_unchanged(w, first_rows):
    out = w.impute_rating(first_rows.copy())
    assert out['Rating'].tolist() == first_rows['Rating'].tolist()


def test_installs(w, first_rows):
    out = w.convert_installs(first_rows)
    assert out['Installs'].tolist() == [500, 50_000, 10_000, 10_000, 50_000]


def test_reviews(w, first_rows):
    out = w.convert_reviews(first_rows)
    assert out['Reviews'].tolist() == [35, 642, 151, 160, 756]
