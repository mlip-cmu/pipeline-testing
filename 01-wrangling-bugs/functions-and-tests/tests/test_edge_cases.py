"""Tests written after thinking about what else is in the data: other units, decimals, missing values, types."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("raw, expected", [
    ("19M", 19_000_000), ("2.4M", 2_400_000), ("201k", 201_000), ("23k", 23_000),
])
def test_size_units(w, raw, expected):
    assert w.convert_size(pd.DataFrame({"Size": [raw]}))['Size'][0] == pytest.approx(expected)


def test_all_missing_sizes_are_filled_with_mean(w):
    df = pd.DataFrame({"Size": ["10M", "Varies with device", "30M"]})
    out = w.fill_missing_size(w.convert_size(df))
    assert out['Size'].tolist() == [10_000_000, 20_000_000, 30_000_000]


def test_apps_without_update_date_are_dropped(w):
    df = pd.DataFrame({"Last Updated": ["January 7, 2018", np.nan, "June 20, 2016"]})
    out = w.add_update_year(df)
    assert out['Update_year'].tolist() == [2018, 2016]


def test_update_year_is_a_number(w):
    out = w.add_update_year(pd.DataFrame({"Last Updated": ["January 7, 2018"]}))
    assert pd.api.types.is_integer_dtype(out['Update_year'])


def test_missing_ratings_get_category_mean(w):
    df = pd.DataFrame({"Category": ["GAME", "GAME", "TOOLS", "TOOLS", "TOOLS"],
                       "Rating": [4.0, np.nan, 3.0, 5.0, np.nan]})
    out = w.impute_rating(df)
    assert out['Rating'].tolist() == [4.0, 4.0, 3.0, 5.0, 4.0]


def test_installs_are_integers(w):
    out = w.convert_installs(pd.DataFrame({"Installs": ["1+", "10,000+", "1,000,000,000+"]}))
    assert pd.api.types.is_integer_dtype(out['Installs'])
    assert out['Installs'].tolist() == [1, 10_000, 1_000_000_000]


@pytest.mark.parametrize("raw, expected", [
    ("159", 159), ("54k", 54_000), ("3.4k", 3_400), ("2.1M", 2_100_000),
])
def test_review_abbreviations(w, raw, expected):
    assert w.convert_reviews(pd.DataFrame({"Reviews": [raw]}))['Reviews'][0] == expected


def test_reviews_are_integers_in_place(w):
    out = w.convert_reviews(pd.DataFrame({"Reviews": ["159", "54k"]}))
    assert list(out.columns) == ["Reviews"]
    assert pd.api.types.is_integer_dtype(out['Reviews'])
