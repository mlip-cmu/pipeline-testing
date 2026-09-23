import numpy as np
import pandas as pd
import pytest

from wrangling import buggy, fixed

BUGGY = pytest.mark.xfail(reason="bug from the slides; some inputs do not trigger it")


def versions(name, *buggy_names):
    buggy_versions = [pytest.param(getattr(buggy, n), marks=BUGGY, id=f"buggy.{n}") for n in buggy_names or [name]]
    return buggy_versions + [pytest.param(getattr(fixed, name), id=f"fixed.{name}")]


@pytest.mark.parametrize("add_join_year", versions("add_join_year"))
def test_join_year_is_integer_and_keeps_missing(add_join_year):
    df = pd.DataFrame({"Joined": ["Jul 1, 2004", np.nan, "Jan 30, 2018"]})
    out = add_join_year(df)
    assert pd.api.types.is_integer_dtype(out["Join_year"])
    assert out["Join_year"].tolist()[0] == 2004
    assert pd.isna(out["Join_year"][1])


@pytest.mark.parametrize("add_join_year", versions("add_join_year"))
def test_join_year_tolerates_unexpected_format(add_join_year):
    df = pd.DataFrame({"Joined": ["Jul 1, 2004", "2018-01-30"]})
    out = add_join_year(df)
    assert out["Join_year"][0] == 2004


@pytest.mark.parametrize("impute_age_by_title", versions("impute_age_by_title"))
def test_age_imputation_fills_all_missing_values(impute_age_by_title):
    df = pd.DataFrame({"Title": ["Mr", "Mr", "Mrs", "Mrs"], "Age": [20.0, np.nan, 40.0, np.nan]})
    out = impute_age_by_title(df)
    assert out["Age"].notna().all()
    assert out["Age"].tolist() == [20.0, 20.0, 40.0, 40.0]


@pytest.mark.parametrize("parse_weight", versions("parse_weight"))
def test_weight_is_numeric(parse_weight):
    df = pd.DataFrame({"Weight": ["159lbs", "192lbs"]})
    out = parse_weight(df)
    assert pd.api.types.is_integer_dtype(out["Weight"])
    assert out["Weight"].sum() == 351


@pytest.mark.parametrize("parse_weight", versions("parse_weight"))
def test_weight_missing_values_do_not_crash(parse_weight):
    df = pd.DataFrame({"Weight": ["159lbs", np.nan]})
    out = parse_weight(df)
    assert out["Weight"][0] == 159
    assert pd.isna(out["Weight"][1])


@pytest.mark.parametrize("parse_reviews", versions("parse_reviews"))
def test_reviews_column_is_converted_in_place(parse_reviews):
    df = pd.DataFrame({"Reviews": ["159", "967"]})
    out = parse_reviews(df)
    assert list(out.columns) == ["Reviews"]
    assert pd.api.types.is_integer_dtype(out["Reviews"])


@pytest.mark.parametrize("parse_release_clause", versions("parse_release_clause"))
@pytest.mark.parametrize(
    "raw, expected",
    [("450k", 450_000), ("1.2k", 1_200), ("€7K", 7_000), ("226.5M", 226_500_000), ("800", 800)],
)
def test_release_clause_units(parse_release_clause, raw, expected):
    out = parse_release_clause(pd.DataFrame({"Release Clause": [raw]}))
    assert out["Release Clause"][0] == pytest.approx(expected)


@pytest.mark.parametrize("parse_size", versions("parse_size", "parse_size_extract", "parse_size_replace"))
@pytest.mark.parametrize(
    "raw, expected",
    [("19M", 19_000_000), ("8.7M", 8_700_000), ("201k", 201_000), ("1.5k", 1_500), ("512", 512)],
)
def test_size_units(parse_size, raw, expected):
    out = parse_size(pd.DataFrame({"Size": [raw]}))
    assert out["Size"][0] == pytest.approx(expected)


@pytest.mark.parametrize("parse_size", versions("parse_size", "parse_size_extract", "parse_size_replace"))
def test_size_unknown_value_becomes_missing(parse_size):
    out = parse_size(pd.DataFrame({"Size": ["19M", "Varies with device"]}))
    assert out["Size"][0] == pytest.approx(19_000_000)
    assert pd.isna(out["Size"][1])
