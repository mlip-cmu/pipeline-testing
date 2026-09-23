import numpy as np
import pandas as pd
import pytest

from datalinter import lint
from datalinter.lint import (
    date_as_string,
    duplicate_rows,
    empty_or_missing,
    enum_as_real,
    number_as_string,
    tailed_distribution,
    tokenizable_string,
    uncommon_sign,
    unnormalized_feature,
    zip_code_as_number,
)

rng = np.random.default_rng(0)
N = 200


def columns(findings):
    return {f.column for f in findings}


@pytest.mark.parametrize("values", [["1,200", "$950", "3.5", "-4"], ["10%", "20%", "5%", "7%"]])
def test_number_as_string(values):
    assert columns(number_as_string(pd.DataFrame({"x": values}))) == {"x"}


def test_real_strings_are_not_numbers():
    assert number_as_string(pd.DataFrame({"x": ["L100", "apt 3", "12b", "n/a"]})) == []


@pytest.mark.parametrize("values", [["2026-01-01", "2026-02-03"], ["1/2/26 10:00", "12/31/2025 23:59"]])
def test_date_as_string(values):
    assert columns(date_as_string(pd.DataFrame({"when": values}))) == {"when"}


def test_enum_as_real():
    df = pd.DataFrame({"rooms": rng.choice([1.0, 2.0, 3.0], N), "size": rng.normal(50, 10, N)})
    assert columns(enum_as_real(df)) == {"rooms"}


def test_tokenizable_string():
    df = pd.DataFrame({"text": [f"A long free text description number {i} with many words" for i in range(N)],
                       "city": rng.choice(["Pittsburgh", "Philadelphia"], N)})
    assert columns(tokenizable_string(df)) == {"text"}


def test_zip_code_as_number():
    df = pd.DataFrame({"zip_code": rng.integers(10000, 99999, N), "price": rng.normal(100, 5, N)})
    assert columns(zip_code_as_number(df)) == {"zip_code"}


def test_unnormalized_feature():
    df = pd.DataFrame({"a": rng.normal(0, 1, N), "b": rng.normal(0, 2, N), "c": rng.normal(0, 5000, N)})
    assert columns(unnormalized_feature(df)) == {"c"}


def test_tailed_distribution():
    values = rng.normal(100, 10, N)
    values[:3] = [5000, 7000, -3000]
    df = pd.DataFrame({"x": values, "y": rng.normal(0, 1, N)})
    assert columns(tailed_distribution(df)) == {"x"}


def test_uncommon_sign():
    values = rng.uniform(1, 10, N)
    values[:4] *= -1
    df = pd.DataFrame({"x": values, "y": rng.normal(0, 1, N)})
    assert columns(uncommon_sign(df)) == {"x"}


def test_duplicate_rows():
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
    assert duplicate_rows(df)[0].message == "1 duplicate rows"
    assert duplicate_rows(df.drop_duplicates()) == []


def test_empty_or_missing():
    df = pd.DataFrame({"empty": [None] * 10, "sparse": ["N/A"] * 5 + ["x"] * 5, "full": list("abcdefghij")})
    findings = {f.column: f.message for f in empty_or_missing(df)}
    assert findings == {"empty": "column is empty", "sparse": "50% of values are missing"}


def test_clean_dataset_has_no_findings():
    df = pd.DataFrame({
        "temperature": rng.normal(15, 5, N),
        "humidity": rng.uniform(20, 90, N),
        "city": rng.choice(["Pittsburgh", "Philadelphia", "Erie"], N),
    })
    assert lint(df) == []
