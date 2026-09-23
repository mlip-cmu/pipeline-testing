"""Corrected versions of the functions in buggy.py."""

import pandas as pd

UNIT_FACTORS = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9}


def _parse_with_unit(values: pd.Series) -> pd.Series:
    parts = values.astype("string").str.strip().str.extract(r"^([\d.]+)\s*([kKmMgG]?)$")
    number = pd.to_numeric(parts[0])
    factor = parts[1].str.lower().map(UNIT_FACTORS)
    return (number * factor).astype(float)


def add_join_year(df: pd.DataFrame) -> pd.DataFrame:
    joined = pd.to_datetime(df["Joined"], format="%b %d, %Y", errors="coerce")
    df["Join_year"] = joined.dt.year.astype("Int64")
    return df


def impute_age_by_title(df: pd.DataFrame) -> pd.DataFrame:
    map_means = df.groupby("Title")["Age"].mean()
    missing = df["Age"].isnull()
    df.loc[missing, "Age"] = df.loc[missing, "Title"].map(map_means)
    return df


def parse_weight(df: pd.DataFrame) -> pd.DataFrame:
    weight = df["Weight"].astype("string").str.removesuffix("lbs")
    df["Weight"] = pd.to_numeric(weight).astype("Int64")
    return df


def parse_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df["Reviews"] = _parse_with_unit(df["Reviews"]).astype("Int64")
    return df


def parse_release_clause(df: pd.DataFrame) -> pd.DataFrame:
    df["Release Clause"] = _parse_with_unit(df["Release Clause"].astype("string").str.lstrip("€"))
    return df


def parse_size(data: pd.DataFrame) -> pd.DataFrame:
    data["Size"] = _parse_with_unit(data["Size"])
    return data
