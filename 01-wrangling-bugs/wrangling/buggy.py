"""Data wrangling code as seen on the slides (collected from public Kaggle notebooks)."""

import pandas as pd


def add_join_year(df: pd.DataFrame) -> pd.DataFrame:
    df["Join_year"] = df.Joined.dropna().map(lambda x: x.split(",")[1].split(" ")[1])
    return df


def impute_age_by_title(df: pd.DataFrame) -> pd.DataFrame:
    map_means = df.groupby("Title")["Age"].mean()
    idx_nan_age = df.loc[df.Age.isnull()].index
    df.loc[idx_nan_age, "Age"].loc[idx_nan_age] = df["Title"].loc[idx_nan_age].map(map_means)
    return df


def parse_weight(df: pd.DataFrame) -> pd.DataFrame:
    df["Weight"] = df["Weight"].str.replace("lbs", "")
    df["Weight"].astype(str).astype(int)
    return df


def parse_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df["Reviws"] = df["Reviews"].apply(int)
    return df


def parse_release_clause(df: pd.DataFrame) -> pd.DataFrame:
    df["Release Clause"] = df["Release Clause"].replace(regex=["k"], value="000")
    df["Release Clause"] = df["Release Clause"].astype(str).astype(float)
    return df


def parse_size_extract(data: pd.DataFrame) -> pd.DataFrame:
    num = data.Size.replace(r"[kM]+$", "", regex=True).astype(float)
    factor = data.Size.str.extract(r"[\d\.]+([KM]+)", expand=False)
    factor = factor.replace(["k", "M"], [10**3, 10**6]).fillna(1)
    data["Size"] = num * factor.astype(int)
    return data


def parse_size_replace(data: pd.DataFrame) -> pd.DataFrame:
    data["Size"] = data["Size"].replace(regex=["k"], value="000")
    data["Size"] = data["Size"].replace(regex=["M"], value="000000")
    data["Size"] = data["Size"].astype(str).astype(float)
    return data
