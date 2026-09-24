"""Corrected versions of the functions in buggy.py."""

import pandas as pd

UNITS = {"": 1, "k": 10**3, "M": 10**6}


def _with_unit(values: pd.Series) -> pd.Series:
    parts = values.astype("string").str.strip().str.extract(r"^([\d.]+)([kM]?)$")
    return pd.to_numeric(parts[0]) * parts[1].map(UNITS)


def parse_size(sizes: pd.Series) -> pd.Series:
    return _with_unit(sizes.replace('Varies with device', None)).astype(float)


def convert_size(df: pd.DataFrame) -> pd.DataFrame:
    df['Size'] = parse_size(df['Size'])
    return df


def fill_missing_size(df: pd.DataFrame) -> pd.DataFrame:
    df['Size'] = df['Size'].fillna(df['Size'].mean())
    return df


def add_update_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Last Updated']).copy()
    df['Update_year'] = pd.to_datetime(df['Last Updated'], format='%B %d, %Y').dt.year
    return df


def impute_rating(df: pd.DataFrame) -> pd.DataFrame:
    map_means = df.groupby('Category')['Rating'].mean()
    missing = df['Rating'].isnull()
    df.loc[missing, 'Rating'] = df.loc[missing, 'Category'].map(map_means)
    return df


def convert_installs(df: pd.DataFrame) -> pd.DataFrame:
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
    return df


def convert_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df['Reviews'] = _with_unit(df['Reviews']).round().astype(int)
    return df
