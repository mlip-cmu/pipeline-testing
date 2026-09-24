"""The notebook cells from wrangling.ipynb as functions, bugs included."""

import numpy as np
import pandas as pd


def convert_size(df: pd.DataFrame) -> pd.DataFrame:
    df['Size'] = df['Size'].replace('Varies with device', np.nan)
    num = df.Size.replace(r'[kM]+$', '', regex=True).astype(float)
    factor = df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
    factor = factor.replace(['k', 'M'], [10**3, 10**6]).fillna(1)
    df['Size'] = num * factor.astype(int)
    return df


def fill_missing_size(df: pd.DataFrame) -> pd.DataFrame:
    df['Size'].fillna(df['Size'].mean(), inplace=True)
    return df


def add_update_year(df: pd.DataFrame) -> pd.DataFrame:
    df['Update_year'] = df['Last Updated'].dropna().map(lambda x: x.split(',')[1].split(' ')[1])
    return df


def impute_rating(df: pd.DataFrame) -> pd.DataFrame:
    map_means = df.groupby('Category')['Rating'].mean()
    idx_nan_rating = df.loc[df.Rating.isnull()].index
    df.loc[idx_nan_rating, 'Rating'].loc[idx_nan_rating] = df['Category'].loc[idx_nan_rating].map(map_means)
    return df


def convert_installs(df: pd.DataFrame) -> pd.DataFrame:
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '')
    df['Installs'].astype(str).astype(int)
    return df


def convert_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df['Reviews'] = df['Reviews'].replace(regex=['k'], value='000')
    df['Reviews'] = df['Reviews'].replace(regex=['M'], value='000000')
    df['Reviews'] = df['Reviews'].astype(str).astype(float)
    df['Reviws'] = df['Reviews'].apply(int)
    return df
