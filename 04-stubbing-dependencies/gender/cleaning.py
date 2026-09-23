"""Decoupled implementation: the caller passes in the gender model."""

from typing import Callable

import pandas as pd

GenderModel = Callable[[str, str, str], str | None]


def clean_gender(df: pd.DataFrame, model: GenderModel) -> pd.DataFrame:
    def clean(row):
        if pd.isnull(row["gender"]):
            row["gender"] = model(row["firstname"], row["lastname"], row["location"])
        return row
    return df.apply(clean, axis=1)
