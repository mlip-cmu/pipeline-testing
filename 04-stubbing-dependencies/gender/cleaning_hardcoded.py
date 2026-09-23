"""Original implementation: hardcodes the external API."""

import pandas as pd

from gender.api_client import gender_api_client


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    def clean(row):
        if pd.isnull(row["gender"]):
            row["gender"] = gender_api_client.predict(row["firstname"], row["lastname"], row["location"])
        return row
    return df.apply(clean, axis=1)
