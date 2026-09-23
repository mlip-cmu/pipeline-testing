"""Fill in missing genders with the real genderize.io API (needs internet access)."""

import numpy as np
import pandas as pd

from gender.api_client import GenderApiClient
from gender.cleaning import clean_gender

df = pd.DataFrame({
    "firstname": ["John", "Jane", "Jim", "Andrea", "Andrea", "Kim"],
    "lastname": ["Doe", "Doe", "Doe", "Rossi", "Miller", "Lee"],
    "location": ["Pittsburgh, PA", "Rome, Italy", "Paris, PA ", "Rome, Italy", "Boston, MA", "Seattle, WA"],
    "gender": [np.nan, "F", np.nan, np.nan, np.nan, np.nan],
})

if __name__ == "__main__":
    print(clean_gender(df, GenderApiClient().predict))
