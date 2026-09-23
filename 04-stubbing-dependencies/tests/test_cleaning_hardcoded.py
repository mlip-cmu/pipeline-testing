"""The hardcoded version can only be tested by patching the module-level client."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from gender import cleaning_hardcoded


def test_do_not_overwrite_gender_with_patched_client():
    df = pd.DataFrame({
        "firstname": ["John", "Jane", "Jim"],
        "lastname": ["Doe", "Doe", "Doe"],
        "location": ["Pittsburgh, PA", "Rome, Italy", "Paris, PA "],
        "gender": [np.nan, "F", np.nan],
    })
    with patch.object(cleaning_hardcoded, "gender_api_client") as client:
        client.predict.return_value = "M"
        out = cleaning_hardcoded.clean_gender(df)
    assert (out["gender"] == ["M", "F", "M"]).all()
    assert client.predict.call_count == 2
