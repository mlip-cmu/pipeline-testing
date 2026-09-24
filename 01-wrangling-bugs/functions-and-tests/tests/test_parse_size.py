"""The unit test from the slide "Anatomy of a Unit Test"."""

import pandas as pd
import pytest


@pytest.fixture
def app_sizes():
    return pd.Series(["19M", "201k", "2.5M", "Varies with device"])


def test_parse_size_converts_units(w, app_sizes):
    """Sizes with k and M suffixes become bytes. Other values become NaN."""
    result = w.parse_size(app_sizes)
    assert result[:3].tolist() == [19_000_000, 201_000, 2_500_000]
    assert pd.isna(result[3])
