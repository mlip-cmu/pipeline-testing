from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gender.cleaning import clean_gender


@pytest.fixture
def people():
    return pd.DataFrame({
        "firstname": ["John", "Jane", "Jim"],
        "lastname": ["Doe", "Doe", "Doe"],
        "location": ["Pittsburgh, PA", "Rome, Italy", "Paris, PA "],
        "gender": [np.nan, "F", np.nan],
    })


def test_do_not_overwrite_gender(people):
    def model_stub(first, last, location):
        return "M"

    out = clean_gender(people, model_stub)
    assert (out["gender"] == ["M", "F", "M"]).all()


def test_model_is_only_asked_for_missing_values(people):
    model = Mock(return_value="M")
    clean_gender(people, model)
    assert model.call_count == 2
    model.assert_any_call("John", "Doe", "Pittsburgh, PA")
    model.assert_any_call("Jim", "Doe", "Paris, PA ")


def test_unknown_gender_stays_missing(people):
    out = clean_gender(people, lambda first, last, location: None)
    assert out["gender"].isna().tolist() == [True, False, True]


def test_model_failure_is_not_hidden(people):
    model = Mock(side_effect=ConnectionError("API unavailable"))
    with pytest.raises(ConnectionError):
        clean_gender(people, model)


def test_input_is_not_modified(people):
    clean_gender(people, lambda *args: "M")
    assert people["gender"].isna().sum() == 2
