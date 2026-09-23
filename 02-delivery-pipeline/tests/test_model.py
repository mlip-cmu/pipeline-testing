import numpy as np
import pandas as pd
import pytest

from delivery.model import NoDataError, evaluate, learn


def test_learning_fails_with_missing_data():
    df = pd.DataFrame({})
    with pytest.raises(NoDataError):
        learn(df, pd.Series([], dtype=float))


def test_learning_fails_with_misaligned_data():
    with pytest.raises(ValueError, match="differ in length"):
        learn(pd.DataFrame({"a": [1, 2, 3]}), pd.Series([1.0, 2.0]))


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_training_on_small_sample(alpha):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
    y = pd.Series(2 * X["a"] - X["c"] + 1)
    model = learn(X, y, alpha)
    assert model.coef_.shape == (3,)
    assert model.predict(X).shape == (20,)
    assert evaluate(model, X, y) > 0.95
