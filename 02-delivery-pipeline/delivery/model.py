import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from delivery.features import decode_count


class NoDataError(ValueError):
    pass


def learn(X: pd.DataFrame, y: pd.Series, alpha: float = 0.0):
    if len(X) == 0 or len(y) == 0:
        raise NoDataError("No training data")
    if len(X) != len(y):
        raise ValueError(f"Features ({len(X)} rows) and labels ({len(y)} rows) differ in length")
    model = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
    return model.fit(X, y)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> float:
    return r2_score(y, model.predict(X))


def mean_absolute_count_error(model, X: pd.DataFrame, y: pd.Series) -> float:
    return mean_absolute_error(decode_count(y), np.clip(decode_count(model.predict(X)), 0, None))
