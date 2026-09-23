from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from covid.features import AUDIO_FEATURES, FEATURES, LABEL, PII_FIELDS

CATEGORICAL = ["phone_model", "sex"]


@dataclass(frozen=True)
class ModelSpec:
    features: tuple[str, ...] = tuple(FEATURES)
    C: float = 0.1
    threshold: float = 0.4
    seed: int = 0
    notes: dict = field(default_factory=dict, compare=False)

    def validate(self) -> None:
        if PII_FIELDS & set(self.features):
            raise ValueError("Personally identifiable information must not be used as a feature")
        unknown = set(self.features) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")
        if not self.features:
            raise ValueError("At least one feature is required")
        if self.C <= 0:
            raise ValueError("C must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")


def build_estimator(spec: ModelSpec) -> Pipeline:
    categorical = [f for f in spec.features if f in CATEGORICAL]
    numeric = [f for f in spec.features if f not in CATEGORICAL]
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    classifier = LogisticRegression(C=spec.C, max_iter=1000, random_state=spec.seed)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


@dataclass
class TrainedModel:
    spec: ModelSpec
    estimator: Pipeline

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(features[list(self.spec.features)])[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features) >= self.spec.threshold).astype(int)

    def contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        preprocess = self.estimator.named_steps["preprocess"]
        coef = self.estimator.named_steps["classifier"].coef_[0]
        transformed = preprocess.transform(features[list(self.spec.features)])
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        return pd.DataFrame(transformed * coef, columns=preprocess.get_feature_names_out(), index=features.index)


def train(data: pd.DataFrame, spec: ModelSpec = ModelSpec()) -> TrainedModel:
    spec.validate()
    if data.empty:
        raise ValueError("No training data")
    if data[LABEL].nunique() < 2:
        raise ValueError("Training data needs positive and negative examples")
    estimator = build_estimator(spec).fit(data[list(spec.features)], data[LABEL])
    return TrainedModel(spec, estimator)


def auc(model: TrainedModel, data: pd.DataFrame) -> float:
    return float(roc_auc_score(data[LABEL], model.predict_proba(data)))


def recall(model: TrainedModel, data: pd.DataFrame) -> float:
    positives = data[data[LABEL] == 1]
    return float(model.predict(positives).mean()) if len(positives) else float("nan")


def audio_only_spec(**kwargs) -> ModelSpec:
    return ModelSpec(features=tuple(AUDIO_FEATURES), **kwargs)
