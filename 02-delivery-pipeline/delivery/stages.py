"""Pipeline steps as separate commands, so that an orchestrator (here: DVC) can run and cache them."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from delivery.model import evaluate, learn, mean_absolute_count_error
from delivery.pipeline import prepare_data

TARGET = "delivery_count"


def prepare(raw_path: str, features_path: str) -> None:
    X, y = prepare_data(pd.read_csv(raw_path))
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    X.assign(**{TARGET: y}).to_csv(features_path, index=False)


def _load_features(path: str):
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def train(features_path: str, model_path: str, alpha: float) -> None:
    model = learn(*_load_features(features_path), alpha=alpha)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def evaluate_model(model_path: str, features_path: str, metrics_path: str) -> None:
    model = joblib.load(model_path)
    X, y = _load_features(features_path)
    metrics = {"r2": evaluate(model, X, y), "mae_deliveries": mean_absolute_count_error(model, X, y)}
    Path(metrics_path).write_text(json.dumps(metrics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("raw")
    p.add_argument("features")
    t = sub.add_parser("train")
    t.add_argument("features")
    t.add_argument("model")
    t.add_argument("--alpha", type=float, default=0.0)
    e = sub.add_parser("evaluate")
    e.add_argument("model")
    e.add_argument("features")
    e.add_argument("metrics")
    args = parser.parse_args()

    if args.stage == "prepare":
        prepare(args.raw, args.features)
    elif args.stage == "train":
        train(args.features, args.model, args.alpha)
    else:
        evaluate_model(args.model, args.features, args.metrics)


if __name__ == "__main__":
    main()
