"""Simulate app uploads, train a model, validate it, and publish it to the registry."""

import argparse

from covid.features import build_feature_table
from covid.model import ModelSpec, auc, recall, train
from covid.recordings import simulate_recordings
from covid.registry import ModelRegistry, ModelRejected
from covid.schema import validate_feature_table
from covid.serving import CovidDetectionService, canary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="build/registry")
    parser.add_argument("--recordings", type=int, default=1500)
    parser.add_argument("--C", type=float, default=ModelSpec.C)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    recordings = simulate_recordings(args.recordings, seed=args.seed)
    split = int(len(recordings) * 0.7)
    train_df, val_df = build_feature_table(recordings[:split]), build_feature_table(recordings[split:])
    if problems := validate_feature_table(train_df):
        raise SystemExit(f"Training data violates schema: {problems}")

    model = train(train_df, ModelSpec(C=args.C, seed=args.seed))
    metrics = {"auc": auc(model, val_df), "recall": recall(model, val_df)}
    print(f"Validation AUC {metrics['auc']:.3f}, recall {metrics['recall']:.3f}")

    registry = ModelRegistry(args.registry)
    try:
        version = registry.publish(model, metrics, canary=canary(recordings[split:split + 20]))
    except ModelRejected as e:
        raise SystemExit(f"Model not deployed: {e}")
    print(f"Deployed model v{version} to {args.registry}")

    service = CovidDetectionService.from_registry(registry)
    example = recordings[-1]
    prediction = service.predict(example)
    print(f"Example: p={prediction.probability:.2f}, label={example.covid_positive}, "
          f"top factors={service.explain(example)}")


if __name__ == "__main__":
    main()
