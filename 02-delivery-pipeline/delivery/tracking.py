"""Train models with different regularization and track the results with MLflow."""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd

from delivery.model import evaluate, learn, mean_absolute_count_error
from delivery.pipeline import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 5.0, 50.0])
    args = parser.parse_args()

    X_train, y_train = prepare_data(pd.read_csv("data/train.csv"))
    X_test, y_test = prepare_data(pd.read_csv("data/test.csv"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("delivery-count")
    for alpha in args.alphas:
        with mlflow.start_run(run_name=f"ridge-alpha-{alpha}"):
            mlflow.log_param("regularization", alpha)
            mlflow.log_param("train_rows", len(X_train))
            model = learn(X_train, y_train, alpha)
            mlflow.log_metric("r2", evaluate(model, X_test, y_test))
            mlflow.log_metric("mae_deliveries", mean_absolute_count_error(model, X_test, y_test))
            mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(3))
            print(f"alpha={alpha}: logged run {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
