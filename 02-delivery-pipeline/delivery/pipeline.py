import argparse

import pandas as pd

from delivery.features import (
    clean_data,
    encode_count,
    encode_day_of_week,
    encode_hour,
    encode_month,
    encode_weather,
)
from delivery.model import evaluate, learn


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_data(df)
    df = encode_day_of_week(df)
    df = encode_month(df)
    df = encode_hour(df)
    df = encode_weather(df)
    df = df.drop(["datetime"], axis=1)
    return df.drop(["delivery_count"], axis=1), encode_count(df["delivery_count"])


def pipeline(train_path: str = "data/train.csv", test_path: str = "data/test.csv", alpha: float = 0.0):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = prepare_data(train)
    X_test, y_test = prepare_data(test)
    model = learn(X_train, y_train, alpha)
    accuracy = evaluate(model, X_test, y_test)
    return model, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the delivery count model")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    _, accuracy = pipeline(args.train, args.test, args.alpha)
    print(f"R^2 on test data: {accuracy:.3f}")


if __name__ == "__main__":
    main()
