"""Generate synthetic hourly delivery counts (raw data with some quality problems)."""

import argparse

import numpy as np
import pandas as pd

HOUR_EFFECT = np.array([-3.0, -3.5, -4.0, -4.0, -4.0, -3.5, -2.5, -1.5, -1.0, -0.5, 0.0, 1.0,
                        2.0, 1.2, 0.3, 0.0, 0.3, 1.0, 2.5, 3.0, 2.2, 1.0, 0.0, -1.5])
DAY_EFFECT = np.array([-0.6, -0.5, -0.4, -0.2, 0.6, 1.2, 1.0])
MONTH_EFFECT = np.array([1.0, 0.8, 0.4, 0.0, -0.3, -0.6, -0.8, -0.6, -0.2, 0.2, 0.6, 1.2])
WEATHER_EFFECT = np.array([0.0, 0.4, 1.2, 2.0])


def generate(start: str, end: str, seed: int, dirty: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(times)
    weather = rng.choice([1, 2, 3, 4], size=n, p=[0.6, 0.25, 0.12, 0.03])
    temp = 12 - 10 * np.cos(2 * np.pi * (times.dayofyear.to_numpy() - 15) / 365) + rng.normal(0, 3, n)
    humidity = np.clip(rng.normal(65, 15, n) + 5 * weather, 5, 100)
    windspeed = np.abs(rng.normal(12, 6, n))

    transformed = (
        12
        + HOUR_EFFECT[times.hour]
        + DAY_EFFECT[times.dayofweek]
        + MONTH_EFFECT[times.month - 1]
        + WEATHER_EFFECT[weather - 1]
        - 0.05 * temp
        + rng.normal(0, 0.6, n)
    )
    count = np.clip((0.4 * transformed + 1) ** (1 / 0.4) - 1, 0, None).round().astype(int)

    df = pd.DataFrame({
        "datetime": times.strftime("%Y-%m-%d %H:%M:%S"),
        "weather": weather,
        "temp": temp.round(1),
        "humidity": humidity.round(0),
        "windspeed": windspeed.round(1),
        "delivery_count": count,
    })
    if dirty:
        idx = rng.choice(n, size=max(n // 200, 3), replace=False)
        df.loc[idx[0::3], "delivery_count"] = -1
        df.loc[idx[1::3], "humidity"] = 250.0
        df = df.astype({"delivery_count": "float"})
        df.loc[idx[2::3], "delivery_count"] = np.nan
        df = pd.concat([df, df.sample(n // 500 + 1, random_state=seed)]).sort_index(kind="stable")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate("2023-01-01", "2024-01-01", args.seed).to_csv("data/train.csv", index=False)
    generate("2024-01-01", "2024-07-01", args.seed + 1).to_csv("data/test.csv", index=False)
    generate("2023-01-01", "2023-04-01", 7).to_csv("tests/data/pipelinetest_training.csv", index=False)
    generate("2024-01-01", "2024-02-01", 8).to_csv("tests/data/pipelinetest_test.csv", index=False)


if __name__ == "__main__":
    main()
