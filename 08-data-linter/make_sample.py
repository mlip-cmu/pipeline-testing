"""Create data/listings.csv, a rental listings dataset with typical problems."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
n = 300
streets = ["Forbes Ave", "Murray Ave", "Walnut St", "Penn Ave", "Liberty Ave", "Craig St"]
df = pd.DataFrame({
    "listing_id": [f"L{100000 + i}" for i in range(n)],
    "listed_on": pd.date_range("2026-01-01", periods=n, freq="7h").strftime("%Y-%m-%d %H:%M"),
    "zip": rng.choice([15213, 15217, 15232, 15206, 15222, 15201, 15224, 15208, 15203, 15210, 15212, 15215,
                       15216, 15218, 15219, 15220, 15221, 15223, 15226, 15227, 15233, 15235], n),
    "bedrooms": rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], n, p=[0.1, 0.35, 0.3, 0.2, 0.05]),
    "sqft": rng.normal(850, 250, n).round(),
    "rent": [f"${v:,.0f}" for v in rng.normal(1500, 400, n)],
    "distance_to_campus_m": rng.gamma(2, 1500, n).round(),
    "days_on_market": rng.poisson(12, n),
    "description": [f"Bright {b}-bedroom apartment on {rng.choice(streets)}, unit {rng.integers(1, 400)}, "
                    f"close to buses, laundry in building, {rng.choice(['pets ok', 'no pets', 'cats only'])}."
                    for b in rng.integers(1, 5, n)],
    "pet_deposit": np.where(rng.random(n) < 0.7, np.nan, rng.choice([200, 300], n)),
    "rating": rng.normal(4.2, 0.4, n).round(1),
    "price_change": rng.normal(20, 5, n).round(),
})
df.loc[[5, 77], "sqft"] = [85_000, 120_000]
df.loc[[10, 50, 90, 130], "price_change"] = [-40, -35, -50, -45]
df = pd.concat([df, df.iloc[[3, 4, 5]]], ignore_index=True)
df.to_csv("data/listings.csv", index=False)
