"""Generate data/googleplaystore.csv: synthetic app data in the format of the Kaggle "Google Play Store Apps" dataset."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(2019)
N = 2500

CATEGORIES = {
    "ART_AND_DESIGN": "Art & Design", "BOOKS_AND_REFERENCE": "Books & Reference", "BUSINESS": "Business",
    "COMMUNICATION": "Communication", "EDUCATION": "Education", "ENTERTAINMENT": "Entertainment",
    "FAMILY": "Casual", "FINANCE": "Finance", "GAME": "Action", "HEALTH_AND_FITNESS": "Health & Fitness",
    "LIFESTYLE": "Lifestyle", "MAPS_AND_NAVIGATION": "Maps & Navigation", "MEDICAL": "Medical",
    "PHOTOGRAPHY": "Photography", "PRODUCTIVITY": "Productivity", "SHOPPING": "Shopping",
    "SOCIAL": "Social", "SPORTS": "Sports", "TOOLS": "Tools", "TRAVEL_AND_LOCAL": "Travel & Local",
}
WORDS_A = ["Smart", "Super", "Easy", "Pocket", "Daily", "Pixel", "Magic", "Quick", "Simple", "Happy", "Pro",
           "Mini", "Ultimate", "Tiny", "Free", "Cool", "Bright", "Star", "Fast", "Secure"]
WORDS_B = ["Photo Editor", "Coloring Book", "Launcher", "Notes", "Budget", "Recipes", "Weather", "Scanner",
           "Flashlight", "Workout", "Translator", "Chess", "Puzzle", "Calendar", "Radio", "Wallpapers",
           "Messenger", "Maps", "Keyboard", "Music Player", "Diet Tracker", "Bible", "Dictionary", "VPN"]
INSTALL_BUCKETS = [1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000,
                   1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def fmt_reviews(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000 and rng.random() < 0.3:
        return f"{n / 1_000:.1f}k" if rng.random() < 0.5 else f"{round(n / 1_000)}k"
    return str(n)


def fmt_size(kind: str) -> str:
    if kind == "varies":
        return "Varies with device"
    if kind == "k":
        return f"{rng.integers(20, 999)}k"
    mb = rng.lognormal(2.6, 0.9)
    return f"{mb:.1f}M" if mb < 10 else f"{min(round(mb), 100)}M"


rows = []
for i in range(N):
    category = rng.choice(list(CATEGORIES))
    reviews = int(rng.lognormal(7, 3))
    installs = next((b for b in INSTALL_BUCKETS if b >= reviews * rng.uniform(5, 40)), INSTALL_BUCKETS[-1])
    paid = rng.random() < 0.07
    rating = np.nan if rng.random() < (0.5 if reviews < 20 else 0.05) else round(float(np.clip(rng.normal(4.2, 0.45), 1, 5)), 1)
    day = pd.Timestamp("2013-01-01") + pd.Timedelta(days=int(rng.beta(5, 1) * 2040))
    rows.append({
        "App": f"{rng.choice(WORDS_A)} {rng.choice(WORDS_B)}" + (f" {rng.integers(2, 20)}" if rng.random() < 0.3 else ""),
        "Category": category,
        "Rating": rating,
        "Reviews": fmt_reviews(reviews),
        "Size": fmt_size(rng.choice(["M", "k", "varies"], p=[0.815, 0.035, 0.15])),
        "Installs": f"{installs:,}+",
        "Type": "Paid" if paid else "Free",
        "Price": f"${rng.choice([0.99, 1.99, 2.99, 4.99, 9.99])}" if paid else "0",
        "Content Rating": rng.choice(["Everyone", "Teen", "Everyone 10+", "Mature 17+"], p=[0.8, 0.11, 0.04, 0.05]),
        "Genres": CATEGORIES[category],
        "Last Updated": np.nan if rng.random() < 0.01 else f"{MONTHS[day.month - 1]} {day.day}, {day.year}",
        "Current Ver": "Varies with device" if rng.random() < 0.12 else f"{rng.integers(1, 9)}.{rng.integers(0, 20)}.{rng.integers(0, 10)}",
        "Android Ver": rng.choice(["4.0.3 and up", "4.1 and up", "4.4 and up", "5.0 and up", "Varies with device"]),
    })

df = pd.DataFrame(rows)
# the first rows look clean: sizes in MB (one "Varies with device"), plain review counts, known ratings and dates
head = df[df.Size.str.endswith("M") & df.Reviews.str.isdigit() & df.Rating.notna() & df["Last Updated"].notna()].head(11)
varies = df[(df.Size == "Varies with device") & df.Reviews.str.isdigit() & df.Rating.notna() & df["Last Updated"].notna()].head(1)
first = pd.concat([head.iloc[:3], varies, head.iloc[3:]])
df = pd.concat([first, df.drop(first.index)]).reset_index(drop=True)
df.to_csv("data/googleplaystore.csv", index=False)

DECIMAL = r"\.\d[kM]$"
print(df.head(12).to_string())
print(f"\n{len(df)} rows; sizes in k: {df.Size.str.endswith('k').mean():.1%}, varies: {(df.Size == 'Varies with device').mean():.1%}, "
      f"missing ratings: {df.Rating.isna().mean():.1%}, reviews with k/M: {df.Reviews.str.contains('[kM]').mean():.1%}, "
      f"decimal k/M reviews: {df.Reviews.str.contains(DECIMAL).mean():.1%}, missing dates: {df['Last Updated'].isna().mean():.1%}")
first_k = df.index[df.Size.str.endswith("k")][0]
print("first 'k' size at row", first_k, "; first abbreviated review at row", df.index[df.Reviews.str.contains("[kM]")][0],
      "; first missing rating at row", df.index[df.Rating.isna()][0], "; first missing date at row", df.index[df["Last Updated"].isna()][0])
