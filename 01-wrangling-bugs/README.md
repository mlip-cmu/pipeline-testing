# 01 – Subtle bugs in data wrangling code

Slides: *Subtle Bugs in Data Wrangling Code*, *Test Data Wrangling Code*.

`wrangling.ipynb` cleans a dataset of Android apps with code from popular
Kaggle notebooks, collected in Yang et al., *Subtle Bugs Everywhere: Generating
Documentation for Data Wrangling Code*, ASE 2021. The original snippets used
different datasets (Google Play Store, FIFA 19, Titanic); here they all use one
dataset. Every cell runs without an exception and shows `df.head()`, but most
cells do not do what they intend. Most problems are not visible in the first
five rows.

`data/googleplaystore.csv` is synthetic data in the format of the Kaggle
[Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps)
dataset, made with `make_dataset.py`.

## Run

```sh
uv run --group notebook jupyter lab wrangling.ipynb
```

## The bugs (answer key)

| Cell | Intent | What happens | Visible in `head()`? |
|---|---|---|---|
| Size | `'19M'`, `'201k'` → bytes | Regex looks for `K`, data has `k`: `'201k'` becomes 201 (3% of rows) | No |
| Fill missing sizes | Replace NaN with the mean | Chained `inplace` call on a column; with pandas ≥ 3 nothing changes (pandas prints a warning) | Yes (row 3) |
| Update year | Drop apps without date, extract the year | `dropna()` on the column does not drop rows; year is a string | No |
| Ratings | Fill missing ratings with the category mean | `.loc[...].loc[...] = ...` assigns to a copy; no rating is filled (pandas ≥ 3 warns) | No |
| Installs | `'10,000+'` → 10000 | Result of `astype(int)` is never assigned; column stays a string | No (looks like a number) |
| Reviews | `'54k'`, `'2.1M'` → numbers | `'3.4k'` becomes `'3.4000'` = 3.4, `'2.1M'` becomes 2.1 | No |
| Reviews as integers | Convert to `int` | Typo `Reviws` creates a new column; `Reviews` stays float | Yes (new column) |

The last cell, `df.describe()`, gives hints for some of the bugs: the smallest
app has 23 bytes, no app has more than a million reviews, and `Installs` is
missing from the numeric columns.

## Tests: which bugs do they find?

`functions-and-tests/` has the same cells as functions (`wrangling/buggy.py`),
corrected versions (`wrangling/fixed.py`), and two groups of tests:

* `test_happy_path.py` – tests written from the first rows of the data, the rows seen in the notebook.
* `test_edge_cases.py` – tests written after thinking about what else is in the data: other units, decimals, missing values, types.
* `test_parse_size.py` – the test from the slide *Anatomy of a Unit Test*.

```sh
uv run pytest                                   # fixed version: all tests pass (and the notebook runs)
uv run pytest functions-and-tests --impl buggy  # buggy version
```

With the buggy version, the happy-path tests find only 3 of the 7 bugs
(missing sizes not filled, year as string, installs as string). Every
happy-path test for the size, rating, and review conversions passes. The
edge-case tests find all bugs, but only because we thought about the problems
when we wrote them.
