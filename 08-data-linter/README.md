# 08 – Bonus: a data linter

Slide: *Bonus: Data Linter at Google* (Hynes, Sculley, and Terry, *The Data
Linter: Lightweight, Automated Sanity Checking for ML Data Sets*, NIPS MLSys
Workshop 2017).

`datalinter/lint.py` has one small function for each check on the slide:

| Group | Checks |
|---|---|
| Miscoding | number as string, date/time as string, enum as real, tokenizable string, zip code as number |
| Outliers and scaling | unnormalized feature, tailed distribution, uncommon sign |
| Packaging | duplicate rows, empty/missing data |

The checks are heuristics: they point to data to look at, not certain errors.
`data/listings.csv` (made with `make_sample.py`) is a rental listings dataset
with one example of each problem.

## Run

```sh
uv run python -m datalinter data/listings.csv   # exit code 1 if there are findings
uv run pytest
```
