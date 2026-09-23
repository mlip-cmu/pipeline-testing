# 01 – Subtle bugs in data wrangling code

Slides: *Subtle Bugs in Data Wrangling Code*, *Test Data Wrangling Code*.

The snippets on the slides come from public Kaggle notebooks (FIFA players,
Titanic passengers, Google Play apps). They look reasonable, but most of them
produce wrong results **silently**.

| File | Content |
|---|---|
| `wrangling/buggy.py` | The code from the slides, wrapped in functions |
| `wrangling/fixed.py` | Corrected versions |
| `tests/test_wrangling.py` | Unit tests that run against both versions |
| `demo.py` | Runs both versions on the sample data in `data/` |

The bugs:

* `add_join_year` – the year is a string, and any other date format crashes.
* `impute_age_by_title` – chained indexing (`df.loc[...].loc[...] = ...`) writes into a copy; no age is filled.
* `parse_weight` – the result of `astype` is never assigned; the column stays a string. Missing values crash.
* `parse_reviews` – typo `Reviws` creates a new column; `Reviews` stays a string.
* `parse_release_clause` – `"1.2k"` becomes `"1.2000"` = 1.2; `M` and `€` are not handled.
* `parse_size_extract` – the regex looks for `K`, the data has `k`; kB values are not scaled.
* `parse_size_replace` – `"8.7M"` becomes `"8.7000000"` = 8.7.

## Run

```sh
uv run python demo.py        # compare buggy and fixed output
uv run pytest -rxX           # buggy versions show as XFAIL/XPASS
```

Tests on the buggy code are marked `xfail`. An `XPASS` shows an input that
does *not* trigger the bug – such inputs are why the bugs go unnoticed.
Good test inputs include decimals, lower- and uppercase units, missing values,
and unexpected formats.
