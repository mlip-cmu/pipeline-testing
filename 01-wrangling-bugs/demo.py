"""Run buggy and fixed wrangling code on the sample data and compare the results."""

import warnings

import pandas as pd

from wrangling import buggy, fixed


def drop_unknown_sizes(df):
    return df[df.Size != "Varies with device"].reset_index(drop=True)


CASES = [
    ("players.csv", "add_join_year", ["add_join_year"], ["Joined", "Join_year"], None),
    ("passengers.csv", "impute_age_by_title", ["impute_age_by_title"], ["Title", "Age"], None),
    ("players.csv", "parse_weight", ["parse_weight"], ["Weight"], None),
    ("apps.csv", "parse_reviews", ["parse_reviews"], None, None),
    ("players.csv", "parse_release_clause", ["parse_release_clause"], ["Release Clause"], None),
    ("apps.csv", "parse_size", ["parse_size_extract", "parse_size_replace"], ["App", "Size"], drop_unknown_sizes),
]


def run(module, function, path, columns, prepare):
    df = pd.read_csv(f"data/{path}", dtype={"Reviews": str})
    if prepare:
        df = prepare(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = getattr(module, function)(df)
        except Exception as e:
            return f"CRASH: {type(e).__name__}: {e}"
    shown = out[columns] if columns else out
    return f"dtypes: {dict(shown.dtypes.astype(str))}\n{shown.to_string()}"


def main():
    for path, fixed_name, buggy_names, columns, prepare in CASES:
        print("=" * 80)
        for name in buggy_names:
            print(f"--- buggy.{name} ({path})\n{run(buggy, name, path, columns, prepare)}\n")
        print(f"--- fixed.{fixed_name} ({path})\n{run(fixed, fixed_name, path, columns, prepare)}\n")


if __name__ == "__main__":
    main()
