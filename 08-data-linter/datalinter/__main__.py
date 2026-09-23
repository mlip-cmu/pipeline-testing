import argparse
import sys

import pandas as pd

from datalinter.lint import CHECKS


def main():
    parser = argparse.ArgumentParser(prog="datalinter", description="Lint a CSV file for common ML data problems")
    parser.add_argument("csv")
    args = parser.parse_args()
    df = pd.read_csv(args.csv, dtype_backend="numpy_nullable", keep_default_na=False, na_values=[""])
    count = 0
    for group, checks in CHECKS.items():
        print(f"== {group}")
        for check in checks:
            for finding in check(df):
                count += 1
                column = f"[{finding.column}] " if finding.column else ""
                print(f"  {finding.check}: {column}{finding.message}")
    print(f"{count} findings in {len(df)} rows, {len(df.columns)} columns")
    sys.exit(1 if count else 0)


if __name__ == "__main__":
    main()
