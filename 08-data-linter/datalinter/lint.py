"""Checks for miscoded data, outliers and scaling problems, and packaging errors."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMBER = re.compile(r"^\s*[-+]?[$€£]?\s*\d[\d,]*(\.\d+)?\s*%?\s*$")
DATE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")
ZIP_NAME = re.compile(r"zip|postal|postcode", re.I)


@dataclass(frozen=True)
class Finding:
    check: str
    column: str | None
    message: str


def _text_columns(df):
    return [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]


def _numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _share(values: pd.Series, pattern: re.Pattern) -> float:
    values = values.dropna().astype(str)
    return values.str.match(pattern).mean() if len(values) else 0.0


def number_as_string(df):
    return [Finding("number_as_string", c, f"{_share(df[c], NUMBER):.0%} of values look like numbers")
            for c in _text_columns(df) if _share(df[c], NUMBER) >= 0.9]


def date_as_string(df):
    return [Finding("date_as_string", c, "values look like dates or times; parse them")
            for c in _text_columns(df) if _share(df[c], DATE) >= 0.9]


def enum_as_real(df, max_levels: int = 10):
    findings = []
    for c in _numeric_columns(df):
        values = df[c].dropna()
        if pd.api.types.is_float_dtype(values) and len(values) > 5 * max_levels \
                and values.nunique() <= max_levels and (values == values.round()).all():
            findings.append(Finding("enum_as_real", c, f"only {values.nunique()} distinct integer values; a category?"))
    return findings


def tokenizable_string(df, min_length: int = 40, min_unique: float = 0.9):
    findings = []
    for c in _text_columns(df):
        values = df[c].dropna().astype(str)
        if len(values) and values.str.len().mean() >= min_length and values.nunique() / len(values) >= min_unique:
            findings.append(Finding("tokenizable_string", c, "long, mostly unique strings; tokenize or embed them"))
    return findings


def zip_code_as_number(df):
    findings = []
    for c in _numeric_columns(df):
        values = df[c].dropna()
        five_digits = len(values) and ((values >= 0) & (values < 100_000) & (values == values.round())).all()
        if ZIP_NAME.search(str(c)) or (five_digits and values.nunique() > 20 and values.min() >= 500 and "code" in str(c).lower()):
            findings.append(Finding("zip_code_as_number", c, "zip codes are categories; leading zeros get lost"))
    return findings


def unnormalized_feature(df, max_ratio: float = 100):
    columns = [c for c in _numeric_columns(df) if df[c].std() > 0]
    if len(columns) < 2:
        return []
    spread = df[columns].std()
    typical = spread.median()
    return [Finding("unnormalized_feature", c, f"standard deviation {spread[c]:.3g} is {spread[c] / typical:.0f}x the median")
            for c in columns if spread[c] / typical > max_ratio]


def tailed_distribution(df, z: float = 6):
    findings = []
    for c in _numeric_columns(df):
        values = df[c].dropna()
        mad = (values - values.median()).abs().median()
        if len(values) < 20 or mad == 0:
            continue
        robust_z = (values - values.median()).abs() / (1.4826 * mad)
        extreme = values[robust_z > z]
        if 0 < len(extreme) <= 0.05 * len(values):
            findings.append(Finding("tailed_distribution", c, f"{len(extreme)} extreme values between "
                                                              f"{extreme.min():.3g} and {extreme.max():.3g}"))
    return findings


def uncommon_sign(df, max_share: float = 0.05):
    findings = []
    for c in _numeric_columns(df):
        values = df[c].dropna()
        values = values[values != 0]
        if len(values) < 20:
            continue
        negative = (values < 0).mean()
        minority = min(negative, 1 - negative)
        if 0 < minority <= max_share:
            sign = "negative" if negative < 0.5 else "positive"
            findings.append(Finding("uncommon_sign", c, f"{minority:.1%} of values are {sign}"))
    return findings


def duplicate_rows(df):
    n = int(df.duplicated().sum())
    return [Finding("duplicate_rows", None, f"{n} duplicate rows")] if n else []


def empty_or_missing(df, max_missing: float = 0.2):
    findings = []
    for c in df.columns:
        values = df[c]
        missing = values.isna() | (values.astype(str).str.strip().isin(["", "NA", "N/A", "null", "None", "?"]))
        if missing.all():
            findings.append(Finding("empty_or_missing", c, "column is empty"))
        elif missing.mean() > max_missing:
            findings.append(Finding("empty_or_missing", c, f"{missing.mean():.0%} of values are missing"))
    return findings


CHECKS = {
    "miscoding": [number_as_string, date_as_string, enum_as_real, tokenizable_string, zip_code_as_number],
    "outliers and scaling": [unnormalized_feature, tailed_distribution, uncommon_sign],
    "packaging": [duplicate_rows, empty_or_missing],
}


def lint(df: pd.DataFrame) -> list[Finding]:
    return [finding for checks in CHECKS.values() for check in checks for finding in check(df)]
