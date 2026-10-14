"""Data preprocessing: loading, cleaning, feature engineering and the scikit-learn transformer.

Two kinds of preprocessing, kept deliberately separate:

* clean()             — fixes or removes BAD DATA (typos, impossible values, duplicates) before the split.
                        Every rule looks at a single row (or at exact duplicates); none uses statistics
                        learned from other rows, so nothing from the test set can influence training.
* build_preprocessor() — LEARNS from data (medians for imputing, means for scaling, categories for
                        one-hot encoding). It lives inside the model pipeline and is fitted on the
                        TRAINING rows only, so nothing from the test set leaks into training.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import schema as S
from .exceptions import DataError

log = logging.getLogger("studentperf")

MIN_ROWS = 100
MAX_FILE_BYTES = 50_000_000


# ====================================================================== loading
def normalise_headers(columns):
    return [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in columns]


def load_dataset(path):
    """Read a CSV and check it has the columns the project needs."""
    path = Path(str(path).strip().strip('"'))
    if not path.exists() or not path.is_file():
        raise DataError(f"Dataset file not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{path.name} is larger than {MAX_FILE_BYTES // 1_000_000} MB.")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise DataError(f"{path.name} is empty.") from None
    except UnicodeDecodeError:
        raise DataError(f"{path.name} is not UTF-8 text. In Excel use 'Save As > CSV UTF-8'.") from None
    except (pd.errors.ParserError, OSError) as exc:
        raise DataError(f"Could not read {path.name}: {exc}") from None
    df.columns = normalise_headers(df.columns)
    missing = [c for c in S.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing column(s): {', '.join(missing)}. "
                        f"Required columns: {', '.join(S.REQUIRED_COLUMNS)}.")
    if len(df) < MIN_ROWS:
        raise DataError(f"{path.name} has only {len(df)} rows; at least {MIN_ROWS} are needed to train a "
                        "reliable model.")
    return df[S.REQUIRED_COLUMNS].copy()


# ====================================================================== cleaning
@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: list = field(default_factory=list)        # (label, count, outcome)
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)

    def add(self, label, count, outcome):
        if count:
            self.steps.append((label, int(count), outcome))

    @property
    def rows_dropped(self):
        return self.rows_in - self.rows_out

    def lines(self):
        out = [f"{self.rows_in} rows read → {self.rows_out} kept ({self.rows_dropped} removed)"]
        out += [f"  • {label}: {count} — {outcome}" for label, count, outcome in self.steps]
        return out

    def to_dict(self):
        return {"rows_in": self.rows_in, "rows_out": self.rows_out,
                "steps": [{"label": a, "count": b, "outcome": c} for a, b, c in self.steps],
                "missing_before": self.missing_before, "missing_after": self.missing_after}


def clean(df):
    """Return (clean_df, CleaningReport). The input is not modified."""
    df = df.copy()
    report = CleaningReport(rows_in=len(df))
    report.missing_before = {c: int(n) for c, n in df.isna().sum().items() if n}

    # 1. categories: fix spelling/case. A student with an unknown course cannot be predicted reliably;
    #    an unknown parent education is just a gap that is filled in later.
    for col in S.CATEGORICAL_FEATURES:
        original = df[col]
        canon = original.map(lambda v, c=col: S.canonical_category(c, v))
        report.add(f"{S.LABELS[col]} spelling standardised", (canon.notna() & (canon != original)).sum(),
                   "e.g. '  science' → 'Science'")
        unknown = canon.isna() & original.notna()
        df[col] = canon
        if col == "course":
            keep = canon.notna()
            report.add("Rows with unknown or missing course", (~keep).sum(), "removed")
            df = df[keep]
        else:
            report.add(f"Unrecognised {S.LABELS[col].lower()} values", unknown.sum(),
                       "set to missing, filled in later")

    # 2. numbers: text → numbers, yes/no → 1/0
    for col in S.NUMERIC_FEATURES + [S.TARGET]:
        before = df[col].isna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        report.add(f"Non-numeric text in {col}", (df[col].isna() & ~before).sum(), "set to missing (e.g. 'AB')")
    for col in S.BINARY_FEATURES:
        flags = df[col].map(S.parse_yes_no)
        report.add(f"Unrecognised {col} values", (flags.isna() & df[col].notna()).sum(), "set to missing")
        df[col] = flags
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. the target: a row without a valid final score (0-100) cannot teach the model anything
    lo, hi = S.SCORE_RANGE
    bad_target = df[S.TARGET].isna() | ~df[S.TARGET].between(lo, hi)
    report.add("Rows with a missing or impossible final score", bad_target.sum(),
               f"removed (valid: {lo}–{hi})")
    df = df[~bad_target]

    # 4. exact duplicates (after spelling was standardised)
    dup = df.duplicated()
    report.add("Duplicate records", dup.sum(), "removed (kept the first copy)")
    df = df[~dup]

    # 5. out-of-range feature values are probably typos, so blank them and impute later
    for col in S.NUMERIC_FEATURES:
        lo, hi = S.NUMERIC[col][1:3]
        bad = df[col].notna() & ~df[col].between(lo, hi)
        report.add(f"Out-of-range {col}", bad.sum(), f"set to missing (valid: {lo}–{hi})")
        df.loc[bad, col] = np.nan

    df = df.reset_index(drop=True)
    report.rows_out = len(df)
    report.missing_after = {c: int(n) for c, n in df.isna().sum().items() if n}
    log.info("Cleaning: %d → %d rows", report.rows_in, report.rows_out)
    if len(df) < MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows remain after cleaning (need {MIN_ROWS}). "
                        "Check that the dataset's columns and units match the schema.")
    return df, report


# ====================================================================== features
def add_features(df):
    """Engineered features. They use only other input columns — never the final score (that would be leakage)."""
    out = pd.DataFrame(df).copy()
    out["effective_study"] = out["study_hours"] * out["attendance_pct"] / 100.0   # study hours that count
    out["score_trend"] = out["internal_score"] - out["previous_score"]            # improving or slipping?
    out["sleep_gap"] = (out["sleep_hours"] - 7.5).abs()                           # distance from healthy sleep
    return out


def build_preprocessor():
    """Impute → scale numbers, impute binary flags, impute → one-hot encode categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("num", numeric, S.NUMERIC_FEATURES + S.ENGINEERED),
        ("bin", binary, S.BINARY_FEATURES),
        ("cat", categorical, S.CATEGORICAL_FEATURES),
    ])


def feature_adder():
    return FunctionTransformer(add_features)


def split_xy(df):
    return df[S.RAW_FEATURES].copy(), df[S.TARGET].astype(float).copy()
