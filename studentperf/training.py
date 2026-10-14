"""The training workflow, step by step.

    raw data → clean → split (train/test) → compare models by cross-validation (train only)
             → tune the best → refit → evaluate ONCE on the untouched test set
             → calibrate score ranges (out-of-fold) → feature importance → save

The test set is used exactly once, for the final report; every choice (which model, which settings)
is made with cross-validation on the training rows.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, cross_validate, train_test_split

from . import __version__
from . import evaluation as E
from . import models as M
from . import preprocessing as P
from . import schema as S
from .exceptions import DataError

log = logging.getLogger("studentperf")


@dataclass
class TrainingResult:
    pipeline: object
    metadata: dict
    comparison: list
    cleaning: P.CleaningReport
    test_frame: pd.DataFrame          # actual / predicted / course for every test row
    importance: list
    reference: pd.DataFrame           # cleaned TRAINING rows (used to find similar students)
    segments: dict = field(default_factory=dict)


def train(df_raw, seed=42, fast=False, tune=True, test_size=0.2, cv=5, progress=None, source="dataset"):
    """Run the whole workflow. `progress(message)` is called as each stage starts."""
    say = progress or (lambda message: None)
    started = datetime.now()

    say("Cleaning the data ...")
    df, report = P.clean(df_raw)
    X, y = P.split_xy(df)

    say("Splitting into training and test sets ...")
    at_risk = (y < S.AT_RISK_BELOW)
    stratify = at_risk if min(at_risk.sum(), (~at_risk).sum()) >= 2 else None   # keep the at-risk share equal
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                        stratify=stratify)
    if len(X_train) < 80:
        raise DataError(f"Only {len(X_train)} training rows; need at least 80 after the split.")

    comparison = M.compare(X_train, y_train, cv=cv, seed=seed, fast=fast, progress=say)
    best = M.best_candidate(comparison)
    chosen, params = best["model"], {}

    tuned_note = ""
    if tune and chosen in M.TUNING_GRIDS:
        _, params = M.tune(chosen, X_train, y_train, seed=seed, fast=fast, progress=say)
        if params:
            say("Checking the tuned settings ...")
            tuned_pipe = M.pipeline_with_params(chosen, params, seed, fast)
            scores = cross_validate(tuned_pipe, X_train, y_train, cv=KFold(cv, shuffle=True, random_state=seed),
                                    scoring=M.SCORING)
            tuned = M._summarise(f"{chosen} (tuned)", scores, 0.0)
            if tuned["rmse"] < best["rmse"]:
                comparison = sorted(comparison + [tuned], key=lambda r: r["rmse"])
                best, tuned_note = tuned, "tuned"
            else:
                params = {}
                tuned_note = "tuning did not improve cross-validation, kept defaults"

    say("Training the final model ...")
    pipeline = (M.pipeline_with_params(chosen, params, seed, fast) if params
                else M.build_pipeline(M.candidates(seed, fast)[chosen]))
    pipeline.fit(X_train, y_train)

    say("Evaluating on the untouched test set ...")
    low_s, high_s = S.SCORE_RANGE
    test_pred = np.clip(pipeline.predict(X_test), low_s, high_s)
    train_pred = np.clip(pipeline.predict(X_train), low_s, high_s)
    test_metrics, train_metrics = E.metrics(y_test, test_pred), E.metrics(y_train, train_pred)
    test_metrics.update(E.pass_fail_metrics(y_test, test_pred))
    baseline_row = next(r for r in comparison if r["model"] == M.BASELINE)

    say("Calibrating score ranges ...")
    low, high = E.interval_offsets(pipeline, X_train, y_train, seed=seed, cv=cv)
    test_metrics["interval_coverage"] = E.coverage(y_test, test_pred, low, high)

    say("Measuring feature importance ...")
    importance = E.feature_importance(pipeline, X_test, y_test, seed=seed, repeats=3 if fast else 8)

    test_frame = X_test.copy()
    test_frame["actual"], test_frame["predicted"] = y_test.to_numpy(), test_pred
    segments = {"course": E.segment_errors(test_frame, by="course"),
                "parent_education": E.segment_errors(test_frame.dropna(subset=["parent_education"]),
                                                     by="parent_education"),
                "score_band": E.score_band_errors(test_frame)}

    reference = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    metadata = {
        "project_version": __version__,
        "trained_at": started.isoformat(timespec="seconds"),
        "training_seconds": round((datetime.now() - started).total_seconds(), 1),
        "source": source,
        "seed": seed,
        "model": chosen,
        "model_note": tuned_note,
        "model_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in params.items()},
        "rows": {"raw": report.rows_in, "clean": report.rows_out, "train": int(len(X_train)), "test": int(len(X_test))},
        "metrics": {"test": test_metrics, "train": train_metrics, "cv": best, "baseline_cv": baseline_row},
        "interval": {"low": low, "high": high, "level": 90},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "platform": sys.platform},
        "features": {"numeric": S.NUMERIC_FEATURES, "binary": S.BINARY_FEATURES, "categorical": S.CATEGORICAL_FEATURES},
        "training_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in S.NUMERIC_FEATURES + [S.TARGET]},
        "typical": {**{c: float(df[c].median()) for c in S.NUMERIC_FEATURES + S.BINARY_FEATURES},
                    **{c: str(df[c].mode().iat[0]) for c in S.CATEGORICAL_FEATURES}},
        "importance": importance,
        "comparison": comparison,
        "cleaning": report.to_dict(),
        "segments": segments,
        "test_points": [[round(float(a), 1), round(float(b), 1)] for a, b in
                        test_frame[["actual", "predicted"]].head(600).itertuples(index=False)],
    }
    log.info("Trained %s: test R²=%.3f MAE=%.2f", chosen, test_metrics["r2"], test_metrics["mae"])
    return TrainingResult(pipeline, metadata, comparison, report, test_frame, importance, reference, segments)
