"""Model evaluation: error metrics, honest score ranges, pass/fail and at-risk checks, segments, importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

from . import schema as S

INTERVAL = (0.05, 0.95)   # a range expected to contain the true score about 90% of the time


def metrics(actual, predicted):
    """MAE/RMSE in marks, R², and how often the prediction is close. All in real score units."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    err = predicted - actual
    ss_res, ss_tot = float((err ** 2).sum()), float(((actual - actual.mean()) ** 2).sum())
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
        "median_ae": float(np.median(np.abs(err))),
        "within_5": float((np.abs(err) <= 5).mean() * 100),
        "within_10": float((np.abs(err) <= 10).mean() * 100),
        "bias": float(err.mean()),
    }


def pass_fail_metrics(actual, predicted):
    """Treat the regression as two classification questions teachers actually care about.

    * Pass or fail (pass mark 40)
    * Needs support: actual score below AT_RISK_BELOW (50). Recall = how many such students the model
      flagged; precision = how many flagged students really were below 50.
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    out = {"pass_accuracy": float(((actual >= S.PASS_MARK) == (predicted >= S.PASS_MARK)).mean() * 100)}
    truth, flag = actual < S.AT_RISK_BELOW, predicted < S.AT_RISK_BELOW
    tp, fp = int((truth & flag).sum()), int((~truth & flag).sum())
    fn, tn = int((truth & ~flag).sum()), int((~truth & ~flag).sum())
    out.update({"risk_tp": tp, "risk_fp": fp, "risk_fn": fn, "risk_tn": tn,
                "risk_precision": 100 * tp / (tp + fp) if tp + fp else 0.0,
                "risk_recall": 100 * tp / (tp + fn) if tp + fn else 0.0,
                "risk_accuracy": 100 * (tp + tn) / len(actual)})
    grades_actual = [S.grade_for(a) for a in actual]
    grades_pred = [S.grade_for(p) for p in predicted]
    out["grade_accuracy"] = float(np.mean([a == b for a, b in zip(grades_actual, grades_pred)]) * 100)
    return out


def interval_offsets(pipeline, X_train, y_train, seed=42, cv=5):
    """Quantiles of (actual − predicted) measured on out-of-fold training predictions.

    Each training row is predicted by a model that never saw it, so these offsets honestly describe how
    far off a prediction can be. A prediction p then gets the range [p + low, p + high]. The TEST set is
    not touched, which lets the test set check the range independently (coverage).
    """
    from sklearn.base import clone
    oof = np.clip(cross_val_predict(clone(pipeline), X_train, y_train, cv=KFold(cv, shuffle=True,
                                                                                  random_state=seed)), 0, 100)
    low, high = np.quantile(np.asarray(y_train, float) - oof, INTERVAL)
    return float(low), float(high)


def coverage(actual, predicted, low, high):
    """Share of true scores inside [predicted + low, predicted + high], in percent."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(((actual >= predicted + low) & (actual <= predicted + high)).mean() * 100)


def segment_errors(df, actual_col="actual", predicted_col="predicted", by="course"):
    """Error metrics for each value of a column (e.g. each course) — a model can be good overall but poor somewhere."""
    rows = []
    for key, group in df.groupby(by, observed=True):
        m = metrics(group[actual_col], group[predicted_col])
        rows.append({by: key, "n": m["n"], "mae": m["mae"], "bias": m["bias"]})
    return sorted(rows, key=lambda r: r["mae"], reverse=True)


def score_band_errors(df, actual_col="actual", predicted_col="predicted"):
    bands = pd.cut(df[actual_col], [-0.1, S.PASS_MARK - 0.01, S.AT_RISK_BELOW - 0.01, 74.99, 100],
                   labels=["Failing (under 40)", "Pass (40-49)", "Middle (50-74)", "Distinction (75+)"])
    out = []
    for label, group in df.groupby(bands, observed=True):
        m = metrics(group[actual_col], group[predicted_col])
        out.append({"band": str(label), "n": m["n"], "mae": m["mae"], "bias": m["bias"]})
    return out


def feature_importance(pipeline, X_test, y_test, seed=42, repeats=8):
    """Permutation importance: how much R² drops when one feature's values are shuffled.

    Computed on the TEST set with the full pipeline, so it reflects what the model really uses on unseen
    students. Importances below zero are reported as 0 (the feature does not help).
    """
    result = permutation_importance(pipeline, X_test, y_test, scoring="r2", n_repeats=repeats, random_state=seed,
                                    n_jobs=1)
    rows = [{"feature": f, "label": S.LABELS[f], "importance": max(0.0, float(m)), "std": float(s)}
            for f, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def noise_ceiling(df_clean):
    """For SYNTHETIC data only: how well the TRUE scoring rule explains the noisy final scores.

    No model can beat this. It is the irreducible randomness, so a model near the ceiling has learned
    everything learnable.
    """
    from .datagen import true_score
    d = df_clean
    previous = d["previous_score"].fillna(62)
    truth = true_score(previous, d["internal_score"].fillna(previous),
                       d["attendance_pct"].fillna(78), d["study_hours"].fillna(11),
                       d["assignments_pct"].fillna(72), d["sleep_hours"].fillna(7), d["screen_hours"].fillna(3),
                       d["tutoring"].fillna(0), d["internet"].fillna(1), d["course"],
                       d["parent_education"].fillna("Graduate"))
    return metrics(d[S.TARGET], truth)
