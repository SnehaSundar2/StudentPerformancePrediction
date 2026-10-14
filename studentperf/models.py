"""Candidate models, cross-validated comparison and light hyper-parameter tuning.

Every candidate is wrapped in the SAME pipeline (feature engineering → imputing/scaling/encoding →
model), so the comparison is fair and nothing learned from one validation fold leaks into another.
The target is the final score out of 100, so every error below is in real marks.
"""

import time

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from . import preprocessing as P

BASELINE = "Baseline (class average)"
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

TUNING_GRIDS = {
    "Random Forest": {"model__max_depth": [None, 8, 14], "model__min_samples_leaf": [1, 3, 6],
                      "model__max_features": [0.5, 0.8, 1.0]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.06, 0.1], "model__max_depth": [None, 3, 5],
                          "model__min_samples_leaf": [10, 20, 40], "model__l2_regularization": [0.0, 1.0]},
    "Ridge Regression": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
}


def build_pipeline(estimator):
    return Pipeline([
        ("features", P.feature_adder()),
        ("prep", P.build_preprocessor()),
        ("model", estimator),
    ])


def candidates(seed=42, fast=False):
    """name -> unfitted estimator (the pipeline is built around it)."""
    return {
        BASELINE: DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=60 if fast else 250, min_samples_leaf=2,
                                               n_jobs=1 if fast else -1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=150 if fast else 300, learning_rate=0.06,
                                                           random_state=seed),
    }


def _summarise(name, scores, seconds):
    return {"model": name, "seconds": round(seconds, 1),
            "rmse": float(-scores["test_rmse"].mean()), "rmse_std": float(scores["test_rmse"].std()),
            "mae": float(-scores["test_mae"].mean()), "r2": float(scores["test_r2"].mean())}


def compare(X, y, cv=5, seed=42, fast=False, progress=None):
    """Cross-validate every candidate on the TRAINING data only. Returns a list sorted by RMSE (best first)."""
    folds = KFold(n_splits=cv, shuffle=True, random_state=seed)
    results = []
    for name, estimator in candidates(seed, fast).items():
        if progress:
            progress(f"Cross-validating {name} ...")
        started = time.perf_counter()
        scores = cross_validate(build_pipeline(estimator), X, y, cv=folds, scoring=SCORING)
        results.append(_summarise(name, scores, time.perf_counter() - started))
    return sorted(results, key=lambda r: r["rmse"])


def best_candidate(results):
    """The best-scoring model that is not the baseline."""
    return next(r for r in results if r["model"] != BASELINE)


def tune(name, X, y, seed=42, cv=3, iterations=8, fast=False, progress=None):
    """Randomised search over a small grid. Returns (fitted_search, tuned_params) or (None, {}) if no grid."""
    grid = TUNING_GRIDS.get(name)
    if not grid:
        return None, {}
    if progress:
        progress(f"Tuning {name} ({iterations} settings × {cv} folds) ...")
    search = RandomizedSearchCV(build_pipeline(candidates(seed, fast)[name]), grid,
                                n_iter=int(min(iterations, np.prod([len(v) for v in grid.values()]))),
                                cv=KFold(cv, shuffle=True, random_state=seed),
                                scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=1, refit=False)
    search.fit(X, y)
    clean = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search, clean


def pipeline_with_params(name, params, seed=42, fast=False):
    """A fresh pipeline for `name` with tuned parameters applied."""
    estimator = candidates(seed, fast)[name]
    estimator.set_params(**params)
    return build_pipeline(estimator)
