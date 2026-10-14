"""Making predictions: score + range, grade, support flag, the factors behind it, study advice,
what-if analysis, similar students and batch files."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import persistence
from . import schema as S
from .exceptions import DataError, NotTrainedError, ValidationError
from .preprocessing import normalise_headers
from .validators import REQUIRED_INPUT, parse_features

MAX_BATCH_ROWS = 5000
LOW, HIGH = S.SCORE_RANGE


@dataclass
class Prediction:
    features: dict
    assumed: dict
    score: float
    low: float
    high: float
    grade: str
    risk: str
    warnings: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    advice: list = field(default_factory=list)
    similar: list = field(default_factory=list)

    def to_dict(self):
        return {"predicted_score": round(self.score, 1), "low": round(self.low, 1), "high": round(self.high, 1),
                "range_level_percent": 90, "grade": self.grade, "risk": self.risk, "features": self.features,
                "assumed_defaults": self.assumed, "warnings": self.warnings, "factors": self.factors,
                "advice": self.advice, "similar_students": self.similar}


def _frame(rows):
    """DataFrame in the exact column order the pipeline was trained on."""
    return pd.DataFrame(rows)[S.RAW_FEATURES]


class Predictor:
    def __init__(self, loaded):
        self.model = loaded.pipeline
        self.meta = loaded.metadata
        self.reference = loaded.reference
        self.load_warnings = list(loaded.warnings)
        self._low, self._high = self.meta["interval"]["low"], self.meta["interval"]["high"]
        self._typical = self.meta["typical"]
        self._ranges = self.meta["training_ranges"]

    @classmethod
    def from_directory(cls, directory):
        return cls(persistence.load(directory))

    # ------------------------------------------------------------------ core
    def _score(self, rows):
        return np.clip(np.asarray(self.model.predict(_frame(rows)), float), LOW, HIGH)

    def _bounds(self, score):
        return float(np.clip(score + self._low, LOW, HIGH)), float(np.clip(score + self._high, LOW, HIGH))

    def _range_warnings(self, f):
        out = []
        for name in S.NUMERIC_FEATURES:
            low, high = self._ranges[name]
            if f[name] < low or f[name] > high:
                out.append(f"{S.LABELS[name]} {f[name]:g} is outside the range the model learned "
                           f"({low:.0f}–{high:.0f}); the estimate is less reliable.")
        if 0 < f["attendance_pct"] <= 1:
            out.append("Attendance is a percentage (0–100). A value like 0.9 means under 1%; "
                       "enter 90 for ninety percent.")
        return out

    def predict(self, raw, explain=True, similar=True):
        features, assumed = parse_features(raw)
        score = float(self._score([features])[0])
        low, high = self._bounds(score)
        result = Prediction(features, assumed, score, low, high, S.grade_for(score), S.risk_level(score, low),
                            self._range_warnings(features))
        if assumed:
            result.warnings.append("Assumed for missing details: " + ", ".join(
                f"{S.LABELS[k].lower()} = {self._show(k, v)}" for k, v in assumed.items()) + ".")
        if explain:
            result.factors = self._factors(features, score)
            result.advice = self._advice(features, score)
        if similar:
            result.similar = self._similar(features)
        return result

    @staticmethod
    def _show(name, value):
        if name in S.BINARY:
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return str(value)

    # ------------------------------------------------------------------ explanations
    def _factors(self, features, score, top=6):
        """How many marks each detail adds or removes compared with a typical student.

        Each detail is swapped, one at a time, for its typical value (the median, or the most common
        category) and the model re-scores the student; the difference is that detail's effect.
        """
        variants, names = [], []
        for name in S.RAW_FEATURES:
            typical = self._typical[name]
            if name in S.BINARY:
                typical = int(round(typical))
            if features[name] != typical:
                variants.append({**features, name: typical})
                names.append((name, typical))
        if not variants:
            return []
        scores = self._score(variants)
        rows = []
        for (name, typical), alt in zip(names, scores):
            effect = score - float(alt)
            rows.append({"feature": name, "label": S.LABELS[name], "value": self._show(name, features[name]),
                         "typical": self._show(name, typical), "effect_marks": round(effect, 1)})
        rows = [r for r in rows if abs(r["effect_marks"]) >= 0.3]
        return sorted(rows, key=lambda r: abs(r["effect_marks"]), reverse=True)[:top]

    def _advice(self, features, score, top=4):
        """Realistic changes a student controls, ranked by the extra marks the model expects from each."""
        options = []
        study = features["study_hours"]
        if study < 30:
            options.append((f"Study 5 more hours a week ({study:g} → {study + 5:g})", {"study_hours": study + 5}))
        attendance = features["attendance_pct"]
        if attendance < 95:
            target = min(100, attendance + 10)
            options.append((f"Raise attendance by {target - attendance:g} points ({attendance:g}% → {target:g}%)",
                            {"attendance_pct": target}))
        assignments = features["assignments_pct"]
        if assignments < 95:
            target = min(100, assignments + 15)
            options.append((f"Submit more assignments ({assignments:g}% → {target:g}%)", {"assignments_pct": target}))
        screen = features["screen_hours"]
        if screen > 3:
            options.append((f"Cut leisure screen time by 2 hours ({screen:g} → {max(0, screen - 2):g} h/day)",
                            {"screen_hours": max(0.0, screen - 2)}))
        sleep = features["sleep_hours"]
        if abs(sleep - 7.5) > 1:
            options.append((f"Sleep closer to 7½ hours ({sleep:g} → 7.5 h)", {"sleep_hours": 7.5}))
        if not features["tutoring"]:
            options.append(("Join extra tutoring or a study group", {"tutoring": 1}))
        if not options:
            return []
        scores = self._score([{**features, **change} for _, change in options])
        rows = [{"action": text, "gain_marks": round(float(s) - score, 1)} for (text, _), s in zip(options, scores)]
        rows = [r for r in rows if r["gain_marks"] >= 0.5]
        return sorted(rows, key=lambda r: r["gain_marks"], reverse=True)[:top]

    def what_if(self, raw, feature, values):
        """Score for the same student with one detail changed. Returns [(value, score), ...]."""
        feature = feature.strip().lower().replace(" ", "_")
        if feature not in S.RAW_FEATURES:
            raise ValidationError(f"Unknown feature '{feature}'. Choose one of: {', '.join(S.RAW_FEATURES)}.")
        features, _ = parse_features(raw)
        variants, shown = [], []
        for value in values:
            try:
                changed, _ = parse_features({**features, feature: value})
            except ValidationError as exc:
                raise ValidationError(f"{value!r} does not work for {S.LABELS[feature].lower()}: "
                                      f"{exc.errors.get(feature) or exc}") from None
            variants.append(changed)
            shown.append(changed[feature])
        return list(zip(shown, self._score(variants).tolist()))

    def comparison_values(self, feature):
        """Sensible values to try in a what-if for a feature."""
        if feature in S.CATEGORICAL:
            return list(S.CATEGORICAL[feature])
        if feature in S.BINARY:
            return [0, 1]
        lo, hi = self._ranges[feature]
        return sorted({round(float(p), 1) for p in np.linspace(lo, hi, 6)})

    def _similar(self, features, n=3):
        ref = self.reference
        pool = ref[ref["course"] == features["course"]]
        if len(pool) < n:
            pool = ref
        cols = ("previous_score", "internal_score", "attendance_pct", "study_hours")
        scale = ref[list(cols)].std().replace(0, 1)
        dist = np.sqrt(sum(((pool[c].fillna(pool[c].median()) - features[c]) / scale[c]) ** 2 for c in cols))
        rows = []
        for idx in dist.nsmallest(n).index:
            r = pool.loc[idx]
            rows.append({"course": r["course"], "previous_score": round(float(r["previous_score"]), 1),
                         "internal_score": None if pd.isna(r["internal_score"]) else round(float(r["internal_score"]), 1),
                         "attendance_pct": None if pd.isna(r["attendance_pct"]) else round(float(r["attendance_pct"]), 1),
                         "study_hours": None if pd.isna(r["study_hours"]) else round(float(r["study_hours"]), 1),
                         "final_score": round(float(r[S.TARGET]), 1)})
        return rows

    # ------------------------------------------------------------------ batch
    def predict_csv(self, in_path, out_path):
        """Predict every row of a CSV. Bad rows are reported (with line numbers), good rows still get scores.

        Returns (ok_count, errors) where errors is a list of 'Line N: message'. The output CSV has the
        original columns plus predicted_score, low, high, grade, risk and a status column.
        """
        in_path, out_path = Path(str(in_path).strip().strip('"')), Path(str(out_path).strip().strip('"'))
        if not in_path.exists():
            raise DataError(f"File not found: {in_path}")
        if out_path.exists():
            raise DataError(f"{out_path} already exists; choose a new file name.")
        try:
            df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            raise DataError(f"{in_path.name} is empty.") from None
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            raise DataError(f"Could not read {in_path.name}: {exc}") from None
        df.columns = normalise_headers(df.columns)
        if len(df) > MAX_BATCH_ROWS:
            raise DataError(f"{in_path.name} has {len(df)} rows; the limit is {MAX_BATCH_ROWS}.")
        missing = [c for c in REQUIRED_INPUT if c not in df.columns]
        if missing:
            raise DataError(f"{in_path.name} is missing required column(s): {', '.join(missing)}.")

        parsed, errors, status = [], [], []
        for position, row in enumerate(df.to_dict("records")):
            line = position + 2   # header is line 1
            try:
                features, assumed = parse_features({k: v for k, v in row.items() if k in S.RAW_FEATURES})
                parsed.append((position, features))
                notes = (["defaults used"] if assumed else []) + \
                        (["outside the range the model learned: less reliable"] if self._range_warnings(features) else [])
                status.append("ok" + (f" ({'; '.join(notes)})" if notes else ""))
            except ValidationError as exc:
                errors.append(f"Line {line}: {exc}")
                status.append(f"error: {exc}")
                parsed.append((position, None))
        valid = [(i, f) for i, f in parsed if f is not None]
        scores = self._score([f for _, f in valid]) if valid else []
        out = df.copy()
        for col in ("predicted_score", "low", "high", "grade", "risk"):
            out[col] = ""
        for (i, _), score in zip(valid, scores):
            low, high = self._bounds(float(score))
            out.loc[i, "predicted_score"] = f"{score:.1f}"
            out.loc[i, "low"], out.loc[i, "high"] = f"{low:.1f}", f"{high:.1f}"
            out.loc[i, "grade"] = S.grade_for(float(score))
            out.loc[i, "risk"] = S.risk_level(float(score), low)
        out["status"] = status
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        except OSError as exc:
            raise DataError(f"Could not write {out_path}: {exc.strerror or exc}") from exc
        return len(valid), errors


def require_predictor(directory):
    """Load the saved model or raise NotTrainedError with instructions."""
    if not persistence.is_trained(directory):
        raise NotTrainedError(f"No trained model found in '{directory}'. Train one first "
                              "(menu option 'Train a model', or: python main.py --train).")
    return Predictor.from_directory(directory)
