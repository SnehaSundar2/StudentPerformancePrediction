"""Input validation, model persistence, predictions, explanations, advice, what-if and batch files."""

import json
import math

import pandas as pd

from studentperf import persistence, schema as S, validators as v
from studentperf.exceptions import DataError, ModelError, NotTrainedError, ValidationError
from studentperf.predictor import Predictor, require_predictor
from tests.helpers import STUDENT, TempDirTest, shared


class TestValidation(TempDirTest):
    def test_accepted_formats(self):
        self.assertEqual(v.parse_number("attendance_pct", "85%"), 85.0)
        self.assertEqual(v.parse_number("study_hours", "12 hours"), 12.0)
        self.assertEqual(v.parse_number("study_hours", " 7.5 hrs "), 7.5)
        self.assertEqual(v.parse_number("previous_score", "62 marks"), 62.0)
        self.assertEqual(v.parse_number("previous_score", 0), 0.0)        # the edges of the valid range are fine
        self.assertEqual(v.parse_number("previous_score", 100), 100.0)

    def test_rejected_values_explain_why(self):
        cases = [("attendance_pct", 101, "between 0 and 100"), ("attendance_pct", -1, "between 0 and 100"),
                 ("sleep_hours", 2, "between 3 and 12"), ("study_hours", "abc", "must be a number"),
                 ("study_hours", "", "must be a number"), ("study_hours", float("nan"), "normal number"),
                 ("study_hours", float("inf"), "normal number"), ("study_hours", True, "must be a number"),
                 ("study_hours", "1e9", "must be a number")]
        for feature, value, hint in cases:
            with self.assertRaises(ValueError, msg=repr(value)) as ctx:
                v.parse_number(feature, value)
            self.assertIn(hint, str(ctx.exception))

    def test_category_aliases_and_suggestions(self):
        self.assertEqual(v.parse_category("course", " pcm "), "Science")
        self.assertEqual(v.parse_category("parent_education", "Masters"), "Postgraduate")
        with self.assertRaises(ValueError) as ctx:
            v.parse_category("course", "Scince")
        self.assertIn("Did you mean 'Science'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            v.parse_category("course", "Medicine")
        self.assertIn("Choose one of: Science, Commerce, Arts", str(ctx.exception))

    def test_defaults_are_reported(self):
        required = {k: STUDENT[k] for k in v.REQUIRED_INPUT}
        features, assumed = v.parse_features(required)
        self.assertEqual(set(assumed), set(v.DEFAULTS))
        self.assertEqual(set(features), set(S.RAW_FEATURES))
        full, assumed = v.parse_features(STUDENT)
        self.assertEqual(assumed, {})
        self.assertEqual((full["tutoring"], full["internet"]), (0, 1))

    def test_keys_are_forgiving(self):
        features, _ = v.parse_features({"Study Hours": "9", "course": "arts", "Previous-Score": 50,
                                        "internal_score": 50, "ATTENDANCE_PCT": 70})
        self.assertEqual((features["study_hours"], features["previous_score"], features["course"]), (9.0, 50.0, "Arts"))

    def test_all_problems_are_reported_together(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({"course": "Medicine", "attendance_pct": 150, "study_hours": "lots", "bogus": 1})
        errors = ctx.exception.errors
        self.assertTrue({"course", "attendance_pct", "study_hours", "previous_score", "internal_score", "_"} <= set(errors))
        with self.assertRaises(ValidationError):
            v.parse_features("not a record")

    def test_blank_values_and_yes_no(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**STUDENT, "study_hours": "  ", "previous_score": None})
        self.assertEqual(set(ctx.exception.errors), {"study_hours", "previous_score"})
        features, _ = v.parse_features({**STUDENT, "tutoring": "Yes", "internet": 0})
        self.assertEqual((features["tutoring"], features["internet"]), (1, 0))
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**STUDENT, "tutoring": "sometimes"})
        self.assertIn("yes or no", ctx.exception.errors["tutoring"])

    def test_settings_validation(self):
        self.assertEqual(v.validate_port("8010"), 8010)
        self.assertEqual(v.validate_int("5", "Rows", 1, 10), 5)
        for bad in ("abc", "80", "70000", ""):
            with self.assertRaises(ValidationError):
                v.validate_port(bad)
        for bad in ("x", "0", "11", ""):
            with self.assertRaises(ValidationError):
                v.validate_int(bad, "Rows", 1, 10)


class TestPersistence(TempDirTest):
    def test_round_trip_gives_identical_predictions(self):
        result, model_dir, _ = shared()
        a = Predictor(persistence.load(model_dir)).predict(STUDENT)
        b = Predictor(persistence.load(model_dir)).predict(STUDENT)
        self.assertEqual(a.score, b.score)
        features, _ = v.parse_features(STUDENT)
        original = float(result.pipeline.predict(pd.DataFrame([features])[S.RAW_FEATURES])[0])
        self.assertAlmostEqual(a.score, min(100.0, max(0.0, original)), places=6)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(meta["files"]), {"model.joblib", "reference.csv"})
        self.assertEqual(len(meta["files"]["model.joblib"]), 64)
        self.assertEqual([p.name for p in model_dir.glob("*.tmp")], [])        # atomic saves leave no temp files

    def test_no_model_gives_instructions(self):
        for call in (lambda: persistence.load(self.dir / "empty"), lambda: require_predictor(self.dir / "empty")):
            with self.assertRaises(NotTrainedError) as ctx:
                call()
            self.assertIn("Train one first", str(ctx.exception))

    def test_tampering_and_damage_are_detected(self):
        model = self.copy_model()
        data = bytearray((model / "model.joblib").read_bytes())
        data[len(data) // 2] ^= 0x01
        (model / "model.joblib").write_bytes(bytes(data))
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("checksum", str(ctx.exception))

        model = self.copy_model()
        (model / "reference.csv").unlink()
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("missing", str(ctx.exception))

        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["files"]["model.joblib"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ModelError):
            persistence.load(model)

        meta_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("damaged", str(ctx.exception))

        meta_path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("no checksums", str(ctx.exception))

    def test_library_version_mismatch_warns(self):
        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["environment"]["sklearn"] = "0.20.0"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        loaded = persistence.load(model)
        self.assertTrue(any("0.20.0" in w for w in loaded.warnings))


class TestPredictor(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def score(self, **changes):
        return self.predictor.predict({**STUDENT, **changes}, explain=False, similar=False).score

    def test_prediction_structure(self):
        p = self.predictor.predict(STUDENT)
        self.assertLessEqual(p.low, p.score)
        self.assertLessEqual(p.score, p.high)
        self.assertTrue(0 <= p.low and p.high <= 100)
        self.assertEqual(p.grade, S.grade_for(p.score))
        self.assertEqual(p.risk, S.risk_level(p.score, p.low))
        data = p.to_dict()
        for key in ("predicted_score", "low", "high", "grade", "risk", "factors", "advice", "similar_students",
                    "assumed_defaults", "warnings", "range_level_percent"):
            self.assertIn(key, data)
        json.dumps(data)                                                  # everything can be sent as JSON

    def test_scores_move_in_the_right_direction(self):
        self.assertGreater(self.score(study_hours=25), self.score(study_hours=2))
        self.assertGreater(self.score(attendance_pct=95), self.score(attendance_pct=50))
        self.assertGreater(self.score(previous_score=85), self.score(previous_score=40))
        self.assertGreater(self.score(internal_score=85), self.score(internal_score=40))
        self.assertGreater(self.score(screen_hours=1), self.score(screen_hours=9))
        self.assertGreater(self.score(sleep_hours=7.5), self.score(sleep_hours=4.5))
        self.assertGreater(self.score(tutoring="yes"), self.score(tutoring="no"))

    def test_strong_and_weak_students_are_flagged_correctly(self):
        strong = self.predictor.predict({**STUDENT, "previous_score": 92, "internal_score": 90, "attendance_pct": 98,
                                         "study_hours": 25, "assignments_pct": 100})
        weak = self.predictor.predict({**STUDENT, "previous_score": 35, "internal_score": 30, "attendance_pct": 50,
                                       "study_hours": 3, "assignments_pct": 30})
        self.assertEqual((strong.grade, strong.risk), ("Distinction", "On track"))
        self.assertEqual(weak.risk, "At risk")
        self.assertLess(weak.score, 45)
        self.assertLessEqual(strong.high, 100)
        self.assertGreaterEqual(weak.low, 0)

    def test_assumed_defaults_are_reported(self):
        p = self.predictor.predict({k: STUDENT[k] for k in v.REQUIRED_INPUT})
        self.assertTrue(any("Assumed for missing details" in w for w in p.warnings))
        self.assertIn("sleep_hours", p.assumed)

    def test_unusual_values_warn(self):
        p = self.predictor.predict({**STUDENT, "study_hours": 59})        # allowed, but far above any student it learned from
        self.assertTrue(any("outside the range the model learned" in w for w in p.warnings))
        p = self.predictor.predict({**STUDENT, "attendance_pct": 0.9})   # looks like a fraction
        self.assertTrue(any("percentage" in w for w in p.warnings))
        self.assertEqual(self.predictor.predict(STUDENT).warnings, [])

    def test_factors_are_sorted_and_labelled(self):
        p = self.predictor.predict({**STUDENT, "previous_score": 90, "study_hours": 2})
        effects = [abs(f["effect_marks"]) for f in p.factors]
        self.assertEqual(effects, sorted(effects, reverse=True))
        top = {f["feature"]: f for f in p.factors}
        self.assertGreater(top["previous_score"]["effect_marks"], 5)
        self.assertLess(top["study_hours"]["effect_marks"], 0)
        self.assertLessEqual(len(p.factors), 6)
        typical = self.predictor.meta["typical"]
        average_student = {k: (("yes" if round(typical[k]) else "no") if k in S.BINARY else typical[k])
                           for k in S.RAW_FEATURES}
        self.assertEqual(self.predictor.predict(average_student).factors, [])   # a typical student has nothing to explain

    def test_advice_is_ranked_and_honest(self):
        weak = self.predictor.predict({**STUDENT, "study_hours": 3, "attendance_pct": 60, "screen_hours": 7,
                                       "sleep_hours": 5})
        gains = [a["gain_marks"] for a in weak.advice]
        self.assertTrue(weak.advice)
        self.assertEqual(gains, sorted(gains, reverse=True))
        self.assertTrue(all(g >= 0.5 for g in gains))
        self.assertLessEqual(len(weak.advice), 4)
        self.assertTrue(any("sleep" in a["action"].lower() for a in weak.advice))
        ideal = self.predictor.predict({**STUDENT, "study_hours": 40, "attendance_pct": 98, "assignments_pct": 99,
                                        "screen_hours": 1, "sleep_hours": 7.5, "tutoring": "yes"})
        self.assertEqual(ideal.advice, [])                                 # nothing left to suggest

    def test_similar_students_come_from_the_same_course(self):
        p = self.predictor.predict({**STUDENT, "course": "Arts"})
        self.assertEqual(len(p.similar), 3)
        self.assertTrue(all(s["course"] == "Arts" for s in p.similar))
        self.assertTrue(all(0 <= s["final_score"] <= 100 for s in p.similar))
        near = min(abs(s["previous_score"] - 62) for s in p.similar)
        self.assertLess(near, 8)

    def test_what_if(self):
        results = self.predictor.what_if(STUDENT, "study_hours", [0, 10, 20])
        self.assertEqual([value for value, _ in results], [0.0, 10.0, 20.0])
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores))
        by_course = dict(self.predictor.what_if(STUDENT, "Course", ["science", "arts"]))
        self.assertEqual(set(by_course), {"Science", "Arts"})
        with self.assertRaises(ValidationError) as ctx:
            self.predictor.what_if(STUDENT, "study_hours", [10, 500])
        self.assertIn("500", str(ctx.exception))
        with self.assertRaises(ValidationError):
            self.predictor.what_if(STUDENT, "favourite_colour", [1])

    def test_comparison_values(self):
        self.assertEqual(self.predictor.comparison_values("course"), S.COURSES)
        self.assertEqual(self.predictor.comparison_values("tutoring"), [0, 1])
        values = self.predictor.comparison_values("study_hours")
        self.assertEqual(values, sorted(values))
        self.assertGreaterEqual(len(values), 4)

    def test_invalid_input_never_reaches_the_model(self):
        for bad in ({**STUDENT, "attendance_pct": 140}, {**STUDENT, "course": "Medicine"},
                    {**STUDENT, "study_hours": float("nan")}, {"course": "Arts"}):
            with self.assertRaises(ValidationError):
                self.predictor.predict(bad)


class TestBatch(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def setUp(self):
        super().setUp()
        self.source = self.dir / "students.csv"
        self.source.write_text(
            "Course,Previous Score,Internal Score,Attendance Pct,Study Hours,Tutoring\n"
            "Science,62,58,80,10,no\n"                  # line 2: good
            "Medicine,62,58,80,10,no\n"                 # line 3: unknown course
            "Arts,70,66,90,12,\n"                       # line 4: good, defaults used
            "Commerce,55,50,150,8,no\n"                 # line 5: impossible attendance
            "Science,60,60,80,abc,no\n"                 # line 6: bad study hours
            "Science,60,60,80,59,yes\n", encoding="utf-8")   # line 7: valid but more study than any student it learned from

    def test_good_rows_are_scored_and_bad_rows_reported(self):
        out = self.dir / "result.csv"
        ok, errors = self.predictor.predict_csv(self.source, out)
        self.assertEqual(ok, 3)
        self.assertEqual([e.split(":")[0] for e in errors], ["Line 3", "Line 5", "Line 6"])
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.assertEqual(len(frame), 6)
        self.assertEqual(list(frame.columns[:6]), ["course", "previous_score", "internal_score", "attendance_pct",
                                                    "study_hours", "tutoring"])      # original columns kept
        self.assertTrue({"predicted_score", "low", "high", "grade", "risk", "status"} <= set(frame.columns))
        self.assertIn("outside the range the model learned", frame["status"][5])    # unusual rows are flagged, not hidden
        self.assertIn("defaults used", frame["status"][2])
        self.assertTrue(frame["status"][1].startswith("error:"))
        self.assertEqual(frame["predicted_score"][1], "")                          # no score for a bad row
        self.assertTrue(0 <= float(frame["predicted_score"][0]) <= 100)
        self.assertIn(frame["risk"][0], {"On track", "Watch", "At risk"})
        self.assertTrue(math.isclose(float(frame["predicted_score"][0]),
                                     self.predictor.predict(STUDENT, explain=False, similar=False).score, abs_tol=0.06))

    def test_file_level_problems(self):
        out = self.dir / "result.csv"
        with self.assertRaises(DataError):
            self.predictor.predict_csv(self.dir / "missing.csv", out)
        no_cols = self.dir / "bad.csv"
        no_cols.write_text("name,marks\nAsha,50\n", encoding="utf-8")
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(no_cols, out)
        self.assertIn("missing required column", str(ctx.exception))
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(DataError):
            self.predictor.predict_csv(empty, out)
        self.predictor.predict_csv(self.source, out)
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(self.source, out)                   # never overwrite
        self.assertIn("already exists", str(ctx.exception))
