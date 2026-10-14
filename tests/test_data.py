"""Synthetic data, schema helpers, dataset loading, cleaning and feature engineering."""

import numpy as np
import pandas as pd

from studentperf import datagen, models as M, preprocessing as P, schema as S
from studentperf.exceptions import DataError
from tests.helpers import TempDirTest


class TestDatagen(TempDirTest):
    def test_reproducible_and_dirty(self):
        a, b = datagen.generate(400, seed=1), datagen.generate(400, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(datagen.generate(400, seed=2)))
        self.assertGreater(a.isna().sum().sum(), 0)
        self.assertGreater(a.duplicated().sum(), 0)
        clean = datagen.generate(400, dirty=False)
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertEqual(set(clean.columns), set(S.REQUIRED_COLUMNS))
        self.assertTrue(clean[S.TARGET].between(0, 100).all())

    def test_scoring_rule_is_sensible(self):
        def score(**kw):
            base = dict(previous=[60.0], internal=[60.0], attendance=[80.0], study=[10.0], assignments=[75.0],
                        sleep=[7.5], screen=[3.0], tutoring=[0], internet=[1], course=["Commerce"],
                        parent=["Graduate"])
            base.update(kw)
            return float(datagen.true_score(**base)[0])
        self.assertGreater(score(study=[25.0]), score(study=[2.0]))
        self.assertGreater(score(attendance=[95.0]), score(attendance=[45.0]))
        self.assertGreater(score(previous=[85.0]), score(previous=[40.0]) + 10)
        self.assertGreater(score(screen=[1.0]), score(screen=[9.0]))
        self.assertGreater(score(sleep=[7.5]), score(sleep=[4.5]))        # too little sleep hurts
        self.assertGreater(score(sleep=[7.5]), score(sleep=[11.0]))       # so does too much
        self.assertGreater(score(tutoring=[1]), score(tutoring=[0]))
        self.assertLessEqual(score(previous=[100.0], internal=[100.0], attendance=[100.0], study=[40.0],
                                   assignments=[100.0], tutoring=[1], parent=["Postgraduate"]), 100)

    def test_sample_data_file_is_dirty_but_cleanable(self):
        cleaned, report = P.clean(datagen.generate(1800))
        self.assertGreater(report.rows_dropped, 15)
        self.assertLess(report.rows_dropped, 120)
        self.assertEqual(cleaned.duplicated().sum(), 0)


class TestSchemaHelpers(TempDirTest):
    def test_categories_and_aliases(self):
        cases = [("course", "  science", "Science"), ("course", "COMMERCE ", "Commerce"), ("course", "humanities", "Arts"),
                 ("parent_education", "post_graduate", "Postgraduate"), ("parent_education", "10th", "School"),
                 ("parent_education", "DEGREE", "Graduate")]
        for feature, raw, expected in cases:
            self.assertEqual(S.canonical_category(feature, raw), expected, raw)
        for raw in ("Medicine", "", None, float("nan")):
            self.assertIsNone(S.canonical_category("course", raw), repr(raw))

    def test_yes_no(self):
        self.assertEqual([S.parse_yes_no(x) for x in ("Yes", "n", True, 0, "1", "TRUE")], [1, 0, 1, 0, 1, 1])
        self.assertIsNone(S.parse_yes_no("maybe"))
        self.assertIsNone(S.parse_yes_no(2))

    def test_grades_and_risk_levels(self):
        self.assertEqual([S.grade_for(x) for x in (100, 75, 74.9, 60, 50, 40, 39.9, 0)],
                         ["Distinction", "Distinction", "First class", "First class", "Second class", "Pass",
                          "Fail", "Fail"])
        self.assertEqual(S.risk_level(45, 38), "At risk")
        self.assertEqual(S.risk_level(55, 47), "Watch")       # under 60
        self.assertEqual(S.risk_level(65, 38), "Watch")       # likely OK, but the range reaches the fail zone
        self.assertEqual(S.risk_level(75, 66), "On track")

    def test_no_personal_attributes_are_used(self):
        forbidden = {"gender", "religion", "caste", "income", "name", "age"}
        self.assertFalse(forbidden & set(S.RAW_FEATURES))


class TestLoading(TempDirTest):
    def write(self, name, text, binary=False):
        path = self.dir / name
        path.write_bytes(text) if binary else path.write_text(text, encoding="utf-8")
        return path

    def test_good_file_and_header_normalisation(self):
        df = datagen.generate(150, dirty=False)
        df.columns = [c.replace("_", " ").title() + " " for c in df.columns]      # "Study Hours ", "Final Score " ...
        path = self.dir / "ok.csv"
        df.to_csv(path, index=False)
        loaded = P.load_dataset(path)
        self.assertEqual(list(loaded.columns), S.REQUIRED_COLUMNS)
        self.assertEqual(len(loaded), 150)

    def test_problem_files_are_explained(self):
        cases = {
            "missing.csv": (None, "not found"),
            "empty.csv": ("", "is empty"),
            "cols.csv": ("course,marks\nArts,50\n", "missing column(s)"),
            "small.csv": (datagen.generate(30, dirty=False).to_csv(index=False), "only 30 rows"),
        }
        for name, (text, hint) in cases.items():
            path = self.dir / name if text is None else self.write(name, text)
            with self.assertRaises(DataError, msg=name) as ctx:
                P.load_dataset(path)
            self.assertIn(hint, str(ctx.exception))
        latin = self.write("latin.csv", "course,marks\nZürich,5\n".encode("latin-1"), binary=True)
        with self.assertRaises(DataError) as ctx:
            P.load_dataset(latin)
        self.assertIn("UTF-8", str(ctx.exception))


class TestCleaning(TempDirTest):
    def setUp(self):
        super().setUp()
        self.base = datagen.generate(300, dirty=False, seed=5)

    def with_row(self, exact=False, **changes):
        """The data plus a copy of row 0 with `changes`. Unless `exact`, its study hours differ slightly so
        that it is not an exact duplicate."""
        row = self.base.iloc[[0]].copy()
        if not exact:
            row["study_hours"] = row["study_hours"] + 0.1
        for key, value in changes.items():
            row[key] = value
        return pd.concat([self.base, row], ignore_index=True)

    def test_clean_data_passes_through_untouched(self):
        out, report = P.clean(self.base)
        self.assertEqual((report.rows_in, report.rows_out), (300, 300))
        self.assertEqual(report.steps, [])

    def test_each_problem_is_handled(self):
        original = self.base.copy()
        removed = {
            "duplicate row": self.with_row(exact=True),
            "missing final score": self.with_row(**{S.TARGET: np.nan}),
            "score typo 720": self.with_row(**{S.TARGET: 720}),
            "negative score": self.with_row(**{S.TARGET: -5}),
            "unknown course": self.with_row(course="Medicine"),
            "missing course": self.with_row(course=np.nan),
        }
        for label, df in removed.items():
            out, report = P.clean(df)
            self.assertEqual(report.rows_out, 300, label)
            self.assertEqual(report.rows_dropped, 1, label)
        blanked = {                     # row is kept, the bad value becomes missing
            "attendance_pct": 105, "study_hours": -5, "previous_score": 450, "sleep_hours": 1,
            "screen_hours": 20, "internal_score": "AB",
        }
        for column, value in blanked.items():
            df = self.with_row()
            df[column] = df[column].astype(object)
            df.loc[300, column] = value
            out, report = P.clean(df)
            self.assertEqual(report.rows_out, 301, column)
            self.assertTrue(pd.isna(out[column].iloc[-1]) or out[column].iloc[-1] != value, column)
            self.assertEqual(out[column].isna().sum(), 1, column)
        pd.testing.assert_frame_equal(self.base, original)         # the input is never modified

    def test_spelling_and_yes_no_text(self):
        df = self.with_row(course="  science", parent_education="POSTGRAD")
        df["tutoring"] = df["tutoring"].astype(object)
        df.loc[300, "tutoring"] = "Yes"
        out, report = P.clean(df)
        self.assertEqual(out["course"].iloc[-1], "Science")
        self.assertTrue(pd.isna(out["parent_education"].iloc[-1]))        # not recognised: a gap, row kept
        self.assertEqual(out["tutoring"].iloc[-1], 1)
        labels = " ".join(step[0] for step in report.steps)
        self.assertIn("Course spelling standardised", labels)

    def test_accounting_adds_up(self):
        out, report = P.clean(datagen.generate(800))
        removed = sum(count for label, count, outcome in report.steps if outcome.startswith(("removed", "removed (")))
        self.assertEqual(report.rows_dropped, removed)
        self.assertEqual(report.rows_out, len(out))
        self.assertEqual(len(report.lines()), len(report.steps) + 1)
        self.assertEqual(out[S.TARGET].between(0, 100).all(), True)

    def test_too_little_usable_data_is_refused(self):
        df = datagen.generate(300, dirty=False)
        df[S.TARGET] = 999.0                     # every score impossible
        with self.assertRaises(DataError) as ctx:
            P.clean(df)
        self.assertIn("usable rows remain", str(ctx.exception))


class TestFeatures(TempDirTest):
    def test_engineered_features_never_use_the_target(self):
        df = datagen.generate(50, dirty=False).drop(columns=[S.TARGET])     # no target column at all
        out = P.add_features(df)
        self.assertTrue(set(S.ENGINEERED) <= set(out.columns))
        self.assertFalse(any(S.TARGET in name for name in out.columns))

    def test_engineered_values(self):
        df = pd.DataFrame([{"study_hours": 10.0, "attendance_pct": 80.0, "internal_score": 70.0,
                            "previous_score": 60.0, "sleep_hours": 6.0}])
        out = P.add_features(df).iloc[0]
        self.assertAlmostEqual(out["effective_study"], 8.0)
        self.assertAlmostEqual(out["score_trend"], 10.0)
        self.assertAlmostEqual(out["sleep_gap"], 1.5)

    def test_pipeline_handles_gaps_and_unknown_categories(self):
        raw = datagen.generate(300, dirty=False)
        X, y = P.split_xy(raw)
        pipe = M.build_pipeline(M.candidates(fast=True)["Ridge Regression"]).fit(X, y)
        odd = X.head(3).copy()
        odd.loc[odd.index[0], ["study_hours", "internal_score", "course"]] = np.nan          # gaps
        odd.loc[odd.index[1], "parent_education"] = "Unheard-of"                             # unseen category
        self.assertTrue(np.isfinite(pipe.predict(odd)).all())
