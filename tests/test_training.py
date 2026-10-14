"""Model training, comparison, tuning and evaluation."""

import numpy as np
import pandas as pd

from studentperf import datagen, evaluation as E, models as M, preprocessing as P, report, schema as S, training
from studentperf.exceptions import DataError
from tests.helpers import TempDirTest, shared


class TestEvaluationMaths(TempDirTest):
    def test_metrics_on_a_tiny_example(self):
        m = E.metrics([60, 80], [65, 70])
        self.assertAlmostEqual(m["mae"], 7.5)
        self.assertAlmostEqual(m["rmse"], np.sqrt(62.5))
        self.assertAlmostEqual(m["bias"], -2.5)
        self.assertEqual((m["within_5"], m["within_10"]), (50.0, 100.0))
        self.assertAlmostEqual(m["r2"], 1 - 125 / 200)
        self.assertEqual(E.metrics([5, 5], [5, 5])["r2"], 0.0)       # no variation: defined, not a crash

    def test_perfect_and_terrible_predictions(self):
        actual = np.array([30.0, 50.0, 70.0, 90.0])
        self.assertAlmostEqual(E.metrics(actual, actual)["r2"], 1.0)
        self.assertLess(E.metrics(actual, np.full(4, 5.0))["r2"], 0)

    def test_pass_fail_and_support_flag_on_a_tiny_example(self):
        m = E.pass_fail_metrics([30, 45, 55, 80], [35, 55, 45, 78])
        self.assertEqual(m["pass_accuracy"], 100.0)                  # nobody crosses the pass mark of 40
        self.assertEqual((m["risk_tp"], m["risk_fp"], m["risk_fn"], m["risk_tn"]), (1, 1, 1, 1))
        self.assertEqual((m["risk_precision"], m["risk_recall"], m["risk_accuracy"]), (50.0, 50.0, 50.0))
        self.assertEqual(m["grade_accuracy"], 50.0)                  # Fail and Distinction right, the middle two wrong
        self.assertEqual(E.pass_fail_metrics([60, 70], [65, 75])["risk_recall"], 0.0)   # nobody at risk: defined

    def test_coverage_and_segments(self):
        self.assertEqual(E.coverage([50, 50, 50, 50], [50, 50, 50, 60], -5, 5), 75.0)
        df = pd.DataFrame({"course": ["A", "A", "B", "B"], "actual": [50, 50, 50, 50], "predicted": [50, 50, 60, 40]})
        rows = E.segment_errors(df)
        self.assertEqual([r["course"] for r in rows], ["B", "A"])       # worst first
        self.assertAlmostEqual(rows[0]["mae"], 10.0)
        actual = [10, 30, 45, 48, 52, 60, 70, 80, 90, 95]
        bands = E.score_band_errors(pd.DataFrame({"actual": actual, "predicted": actual}))
        self.assertEqual(len(bands), 4)
        self.assertTrue(all(b["mae"] == 0 for b in bands))
        self.assertEqual(sum(b["n"] for b in bands), len(actual))


class TestTrainedModel(TempDirTest):
    def test_comparison_table(self):
        result, _, _ = shared()
        names = [c["model"] for c in result.comparison]
        self.assertEqual(set(names), {M.BASELINE, "Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"})
        rmse = [c["rmse"] for c in result.comparison]
        self.assertEqual(rmse, sorted(rmse))                           # best first
        self.assertEqual(result.comparison[-1]["model"], M.BASELINE)   # the baseline is the worst
        self.assertGreater(result.comparison[-1]["rmse"], 1.8 * result.comparison[0]["rmse"])
        self.assertNotEqual(result.metadata["model"], M.BASELINE)

    def test_split_sizes_and_data_flow(self):
        result, _, raw = shared()
        rows = result.metadata["rows"]
        self.assertEqual(rows["raw"], len(raw))
        self.assertEqual(rows["train"] + rows["test"], rows["clean"])
        self.assertAlmostEqual(rows["test"] / rows["clean"], 0.2, delta=0.02)
        self.assertEqual(len(result.test_frame), rows["test"])
        self.assertEqual(len(result.reference), rows["train"])          # similar students come from TRAINING rows only

    def test_at_risk_share_is_kept_equal_in_train_and_test(self):
        result, _, _ = shared()
        train_share = (result.reference[S.TARGET] < S.AT_RISK_BELOW).mean()
        test_share = (result.test_frame["actual"] < S.AT_RISK_BELOW).mean()
        self.assertAlmostEqual(train_share, test_share, delta=0.03)

    def test_accuracy_on_unseen_students(self):
        result, _, _ = shared()
        test, train = result.metadata["metrics"]["test"], result.metadata["metrics"]["train"]
        self.assertGreater(test["r2"], 0.78)
        self.assertLess(test["mae"], 5.0)
        self.assertGreater(test["within_10"], 90)
        self.assertGreater(test["pass_accuracy"], 93)
        self.assertGreater(test["risk_recall"], 65)                     # most students who need help are found
        self.assertGreater(test["risk_precision"], 65)
        self.assertLess(train["r2"] - test["r2"], 0.08)                 # no serious overfitting
        base = result.metadata["metrics"]["baseline_cv"]
        self.assertGreater(base["mae"], 1.8 * test["mae"])              # clearly better than guessing the average

    def test_predictions_stay_on_the_0_to_100_scale(self):
        result, _, _ = shared()
        self.assertTrue(result.test_frame["predicted"].between(0, 100).all())

    def test_score_range_is_honest(self):
        result, _, _ = shared()
        iv, test = result.metadata["interval"], result.metadata["metrics"]["test"]
        self.assertLess(iv["low"], 0)
        self.assertGreater(iv["high"], 0)
        self.assertGreater(test["interval_coverage"], 80)               # claimed 90%, small test set
        self.assertLess(test["interval_coverage"], 99)

    def test_importance_matches_the_scoring_rule(self):
        result, _, _ = shared()
        imp = result.importance
        self.assertEqual([i["importance"] for i in imp], sorted((i["importance"] for i in imp), reverse=True))
        self.assertTrue(all(i["importance"] >= 0 for i in imp))
        self.assertEqual({i["feature"] for i in imp[:2]}, {"previous_score", "internal_score"})
        self.assertEqual({i["feature"] for i in imp}, set(S.RAW_FEATURES))

    def test_noise_ceiling_for_synthetic_data(self):
        clean = datagen.generate(800, dirty=False)
        ceiling = E.noise_ceiling(clean)
        self.assertGreater(ceiling["r2"], 0.80)
        self.assertAlmostEqual(ceiling["mae"], 3.2, delta=0.6)          # about 4 marks of random noise

    def test_metadata_is_complete(self):
        result, _, _ = shared()
        meta = result.metadata
        for key in ("trained_at", "model", "metrics", "interval", "environment", "training_ranges", "typical",
                    "importance", "comparison", "cleaning", "segments", "test_points", "seed"):
            self.assertIn(key, meta, key)
        self.assertEqual(set(meta["segments"]), {"course", "parent_education", "score_band"})
        self.assertEqual(set(meta["typical"]), set(S.RAW_FEATURES))
        self.assertGreaterEqual(len(meta["test_points"]), 50)


class TestReproducibilityAndGuards(TempDirTest):
    def test_same_seed_same_result(self):
        raw = datagen.generate(500, seed=11)
        a = training.train(raw, fast=True, tune=False, cv=3)
        b = training.train(raw, fast=True, tune=False, cv=3)
        self.assertEqual(a.metadata["metrics"]["test"]["mae"], b.metadata["metrics"]["test"]["mae"])
        np.testing.assert_allclose(a.test_frame["predicted"], b.test_frame["predicted"])
        self.assertEqual(list(a.test_frame.index), list(b.test_frame.index))

    def test_tiny_or_unusable_data_is_refused(self):
        with self.assertRaises(DataError):
            training.train(datagen.generate(60, dirty=False), fast=True, tune=False)
        wrong_scale = datagen.generate(300, dirty=False)
        wrong_scale[S.TARGET] *= 10                  # scores out of 1000 instead of 100
        with self.assertRaises(DataError):
            training.train(wrong_scale, fast=True, tune=False)

    def test_tuning_search(self):
        raw = datagen.generate(400, dirty=False)
        X, y = P.split_xy(raw)
        search, params = M.tune("Ridge Regression", X, y, cv=3, iterations=4)
        self.assertIn(params["alpha"], M.TUNING_GRIDS["Ridge Regression"]["model__alpha"])
        self.assertEqual(M.tune(M.BASELINE, X, y), (None, {}))          # nothing to tune
        pipe = M.pipeline_with_params("Ridge Regression", params)
        self.assertEqual(pipe.fit(X, y).predict(X.head(3)).shape, (3,))


class TestReports(TempDirTest):
    def test_text_and_html_reports(self):
        meta = shared()[0].metadata
        text = report.model_text(meta)
        for part in ("ACCURACY ON THE UNSEEN TEST SET", "PASS / FAIL AND SUPPORT FLAG", "WHAT THE MODEL RELIES ON",
                     "ERROR BY COURSE", "R²"):
            self.assertIn(part, text)
        self.assertIn("chosen", report.comparison_text(meta))
        page = report.render_html(meta)
        self.assertEqual(page.count('<details class="tv">'), 4)         # every chart has a table view
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertIn("Data cleaning log", page)
        self.assertNotIn("<script>alert", page)
        path = report.save_html(meta, self.dir / "out" / "report.html")
        self.assertTrue(path.exists())

    def test_axis_ceilings_have_whole_ticks(self):
        for value in (7, 45, 130, 999):
            top = report._tick_ceiling(value, 5)
            self.assertGreaterEqual(top, value)
            self.assertEqual(top / 5, round(top / 5, 1))

    def test_scatter_and_error_histogram(self):
        points = [(40 + i, 42 + i) for i in range(50)]
        svg = report.scatter_chart(points, "t")
        self.assertEqual(svg.count("<circle"), 50)
        bins, outside = report.histogram_of_errors([(50, 51), (50, 49), (50, 60), (50, 30)])
        self.assertEqual(sum(count for _, count in bins), 3)           # +1, -1 and +10 are inside the axis
        self.assertEqual(outside, 1)                                    # -20 is beyond ±12
