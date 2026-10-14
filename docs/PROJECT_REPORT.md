# Project Report: Student Performance Prediction

**Program:** Virtual Internship – Skyrovix IT Solutions Pvt. Ltd. (AI & Machine Learning project)
**Project:** Student Performance Prediction
**Developed and submitted by:** Sneha S (Intern ID: SKX-2026-6425)
**Due date:** 26 October 2026

---

## 1. Introduction

Every institution has students who quietly fall behind. By the time the final result arrives it
is too late to help. A model that estimates the final score from information available *during*
the term (past marks, attendance, study habits) lets teachers and mentors start support early.

This project builds that system end to end: it cleans messy student records, trains and compares
several models, evaluates the winner honestly on students it has never seen, and offers prediction
interfaces (a menu, a command line, a web form and a JSON API). Every prediction comes with a likely
score range, a grade band, a support flag, the reasons behind the number and ranked suggestions for
improvement.

**About the data.** Real student records are private and cannot be shared, so the project generates
a realistic but *artificial* dataset with a known scoring rule, random noise and deliberate data
problems. This makes the work reproducible offline, protects privacy, and allows an extra check:
how close the model gets to the best result that is possible at all. The results therefore
describe the workflow, not real-world accuracy, and the tool must never be used to judge a real student.

**Responsible use.** The model uses only study habits, attendance, past marks and two background
facts (course, parents' education). It does not use gender, religion, caste, family income or
any other personal attribute. Its output is a prompt to offer help, not a verdict.

## 2. Objectives

1. **Data preprocessing:** load, validate, clean and transform data, leaving a record of every change.
2. **Model training:** compare several algorithms fairly, tune the best one, and avoid leaking test data into training.
3. **Evaluation:** measure accuracy honestly, check for overfitting, check the decisions teachers care about (pass/fail, who needs support), find where the model is weak, and report how uncertain a prediction is.
4. **Prediction interface:** make the model usable through easy interfaces, with strict input validation and clear explanations.
5. **Quality:** handle errors safely, document the workflow, and verify the behaviour with automated tests.

## 3. Tools and technologies

| Item | Used |
|---|---|
| Language | Python 3 (tested on 3.13) |
| ML libraries | scikit-learn (pipelines, models, cross-validation, tuning), pandas, numpy, joblib |
| Standard library | `http.server` (web interface), `hashlib` (checksums), `json`, `csv`, `unittest` |
| Output | Terminal reports, self-contained HTML report with SVG charts, JSON API |

## 4. System design

### 4.1 Workflow

1. **Load** the CSV and check that it has the right columns and enough rows.
2. **Clean** the data and record every action.
3. **Split** 80% / 20% into training and test sets, keeping the share of struggling students (score under 50) equal in both. The test set is not touched again until the end.
4. **Compare** five candidate models by 5-fold cross-validation on the training rows: a baseline, linear regression, ridge regression, random forest, gradient boosting.
5. **Tune** the best model with a randomised search; keep the tuned version only if cross-validation says it is better.
6. **Evaluate once** on the test set; calibrate the score range using out-of-fold predictions; measure feature importance.
7. **Save** the model with checksums, then **predict** through any interface.

### 4.2 The data

| Feature | Meaning | Valid range |
|---|---|---|
| `previous_score`, `internal_score` | Marks in the previous exam and internal tests | 0–100 |
| `attendance_pct` | Class attendance | 0–100 % |
| `study_hours` | Self-study per week | 0–60 h |
| `assignments_pct` | Assignments submitted | 0–100 % |
| `sleep_hours`, `screen_hours` | Sleep per night, leisure screen time per day | 3–12 h, 0–12 h |
| `tutoring`, `internet` | Extra tutoring, internet at home | yes / no |
| `course`, `parent_education` | Science / Commerce / Arts; School / Graduate / Postgraduate | – |
| **`final_score`** (target) | Final exam score | 0–100 |

Three features are engineered from the others, never from the target: *effective study* (study
hours × attendance), *score trend* (internal minus previous score) and *sleep gap* (distance from 7½ hours).

### 4.3 Modules

| Module | Responsibility |
|---|---|
| `schema.py` | Features, valid ranges, grade bands, support flag, category spellings |
| `datagen.py` | Synthetic data with a known scoring rule and injected problems |
| `preprocessing.py` | Loading, cleaning with a report, feature engineering, imputing/scaling/encoding |
| `models.py` | Candidate models, cross-validated comparison, tuning |
| `evaluation.py` | Metrics, pass/fail and support-flag checks, ranges, segment errors, importance, noise ceiling |
| `training.py` | The workflow end to end |
| `persistence.py` | Saving and loading with SHA-256 integrity checks |
| `validators.py` | Validation of student records |
| `predictor.py` | Score, explanation, advice, what-if, similar students, batch CSV |
| `report.py` | Text and HTML reports |
| `web.py`, `cli.py`, `main.py` | Web form and API, menu, command line |
| `exceptions.py` | Custom error types |

### 4.4 The pipeline

Every model is wrapped in the same scikit-learn pipeline: **add engineered features → impute
missing values (median / most frequent) → scale numbers and one-hot encode categories → model**.
Because the whole chain lives in one pipeline, anything that learns from data (medians, scaling,
categories) is fitted on training rows only. Predictions are limited to the 0–100 scale.

## 5. Implementation highlights

**Cleaning with an audit trail.** The generator deliberately produces about 8% problem rows. The
cleaner found and handled all of them, and said what it did:

```
1821 rows read → 1788 kept (33 removed)
  • Course spelling standardised: 8 — e.g. '  science' → 'Science'
  • Rows with unknown or missing course: 4 — removed
  • Non-numeric text in internal_score: 5 — set to missing (e.g. 'AB')
  • Rows with a missing or impossible final score: 8 — removed (valid: 0–100)
  • Duplicate records: 21 — removed (kept the first copy)
  • Out-of-range previous_score: 4 — set to missing (valid: 0–100)
  • Out-of-range attendance_pct: 5 — set to missing (valid: 0–100)
  • Out-of-range study_hours: 4 — set to missing (valid: 0–60)
```

A dataset on the wrong scale (marks out of 1000) is refused outright, because every final score
would be flagged as impossible. Unlike the house-price project, no cleaning rule uses statistics
from other rows, so nothing from the test set can influence training.

**No leakage.** Engineered features use only other inputs, never the final score. The test set is
used once. A test checks that the features can be built from a table that has no target column.

**From scores to decisions.** A teacher does not only want "62.6". The evaluation therefore also asks
two classification questions of the same predictions: *pass or fail?* (pass mark 40) and *does this
student need support?* (score under 50). Precision and recall are reported for the second question,
because missing a struggling student costs more than a false alarm.

**Honest score ranges.** The range is built from out-of-fold training predictions: the 5th and 95th
percentiles of (actual − predicted) are added to each prediction. The test set then checks the claim
independently: the "90%" range contained the true score for **92%** of test students.

**Explanations anyone can follow.** For each detail, the model re-scores the student with that detail
swapped for its typical value; the difference is that detail's effect in marks. A separate step tries
realistic changes a student controls (5 more study hours, +10 attendance points, fewer screen hours,
better sleep, tutoring) and ranks them by the extra marks the model expects. These are estimates from
patterns in the data, not guarantees, and the interface says so. The three most similar training
students are shown beside the prediction.

**A safe model file.** `metadata.json` stores the SHA-256 of the model and reference files. Loading
verifies them first, so a damaged or modified file is refused with a clear message instead of being unpickled.

## 6. Results

| Model (5-fold CV on training rows) | Average miss | RMSE | R² |
|---|---|---|---|
| **Ridge Regression** ← chosen | **3.6 marks** | 4.5 | 0.850 |
| Linear Regression | 3.6 marks | 4.5 | 0.850 |
| Gradient Boosting | 3.9 marks | 4.9 | 0.823 |
| Random Forest | 4.0 marks | 5.1 | 0.810 |
| Baseline (always the class average) | 9.3 marks | 11.6 | −0.008 |

Tuning ridge regression did not improve cross-validation, so the defaults were kept (the program
reports this instead of hiding it).

**On the 358 unseen test students:** R² **0.858**; average miss 3.5 marks (median 3.0); RMSE 4.4;
73% within 5 marks and 98% within 10; bias +0.4 marks. Training R² is 0.855, so there is no overfitting.

| Decision | Result |
|---|---|
| Pass or fail called correctly (pass mark 40) | 99% |
| Grade band (Fail / Pass / Second / First / Distinction) correct | 74% |
| Students who really scored under 50 that were flagged (recall) | 82% (37 of 45) |
| Flags that were right (precision) | 86% (37 of 43) |
| 90% score range contained the true score | 92% |

**Interpretation.**
- The model misses by well under half as many marks as guessing the class average would (3.5 vs 9.3).
- With 4 marks of random noise built into the data, the best any model could do is R² 0.861 and a
  3.4-mark average miss. The model is at that ceiling.
- Previous exam score (importance 0.33) and internal test score (0.29) drive the result, then study
  hours (0.11) and attendance (0.05); sleep, parents' education, course and screen time add a little.
- Errors are even across courses: 3.3 marks (Science) to 3.7 (Commerce).
- The simple linear models win because the scoring rule is nearly linear; on real data flexible models
  usually win by more, which is why they stay in the comparison.
- Grade bands are only 74% right because a 3.5-mark miss easily crosses a boundary 10 marks wide:
  the score and its range are more reliable than the band label.
- Predictions pull towards the middle: the weakest and strongest students are the hardest to predict.

## 7. Testing

78 automated tests were written and **all pass** (about 25 seconds). The whole suite was run 10 times
in a row without a failure. One small model is trained once and shared by the tests.

| # | Test case | Expected | Result |
|---|---|---|---|
| 1 | Generator run twice with the same seed | Identical data; different with another seed | Pass |
| 2 | Scoring rule: more study, attendance, sleep near 7½ h, less screen time, tutoring | Scores move the right way | Pass |
| 3 | `  science`, `COMMERCE `, `humanities`, `post_graduate`, `10th` | Standardised to the supported spellings | Pass |
| 4 | Grade bands and support flag at their boundaries | Correct band and On track / Watch / At risk | Pass |
| 5 | No gender, caste, religion or income feature | None in the schema | Pass |
| 6 | CSV: missing, empty, wrong columns, 30 rows, not UTF-8 | Specific `DataError` for each | Pass |
| 7 | One data problem at a time (duplicate, score 720, −5, unknown course, attendance 105, study −5, 'AB' …) | Row removed or value blanked exactly as documented | Pass |
| 8 | Scores recorded out of 1000 | Dataset refused ("usable rows remain") | Pass |
| 9 | Cleaning report accounting | Rows dropped = rows removed by the listed steps | Pass |
| 10 | Features built from a table with no target column | No leakage | Pass |
| 11 | Gaps and an unseen category sent through the pipeline | Finite predictions, no crash | Pass |
| 12 | Metrics on hand-worked examples | MAE 7.5, RMSE √62.5, bias −2.5, R² 0.375 | Pass |
| 13 | Pass/fail and support flag on a four-student example | Counts, precision 50%, recall 50% | Pass |
| 14 | Chosen model vs baseline on unseen students | R² > 0.78; baseline miss at least 1.8× larger | Pass |
| 15 | Support flag quality | Recall and precision above 65%, pass/fail above 93% | Pass |
| 16 | Training vs test R² | Gap below 0.08 (no serious overfitting) | Pass |
| 17 | "90%" score range on test students | Coverage between 80% and 99% | Pass |
| 18 | Top 2 important features | Previous score and internal score | Pass |
| 19 | Share of struggling students in train vs test | Within 3 points | Pass |
| 20 | Training twice with the same seed | Identical predictions | Pass |
| 21 | `attendance_pct=150`, `"abc"`, `nan`, `True`, empty | Rejected with the reason | Pass |
| 22 | Unknown course `Scince` | "Did you mean 'Science'?" | Pass |
| 23 | One byte flipped in `model.joblib`; missing file; bad JSON; no or wrong checksum | Refused with a clear `ModelError` | Pass |
| 24 | Strong and weak students | Distinction / On track vs At risk, ranges within 0–100 | Pass |
| 25 | Typical student | Nothing to explain (no factors) | Pass |
| 26 | Study 59 h/week (never seen in training); attendance 0.9 | Warning: outside learned range / enter a percentage | Pass |
| 27 | Weak student's advice | Ranked, at least 0.5 marks each, at most 4; ideal student gets none | Pass |
| 28 | Batch CSV with 3 bad rows and 1 unusual student | 3 scored, 3 reported by line number, unusual one flagged | Pass |
| 29 | Web: `"><script>alert(1)</script>` in a field | Escaped; never appears as markup | Pass |
| 30 | Web: wrong content type / bad JSON / 30 KB body / foreign Host header | 415 / 400 / 413 / 403 | Pass |
| 31 | Menu: bad course then good course; attendance 150; required answer skipped | Asked again with a clear message | Pass |
| 32 | Menu with no trained model | Friendly instruction; training from CSV works | Pass |

## 8. Challenges and solutions

| Challenge | Solution |
|---|---|
| Real student data is private | A generator with a known scoring rule, so results are reproducible and can be compared with the best possible score |
| Preventing data leakage | One pipeline holds all learned preprocessing; no cleaning rule uses other rows; the test set is used once |
| A score alone is not a decision | Pass/fail and support-flag precision/recall, plus a grade band and risk level |
| A single number looks more certain than it is | A calibrated 90% range, verified on the test set (92% coverage) |
| A model should say *why* and *what next* | Per-detail "swap for typical" effects, ranked advice, similar students |
| Fairness | No personal attributes in the model; errors reported by course and parent education; output described as a prompt for help |
| Grade labels overstate precision | Reported honestly (74% band accuracy) and the range is shown beside the label |
| Pickled models can be altered or damaged | SHA-256 verified before loading, plus a clear warning to load only your own models |
| A rejected oversized web request could be lost on Windows | The server reads and discards a bounded amount of the body before sending the 413 reply |

## 9. Future enhancements

- Train on a real, anonymised, consented dataset and compare with the synthetic results.
- Add a second random seed and a learning curve (error versus training-set size) to show stability.
- Predict at several points in the term and show how the forecast improves as marks arrive.
- Add a fairness audit across more groups once real data is available.
- Keep a log of predictions and retrain when results drift.

## 10. Conclusion

The Student Performance Prediction project meets all its objectives. It cleans messy records and
records every change. It compares models fairly, evaluates the winner honestly on unseen students
(R² 0.858, 3.5-mark average miss, at the 3.4-mark noise limit of the data), and checks the decisions
that matter: 99% correct pass/fail and 82% of struggling students flagged. Predictions come with a
verified score range, the factors behind them, ranked suggestions and similar students, through a
menu, a command line, a web form and a JSON API. Inputs are validated strictly, errors are handled
clearly, saved models are integrity-checked, and 78 automated tests verify the behaviour.

## 11. How to run

```bash
cd StudentPerformancePrediction
pip install -r requirements.txt
python main.py --train
python main.py
python main.py --serve
python -m unittest discover -s tests -t . -v
```
