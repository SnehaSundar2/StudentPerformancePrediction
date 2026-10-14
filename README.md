# Student Performance Prediction

A complete machine-learning project that **predicts a student's final exam score** from study
habits, attendance and past marks, so teachers and mentors can offer help *early*.

It covers the whole workflow the internship brief asks for:

| Brief | Where it lives |
|---|---|
| **Data preprocessing** | `studentperf/preprocessing.py` – loading, validation, cleaning with an audit log, feature engineering, imputing / scaling / encoding |
| **Model training** | `studentperf/models.py`, `training.py` – five models compared by cross-validation, tuning, leak-free split |
| **Evaluation** | `studentperf/evaluation.py`, `report.py` – R², average miss in marks, pass/fail and "needs support" checks, honest score ranges, feature importance, HTML report |
| **Prediction interface** | `cli.py` (menu), `web.py` (web form + JSON API), `main.py` (command line), batch CSV |

Developed and submitted by **Sneha S** (Intern ID SKX-2026-6425) · Skyrovix virtual internship ·
AI & Machine Learning · due 26 Oct 2026.

> **Data note.** Real student records are private, so the project generates a realistic but
> **artificial** dataset with a known scoring rule, random noise and deliberate data problems.
> Results show that the *workflow* works; they say nothing about real students, and the tool must
> never be used to judge one. The model never uses gender, religion, caste, income or any other
> personal attribute.

## Features

- **Cleaning with an audit trail.** Fixes spelling and yes/no text, blanks impossible values,
  removes duplicates and rows with a missing or impossible final score, and prints exactly what it did.
- **Fair model comparison.** Baseline (class average), linear, ridge, random forest and gradient boosting,
  all inside the same pipeline, scored by 5-fold cross-validation on training rows only.
- **No data leakage.** The test set is used once. Imputing, scaling and encoding are fitted inside
  the pipeline on training rows only; engineered features never use the final score.
- **Answers that teachers care about.** Besides the predicted score: a grade band, a support flag
  (*On track / Watch / At risk*), pass/fail accuracy and how many struggling students were found.
- **A range, not a single number.** A 90% score range built from out-of-fold predictions and
  checked independently on the test set (92% coverage).
- **Explanations and advice.** "What moves this score" (each detail swapped for the typical value),
  "what could raise this score" (ranked extra marks), three similar students, and a what-if tool.
- **Safe by design.** Every input is validated with a message per field; saved models are checked with
  SHA-256 before loading; the web server only answers to localhost, limits request size and escapes output.

## Results (synthetic data, unseen test students)

| Model (5-fold CV on training rows) | Average miss | RMSE | R² |
|---|---|---|---|
| **Ridge Regression** ← chosen | **3.6 marks** | 4.5 | 0.850 |
| Linear Regression | 3.6 marks | 4.5 | 0.850 |
| Gradient Boosting | 3.9 marks | 4.9 | 0.823 |
| Random Forest | 4.0 marks | 5.1 | 0.810 |
| Baseline (always the class average) | 9.3 marks | 11.6 | −0.008 |

On the 358 test students: **R² 0.858**, average miss **3.5 marks**, 73% within 5 marks and 98% within
10 marks; pass/fail called correctly for **99%**; **82%** of students who really scored under 50
were flagged (86% of flags were right); the 90% range held the true score for **92%**. Training R²
is 0.855, so there is no overfitting. The best any model could do on this data is R² 0.861 and a
3.4-mark miss (the built-in random noise), so the model is at the ceiling.

The simple models win because the scoring rule is almost linear. On messier real data the flexible
models usually pull ahead, which is why they are kept in the comparison.

## Project structure

```
StudentPerformancePrediction/
├── main.py                  entry point (menu or command line)
├── requirements.txt
├── studentperf/
│   ├── schema.py            features, valid ranges, grades, support flag
│   ├── datagen.py           synthetic data with a known scoring rule and injected problems
│   ├── preprocessing.py     loading, cleaning, feature engineering, transformers
│   ├── models.py            candidate models, cross-validation, tuning
│   ├── evaluation.py        metrics, pass/fail checks, ranges, importance, noise ceiling
│   ├── training.py          the workflow end to end
│   ├── persistence.py       save / load with SHA-256 checks
│   ├── validators.py        validation of student records
│   ├── predictor.py         score, factors, advice, what-if, similar students, batch CSV
│   ├── report.py            text and HTML (SVG charts) reports
│   ├── web.py               local web form and JSON API
│   ├── cli.py               interactive menu
│   └── exceptions.py        custom errors
├── sample_data/             students_dirty.csv (practice cleaning), students_to_score.csv (batch example)
├── tests/                   78 automated tests
└── docs/PROJECT_REPORT.md   full project report
```

## How to run

Python 3.10+ is needed.

```bash
cd StudentPerformancePrediction
pip install -r requirements.txt
python main.py --train        # about 25-40 seconds; creates models/
python main.py                # interactive menu
python main.py --serve        # web form at http://127.0.0.1:8010/
```

Other commands:

```bash
python main.py --predict course=Science previous_score=62 internal_score=58 attendance_pct=80 study_hours=10
python main.py --report reports/model_report.html
python main.py --train --data sample_data/students_dirty.csv
python main.py --generate sample_data/more_students.csv --rows 2500
```

JSON API (while the web server runs):

```bash
curl -X POST http://127.0.0.1:8010/api/predict -H "Content-Type: application/json" \
  -d '{"course":"Science","previous_score":62,"internal_score":58,"attendance_pct":80,"study_hours":10}'
```

Required details: course, previous score, internal score, attendance and weekly study hours. The rest
(assignments, sleep, screen time, tutoring, internet, parents' education) fall back to typical values,
and the result says which defaults were used.

## Concepts demonstrated

Train/test split with stratification on the at-risk share · scikit-learn `Pipeline` and
`ColumnTransformer` · imputation, scaling, one-hot encoding · feature engineering · k-fold
cross-validation · hyper-parameter search · regression metrics (MAE, RMSE, R²) · turning a regression
into pass/fail and "needs support" decisions (precision and recall) · overfitting check · permutation
feature importance · calibrated prediction ranges · model persistence with integrity checks · input
validation · responsible use of ML on student data.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

78 tests (about 25 seconds) cover data generation, schema helpers, file loading, every cleaning rule,
metric maths, model quality, reproducibility, persistence and tamper detection, validation,
predictions and their direction, advice, what-if, batch files, the web server (HTML and JSON,
malformed requests, escaping) and the menu. A small model is trained once and shared.

## Limitations

- The data is synthetic; accuracy on real students would be lower and would need real records,
  consent and a fairness review.
- A prediction is an estimate with a range, not a verdict. "What could raise this score" shows patterns
  in the data, not guaranteed cause and effect.
- Very low or very high scorers are the hardest to predict; predictions pull towards the middle.
- Pickle-based model files must only be loaded if you trained them yourself.
