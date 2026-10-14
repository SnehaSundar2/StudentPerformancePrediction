"""Synthetic student data for the project.

Real student records are private and cannot be shared, so the project ships a generator that
produces a realistic dataset with a KNOWN, plausible scoring rule plus noise. That makes the
project fully reproducible offline, protects privacy, and lets the evaluation show whether a model
recovered the true relationships. The data is clearly SYNTHETIC: it teaches the workflow and must
never be used to judge a real student.

The generated file is also deliberately DIRTY (missing values, duplicates, impossible entries,
text where numbers belong) so the preprocessing stage has real work to do.
"""

import numpy as np
import pandas as pd

from . import schema as S

PARENT_BONUS = {"School": 0.0, "Graduate": 1.4, "Postgraduate": 2.6}
COURSE_BONUS = {"Science": -1.2, "Commerce": 0.0, "Arts": 0.9}
NOISE_SD = 4.0       # marks of "unseen factors" (exam-day luck, question paper, health ...)


def true_score(previous, internal, attendance, study, assignments, sleep, screen, tutoring, internet, course,
               parent):
    """The 'ground truth' scoring rule (vectorised over numpy arrays / pandas columns), before noise."""
    previous, internal, attendance = (np.asarray(x, float) for x in (previous, internal, attendance))
    study, assignments, sleep, screen = (np.asarray(x, float) for x in (study, assignments, sleep, screen))
    score = (5.0
             + 0.30 * previous + 0.30 * internal
             + 0.12 * attendance + 0.05 * assignments
             + 13.0 * (1 - np.exp(-study / 10.0))            # studying helps, with diminishing returns
             - 0.15 * np.maximum(0, 65 - attendance)           # missing many classes hurts extra
             - 0.9 * np.maximum(0, screen - 4)                 # long leisure screen time
             - 1.4 * np.minimum(np.abs(sleep - 7.5), 4)        # too little or too much sleep
             + 2.5 * np.asarray(tutoring, float) + 1.0 * np.asarray(internet, float)
             + np.array([PARENT_BONUS[p] for p in parent]) + np.array([COURSE_BONUS[c] for c in course]))
    return np.clip(score, 0, 100)


def generate(rows=1800, seed=2026, dirty=True):
    """Return a DataFrame with the schema's columns. `dirty=False` gives a perfectly clean table."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, rows)
    diligence = rng.normal(0, 1, rows)
    previous = np.clip(62 + 14 * ability + rng.normal(0, 6, rows), 8, 99).round(1)
    internal = np.clip(62 + 12 * ability + 4 * diligence + rng.normal(0, 7, rows), 5, 100).round(1)
    attendance = np.clip(78 + 9 * diligence + rng.normal(0, 8, rows), 25, 100).round(1)
    assignments = np.clip(72 + 14 * diligence + rng.normal(0, 10, rows), 10, 100).round(1)
    study = np.clip(rng.normal(11 + 4 * diligence + 1.5 * ability, 4, rows), 0, 45).round(1)
    sleep = np.clip(rng.normal(7.0, 1.1, rows), 4, 10).round(1)
    screen = np.clip(rng.gamma(3.0, 1.0, rows), 0, 10).round(1)
    tutoring = (rng.random(rows) < 0.30).astype(int)
    internet = (rng.random(rows) < 0.88).astype(int)
    course = rng.choice(S.COURSES, rows, p=[0.45, 0.35, 0.20])
    parent = rng.choice(S.PARENT_EDUCATION, rows, p=[0.35, 0.45, 0.20])
    clean_score = true_score(previous, internal, attendance, study, assignments, sleep, screen, tutoring,
                             internet, course, parent)
    final = np.clip(clean_score + rng.normal(0, NOISE_SD, rows), 0, 100).round(1)
    df = pd.DataFrame({"previous_score": previous, "internal_score": internal, "attendance_pct": attendance,
                       "study_hours": study, "assignments_pct": assignments, "sleep_hours": sleep,
                       "screen_hours": screen, "tutoring": tutoring, "internet": internet, "course": course,
                       "parent_education": parent, S.TARGET: final})
    return make_dirty(df, rng) if dirty else df


def make_dirty(df, rng):
    """Inject realistic data-quality problems (about 8% of rows are affected)."""
    df = df.copy()
    n = len(df)
    for col, rate in (("study_hours", 0.04), ("sleep_hours", 0.05), ("screen_hours", 0.05),
                      ("assignments_pct", 0.03), ("attendance_pct", 0.02), ("internal_score", 0.02),
                      ("parent_education", 0.04)):
        df.loc[rng.random(n) < rate, col] = np.nan                                     # missing values
    for col in ("tutoring", "internet"):
        df[col] = df[col].astype("float")
        df.loc[rng.random(n) < 0.02, col] = np.nan
    df["internal_score"] = df["internal_score"].astype(object)
    for idx in rng.choice(n, 5, replace=False):
        df.loc[idx, "internal_score"] = "AB"                                           # 'absent' typed in a number column
    for idx in rng.choice(n, 5, replace=False):                                        # impossible attendance
        df.loc[idx, "attendance_pct"] = float(rng.choice([105.0, 120.0, 250.0]))
    for idx in rng.choice(n, 4, replace=False):                                        # impossible study hours
        df.loc[idx, "study_hours"] = float(rng.choice([-5.0, 90.0, 168.0]))
    for idx in rng.choice(n, 4, replace=False):                                        # previous score not out of 100
        df.loc[idx, "previous_score"] = float(rng.choice([450.0, 720.0]))
    for idx in rng.choice(n, 4, replace=False):                                        # final score typo
        df.loc[idx, S.TARGET] = float(rng.choice([720.0, -5.0, 1000.0]))
    for idx in rng.choice(n, 4, replace=False):                                        # unknown final score
        df.loc[idx, S.TARGET] = np.nan
    for idx in rng.choice(n, 8, replace=False):                                        # messy category text
        df.loc[idx, "course"] = str(rng.choice(["  science", "COMMERCE ", "arts", "Sci "]))
    for idx in rng.choice(n, 4, replace=False):                                        # unknown course
        df.loc[idx, "course"] = "Medicine"
    for col in ("tutoring", "internet"):                                               # yes/no written as text
        df[col] = df[col].astype(object)
        as_text = (rng.random(n) < 0.15) & df[col].notna()
        df.loc[as_text, col] = df.loc[as_text, col].map(lambda x: "Yes" if x == 1 else "No")
    dupes = df.sample(int(n * 0.012), random_state=1)                                  # repeated records
    return pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
