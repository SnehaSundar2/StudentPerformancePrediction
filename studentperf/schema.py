"""The dataset schema: every feature, its type, valid range and allowed values.

Training-data cleaning and prediction-input validation both read from here, so
the two can never disagree about what a valid student record looks like.

Fairness note: the model deliberately does NOT use gender, religion, caste, family income
or any other personal attribute. It uses only study habits, attendance and past marks, plus two
background facts (course and parent education) that help explain differences between groups.
"""

TARGET = "final_score"          # final exam score out of 100

COURSES = ["Science", "Commerce", "Arts"]
PARENT_EDUCATION = ["School", "Graduate", "Postgraduate"]

# name -> (type, low, high, description). Numeric limits are hard limits for valid data.
NUMERIC = {
    "previous_score": ("float", 0, 100, "Score in the previous exam (out of 100)"),
    "internal_score": ("float", 0, 100, "Internal test / assessment score (out of 100)"),
    "attendance_pct": ("float", 0, 100, "Class attendance in percent"),
    "study_hours": ("float", 0, 60, "Self-study hours per week"),
    "assignments_pct": ("float", 0, 100, "Assignments submitted, in percent"),
    "sleep_hours": ("float", 3, 12, "Average sleep per night, in hours"),
    "screen_hours": ("float", 0, 12, "Leisure screen time per day (phone, games, social media), in hours"),
}
BINARY = {
    "tutoring": "Takes extra tutoring or coaching (yes/no)",
    "internet": "Has internet access at home (yes/no)",
}
CATEGORICAL = {
    "course": COURSES,
    "parent_education": PARENT_EDUCATION,
}

NUMERIC_FEATURES = list(NUMERIC)
BINARY_FEATURES = list(BINARY)
CATEGORICAL_FEATURES = list(CATEGORICAL)
RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = RAW_FEATURES + [TARGET]

# Engineered features added by preprocessing.add_features (never uses the target).
ENGINEERED = ["effective_study", "score_trend", "sleep_gap"]
MODEL_NUMERIC = NUMERIC_FEATURES + BINARY_FEATURES + ENGINEERED

# Result bands (out of 100) used for grades and the support flag
SCORE_RANGE = (0, 100)
PASS_MARK = 40
AT_RISK_BELOW = 50      # a predicted score under this is flagged "At risk" (needs support)
WATCH_BELOW = 60        # under this (or a range that reaches the fail zone) is "Watch"
GRADE_BANDS = [(75, "Distinction"), (60, "First class"), (50, "Second class"), (PASS_MARK, "Pass"), (0, "Fail")]


def grade_for(score):
    for floor, name in GRADE_BANDS:
        if score >= floor:
            return name
    return "Fail"


def risk_level(score, low):
    """'At risk' / 'Watch' / 'On track' from the predicted score and the low end of its range."""
    if score < AT_RISK_BELOW:
        return "At risk"
    if score < WATCH_BELOW or low < PASS_MARK:
        return "Watch"
    return "On track"


def describe(feature):
    if feature in NUMERIC:
        return NUMERIC[feature][3]
    if feature in BINARY:
        return BINARY[feature]
    return {"course": "Course stream (Science, Commerce or Arts)",
            "parent_education": "Highest education of a parent (School, Graduate or Postgraduate)"}.get(feature, feature)


LABELS = {
    "previous_score": "Previous exam score", "internal_score": "Internal test score", "attendance_pct": "Attendance",
    "study_hours": "Study hours per week", "assignments_pct": "Assignments submitted", "sleep_hours": "Sleep per night",
    "screen_hours": "Leisure screen time", "tutoring": "Extra tutoring", "internet": "Internet at home",
    "course": "Course", "parent_education": "Parents' education",
}
UNITS = {"previous_score": "marks", "internal_score": "marks", "attendance_pct": "%", "study_hours": "hours",
         "assignments_pct": "%", "sleep_hours": "hours", "screen_hours": "hours/day"}

# Alternative spellings people actually use (all lower case, '-' and '_' treated as spaces)
ALIASES = {
    "course": {"sci": "Science", "pcm": "Science", "pcb": "Science", "maths": "Science", "com": "Commerce",
               "business": "Commerce", "humanities": "Arts", "art": "Arts", "liberal arts": "Arts"},
    "parent_education": {"school level": "School", "secondary": "School", "high school": "School", "10th": "School",
                         "12th": "School", "degree": "Graduate", "bachelors": "Graduate", "ug": "Graduate",
                         "graduation": "Graduate", "masters": "Postgraduate", "pg": "Postgraduate",
                         "post graduate": "Postgraduate", "phd": "Postgraduate"},
}


def _key(text):
    return " ".join(str(text).replace("_", " ").replace("-", " ").split()).lower()


def canonical_category(feature, value):
    """'  science ' -> 'Science'. Returns None when the value is missing or not recognised."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    key = _key(value)
    lookup = {_key(name): name for name in CATEGORICAL[feature]}
    lookup.update({_key(alias): name for alias, name in ALIASES.get(feature, {}).items()})
    return lookup.get(key)


def parse_yes_no(value):
    """'Yes' / 'y' / true / 1 -> 1;  'no' / 'n' / false / 0 -> 0;  anything else -> None."""
    if isinstance(value, (int, float)) and value == value:
        return {1: 1, 0: 0}.get(int(value)) if value in (0, 1) else None
    return {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}.get(str(value).strip().lower())
