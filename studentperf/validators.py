"""Validation of student records (prediction input).

parse_features() turns raw text values (from the menu, a web form, a JSON request or a CSV row) into a
clean, consistent description of a student — or raises ValidationError listing EVERY problem at once, with
a message per field. It uses the same schema as training-data cleaning, so both agree on what is valid.
"""

import difflib
import math
import re

from . import schema as S
from .exceptions import ValidationError

REQUIRED_INPUT = ["course", "previous_score", "internal_score", "attendance_pct", "study_hours"]
DEFAULTS = {"assignments_pct": 75.0, "sleep_hours": 7.0, "screen_hours": 3.0, "tutoring": 0, "internet": 1,
            "parent_education": "Graduate"}

_NUMBER = re.compile(r"^([-+]?\d+(?:\.\d+)?|[-+]?\.\d+)\s*(%|marks?|hours?|hrs?|h)?$", re.I)


def normalise_key(key):
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def parse_number(feature, value):
    """'85%' → 85.0 ; '12 hours' → 12.0. Raises ValueError with a message that names the field."""
    _, low, high, _ = S.NUMERIC[feature]
    label = S.LABELS[feature]
    unit = f" {S.UNITS[feature]}" if feature in S.UNITS else ""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _NUMBER.match(str(value).strip().replace(",", ""))
        if not match:
            raise ValueError(f"{label} must be a number (you entered '{str(value).strip()[:30]}').")
        number = float(match.group(1))
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a normal number.")
    if not low <= number <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}{unit} (you entered {number:g}).")
    return number


def parse_category(feature, value):
    canon = S.canonical_category(feature, value)
    if canon is None:
        allowed = S.CATEGORICAL[feature]
        guess = difflib.get_close_matches(str(value).strip().title(), allowed, n=1, cutoff=0.6)
        hint = f" Did you mean '{guess[0]}'?" if guess else ""
        raise ValueError(f"{S.LABELS[feature]} '{str(value).strip()[:30]}' is not supported.{hint} "
                         f"Choose one of: {', '.join(allowed)}.")
    return canon


def parse_features(raw, apply_defaults=True):
    """Validate a student record. Returns (features, assumed) where `assumed` lists any defaults used.

    Raises ValidationError(message, errors={field: message}) listing all problems.
    """
    if not isinstance(raw, dict):
        raise ValidationError("The student record must be a set of named values.")
    given = {normalise_key(k): v for k, v in raw.items()}
    errors = {}
    unknown = sorted(set(given) - set(S.RAW_FEATURES))
    if unknown:
        errors["_"] = f"Unknown field(s): {', '.join(unknown)}. Valid fields: {', '.join(S.RAW_FEATURES)}."
    features, assumed = {}, {}

    for name in S.RAW_FEATURES:
        value = given.get(name)
        blank = value is None or (isinstance(value, str) and not value.strip()) or \
            (isinstance(value, float) and value != value)
        if blank:
            if name in REQUIRED_INPUT or not apply_defaults:
                errors[name] = f"{S.LABELS[name]} is required."
                continue
            features[name] = DEFAULTS[name]
            assumed[name] = DEFAULTS[name]
            continue
        try:
            if name in S.NUMERIC:
                features[name] = parse_number(name, value)
            elif name in S.CATEGORICAL:
                features[name] = parse_category(name, value)
            else:
                flag = S.parse_yes_no(value)
                if flag is None:
                    raise ValueError(f"{S.LABELS[name]} must be yes or no.")
                features[name] = flag
        except ValueError as exc:
            errors[name] = str(exc)

    if errors:
        raise ValidationError(" ".join(message.rstrip(".") + "." for message in errors.values()), errors)
    return features, assumed


def validate_port(value):
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError("Port must be a whole number.") from None
    if not 1024 <= port <= 65535:
        raise ValidationError("Port must be between 1024 and 65535.")
    return port


def validate_int(value, field, low, high):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a whole number.") from None
    if not low <= number <= high:
        raise ValidationError(f"{field} must be between {low} and {high}.")
    return number
