"""Student Performance Prediction.

An end-to-end machine-learning project: data generation and cleaning, feature
engineering, model comparison with cross-validation, honest evaluation on a
held-out test set, prediction with score ranges, explanations and study advice,
and prediction interfaces (command-line menu, local web form and JSON API).
"""

import logging

__version__ = "1.0.0"

logging.getLogger("studentperf").addHandler(logging.NullHandler())
