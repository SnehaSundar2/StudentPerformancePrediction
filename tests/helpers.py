"""Shared fixtures. Training takes a few seconds, so ONE small model is trained for the whole test run."""

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from studentperf import datagen, persistence, training

_SHARED = {}

STUDENT = {"course": "Science", "previous_score": 62, "internal_score": 58, "attendance_pct": 80, "study_hours": 10,
           "assignments_pct": 75, "sleep_hours": 7, "screen_hours": 3, "tutoring": "no", "internet": "yes",
           "parent_education": "Graduate"}


def shared():
    """(TrainingResult, saved model directory, raw training DataFrame) — trained once, in fast mode."""
    if not _SHARED:
        folder = Path(tempfile.mkdtemp(prefix="studentperf_tests_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        raw = datagen.generate(700)
        result = training.train(raw, fast=True, tune=False, cv=3, source="test data")
        persistence.save(result, folder / "model")
        _SHARED.update(result=result, model_dir=folder / "model", raw=raw, folder=folder)
    return _SHARED["result"], _SHARED["model_dir"], _SHARED["raw"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def copy_model(self):
        """A private copy of the trained model that a test may damage (each call makes a fresh copy)."""
        _, model_dir, _ = shared()
        self._copies = getattr(self, "_copies", 0) + 1
        target = self.dir / f"model_copy_{self._copies}"
        shutil.copytree(model_dir, target)
        return target
