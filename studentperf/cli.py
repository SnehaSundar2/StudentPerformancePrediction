"""Interactive menu-driven command-line interface.

All StudentPerfError subclasses are caught here and shown as friendly messages; unexpected errors are
logged and the menu keeps running.
"""

import logging
from pathlib import Path

from . import datagen, persistence, report, schema as S, training
from . import preprocessing as P
from . import validators as v
from .exceptions import StudentPerfError, ValidationError
from .predictor import Predictor
from .web import serve

log = logging.getLogger("studentperf")

MENU = """
============== STUDENT PERFORMANCE PREDICTION ==============
 PREDICT                            MODEL
  1. Predict a student's score       5. Train a new model
  2. What-if: change one detail      6. Model report (accuracy, factors)
  3. Predict scores from a CSV       7. Compare models
  4. Start the web interface         8. Save an HTML report
 DATA
  9. Create a sample dataset (CSV)  10. Check a dataset's quality
  0. Exit
============================================================"""

# (feature, prompt, hint) in the order they are asked
QUESTIONS = [
    ("course", "Course", ", ".join(S.COURSES)),
    ("previous_score", "Previous exam score", "0-100"),
    ("internal_score", "Internal test score", "0-100"),
    ("attendance_pct", "Attendance in percent", "0-100"),
    ("study_hours", "Self-study hours per week", "0-60"),
    ("assignments_pct", "Assignments submitted in percent", "0-100"),
    ("sleep_hours", "Sleep per night in hours", "3-12"),
    ("screen_hours", "Leisure screen time per day in hours", "0-12"),
    ("tutoring", "Takes extra tutoring? yes/no", ""),
    ("internet", "Internet at home? yes/no", ""),
    ("parent_education", "Parents' education", ", ".join(S.PARENT_EDUCATION)),
]
SYNTHETIC_ROWS = 1800


def ask(prompt):
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default=False):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
    return default if not answer else answer in {"y", "yes"}


def _default_text(name):
    default = v.DEFAULTS.get(name)
    if default is None:
        return ""
    if name in S.BINARY:
        return " [yes]" if default else " [no]"
    return f" [{default:g}]" if isinstance(default, float) else f" [{default}]"


class StudentPerfCLI:
    def __init__(self, model_dir="models", reports_dir="reports"):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self._predictor = None
        self.actions = {"1": self.predict, "2": self.what_if, "3": self.predict_csv, "4": self.web,
                        "5": self.train, "6": self.model_report, "7": self.compare, "8": self.save_report,
                        "9": self.make_dataset, "10": self.check_quality}

    # ---------------------------------------------------------- plumbing
    def predictor(self):
        if self._predictor is None:
            self._predictor = Predictor.from_directory(self.model_dir)
            for warning in self._predictor.load_warnings:
                print(f"  ! {warning}")
        return self._predictor

    def run(self):
        if not persistence.is_trained(self.model_dir):
            print("\n  No trained model yet.")
            if confirm("Train one now on the built-in sample data (about a minute)?", default=True):
                self._run_action(self.train_default)
        while True:
            print(MENU)
            choice = ask("Choose an option")
            if choice == "0":
                print("  Goodbye!")
                return
            action = self.actions.get(choice)
            if action is None:
                print("  ! Invalid option, please enter a number from the menu.")
                continue
            self._run_action(action)

    def _run_action(self, action):
        try:
            action()
        except StudentPerfError as exc:
            log.warning("%s failed: %s", action.__name__, exc)
            print(f"  ! Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
        except Exception:  # safety net; the full traceback goes to the log file
            log.exception("Unexpected error in %s", action.__name__)
            print("  ! An unexpected error occurred. See studentperf.log for details.")

    # ---------------------------------------------------------- prediction
    def ask_student(self):
        """Ask for each detail, re-asking (up to 3 times) when a value is invalid. Blank = default for optional ones."""
        print("  Describe the student (press Enter to use the value in brackets):")
        raw = {}
        # Each answer is validated together with sensible values for the rest, so the message names
        # the field that is wrong and nothing else.
        context = {"course": "Science", "previous_score": 60, "internal_score": 60, "attendance_pct": 80,
                   "study_hours": 10}
        for name, prompt, hint in QUESTIONS:
            shown = _default_text(name)
            for attempt in range(3):
                value = ask(f"{prompt}{shown}" + (f"  ({hint})" if hint else ""))
                if value == "" and name not in v.REQUIRED_INPUT:
                    break   # keep the default
                try:
                    if value == "":
                        raise ValidationError(f"{S.LABELS[name]} is required.")
                    v.parse_features({**context, name: value})
                except ValidationError as exc:
                    print(f"    ! {exc.errors.get(name) or exc}")
                    if attempt == 2:
                        raise ValidationError("Too many invalid answers; prediction cancelled.") from None
                else:
                    raw[name] = context[name] = value
                    break
        return raw

    def show_prediction(self, p):
        print(f"\n  Predicted final score:  {p.score:.1f} / 100   ({p.grade})")
        print(f"  Likely range:           {p.low:.1f}  to  {p.high:.1f}   (90% confidence)")
        print(f"  Support flag:           {p.risk}")
        for warning in p.warnings:
            print(f"  ! {warning}")
        if p.advice:
            print("\n  What could raise this score (estimates, not guarantees):")
            for a in p.advice:
                print(f"   +{a['gain_marks']:4.1f} marks   {a['action']}")
        if p.factors:
            print("\n  What moves this score (compared with a typical student):")
            top = max(abs(f["effect_marks"]) for f in p.factors) or 1
            for f in p.factors:
                sign = "+" if f["effect_marks"] >= 0 else "-"
                print(f"   {f['label']:<22} {f['value']:<8} vs {f['typical']:<8} {sign}{abs(f['effect_marks']):5.1f} marks  "
                      f"{report.bar(abs(f['effect_marks']), top, 16)}")
        if p.similar:
            print("\n  Similar students in the training data:")
            for c in p.similar:
                print(f"   {c['course']:<9} previous {c['previous_score']:>5}  attendance {c['attendance_pct'] or '-':>5}  "
                      f"study {c['study_hours'] or '-':>4} h  → final {c['final_score']}")

    def predict(self):
        predictor = self.predictor()
        self.show_prediction(predictor.predict(self.ask_student()))

    def what_if(self):
        predictor = self.predictor()
        raw = self.ask_student()
        for i, name in enumerate(S.RAW_FEATURES, 1):
            print(f"   {i:>2}. {S.LABELS[name]}")
        choice = ask("Which detail to vary? (number or name)")
        name = S.RAW_FEATURES[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(S.RAW_FEATURES) else choice
        key = name.strip().lower().replace(" ", "_")
        suggestions = predictor.comparison_values(key) if key in S.RAW_FEATURES else []
        typed = ask(f"Values to try, comma-separated (Enter = {', '.join(map(str, suggestions)) or '?'})")
        values = [x.strip() for x in typed.split(",") if x.strip()] or suggestions
        results = predictor.what_if(raw, name, values)
        label = S.LABELS[key]
        print(f"\n  Predicted score of the same student with a different {label.lower()}:")
        for value, score in results:
            print(f"   {str(value):<16} {score:5.1f}  {report.bar(score, 100, 28)}")

    def predict_csv(self):
        predictor = self.predictor()
        print(f"  The CSV needs the columns {', '.join(v.REQUIRED_INPUT)}; other details are optional.")
        source = ask("CSV file with students")
        default = Path("predictions") / f"{Path(source).stem}_predicted.csv"
        target = ask(f"Save predictions to [{default}]") or default
        ok, errors = predictor.predict_csv(source, target)
        print(f"  Predicted {ok} student(s); saved to {target}.")
        if errors:
            print(f"  {len(errors)} row(s) could not be predicted:")
            for line in errors[:15]:
                print(f"    - {line}")

    def web(self):
        port = v.validate_port(ask("Port [8010]") or 8010)
        serve(self.predictor(), port)

    # ---------------------------------------------------------- training and reports
    def train_default(self):
        self._train(datagen.generate(SYNTHETIC_ROWS), "synthetic (generated)", fast=False, tune=True)

    def train(self):
        source = ask("CSV file to train on (Enter = generate the built-in sample data)")
        if source:
            df, label = P.load_dataset(source), Path(source.strip('"')).name
        else:
            rows = v.validate_int(ask(f"Rows to generate [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 300, 20000)
            df, label = datagen.generate(rows), "synthetic (generated)"
        fast = confirm("Quick mode (smaller models, faster, a little less accurate)?")
        tune = False if fast else confirm("Tune the best model? (adds about 20 seconds)", default=True)
        self._train(df, label, fast, tune)

    def _train(self, df, label, fast, tune):
        print()
        result = training.train(df, fast=fast, tune=tune, source=label, progress=lambda m: print(f"   {m}"))
        persistence.save(result, self.model_dir)
        self._predictor = None
        print("\n  Data cleaning:")
        print("\n".join("  " + line for line in result.cleaning.lines()))
        print()
        print(report.comparison_text(result.metadata))
        print()
        print(report.model_text(result.metadata))
        print(f"\n  Model saved to {self.model_dir}/ — you can now make predictions.")

    def model_report(self):
        print(report.model_text(self.predictor().meta))

    def compare(self):
        print(report.comparison_text(self.predictor().meta))

    def save_report(self):
        default = self.reports_dir / "model_report.html"
        path = Path(ask(f"File name [{default}]") or default)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        report.save_html(self.predictor().meta, path)
        print(f"  Report saved to {path} — open it in a web browser.")

    # ---------------------------------------------------------- data
    def make_dataset(self):
        rows = v.validate_int(ask(f"Number of students [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 100, 50000)
        dirty = not ask("Include data problems for practising cleaning? [Y/n]").lower().startswith("n")
        default = Path("sample_data") / ("students_dirty.csv" if dirty else "students_clean.csv")
        path = Path(ask(f"File name [{default}]") or default)
        if path.exists():
            raise ValidationError(f"{path} already exists; choose another name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        datagen.generate(rows, dirty=dirty).to_csv(path, index=False)
        print(f"  Saved {rows} students to {path} (synthetic data).")

    def check_quality(self):
        df = P.load_dataset(ask("CSV file to check"))
        _, cleaning = P.clean(df)
        print("\n".join("  " + line for line in cleaning.lines()))
        if cleaning.missing_before:
            print("  Missing values before cleaning: " + ", ".join(f"{c} {n}" for c, n in cleaning.missing_before.items()))
            print("  (remaining gaps are filled during training using values learned from the training rows only)")
