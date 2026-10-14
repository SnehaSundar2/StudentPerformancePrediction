"""The web interface (over real HTTP on a random local port) and the interactive menu."""

import http.client
import io
import json
import threading
import urllib.parse
from contextlib import redirect_stdout
from unittest.mock import patch

from studentperf import datagen, web
from studentperf.cli import StudentPerfCLI
from studentperf.predictor import Predictor
from tests.helpers import STUDENT, TempDirTest, shared

FORM = {k: str(v) for k, v in STUDENT.items()}


class TestWeb(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.server = web.make_server(Predictor.from_directory(shared()[1]), port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}

    def post_form(self, values):
        return self.request("POST", "/", urllib.parse.urlencode(values),
                            {"Content-Type": "application/x-www-form-urlencoded"})

    def post_json(self, payload):
        return self.request("POST", "/api/predict", json.dumps(payload), {"Content-Type": "application/json"})

    def test_pages_and_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Predict final score", body)
        self.assertNotIn("<script", body.lower())                          # the page needs no scripts at all
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/elsewhere", b"")[0], 404)

    def test_form_prediction(self):
        status, body, _ = self.post_form(FORM)
        self.assertEqual(status, 200)
        for part in ("Predicted final score", "Likely range", "What moves this score", "Similar students",
                     "pass mark 40"):
            self.assertIn(part, body)

    def test_weak_student_shows_advice_and_flag(self):
        status, body, _ = self.post_form({**FORM, "previous_score": "35", "internal_score": "30",
                                          "attendance_pct": "55", "study_hours": "3", "sleep_hours": "5"})
        self.assertEqual(status, 200)
        self.assertIn("At risk", body)
        self.assertIn("What could raise this score", body)

    def test_form_errors_keep_the_values_and_name_the_field(self):
        status, body, _ = self.post_form({**FORM, "attendance_pct": "150", "course": "Medicine"})
        self.assertEqual(status, 422)
        self.assertIn("Attendance must be between 0 and 100", body)
        self.assertIn("is not supported", body)
        self.assertIn('value="150"', body)                                  # what the user typed is still there
        self.assertNotIn("Predicted final score", body)

    def test_scripts_in_input_are_escaped(self):
        evil = '"><script>alert(1)</script>'
        status, body, _ = self.post_form({**FORM, "previous_score": evil, "course": evil})
        self.assertEqual(status, 422)
        self.assertNotIn("<script>alert", body)
        self.assertIn("&lt;script&gt;", body)

    def test_json_api(self):
        status, body, headers = self.post_json({**STUDENT})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertLessEqual(data["low"], data["predicted_score"])
        self.assertLessEqual(data["predicted_score"], data["high"])
        self.assertIn(data["risk"], {"On track", "Watch", "At risk"})
        status, body, _ = self.post_json({"course": "Medicine", "attendance_pct": 500})
        self.assertEqual(status, 422)
        self.assertIn("course", json.loads(body)["fields"])
        health = json.loads(self.request("GET", "/api/health")[1])
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["courses"], ["Science", "Commerce", "Arts"])

    def test_bad_requests_are_refused(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{not json", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "x" * 30_000, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/", "x" * 30_000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example.com"})[0], 403)   # DNS-rebinding guard
        self.assertEqual(self.request("GET", "/", headers={"Host": f"localhost:{self.port}"})[0], 200)


def run_cli(cli, inputs):
    out = io.StringIO()
    with patch("builtins.input", side_effect=inputs), redirect_stdout(out):
        cli.run()
    return out.getvalue()


# answers for "Predict a student's score": course, previous, internal, attendance, study hours, assignments,
# sleep, screen time, tutoring, internet, parents' education
ANSWERS = ["Science", "62", "58", "80", "10", "75", "7", "3", "no", "yes", "Graduate"]


class TestCLI(TempDirTest):
    def make(self):
        return StudentPerfCLI(shared()[1], self.dir / "reports")

    def test_predict_session(self):
        out = run_cli(self.make(), ["1", *ANSWERS, "0"])
        for part in ("Predicted final score:", "Likely range:", "Support flag:", "What moves this score",
                     "Similar students"):
            self.assertIn(part, out)

    def test_blank_optional_answers_use_defaults(self):
        out = run_cli(self.make(), ["1", "Arts", "70", "66", "90", "12", "", "", "", "", "", "", "0"])
        self.assertIn("Assumed for missing details", out)
        self.assertIn("Predicted final score:", out)

    def test_invalid_answers_are_asked_again(self):
        answers = ["Medicine", "Science", "62", "58", "150", "80", *ANSWERS[4:]]      # bad course, then bad attendance
        out = run_cli(self.make(), ["1", *answers, "0"])
        self.assertIn("is not supported", out)
        self.assertIn("Attendance must be between 0 and 100", out)
        self.assertIn("Predicted final score:", out)

    def test_three_bad_answers_cancel_the_prediction(self):
        out = run_cli(self.make(), ["1", "Medicine", "Law", "Dance", "0"])
        self.assertIn("Too many invalid answers", out)
        self.assertNotIn("Predicted final score:", out)

    def test_required_answers_cannot_be_skipped(self):
        out = run_cli(self.make(), ["1", "Science", "", "62", "58", "80", "10", *[""] * 6, "0"])
        self.assertIn("Previous exam score is required", out)
        self.assertIn("Predicted final score:", out)

    def test_what_if_session(self):
        out = run_cli(self.make(), ["2", *ANSWERS, "study_hours", "0, 20", "0"])
        self.assertIn("different study hours per week", out)
        lines = [line for line in out.splitlines() if line.strip().startswith(("0.0 ", "20.0 "))]
        self.assertGreaterEqual(len(lines), 2)

    def test_reports_and_menu_errors(self):
        out = run_cli(self.make(), ["6", "7", "8", "", "99", "0"])
        self.assertIn("MODEL REPORT", out)
        self.assertIn("MODEL COMPARISON", out)
        self.assertIn("Report saved to", out)
        self.assertTrue((self.dir / "reports" / "model_report.html").exists())
        self.assertIn("Invalid option", out)

    def test_batch_prediction_through_the_menu(self):
        source = self.dir / "in.csv"
        source.write_text("course,previous_score,internal_score,attendance_pct,study_hours\n"
                          "Science,62,58,80,10\nXYZ,62,58,80,10\n", encoding="utf-8")
        target = self.dir / "out.csv"
        out = run_cli(self.make(), ["3", str(source), str(target), "0"])
        self.assertIn("Predicted 1 student(s)", out)
        self.assertIn("Line 3:", out)
        self.assertTrue(target.exists())

    def test_dataset_tools(self):
        path = self.dir / "mine.csv"
        out = run_cli(self.make(), ["9", "300", "", str(path), "10", str(path), "0"])
        self.assertIn("Saved 300 students", out)
        self.assertIn("rows read", out)
        self.assertIn("Missing values before cleaning", out)
        out = run_cli(self.make(), ["9", "300", "", str(path), "0"])
        self.assertIn("already exists", out)

    def test_no_model_gives_instructions_and_training_works(self):
        empty = StudentPerfCLI(self.dir / "nomodel", self.dir / "reports")
        out = run_cli(empty, ["n", "1", "0"])
        self.assertIn("No trained model yet", out)
        self.assertIn("Train one first", out)
        data = self.dir / "train.csv"
        datagen.generate(400).to_csv(data, index=False)
        out = run_cli(empty, ["n", "5", str(data), "y", "0"])             # quick mode training from a CSV
        self.assertIn("MODEL REPORT", out)
        self.assertIn("Model saved to", out)
        self.assertTrue((self.dir / "nomodel" / "model.joblib").exists())
