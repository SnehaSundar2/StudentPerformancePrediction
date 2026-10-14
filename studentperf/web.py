"""Local web interface for predictions (standard library only).

    GET  /              the form
    POST /              the form submitted → result page
    POST /api/predict   JSON in, JSON out
    GET  /api/health    model information

Safety measures: binds to 127.0.0.1 only; the Host header must be a local name (blocks DNS-rebinding);
request bodies are size-limited; every value is validated by the same code as the command line;
everything shown is HTML-escaped; the page has no scripts and a strict Content-Security-Policy.
"""

import html
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import schema as S
from .exceptions import StudentPerfError, ValidationError

log = logging.getLogger("studentperf")

MAX_BODY = 20_000
DRAIN_LIMIT = 1_000_000   # a rejected body up to this size is read and discarded before the error is sent
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                               "base-uri 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
}
FORM_DEFAULTS = {"course": "Science", "previous_score": "62", "internal_score": "60", "attendance_pct": "80",
                 "study_hours": "10", "assignments_pct": "75", "sleep_hours": "7", "screen_hours": "3",
                 "tutoring": "no", "internet": "yes", "parent_education": "Graduate"}
RISK_CLASS = {"On track": "ok", "Watch": "watch", "At risk": "risk"}

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--border:rgba(11,11,11,.12);--err:#b3261e;--errbg:#fdecea;--warnbg:#fff4e0;--okbg:#e6f4ea;--ok:#1e6b34}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--border:rgba(255,255,255,.12);--err:#ff8a80;--errbg:#3d1717;--warnbg:#3a2c12;
--okbg:#16301d;--ok:#8fd9a0}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 8px}h3{font-size:14px;margin:16px 0 6px}
.sub{color:var(--ink2);font-size:13px}.badge{font-size:12px;border:1px solid var(--border);border-radius:999px;padding:2px 8px;margin-left:8px;color:var(--ink2)}
.layout{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:16px;margin-top:20px;align-items:start}
@media (max-width:760px){.layout{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
form{display:grid;grid-template-columns:1fr 1fr;gap:10px 12px}label{display:block;font-size:13px;color:var(--ink2)}
.full{grid-column:1/-1}input,select{width:100%;margin-top:3px;padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--page);
color:var(--ink);font:inherit}input:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
.err{color:var(--err);font-size:12px;margin-top:2px}.invalid input,.invalid select{border-color:var(--err)}
button{grid-column:1/-1;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.hero{font-size:48px;font-weight:600;line-height:1.1;margin:4px 0}.hero small{font-size:18px;color:var(--ink2);font-weight:400}
.range{color:var(--ink2)}.chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.chip{font-size:13px;font-weight:600;border-radius:999px;padding:3px 12px;border:1px solid var(--border)}
.chip.ok{background:var(--okbg);color:var(--ok)}.chip.watch{background:var(--warnbg)}.chip.risk{background:var(--errbg);color:var(--err)}
.rangebar{position:relative;height:8px;border-radius:4px;background:var(--line);margin:14px 0 4px}
.rangebar i{position:absolute;top:0;bottom:0;border-radius:4px;background:var(--accent);opacity:.35}
.rangebar b{position:absolute;top:-3px;width:4px;height:14px;border-radius:2px;background:var(--accent)}
.rangebar u{position:absolute;top:-5px;width:2px;height:18px;background:var(--muted);text-decoration:none}
.rangelab{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.factor{display:grid;grid-template-columns:170px 1fr 90px;gap:8px;align-items:center;font-size:13px;margin:5px 0}
.factor .track{height:10px;background:var(--line);border-radius:5px;position:relative}.factor .fill{position:absolute;top:0;bottom:0;border-radius:5px;background:var(--accent)}
.factor .fill.neg{background:var(--muted)}.factor .amt{text-align:right;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--line)}th{color:var(--ink2)}
.num{text-align:right;font-variant-numeric:tabular-nums}.note{background:var(--warnbg);border-radius:8px;padding:8px 10px;margin:8px 0;font-size:13px}
ul.advice{margin:6px 0;padding-left:20px;font-size:14px}ul.advice li{margin:3px 0}
.error{background:var(--errbg);color:var(--err);border-radius:8px;padding:10px;margin-bottom:12px}
footer{margin-top:24px;color:var(--muted);font-size:12px}@media (max-width:520px){.factor{grid-template-columns:1fr 70px}.factor .track{display:none}}
"""


def e(value):
    return html.escape(str(value), quote=True)


def select(name, options, current):
    items = [f'<option{" selected" if o == current else ""}>{e(o)}</option>' for o in options]
    return f'<select name="{name}" id="{name}">{"".join(items)}</select>'


def number_input(name, current):
    _, low, high, _ = S.NUMERIC[name]
    return (f'<input type="number" name="{name}" id="{name}" min="{low}" max="{high}" step="0.1" '
            f'value="{e(current)}" inputmode="decimal" required>')


def field(name, control, errors, wide=False):
    label = S.LABELS[name] + (f" ({S.UNITS[name]})" if name in S.UNITS else "")
    message = f'<div class="err">{e(errors[name])}</div>' if name in errors else ""
    return f'<div class="{"full " if wide else ""}{"invalid" if name in errors else ""}"><label for="{name}">{e(label)}{control}</label>{message}</div>'


def form_html(values, errors):
    v = {**FORM_DEFAULTS, **values}
    parts = [field("course", select("course", S.COURSES, v["course"]), errors),
             field("parent_education", select("parent_education", S.PARENT_EDUCATION, v["parent_education"]), errors)]
    parts += [field(name, number_input(name, v[name]), errors) for name in S.NUMERIC_FEATURES]
    parts += [field("tutoring", select("tutoring", ["no", "yes"], v["tutoring"]), errors),
              field("internet", select("internet", ["yes", "no"], v["internet"]), errors)]
    return f'<form method="post" action="/" novalidate>{"".join(parts)}<button type="submit">Predict final score</button></form>'


def result_html(p):
    lo, hi = p.low, p.high
    out = [f'<h2>Predicted final score</h2><div class="hero">{p.score:.1f}<small> / 100</small></div>',
           f'<div class="chips"><span class="chip">{e(p.grade)}</span>'
           f'<span class="chip {RISK_CLASS[p.risk]}">{e(p.risk)}</span></div>',
           f'<div class="range">Likely range (90%): <b>{lo:.1f}</b> to <b>{hi:.1f}</b></div>',
           f'<div class="rangebar" aria-hidden="true"><i style="left:{lo:.1f}%;right:{100 - hi:.1f}%"></i>'
           f'<u style="left:{S.PASS_MARK}%"></u><b style="left:calc({p.score:.1f}% - 2px)"></b></div>'
           f'<div class="rangelab"><span>0</span><span>pass mark {S.PASS_MARK}</span><span>100</span></div>']
    for w in p.warnings:
        out.append(f'<div class="note">⚠ {e(w)}</div>')
    if p.advice:
        out.append("<h3>What could raise this score</h3><ul class='advice'>"
                   + "".join(f'<li>{e(a["action"])} — about <b>+{a["gain_marks"]:.1f}</b> marks</li>' for a in p.advice)
                   + "</ul><div class='sub'>Estimates from patterns in the training data, not guarantees.</div>")
    if p.factors:
        top = max(abs(f["effect_marks"]) for f in p.factors) or 1
        out.append("<h3>What moves this score (compared with a typical student)</h3>")
        for f in p.factors:
            width = max(3, 100 * abs(f["effect_marks"]) / top)
            sign = "+" if f["effect_marks"] >= 0 else "−"
            out.append(f'<div class="factor"><span>{e(f["label"])}: {e(f["value"])}</span>'
                       f'<div class="track"><div class="fill{" neg" if f["effect_marks"] < 0 else ""}" style="width:{width:.0f}%"></div></div>'
                       f'<span class="amt">{sign}{abs(f["effect_marks"]):.1f} marks</span></div>')
        out.append('<div class="sub">A longer bar means a bigger effect; “−” lowers the score compared with the typical value.</div>')
    if p.similar:
        def cell(value):
            return "—" if value is None else value
        rows = "".join(f'<tr><td>{e(c["course"])}</td><td class="num">{c["previous_score"]}</td><td class="num">{cell(c["internal_score"])}</td>'
                       f'<td class="num">{cell(c["attendance_pct"])}</td><td class="num">{cell(c["study_hours"])}</td>'
                       f'<td class="num"><b>{c["final_score"]}</b></td></tr>' for c in p.similar)
        out.append("<h3>Similar students in the training data</h3><table><thead><tr><th>Course</th><th class='num'>Previous</th>"
                   "<th class='num'>Internal</th><th class='num'>Attend. %</th><th class='num'>Study h</th><th class='num'>Final</th></tr></thead>"
                   f"<tbody>{rows}</tbody></table>")
    return "".join(out)


def page(meta, values=None, errors=None, prediction=None, message=None):
    errors = errors or {}
    test = meta["metrics"]["test"]
    if prediction is not None:
        right = result_html(prediction)
    elif message:
        right = f'<div class="error">{e(message)}</div>'
    else:
        right = ('<h2>Describe a student</h2><p class="sub">Fill in the details on the left. The model predicts the final '
                 'exam score, gives a likely range, and shows what helps or hurts the score.</p>'
                 f'<p class="sub">Model: {e(meta["model"])} · average miss {test["mae"]:.1f} marks on unseen students · '
                 f'R² {test["r2"]:.3f}</p>')
    summary = "".join(f"<li>{e(m)}</li>" for m in ([errors["_"]] if "_" in errors else []))
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Student Performance Prediction</title><style>{CSS}</style></head><body><main>"
            "<h1>Student Performance Prediction<span class='badge'>demo · synthetic data</span></h1>"
            "<div class='sub'>Predicts a final exam score from study habits, attendance and past marks, so support can start early.</div>"
            f"<div class='layout'><div class='card'>{('<div class=error>' + summary + '</div>') if summary else ''}"
            f"{form_html(values or {}, errors)}</div><div class='card' aria-live='polite'>{right}</div></div>"
            "<footer>Developed by Sneha S · Trained on generated data to learn the machine-learning workflow — "
            "a guide for offering help, never a judgement of a real student.</footer></main></body></html>")


def make_server(predictor, host="127.0.0.1", port=8010):
    meta = predictor.meta

    class Handler(BaseHTTPRequestHandler):
        server_version = "StudentPerf/1.0"

        def log_message(self, fmt, *args):   # keep the terminal quiet; real problems go to the log file
            log.info("web: " + fmt, *args)

        # ----- helpers
        def send(self, status, body, content_type="text/html; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for key, value in SECURITY_HEADERS.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, payload):
            self.send(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

        def host_ok(self):
            host = (self.headers.get("Host") or "").strip().lower()
            name = host[:host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]
            return name in LOCAL_HOSTS

        def read_body(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise ValidationError("Content-Length header is required.") from None
            if length < 0 or length > MAX_BODY:
                self.discard_body(length)
                raise OverflowError
            return self.rfile.read(length)

        def discard_body(self, length):
            """Read and throw away (a bounded amount of) a rejected body before replying.

            If the server answers and closes while the client is still sending, the operating system may
            reset the connection and the client never sees the 413 message. Huge bodies are not drained
            (that would let one request tie the server up); for those the connection is simply closed.
            """
            remaining = min(max(length, 0), DRAIN_LIMIT)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True

        def guard(self):
            if not self.host_ok():
                self.send(403, "Forbidden: this server only answers to localhost.", "text/plain; charset=utf-8")
                return False
            return True

        # ----- routes
        def do_GET(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, page(meta))
            elif path == "/api/health":
                self.send_json(200, {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"],
                                     "test_r2": round(meta["metrics"]["test"]["r2"], 4),
                                     "test_mae_marks": round(meta["metrics"]["test"]["mae"], 2),
                                     "features": S.RAW_FEATURES, "courses": S.COURSES})
            else:
                self.send(404, page(meta, message="Page not found."))

        def do_POST(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path not in ("/", "/api/predict"):
                return self.send(404, page(meta, message="Page not found."))
            api = path == "/api/predict"
            try:
                body = self.read_body()
            except OverflowError:
                return self.send_json(413, {"error": "Request too large."}) if api else \
                    self.send(413, page(meta, message="That request is too large."))
            except ValidationError as exc:
                return self.send_json(411, {"error": str(exc)}) if api else self.send(411, page(meta, message=str(exc)))
            try:
                if api:
                    if "application/json" not in (self.headers.get("Content-Type") or ""):
                        return self.send_json(415, {"error": "Send JSON with Content-Type: application/json."})
                    try:
                        raw = json.loads(body.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        return self.send_json(400, {"error": "The request body is not valid JSON."})
                    prediction = predictor.predict(raw)
                    return self.send_json(200, prediction.to_dict())
                form = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True, max_num_fields=30)
                raw = {k: v[0] for k, v in form.items()}
                try:
                    prediction = predictor.predict(raw)
                except ValidationError as exc:
                    return self.send(422, page(meta, raw, exc.errors or {"_": str(exc)}))
                self.send(200, page(meta, raw, prediction=prediction))
            except ValidationError as exc:
                self.send_json(422, {"error": str(exc), "fields": exc.errors})
            except StudentPerfError as exc:
                self.send_json(500, {"error": str(exc)}) if api else self.send(500, page(meta, message=str(exc)))
            except Exception:  # noqa: BLE001 - never leak internals to the browser
                log.exception("Unexpected error handling %s", path)
                self.send_json(500, {"error": "Something went wrong."}) if api else \
                    self.send(500, page(meta, message="Something went wrong. See the log file."))

    return ThreadingHTTPServer((host, port), Handler)


def serve(predictor, port=8010, host="127.0.0.1"):
    server = make_server(predictor, host, port)
    print(f"  Student Performance Prediction is running at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Web interface stopped.")
    finally:
        server.server_close()
