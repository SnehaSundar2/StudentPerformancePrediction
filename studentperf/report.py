"""Model reports: a text summary for the terminal and a self-contained HTML report with charts.

Chart conventions: one validated colour per chart, thin bars with a rounded data end, solid hairline
gridlines, text in text colours (never the series colour), a tooltip on every mark, a "show as table"
view for every chart, and light/dark themes.
"""

import html
import math
from datetime import datetime
from pathlib import Path

from . import schema as S
from .exceptions import ModelError

WIDTH = 78


# ====================================================================== terminal
def bar(value, top, width=24):
    return "█" * max(1 if value > 0 else 0, round(width * value / top)) if top else ""


def _verdict(meta):
    test, train = meta["metrics"]["test"], meta["metrics"]["train"]
    gap = train["r2"] - test["r2"]
    lines = []
    if test["r2"] >= 0.9:
        lines.append(f"The model explains {test['r2'] * 100:.1f}% of the variation in final scores of students it has never seen.")
    elif test["r2"] >= 0.7:
        lines.append(f"The model explains {test['r2'] * 100:.1f}% of score variation — useful, but with clear room to improve.")
    else:
        lines.append(f"The model explains only {test['r2'] * 100:.1f}% of score variation — treat predictions with care.")
    lines.append(f"A typical prediction is off by {test['median_ae']:.1f} marks (average {test['mae']:.1f}); "
                 f"{test['within_5']:.0f}% of predictions are within 5 marks of the real score.")
    if gap > 0.05:
        lines.append(f"Overfitting warning: R² is {train['r2']:.3f} on training data but {test['r2']:.3f} on test data.")
    else:
        lines.append(f"No serious overfitting: R² is {train['r2']:.3f} on training data and {test['r2']:.3f} on test data.")
    base = meta["metrics"]["baseline_cv"]
    lines.append(f"A model that always guesses the class average is off by {base['mae']:.1f} marks on average, "
                 f"so this model is {base['mae'] / max(test['mae'], 0.1):.1f}× more accurate.")
    if "pass_accuracy" in test:
        lines.append(f"It called pass or fail correctly for {test['pass_accuracy']:.0f}% of test students, and it "
                     f"flagged {test['risk_recall']:.0f}% of the students who really scored under {S.AT_RISK_BELOW}.")
    cover = test.get("interval_coverage")
    if cover is not None:
        lines.append(f"The 90% score range contained the true score for {cover:.0f}% of test students.")
    return lines


def model_text(meta):
    test = meta["metrics"]["test"]
    rows = meta["rows"]
    out = ["=" * WIDTH, "  MODEL REPORT",
           f"  {meta['model']}{' (' + meta['model_note'] + ')' if meta.get('model_note') else ''} · trained "
           f"{meta['trained_at'].replace('T', ' ')} in {meta['training_seconds']} s · data: {meta['source']}",
           f"  {rows['raw']} rows read → {rows['clean']} after cleaning → {rows['train']} to train, {rows['test']} to test",
           "=" * WIDTH, "", "  ACCURACY ON THE UNSEEN TEST SET",
           f"   R²             {test['r2']:.3f}       (1.0 = perfect, 0 = no better than guessing the average)",
           f"   Average miss   {test['mae']:.1f} marks   (median {test['median_ae']:.1f})",
           f"   RMSE           {test['rmse']:.1f} marks   (penalises big misses more)",
           f"   Within 5 marks {test['within_5']:.0f}% of students · within 10 marks: {test['within_10']:.0f}%",
           f"   Bias           {test['bias']:+.2f} marks   (positive = over-predicts on average)"]
    if "pass_accuracy" in test:
        out += ["", "  PASS / FAIL AND SUPPORT FLAG",
                f"   Pass or fail correct       {test['pass_accuracy']:.0f}%   (pass mark {S.PASS_MARK})",
                f"   Grade band correct         {test['grade_accuracy']:.0f}%",
                f"   Under-{S.AT_RISK_BELOW} students found    {test['risk_recall']:.0f}%   "
                f"({test['risk_tp']} of {test['risk_tp'] + test['risk_fn']})",
                f"   Flags that were right      {test['risk_precision']:.0f}%   "
                f"({test['risk_tp']} of {test['risk_tp'] + test['risk_fp']})"]
    out += [""]
    out += [f"  → {line}" for line in _verdict(meta)]
    out += ["", "  WHAT THE MODEL RELIES ON (drop in R² when a feature is scrambled)"]
    top = meta["importance"][0]["importance"] or 1
    for item in meta["importance"][:8]:
        out.append(f"   {item['label']:<22} {item['importance']:6.3f}  {bar(item['importance'], top)}")
    out += ["", "  ERROR BY COURSE (test set, average miss in marks)"]
    top = max(s["mae"] for s in meta["segments"]["course"])
    for s in meta["segments"]["course"]:
        out.append(f"   {s['course']:<10} {s['mae']:5.1f}  {bar(s['mae'], top, 20)}  ({s['n']} students)")
    return "\n".join(out)


def comparison_text(meta):
    rows = ["  MODEL COMPARISON (5-fold cross-validation on training data; lower error is better)",
            f"  {'MODEL':<38} {'AVG MISS':>9} {'RMSE':>8} {'R²':>7}", f"  {'-' * 38} {'-' * 9} {'-' * 8} {'-' * 7}"]
    for c in meta["comparison"]:
        mark = "  ← chosen" if c["model"] == meta["metrics"]["cv"]["model"] else ""
        rows.append(f"  {c['model']:<38} {c['mae']:>6.1f} mk {c['rmse']:>6.1f} {c['r2']:>8.3f}{mark}")
    return "\n".join(rows)


# ====================================================================== HTML
CSS = """
.rep{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--series:#2a78d6;--border:rgba(11,11,11,.10);--other:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}}
:root[data-theme="dark"] .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}
*{box-sizing:border-box}html,body{margin:0}
body.rep{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 56px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 4px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px}section{margin-top:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile.hero .value{font-size:48px}.tile .note{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-top:12px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg.chart text{fill:var(--muted);font-size:13px;font-family:inherit}
svg.chart .label{fill:var(--ink2)}svg.chart .value{fill:var(--ink);font-weight:600}
svg.chart .gridline{stroke:var(--grid)}svg.chart .baseline{stroke:var(--axis)}svg.chart .ref{stroke:var(--axis);stroke-width:1.5}
.mark{fill:var(--series);outline:none}.mark.other{fill:var(--other)}.mark:hover,.mark:focus-visible{fill-opacity:.75}
.mark:focus-visible{stroke:var(--ink);stroke-width:1.5}circle.dot{fill:var(--series);fill-opacity:.7;stroke:var(--surface);stroke-width:1.5}
ul.notes{margin:8px 0 0;padding-left:20px}ul.notes li{margin:4px 0}
details.tv{margin-top:8px}details.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:10;max-width:280px}#tip b{display:block;font-size:15px}
footer{margin-top:32px;color:var(--muted);font-size:12px}@media (max-width:520px){.tile.hero .value{font-size:40px}.grid{grid-template-columns:1fr}}
"""
TOOLTIP_JS = """(function(){var tip=document.getElementById('tip');
function show(el,x,y){tip.textContent='';var b=document.createElement('b');b.textContent=el.getAttribute('data-value');
var s=document.createElement('span');s.textContent=el.getAttribute('data-label');tip.appendChild(b);tip.appendChild(s);
tip.style.display='block';var w=tip.offsetWidth;tip.style.left=Math.min(x+14,window.innerWidth-w-8)+'px';tip.style.top=(y+14)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-value]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.right,r.top);});el.addEventListener('blur',hide);});})();"""


def _e(value):
    return html.escape(str(value), quote=True)


def _tick_ceiling(value, intervals=4):
    """Smallest axis maximum with a 'nice' step (1, 2, 2.5, 5, 10 × a power of ten) that covers `value`."""
    raw = max(value, 1e-9) / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * magnitude:
            return m * magnitude * intervals
    return 10 * magnitude * intervals


def _end_rounded(x, y, w, h, r=4):
    r = min(r, h / 2, w)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h - r:.1f}"
            f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z")


def _top_rounded(x, y, w, h, r=4):
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}"
            f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z")


def hbar_chart(points, aria, highlight=None, fmt="{:g}"):
    """points [(label, value)]; with `highlight`, that bar is the series colour and the rest recede."""
    w, row, label_w, right = 420, 30, 190, 58
    h = row * len(points) + 8
    vmax = max(v for _, v in points) or 1
    pw = w - label_w - right
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">',
             f'<line class="baseline" x1="{label_w}" x2="{label_w}" y1="0" y2="{h - 4}"/>']
    for i, (label, value) in enumerate(points):
        y = 4 + i * row
        bw = max(pw * value / vmax, 2)
        cls = "mark" + ("" if highlight in (None, label) else " other")
        shown = fmt.format(value)
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 19}" text-anchor="end">{_e(label[:28])}</text>'
                     f'<path class="{cls}" tabindex="0" d="{_end_rounded(label_w, y + 6, bw, 18)}" data-value="{_e(shown)}" '
                     f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>'
                     f'<text class="value" x="{label_w + bw + 6:.1f}" y="{y + 19}">{_e(shown)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def column_chart(points, aria, x_title=""):
    w, h, left, right, top, bottom = 420, 236, 40, 6, 20, 52
    pw, ph = w - left - right, h - top - bottom
    vmax = _tick_ceiling(max((v for _, v in points), default=0), 5)
    band = pw / max(len(points), 1)
    bar_w = min(24, band - 2)
    tallest = max(range(len(points)), key=lambda i: points[i][1]) if points else -1
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        y = top + ph - ph * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{vmax * k / 5:g}</text>')
    for i, (label, value) in enumerate(points):
        x = left + band * i + (band - bar_w) / 2
        bh = ph * value / vmax if vmax else 0
        if value:
            parts.append(f'<path class="mark" tabindex="0" d="{_top_rounded(x, top + ph - bh, bar_w, bh)}" data-value="{value} students" '
                         f'data-label="{_e(label)}" aria-label="{_e(label)}: {value}"/>')
        if i % 2 == 0:
            parts.append(f'<text class="label" x="{x + bar_w / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle">{_e(label)}</text>')
        if i == tallest and value:
            parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{top + ph - bh - 5:.1f}" text-anchor="middle">{value}</text>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text></svg>')
    return "".join(parts)


def scatter_chart(points, aria):
    """Actual (x) vs predicted (y) final score, both on the 0-100 scale, with the 'perfect prediction' diagonal."""
    w, h, left, bottom, top, right = 420, 340, 46, 40, 10, 10
    pw, ph = w - left - right, h - top - bottom
    vmax = 100
    sx = lambda v: left + pw * v / vmax
    sy = lambda v: top + ph - ph * v / vmax
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        v = vmax * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
                     f'<text x="{sx(v):.1f}" y="{h - bottom + 16}" text-anchor="middle">{v:g}</text>')
    parts.append(f'<line class="ref" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(vmax):.1f}" y2="{sy(vmax):.1f}"/>')
    for a, b in points:
        label = f"actual {a:.1f}, predicted {b:.1f} ({b - a:+.1f} marks)"
        parts.append(f'<circle class="dot" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="3.5" data-value="{b:.1f} predicted" '
                     f'data-label="{_e(label)}" tabindex="0" aria-label="{_e(label)}"/>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">Actual final score</text>'
                 f'<text class="label" x="12" y="{top + ph / 2}" text-anchor="middle" transform="rotate(-90 12 {top + ph / 2})">Predicted score</text></svg>')
    return "".join(parts)


def _table(headers, rows, numeric=()):
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table_view(headers, rows, numeric=(1,)):
    return f'<details class="tv"><summary>Show as table</summary><div class="scroll">{_table(headers, rows, numeric)}</div></details>'


def _tile(label, value, note="", hero=False):
    return (f'<div class="card tile{" hero" if hero else ""}"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            + (f'<div class="note">{_e(note)}</div>' if note else "") + "</div>")


def histogram_of_errors(points, lo=-12, hi=12, step=2):
    """Prediction error (predicted − actual) in marks, binned. Returns (bins, count_outside_the_axis)."""
    edges = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
    counts = [0] * (len(edges) - 1)
    for a, b in points:
        e = b - a
        if lo <= e < hi:
            counts[int((e - lo) // step)] += 1
    return [(f"{edges[i]:+g}", c) for i, c in enumerate(counts)], sum(1 for a, b in points if not lo <= b - a < hi)


def render_html(meta):
    if "test_points" not in meta:
        raise ModelError("This model was saved without test results; train it again to build a report.")
    test, cv, rows = meta["metrics"]["test"], meta["metrics"]["cv"], meta["rows"]
    points = meta["test_points"]
    body = [f'<header><h1>Student Performance Prediction — model report</h1><div class="sub">{_e(meta["model"])}'
            f'{" (" + _e(meta["model_note"]) + ")" if meta.get("model_note") else ""} · trained {_e(meta["trained_at"].replace("T", " "))} · '
            f'data: {_e(meta["source"])} · {rows["raw"]} rows read, {rows["clean"]} after cleaning, {rows["train"]} train / {rows["test"]} test</div></header>']
    body.append('<section aria-label="Summary"><div class="tiles">' + "".join([
        _tile("Test R²", f"{test['r2']:.3f}", "1.0 would be perfect", hero=True),
        _tile("Average miss", f"{test['mae']:.1f} marks", f"median {test['median_ae']:.1f}"),
        _tile("Within 5 marks", f"{test['within_5']:.0f}%", f"within 10 marks: {test['within_10']:.0f}%"),
        _tile("Pass / fail correct", f"{test.get('pass_accuracy', 0):.0f}%", f"pass mark {S.PASS_MARK}"),
        _tile(f"Under-{S.AT_RISK_BELOW} students found", f"{test.get('risk_recall', 0):.0f}%", "support flag recall"),
        _tile("90% range coverage", f"{test.get('interval_coverage', 0):.0f}%", "target ≈ 90%")]) + "</div></section>")
    body.append('<section class="card"><h2>What this means</h2><ul class="notes">'
                + "".join(f"<li>{_e(line)}</li>" for line in _verdict(meta)) + "</ul></section>")

    scatter_svg = scatter_chart(points, "Actual versus predicted final score of test students")
    comparison_rows = [[c["model"], f"{c['mae']:.2f}", f"{c['rmse']:.2f}", f"{c['r2']:.3f}"] for c in meta["comparison"]]
    comp = [(c["model"], c["mae"]) for c in meta["comparison"]]
    chosen = next((c["model"] for c in meta["comparison"] if c["model"] == cv["model"]), None)
    err_bins, outside = histogram_of_errors(points)
    imp = [(i["label"], round(i["importance"], 3)) for i in meta["importance"][:8]]
    cards = [
        f'<div class="card"><h3>Model comparison</h3><div class="sub">Cross-validated average miss in marks — lower is better</div>'
        f'{hbar_chart(comp, "Cross-validated error of each model", chosen, "{:.1f}")}'
        f'{_table_view(["Model", "Average miss (marks)", "RMSE (marks)", "R²"], comparison_rows, (1, 2, 3))}</div>',
        f'<div class="card"><h3>Actual vs predicted final score</h3><div class="sub">{len(points)} test students; dots on the line are perfect predictions</div>'
        f'{scatter_svg}{_table_view(["Actual", "Predicted"], [[a, b] for a, b in points[:40]], (0, 1))}</div>',
        f'<div class="card"><h3>How far off are predictions?</h3><div class="sub">Predicted minus actual, in marks'
        f'{f" ({outside} beyond ±12 not shown)" if outside else ""}</div>{column_chart(err_bins, "Distribution of prediction errors", "Prediction error (marks)")}'
        f'{_table_view(["Error (marks)", "Students"], err_bins)}</div>',
        f'<div class="card"><h3>What the model relies on</h3><div class="sub">Drop in R² when a feature is scrambled</div>'
        f'{hbar_chart(imp, "Feature importance", None, "{:.3f}")}{_table_view(["Feature", "R² drop"], imp)}</div>',
    ]
    body.append(f'<section><h2>Charts</h2><div class="grid">{"".join(cards)}</div></section>')

    def seg_rows(key, name):
        return [[s[key], s["n"], f"{s['mae']:.1f}", f"{s['bias']:+.1f}"] for s in meta["segments"][name]]
    band_rows = [[s["band"], s["n"], f"{s['mae']:.1f}", f"{s['bias']:+.1f}"] for s in meta["segments"]["score_band"]]
    head = ["Students", "Average miss (marks)", "Bias (marks)"]
    course_table = _table(["Course"] + head, seg_rows("course", "course"), (1, 2, 3))
    parent_table = _table(["Parent education"] + head, seg_rows("parent_education", "parent_education"), (1, 2, 3))
    band_table = _table(["Score band"] + head, band_rows, (1, 2, 3))
    body.append('<section><h2>Where is the model weaker?</h2><div class="grid">'
                f'<div class="card"><h3>By course</h3>{course_table}</div>'
                f'<div class="card"><h3>By parent education</h3>{parent_table}</div>'
                f'<div class="card"><h3>By real score band</h3>{band_table}</div>'
                '</div><p class="sub">Predictions pull towards the middle: the very best and the weakest students are '
                'the hardest to predict, so the model tends to over-predict for failing students and under-predict '
                'for distinctions.</p></section>')

    steps = meta["cleaning"]["steps"]
    body.append('<section class="card"><h2>Data cleaning log</h2><div class="sub">'
                f'{meta["cleaning"]["rows_in"]} rows read → {meta["cleaning"]["rows_out"]} kept</div>'
                + _table(["Problem found", "Rows", "Action"], [[s["label"], s["count"], s["outcome"]] for s in steps], (1,))
                + "</section>")
    params = ", ".join(f"{k} = {v}" for k, v in meta.get("model_params", {}).items()) or "defaults"
    env = meta["environment"]
    body.append(f'<footer>Generated {datetime.now():%d %b %Y, %H:%M} · settings: {_e(params)} · seed {meta["seed"]} · '
                f'Python {_e(env["python"])}, scikit-learn {_e(env["sklearn"])} · developed by Sneha S · '
                'Trained on synthetic data: for learning only, never to judge a real student.</footer>')
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Student performance model report</title><style>{CSS}</style></head><body class='rep'><main>" + "\n".join(body)
            + f"</main><div id='tip' role='status' aria-live='polite'></div><script>{TOOLTIP_JS}</script></body></html>")


def save_html(meta, path):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(meta), encoding="utf-8")
    except OSError as exc:
        raise ModelError(f"Could not write {path}: {exc.strerror or exc}") from exc
    return path
