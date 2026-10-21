"""Comparison board across candidates and experiments (PLAN.md §4.2, §12).

One static HTML page over several scored runs:

- scorecard: one row per run, no composite score, n and intervals shown
- heatmaps: leak rate by PII type, document condition and page encoding (candidate × slice),
  and the failure-category mix
- failure decomposition: for each candidate with oracle runs on the same dataset, the share
  of the end-to-end leak rate attributable to each stage (telescoping differences)
- degradation ladders: leak rate against each controlled factor (DPI, skew, JPEG, noise, blur)

Charts follow the benchmark's chart rules: one axis, one hue for magnitude, fixed categorical
order (validated for colour-vision deficiency), direct labels plus a table view.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..dataset import load_manifest
from ..evaluate import HIDDEN_VALUES, SAFE_VALUES
from ..runner.runner import load_run
from ..runner.scoring import results_root

# Leak = original content still there; failure = leak or hidden without the required replacement.
HIDDEN = HIDDEN_VALUES
PASS = SAFE_VALUES
MODE_ORDER = ["e2e", "oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions"]
STAGE_BETWEEN = {("e2e", "oracle_text"): "text acquisition", ("oracle_text", "oracle_mentions"): "detection",
                 ("oracle_mentions", "oracle_entities"): "resolution",
                 ("oracle_entities", "oracle_regions"): "localization"}
STAGES = ["text acquisition", "detection", "resolution", "localization", "application"]
# Categorical slots 1-4 (validated light and dark, adjacent pairs), fixed order by stage/candidate.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _rows(run_id: str, name: str) -> list[dict]:
    p = results_root() / run_id / "tables" / f"{name}.parquet"
    return pq.read_table(p).to_pylist() if p.exists() else []


class RunData:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run = load_run(run_id)
        self.candidate = self.run.candidate["name"]
        self.version = self.run.candidate["version"]
        self.mode = self.run.experiment.mode
        self.dataset = f"{self.run.dataset['name']}@{self.run.dataset['version']}"
        self.mentions = [m for m in _rows(run_id, "mentions") if m["requirement"] == "MUST" and m["repeat"] == 0
                         and m["status"] != "NOT_EVALUATED"]
        self.words = [w for w in _rows(run_id, "words") if w["repeat"] == 0 and w["status"] != "NOT_EVALUATED"]
        self.documents = [d for d in _rows(run_id, "documents") if d.get("repeat") == 0]
        self.determinism = _rows(run_id, "determinism")
        self.matters = _rows(run_id, "matters")
        self.label = f"{self.candidate} · {self.run.experiment.name}"

    def leak(self, mentions=None) -> float | None:
        ms = self.mentions if mentions is None else mentions
        return sum(m["status"] not in HIDDEN for m in ms) / len(ms) if ms else None

    def fail(self, mentions=None) -> float | None:
        ms = self.mentions if mentions is None else mentions
        return sum(m["status"] not in PASS for m in ms) / len(ms) if ms else None


def _ci(ci) -> str:
    return "" if not ci else f"{ci[0] * 100:.1f}–{ci[1] * 100:.1f}%"


def _verdicts(v: dict) -> str:
    """The tool's own safety check against the evaluator: right calls and wrong ones."""
    if not v:
        return "N/A"
    labels = (("correct_release", "released ok"), ("correct_hold", "held ok"), ("false_alarm", "false alarm"),
              ("unsafe_release", "unsafe release"))
    return " · ".join(f"{name} {v.get(k, 0)}" for k, name in labels if v.get(k))


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _bootstrap(per_doc: list[tuple[int, int]], n: int = 2000) -> tuple[float, float] | None:
    if len(per_doc) < 2:
        return None
    rng = np.random.default_rng(0)
    arr = np.array(per_doc, float)
    vals = [s[:, 0].sum() / s[:, 1].sum() for s in (arr[rng.integers(0, len(arr), len(arr))] for _ in range(n))
            if s[:, 1].sum() > 0]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else None


# -- sections ------------------------------------------------------------------------------------


def scorecard(runs: list[RunData]) -> list[dict]:
    rows = []
    for r in runs:
        per_doc = defaultdict(lambda: [0, 0])
        for m in r.mentions:
            per_doc[m["doc_id"]][1] += 1
            per_doc[m["doc_id"]][0] += m["status"] not in HIDDEN
        done = [m for m in r.mentions if m["status"] != "NO_OUTPUT"]
        not_replaced = [m for m in r.mentions if m["status"] == "NOT_REPLACED"]
        searchable = [d["searchable_rate"] for d in r.documents if isinstance(d.get("searchable_rate"), (int, float))]
        verdicts = Counter(d["self_check.verdict"] for d in r.documents if d.get("self_check.verdict"))
        l12 = [m for m in r.mentions if m["status"] in ("PARTIAL", "LEAK_VISIBLE", "LEAK_RECOVERABLE")]
        ordinary = [w for w in r.words if w["requirement"] == "ORDINARY"]
        protected = [w for w in r.words if w["requirement"] == "MUST_NOT"]
        must_w = [w for w in r.words if w["requirement"] == "MUST"]

        def red(w):
            return w["visible_fraction"] is not None and w["visible_fraction"] < 0.5

        tp, fp = sum(map(red, must_w)), sum(map(red, ordinary + protected))
        flips = [d["flip_rate"] for d in r.determinism if d.get("flip_rate") is not None]
        rows.append({
            "run": r.run_id, "candidate": r.candidate, "version": r.version, "experiment": r.run.experiment.name,
            "mode": r.mode, "dataset": r.dataset, "docs": len({m["doc_id"] for m in r.mentions}),
            "n": len(r.mentions), "leak": r.leak(), "ci": _bootstrap([tuple(v) for v in per_doc.values()]),
            "leak_completed": r.leak(done), "l12": len(l12) / len(r.mentions) if r.mentions else None,
            "not_replaced": len(not_replaced) / len(r.mentions) if r.mentions else None, "fail": r.fail(),
            "searchable": float(np.mean(searchable)) if searchable else None, "self_check": dict(verdicts),
            "matter": (sum(1 for m in r.matters if m.get("consistent")), len(r.matters)),
            "clean": f"{sum(1 for d in r.documents if d.get('clean_document'))}/{len(r.documents)}",
            "collateral": (sum(w["status"] != "PRESERVED" for w in ordinary) / len(ordinary)) if ordinary else None,
            "protected": sum(w["status"] != "PRESERVED" for w in protected),
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / len(must_w) if must_w else None,
            "flip": float(np.mean(flips)) if flips else None,
        })
    return rows


def heat(runs: list[RunData], key) -> tuple[list[str], list[str], dict]:
    """Rows = run labels, columns = slices, cells = (leak rate, leaking, n)."""
    cells: dict = {}
    cols: Counter = Counter()
    for r in runs:
        groups = defaultdict(lambda: [0, 0])
        for m in r.mentions:
            for k in key(m):
                groups[k][0] += 1
                groups[k][1] += m["status"] not in HIDDEN
        for k, (n, bad) in groups.items():
            cells[(r.label, k)] = (bad / n, bad, n)
            cols[k] += n
    return [r.label for r in runs], [c for c, _ in cols.most_common()], cells


def decomposition(runs: list[RunData]) -> list[dict]:
    """Telescoping failure-rate decomposition per (candidate, dataset), on mentions common to all modes.

    The failure rate counts leaks and required replacements that were not made, so a tool that
    hides everything but replaces nothing shows its failures in the stage that should replace."""
    groups = defaultdict(dict)
    for r in runs:
        groups[(r.candidate, r.dataset)].setdefault(r.mode, r)
    out = []
    for (cand, ds), by_mode in sorted(groups.items()):
        modes = [m for m in MODE_ORDER if m in by_mode]
        if len(modes) < 2 or modes[0] != "e2e":
            continue
        keys = set.intersection(*({(m["doc_id"], m["mention_id"]) for m in by_mode[md].mentions} for md in modes))
        if not keys:
            continue
        rates = {}
        for md in modes:
            ms = [m for m in by_mode[md].mentions if (m["doc_id"], m["mention_id"]) in keys]
            rates[md] = by_mode[md].fail(ms)
        parts = []
        for a, b in zip(modes, modes[1:]):
            i, j = MODE_ORDER.index(a), MODE_ORDER.index(b)
            label = " + ".join(STAGE_BETWEEN[(MODE_ORDER[k], MODE_ORDER[k + 1])] for k in range(i, j))
            parts.append((label, rates[a] - rates[b]))
        last = modes[-1]
        tail = "application" if last == "oracle_regions" else \
            " + ".join(STAGES[MODE_ORDER.index(last):])
        parts.append((tail, rates[last]))
        out.append({"candidate": cand, "dataset": ds, "n": len(keys), "rates": rates, "parts": parts,
                    "negative": any(v < -1e-9 for _, v in parts)})
    return out


def ladders(runs: list[RunData]) -> dict[str, dict[str, list[tuple[float, float, int]]]]:
    """factor -> candidate -> [(value, leak rate, n)], from runs on datasets with derived_from."""
    out: dict = defaultdict(dict)
    manifests = {}
    for r in runs:
        if r.mode != "e2e":
            continue
        name, ver = r.dataset.split("@")
        if (name, ver) not in manifests:
            try:
                manifests[(name, ver)] = load_manifest(name, ver)
            except FileNotFoundError:
                manifests[(name, ver)] = None
        m = manifests[(name, ver)]
        if m is None:
            continue
        derived = {d.doc_id: d.derived_from for d in m.documents if d.derived_from}
        if not derived:
            continue
        steps = defaultdict(lambda: [0, 0])
        factor = None
        for mm in r.mentions:
            df = derived.get(mm["doc_id"])
            if df is None:
                continue
            factor = df.factor
            steps[df.value][0] += 1
            steps[df.value][1] += mm["status"] not in HIDDEN
        if factor:
            out[factor][r.candidate] = [(v, bad / n, n) for v, (n, bad) in steps.items()]
    return out


# -- rendering -------------------------------------------------------------------------------------


def _esc(s) -> str:
    return html.escape(str(s))


def _heat_table(title: str, rows, cols, cells, note: str = "") -> str:
    if not cols:
        return ""
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            if (r, c) not in cells:
                tds.append('<td class="na">–</td>')
                continue
            rate, bad, n = cells[(r, c)]
            step = min(len(RAMP) - 1, int(round(rate * (len(RAMP) - 1))))
            cls = "h z" if bad == 0 else f"h s{step}"  # zero is "nothing": neutral, so leaks stand out
            tds.append(f'<td class="{cls}" title="{_esc(r)} · {_esc(c)}: {bad}/{n} leaking ({rate*100:.1f}%)">'
                       f'{rate*100:.0f}%<small>n={n}</small></td>')
        body.append(f"<tr><th>{_esc(r)}</th>{''.join(tds)}</tr>")
    return (f"<h3>{_esc(title)}</h3>{f'<p class=mut>{note}</p>' if note else ''}<div class=wrap><table class=heat>"
            f"<tr><th></th>{head}</tr>{''.join(body)}</table></div>")


def _decomp_svg(d: dict) -> str:
    w, h = 640, 30
    total = d["rates"]["e2e"] or 0
    x, segs, labels = 0.0, [], []
    for i, (label, v) in enumerate(d["parts"]):
        if v <= 0 or total <= 0:
            continue
        width = v / max(total, 1e-9) * (w - 2 * len(d["parts"]))
        slot = STAGES.index(label) if label in STAGES else min(i, 3)
        segs.append(f'<rect class="c{slot}" x="{x:.1f}" y="0" width="{max(width, 1):.1f}" height="{h}" rx="4">'
                    f'<title>{_esc(label)}: {v*100:.1f} points of the {total*100:.1f}% end-to-end leak rate</title></rect>')
        if width >= 70:
            labels.append(f'<text x="{x + 6:.1f}" y="{h / 2 + 4}" class="seglabel">{_esc(label)} {v*100:.1f}</text>')
        x += width + 2  # 2px surface gap between fills
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="decomposition of '
            f'{_esc(d["candidate"])}">{"".join(segs)}{"".join(labels)}</svg>')


def _ladder_svg(factor: str, series: dict[str, list], colour_of: dict[str, int]) -> str:
    w, h, ml, mr, mt, mb = 700, 240, 46, 220, 14, 34
    values = sorted({v for pts in series.values() for v, _, _ in pts})
    if len(values) < 2:
        return ""
    xs = {v: ml + i * (w - ml - mr) / (len(values) - 1) for i, v in enumerate(values)}

    def y(rate):
        return mt + (1 - rate) * (h - mt - mb)

    grid = "".join(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{int(t*100)}%</text>'
                   for t in (0, 0.25, 0.5, 0.75, 1.0))
    xt = "".join(f'<text class="tick" x="{xs[v]:.1f}" y="{h - mb + 16}" text-anchor="middle">{v:g}</text>'
                 for v in values)
    lines = []
    for cand, pts in sorted(series.items(), key=lambda kv: colour_of[kv[0]]):
        pts = sorted(pts)
        c = colour_of[cand]
        path = " ".join(f"{'M' if i == 0 else 'L'}{xs[v]:.1f},{y(r):.1f}" for i, (v, r, _) in enumerate(pts))
        marks = "".join(f'<circle class="c{c} mk" cx="{xs[v]:.1f}" cy="{y(r):.1f}" r="4.5"><title>{_esc(cand)} · '
                        f'{factor} {v:g}: {r*100:.1f}% leaking (n={n})</title></circle>' for v, r, n in pts)
        lv, lr, _ = pts[-1]
        lines.append(f'<path class="ln s{c}" d="{path}"/>{marks}'
                     f'<text class="dl" x="{xs[lv] + 10:.1f}" y="{y(lr) + 4:.1f}">{_esc(cand)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" '
            f'aria-label="leak rate against {factor}">{grid}<line class="axis" x1="{ml}" x2="{w - mr}" '
            f'y1="{y(0):.1f}" y2="{y(0):.1f}"/>{xt}<text class="tick" x="{(ml + w - mr) / 2:.1f}" y="{h - 4}" '
            f'text-anchor="middle">{_esc(factor)}</text>{"".join(lines)}</svg>')


def _theme_rules(scope: str, dark: bool) -> str:
    """Colour rules for one theme under a selector prefix (empty for the light default)."""
    ramp = list(reversed(RAMP)) if dark else RAMP  # near-zero recedes toward the surface in both themes
    series = SERIES_DARK if dark else SERIES_LIGHT
    tokens = ("--bg:#0d0d0d;--card:#1a1a19;--fg:#ffffff;--mut:#c3c2b7;--line:#2c2c2a;--axis:#383835" if dark else
              "--bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e1e0d9;--axis:#c3c2b7")
    root = scope or ":root"
    out = [f"{root}{{{tokens}}}"]
    pre = f"{scope} " if scope else ""
    for i, v in enumerate(ramp):
        dark_ink = (i < 7) if not dark else (i >= 6)
        out.append(f"{pre}.h.s{i}{{background:{v};color:{'#0b0b0b' if dark_ink else '#ffffff'}}}")
    for i, v in enumerate(series):
        out.append(f"{pre}.c{i}{{fill:{v}}}{pre}.ln.s{i}{{stroke:{v}}}{pre}.k{i}{{background:{v}}}")
    return "".join(out)


def _css() -> str:
    return (_theme_rules("", False)
            + "@media (prefers-color-scheme:dark){" + _theme_rules(':root:not([data-theme="light"])', True) + "}"
            + _theme_rules(':root[data-theme="dark"]', True) + """
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 auto;
max-width:1180px;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:30px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:14px;margin:18px 0 6px}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}th{font-weight:600}
.wrap{overflow-x:auto}table.heat td{text-align:center;min-width:64px;border:2px solid var(--card)}
table.heat small{display:block;font-size:10px;opacity:.8}td.na{color:var(--mut)}td.h.z{background:var(--card);color:var(--mut)}
.card svg{display:block;margin:8px 0}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 8px;color:var(--mut)}.legend span{display:inline-flex;
align-items:center;gap:6px}.legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
svg .grid{stroke:var(--line);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}
svg .tick{fill:var(--mut);font-size:11px}svg .ln{fill:none;stroke-width:2}svg .mk{stroke:var(--card);stroke-width:2}
svg .dl{fill:var(--fg);font-size:12px}svg .seglabel{fill:#0b0b0b;font-size:11px;font-weight:600}
""")


def build_board(run_ids: list[str], out: Path, title: str = "Candidate comparison") -> Path:
    runs = [RunData(r) for r in run_ids]
    cands = sorted({r.candidate for r in runs})
    colour_of = {c: i % len(SERIES_LIGHT) for i, c in enumerate(cands)}  # colour follows the candidate
    e2e = [r for r in runs if r.mode == "e2e" and not r.dataset.startswith("ladder-")]
    parts = [f"<h1>{_esc(title)}</h1><p class=mut>{len(runs)} scored run(s) · candidates: "
             f"{', '.join(_esc(c) for c in cands)}. No composite score: each column answers a different question. "
             f"Leak = required mention still recoverable (visible, partial, recoverable from the file, latent, or "
             f"no output). Not replaced = hidden although a replacement was required. Failure = either. "
             f"Intervals: 95% document-cluster bootstrap.</p>"]

    rows = scorecard(runs)
    parts.append("<h2>Scorecard</h2><div class=wrap><table><tr><th>candidate</th><th>experiment</th><th>dataset</th>"
                 "<th>docs</th><th>required</th><th>leak (strict)</th><th>95% CI</th><th>leak (completed)</th>"
                 "<th>visible/recoverable</th><th>not replaced</th><th>failure</th><th>clean docs</th>"
                 "<th>collateral</th><th>protected dmg</th><th>searchable</th>"
                 "<th>token P / R</th><th>flip rate</th><th>same fake across matter</th>"
                 "<th>tool's own safety check</th></tr>" + "".join(
                     f"<tr><td>{_esc(x['candidate'])} <span class=mut>{_esc(x['version'])}</span></td>"
                     f"<td>{_esc(x['experiment'])}</td><td>{_esc(x['dataset'])}</td><td>{x['docs']}</td><td>{x['n']}</td>"
                     f"<td>{_pct(x['leak'])}</td><td>{_ci(x['ci'])}</td>"
                     f"<td>{_pct(x['leak_completed'])}</td><td>{_pct(x['l12'])}</td>"
                     f"<td>{_pct(x['not_replaced'])}</td><td>{_pct(x['fail'])}</td><td>{x['clean']}</td>"
                     f"<td>{_pct(x['collateral'])}</td><td>{x['protected']}</td><td>{_pct(x['searchable'])}</td>"
                     f"<td>{_pct(x['precision'])} / {_pct(x['recall'])}</td><td>{_pct(x['flip'])}</td>"
                     f"<td>{x['matter'][0]}/{x['matter'][1] if x['matter'][1] else 'N/A'}</td>"
                     f"<td>{_esc(_verdicts(x['self_check']))}</td></tr>"
                     for x in rows) + "</table></div>")

    if e2e:
        parts.append("<h2>Where each candidate leaks</h2><p class=mut>Share of required mentions leaking, end-to-end "
                     "runs only. Stronger colour = more leakage; neutral cells have no leaks. Hover a cell for counts.</p>")
        parts.append(_heat_table("By PII type", *heat(e2e, lambda m: [m["type"]])))
        parts.append(_heat_table("By document condition",
                                 *heat(e2e, lambda m: [c for c in (m.get("conditions") or "").split(",") if c]
                                       or ["(none)"])))
        parts.append(_heat_table("By page encoding", *heat(e2e, lambda m: (m.get("encodings") or "?").split(","))))
        mix_rows, mix_cols, mix = [], Counter(), {}
        for r in e2e:
            fails = [m for m in r.mentions if m["status"] not in PASS]
            cats = Counter(m.get("attr_category") or "?" for m in fails)
            mix_rows.append(r.label)
            for k, v in cats.items():
                mix[(r.label, k)] = (v / len(fails), v, len(fails))
                mix_cols[k] += v
        parts.append(_heat_table("Failure root causes (share of each run's failures)", mix_rows,
                                 [c for c, _ in mix_cols.most_common()], mix,
                                 "Confirmed and inferred attributions are pooled here; see each run report to "
                                 "separate them."))

    dec = decomposition(runs)
    parts.append("<h2>Where failures originate</h2>")
    if not dec:
        parts.append("<p class=mut>No candidate has an end-to-end run and oracle runs on the same dataset in this "
                     "board. Run <code>experiments/oracle-*.yaml</code> to enable the decomposition.</p>")
    else:
        legend = "".join(f'<span><i class="k{i}"></i>{_esc(s)}</span>' for i, s in enumerate(STAGES))
        parts.append(f"<p class=mut>Telescoping differences between runs with perfect upstream input injected: "
                     f"text = e2e − oracle_text, detection = oracle_text − oracle_mentions, resolution = "
                     f"oracle_mentions − oracle_entities, localization = oracle_entities − oracle_regions, "
                     f"application (incl. replacement) = oracle_regions. Points of failure rate (leaks plus "
                     f"required replacements not made); a missing oracle run merges its neighbours; a negative "
                     f"part means the stages interact.</p><div class=legend>{legend}</div>")
        for d in dec:
            parts.append(f"<div class=card><b>{_esc(d['candidate'])}</b> <span class=mut>{_esc(d['dataset'])} · "
                         f"{d['n']} required mentions · end-to-end failure {_pct(d['rates']['e2e'])}</span>"
                         f"{_decomp_svg(d)}<table><tr>" + "".join(f"<th>{_esc(k)}</th>" for k, _ in d["parts"])
                         + "</tr><tr>" + "".join(f"<td>{v*100:+.1f} pts</td>" for _, v in d["parts"]) + "</tr></table>"
                         + ("<p class=mut>Negative part: a stage does better with its own upstream output than with "
                            "perfect input (interaction).</p>" if d["negative"] else "") + "</div>")

    lad = ladders(runs)
    parts.append("<h2>Robustness to document conditions</h2>")
    if not lad:
        parts.append("<p class=mut>No ladder runs in this board. Register one with <code>rb synth ladder</code> and "
                     "run candidates on it.</p>")
    for factor, series in sorted(lad.items()):
        legend = "".join(f'<span><i class="k{colour_of[c]}"></i>{_esc(c)}</span>' for c in sorted(series))
        table = "".join(f"<tr><td>{_esc(c)}</td>" + "".join(f"<td>{v:g}: {r*100:.1f}% (n={n})</td>"
                                                            for v, r, n in sorted(pts)) + "</tr>"
                        for c, pts in sorted(series.items()))
        parts.append(f"<div class=card><b>Leak rate vs {_esc(factor)}</b><div class=legend>{legend}</div>"
                     f"{_ladder_svg(factor, series, colour_of)}<details><summary class=mut>table</summary>"
                     f"<table>{table}</table></details></div>")

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Redaction benchmark board</title>'
            f"<style>{_css()}</style></head><body>{''.join(parts)}"
            f"<p class=mut>Runs: {', '.join(_esc(r) for r in run_ids)}</p></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({"runs": run_ids, "scorecard": rows, "decomposition": dec,
                                                      "ladders": lad}, indent=1, default=str), encoding="utf-8")
    return out
