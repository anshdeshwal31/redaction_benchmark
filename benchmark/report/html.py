"""Static HTML run report (PLAN.md §12). Offline, no external assets.

Reports are sensitive by default: the failure gallery shows crops of the original and
the output. ``shareable=True`` paints GT spans over both crops (geometry and type only).
"""

from __future__ import annotations

import base64
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pymupdf
from jinja2 import Template

from ..dataset import document_pdf, load_manifest
from ..geometry import as_poly, bbox
from ..gt.io import find_gt
from ..runner.runner import load_run, runs_root

SAFE = {"SAFE_REMOVED", "SAFE_REPLACED"}


def _rows(results: Path, name: str) -> list[dict]:
    p = results / "tables" / f"{name}.parquet"
    return pq.read_table(p).to_pylist() if p.exists() else []


def _bootstrap(per_doc: list[tuple[int, int]], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% document-cluster bootstrap CI of a pooled rate from (numerator, denominator) per document."""
    if len(per_doc) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(per_doc, dtype=float)
    vals = []
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))]
        if s[:, 1].sum() > 0:
            vals.append(s[:, 0].sum() / s[:, 1].sum())
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else None


def _rate(num, den):
    return None if not den else num / den


def _fmt(v, pct=False):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}"
    return html.escape(str(v))


def _crop(pdf: bytes, page: int, box, dpi: float, polys, mask_polys: bool, label: str) -> str:
    doc = pymupdf.open("pdf", pdf)
    if page >= doc.page_count:
        return ""
    s = dpi / 72.0
    pg = doc[page]
    x0, y0, x1, y1 = box
    clip = pymupdf.Rect(max(0, x0), max(0, y0), min(pg.rect.width, x1), min(pg.rect.height, y1))
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(s, s), clip=clip, colorspace=pymupdf.csRGB)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3).copy()
    for p in polys:
        pts = np.round((as_poly(p) - [clip.x0, clip.y0]) * s).astype(np.int32)
        if mask_polys:
            cv2.fillPoly(img, [pts], (150, 150, 150))
            cv2.putText(img, label, tuple(pts.min(axis=0) + [2, 12]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        else:
            cv2.polylines(img, [pts], True, (220, 30, 30), 1)
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else ""


TEMPLATE = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Redaction benchmark run</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#ddddd8;--bad:#b42318;--ok:#067647;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#a3a39e;--line:#33332f;--bad:#f97066;--ok:#47cd89;--card:#1e1e1c}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1180px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}th{font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:20px}.bad{color:var(--bad)}.ok{color:var(--ok)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}
.gal img{max-width:100%;border:1px solid var(--line)}.pair{display:flex;gap:6px}.pair div{flex:1}
code{font-size:12px}.wrap{overflow-x:auto}
</style></head><body>
<h1>Run {{ run.run_id }}</h1>
<div class="mut">candidate <b>{{ run.candidate.name }}</b> {{ run.candidate.version }} · experiment {{ run.experiment.name }} ({{ run.experiment.mode }}, {{ run.experiment.repeats }} repeat(s)) · dataset {{ run.dataset.name }}@{{ run.dataset.version }} · eval {{ scoring.eval_config.version }}/{{ scoring.eval_digest }} · benchmark {{ run.benchmark.version }} {{ (run.benchmark.commit or 'uncommitted')[:10] }}{% if run.benchmark.dirty %} (dirty){% endif %}</div>
{% if shareable %}<p class="mut">Shareable mode: GT spans are masked in all crops.</p>{% else %}<p class="bad">Sensitive: contains crops of real documents. Do not circulate; use <code>--shareable</code>.</p>{% endif %}
{% if pending %}<p class="mut">Policy decisions pending (scored as MAY, neither rewarded nor penalised): {% for q, n in pending %}{{ q }} ({{ n }} mentions){% if not loop.last %}, {% endif %}{% endfor %}.</p>{% endif %}
{% if skipped %}<p class="mut">Not scored: {% for s in skipped %}{{ s.doc_id }}{% if s.repeat is defined %}/r{{ s.repeat }}{% endif %} ({{ s.reason }}){% if not loop.last %}, {% endif %}{% endfor %}</p>{% endif %}

<h2>Scorecard</h2>
<div class="grid">{% for c in cards %}<div class="card"><span class="mut">{{ c.label }}</span><b class="{{ c.cls }}">{{ c.value }}</b><span class="mut">{{ c.note }}</span></div>{% endfor %}</div>

<h2>Components</h2><div class="wrap"><table><tr><th>stage</th><th>metric</th><th>value</th></tr>
{% for row in components %}<tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td></tr>{% endfor %}</table></div>

<h2>Documents</h2><div class="wrap"><table><tr><th>doc</th><th>rep</th><th>required</th><th>safe</th><th>partial</th><th>visible</th><th>recoverable</th><th>latent</th><th>collateral</th><th>protected dmg</th><th>findings</th><th>clean</th><th>integrity</th></tr>
{% for d in docs %}<tr><td>{{ d.doc_id }}</td><td>{{ d.repeat }}</td><td>{{ d.must }}</td><td>{{ d.safe }}</td><td>{{ d.partial }}</td><td>{{ d.visible }}</td><td>{{ d.recoverable }}</td><td>{{ d.latent }}</td><td>{{ d.collateral }}</td><td>{{ d.protected }}</td><td>{{ d.findings }}</td><td class="{{ 'ok' if d.clean else 'bad' }}">{{ d.clean }}</td><td>{{ d.integrity }}</td></tr>{% endfor %}</table></div>

{% for title, rows in breakdowns %}<h2>{{ title }}</h2><table><tr><th>slice</th><th>required mentions</th><th>leaking</th><th>leak rate</th></tr>
{% for r in rows %}<tr><td>{{ r[0] }}</td><td>{{ r[1] }}</td><td>{{ r[2] }}</td><td>{{ r[3] }}</td></tr>{% endfor %}</table>{% endfor %}

<h2>Failure funnel and attribution</h2>
<table><tr>{% for k in funnel %}<th>{{ k }}</th>{% endfor %}</tr><tr>{% for k, v in funnel.items() %}<td>{{ v }}</td>{% endfor %}</tr></table>
<table><tr><th>stage</th><th>category</th><th>confidence</th><th>mentions</th></tr>{% for r in attribution %}<tr><td>{{ r[0] }}</td><td>{{ r[1] }}</td><td>{{ r[2] }}</td><td>{{ r[3] }}</td></tr>{% endfor %}</table>
{% if over %}<p>Over-redaction: {% for k, v in over %}{{ k }} {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}</p>{% endif %}

{% if determinism %}<h2>Determinism</h2><table><tr><th>doc</th><th>runs</th><th>D0 bytes</th><th>D1 structural</th><th>D2 visual</th><th>mask IoU</th><th>D3 decisions</th><th>flip rate</th><th>unstable</th></tr>
{% for d in determinism %}<tr><td>{{ d.doc_id }}</td><td>{{ d.runs }}</td><td>{{ d.d0_exact }}</td><td><code>{{ d.d1 }}</code></td><td>{{ d.d2_exact }}/{{ d.d2_tolerant }}</td><td>{{ d.d2_min_mask_iou }}</td><td>{{ d.d3_identical }}</td><td>{{ d.flip_rate }}</td><td>{{ d.unstable }}</td></tr>{% endfor %}</table>{% endif %}

<h2>Failure gallery</h2>{% if not gallery %}<p class="mut">No failing required mentions.</p>{% endif %}
<div class="gal">{% for g in gallery %}<div class="card"><div><b style="font-size:14px;display:inline">{{ g.doc_id }} · {{ g.mention_id }}</b> <span class="mut">{{ g.type }} · {{ g.status }} · {{ g.category }} ({{ g.confidence }})</span></div>
<div class="mut">{{ g.detail }}</div><div class="pair"><div><span class="mut">original</span><img src="{{ g.orig }}" alt="original crop"></div><div><span class="mut">output</span><img src="{{ g.out }}" alt="output crop"></div></div></div>{% endfor %}</div>
<p class="mut">Generated by redaction_benchmark. Rates pool required mentions across documents; intervals are 95% document-cluster bootstrap and are wide when few documents are scored.</p>
</body></html>""")


def build_report(run_id: str, results_dir: Path, shareable: bool = False, gallery_limit: int = 24) -> Path:
    run = load_run(run_id)
    scoring = json.loads((results_dir / "scoring.json").read_text(encoding="utf-8"))
    docs_t, mentions, words, comps = (_rows(results_dir, n) for n in ("documents", "mentions", "words", "components"))
    determinism = _rows(results_dir, "determinism")
    must = [m for m in mentions if m["requirement"] == "MUST" and m["repeat"] == 0]
    leaking = [m for m in must if m["status"] not in SAFE and m["status"] != "NOT_EVALUATED"]
    per_doc = defaultdict(lambda: [0, 0])
    for m in must:
        if m["status"] == "NOT_EVALUATED":
            continue
        per_doc[m["doc_id"]][1] += 1
        per_doc[m["doc_id"]][0] += m["status"] not in SAFE
    evaluated = sum(v[1] for v in per_doc.values())
    leak_rate = _rate(len(leaking), evaluated)
    ci = _bootstrap([tuple(v) for v in per_doc.values()])
    l12 = [m for m in leaking if m["status"] in ("PARTIAL", "LEAK_VISIBLE", "LEAK_RECOVERABLE")]
    by_status = Counter(m["status"] for m in must)
    doc0 = [d for d in docs_t if d.get("repeat") == 0 and "must_mentions" in d]
    clean = sum(1 for d in doc0 if d.get("clean_document"))
    ent_total = sum(int(d.get("entities_with_must_mentions") or 0) for d in doc0)
    ent_leaked = sum(len([e for e in str(d.get("entities_leaked") or "").split(",") if e]) for d in doc0)
    w0 = [w for w in words if w["repeat"] == 0 and w["status"] != "NOT_EVALUATED"]
    ordinary = [w for w in w0 if w["requirement"] == "ORDINARY"]
    protected = [w for w in w0 if w["requirement"] == "MUST_NOT"]
    must_w = [w for w in w0 if w["requirement"] == "MUST"]
    red = lambda w: w["visible_fraction"] is not None and w["visible_fraction"] < 0.5  # noqa: E731
    tp = sum(1 for w in must_w if red(w))
    fp = sum(1 for w in ordinary + protected if red(w))
    no_output = [m for m in must if m["status"] == "NO_OUTPUT"]
    completed = [m for m in leaking if m["status"] != "NO_OUTPUT"]
    cards = [
        {"label": "required mentions leaking (strict)", "value": _fmt(leak_rate, True), "cls": "bad" if leaking else "ok",
         "note": f"{len(leaking)}/{evaluated}" + (f" · 95% CI {ci[0]*100:.1f}-{ci[1]*100:.1f}%" if ci else "")
                 + (f" · {len(no_output)} from failed runs" if no_output else "")},
        {"label": "leaking, completed runs only", "value": _fmt(_rate(len(completed), evaluated - len(no_output)), True),
         "cls": "bad" if completed else "ok", "note": f"{len(completed)}/{evaluated - len(no_output)}"},
        {"label": "visible or recoverable (L1+L2)", "value": _fmt(_rate(len(l12), evaluated), True),
         "cls": "bad" if l12 else "ok", "note": f"partial {by_status['PARTIAL']} · visible {by_status['LEAK_VISIBLE']} · recoverable {by_status['LEAK_RECOVERABLE']}"},
        {"label": "latent only (L3)", "value": str(by_status["LEAK_LATENT"]), "cls": "bad" if by_status["LEAK_LATENT"] else "ok", "note": "recoverable by enhancement"},
        {"label": "entities with any leak", "value": _fmt(_rate(ent_leaked, ent_total), True), "cls": "bad" if ent_leaked else "ok", "note": f"{ent_leaked}/{ent_total}"},
        {"label": "clean documents", "value": f"{clean}/{len(doc0)}", "cls": "ok" if clean == len(doc0) else "bad", "note": "no L1/L2 leak, no protected damage"},
        {"label": "collateral ordinary words", "value": _fmt(_rate(sum(1 for w in ordinary if w['status'] != 'PRESERVED'), len(ordinary)), True), "cls": "", "note": f"of {len(ordinary)}"},
        {"label": "protected words damaged", "value": str(sum(1 for w in protected if w['status'] != 'PRESERVED')), "cls": "", "note": f"of {len(protected)}"},
        {"label": "effective token precision / recall", "value": f"{_fmt(_rate(tp, tp + fp))} / {_fmt(_rate(tp, len(must_w)))}", "cls": "", "note": "from the output pixels"},
    ]
    comp_rows = []
    c0 = [c for c in comps if c.get("repeat") == 0]

    def mean(key):
        vals = [c[key] for c in c0 if isinstance(c.get(key), (int, float)) and not isinstance(c.get(key), bool)]
        return sum(vals) / len(vals) if vals else None

    for stage, metric, key in [
        ("text", "CER (raw)", "text.general.cer_raw"), ("text", "PII CER (raw)", "text.pii.cer_raw"),
        ("text", "PII missed rate", "text.pii.missed_rate"), ("text", "mention recognisability", "text.recognisability"),
        ("text", "reading order Kendall tau", "text.order.kendall_tau_mean"),
        ("detection", "token precision", "det.token.precision"), ("detection", "token recall", "det.token.recall"),
        ("detection", "tolerant mention recall", "det.mention.tolerant.recall"),
        ("detection", "strict mention recall", "det.mention.strict.recall"),
        ("resolution", "B3 F1", "res.b_cubed.f1"), ("resolution", "CEAF-e F1", "res.ceaf_e.f1"),
        ("resolution", "pairwise F1", "res.pairwise.f1"),
        ("localization", "required mentions fully covered", "loc.full_coverage_rate"),
        ("localization", "mean coverage", "loc.mean_coverage"),
        ("application", "planned regions not applied", "fid.not_applied"),
        ("pseudonyms", "value consistency", "pseud.value_consistency_rate"),
        ("pseudonyms", "collisions", "pseud.collisions"),
        ("pseudonyms", "honorific/gender mismatches", "pseud.honorific_gender_mismatch"),
    ]:
        comp_rows.append((stage, metric, _fmt(mean(key))))
    diag = Counter(c.get("loc.diagnosis") for c in c0 if c.get("loc.diagnosis"))
    if diag:
        comp_rows.append(("localization", "diagnosis", ", ".join(f"{k} ({v})" for k, v in diag.items())))

    def breakdown(keyfn):
        groups = defaultdict(lambda: [0, 0])
        for m in must:
            if m["status"] == "NOT_EVALUATED":
                continue
            for k in keyfn(m):
                groups[k][0] += 1
                groups[k][1] += m["status"] not in SAFE
        return [(k, n, l, _fmt(_rate(l, n), True)) for k, (n, l) in sorted(groups.items(), key=lambda kv: -kv[1][1])]

    breakdowns = [("By PII type", breakdown(lambda m: [m["type"]])),
                  ("By condition", breakdown(lambda m: [c for c in (m.get("conditions") or "").split(",") if c] or ["(none)"])),
                  ("By page encoding", breakdown(lambda m: (m.get("encodings") or "?").split(",")))]
    funnel = {}
    for d in doc0:
        for k, v in d.items():
            if k.startswith("funnel.") and isinstance(v, (int, float)):
                funnel[k[7:]] = funnel.get(k[7:], 0) + int(v)
    over = Counter()
    for d in doc0:
        for k, v in d.items():
            if k.startswith("over.") and isinstance(v, (int, float)):
                over[k[5:]] += int(v)
    attribution = Counter((m["attr_stage"], m["attr_category"], m["attr_confidence"]) for m in must
                          if m.get("attr_category") and m["attr_category"] != "SAFE")
    pending = Counter(m["pending"] for m in mentions if m["repeat"] == 0 and m.get("pending"))
    docs = []
    for d in doc0:
        dm = [m for m in must if m["doc_id"] == d["doc_id"]]
        st = Counter(m["status"] for m in dm)
        docs.append({"doc_id": d["doc_id"], "repeat": d["repeat"], "must": len(dm),
                     "safe": st["SAFE_REMOVED"] + st["SAFE_REPLACED"], "partial": st["PARTIAL"],
                     "visible": st["LEAK_VISIBLE"], "recoverable": st["LEAK_RECOVERABLE"], "latent": st["LEAK_LATENT"],
                     "collateral": d.get("collateral_words"), "protected": len([x for x in str(d.get("protected_damaged") or "").split(",") if x]),
                     "findings": d.get("document_findings"), "clean": bool(d.get("clean_document")),
                     "integrity": "ok" if d.get("integrity_ok") else "FAILED"})

    gallery = []
    ds = load_manifest(run.dataset["name"], run.dataset["version"])
    gts = {}
    for m in [x for x in leaking if x["status"] != "NO_OUTPUT"][:gallery_limit]:
        doc_id = m["doc_id"]
        entry = ds.document(doc_id)
        if doc_id not in gts:
            gts[doc_id] = find_gt(doc_id, entry.sha256)
        gt = gts[doc_id]
        if gt is None:
            continue
        gm = next(x for x in gt.mentions if x.id == m["mention_id"])
        words = gt.word_index()
        regions = gt.region_index()
        seg = gm.segments[0]
        polys = [words[w][1].poly for w in seg.words] + [regions[r][1].poly for r in seg.regions]
        x0, y0, x1, y1 = bbox(np.vstack([as_poly(p) for p in polys]))
        box = (x0 - 30, y0 - 20, x1 + 30, y1 + 20)
        item = next(i for i in run.items if i.doc_id == doc_id and i.repeat == 0)
        out_pdf = runs_root() / run_id / item.workdir / "out" / "output.pdf"
        src = document_pdf(entry).read_bytes()
        gallery.append({"doc_id": doc_id, "mention_id": m["mention_id"], "type": m["type"], "status": m["status"],
                        "category": m.get("attr_category"), "confidence": m.get("attr_confidence"),
                        "detail": m.get("attr_detail") or "",
                        "orig": _crop(src, seg.page, box, 150, polys, shareable, m["type"]),
                        "out": _crop(out_pdf.read_bytes(), seg.page, box, 150, polys, shareable, m["type"])
                        if out_pdf.exists() else ""})
    page = TEMPLATE.render(run=run.model_dump(mode="json"), scoring=scoring, shareable=shareable, cards=cards,
                           components=comp_rows, docs=docs, breakdowns=breakdowns, funnel=funnel,
                           attribution=[(*k, v) for k, v in attribution.most_common()],
                           over=sorted(over.items()), determinism=determinism, gallery=gallery,
                           pending=sorted(pending.items()), skipped=scoring.get("skipped", []))
    out = results_dir / ("report-shareable.html" if shareable else "report.html")
    out.write_text(page, encoding="utf-8")
    return out
