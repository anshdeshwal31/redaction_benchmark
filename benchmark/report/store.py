"""Results store: tidy long tables in Parquet, queryable with DuckDB (PLAN.md §12).

Any new question is a new group-by over these tables, not new code. Tables contain GT
ids and verdicts; document-level findings may contain matched PII terms, so the results
directory is sensitive by default (git-ignored).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ("documents", "mentions", "words", "components", "determinism", "findings")


def _flat(prefix: str, d: dict | None, out: dict) -> None:
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flat(f"{key}.", v, out)
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        else:
            out[key] = v


def rows_for_score(run: dict, item: dict, score, doc_entry=None) -> dict[str, list[dict]]:
    """Rows for one scored (document, repeat). ``run``/``item`` carry identifying fields."""
    base = {"run_id": run["run_id"], "candidate": run["candidate"], "candidate_version": run["candidate_version"],
            "experiment": run["experiment"], "mode": run["mode"], "dataset": run["dataset"],
            "doc_id": item["doc_id"], "repeat": item["repeat"]}
    ev = score.evaluation
    page_enc = {}
    if doc_entry is not None:
        page_enc = {c.index: c.encoding.value for c in doc_entry.page_conditions}
    doc = dict(base)
    _flat("", ev.summary, doc)
    doc.update({"integrity_ok": ev.integrity.ok, "revisions_checked": ev.revisions_checked,
                "eval_version": ev.eval_version, "eval_digest": ev.eval_digest, "policy": ev.policy,
                "output_sha256": ev.output_sha256, "stages_exposed": ",".join(score.stages_exposed)})
    if score.attribution is not None:
        _flat("funnel.", score.attribution.funnel, doc)
        _flat("over.", score.attribution.counts.get("over_redaction", {}), doc)
    det = {x.mention_id: x for x in score.detection.matches} if score.detection else {}
    cov = {c.mention_id: c.coverage for c in score.localization.mentions} if score.localization else {}
    att = {a.mention_id: a for a in score.attribution.mentions} if score.attribution else {}
    mentions = []
    for m in ev.mentions:
        r = dict(base)
        r.update({k: getattr(m, k) for k in ("mention_id", "type", "subtype", "entity", "role", "requirement",
                                              "pending", "cells", "visible_residual", "worst_config",
                                              "latent_residual", "text_residual", "image_residual",
                                              "image_latent_residual", "vector_residual", "revision_residual",
                                              "replaced")})
        r["status"] = m.status.value
        r["conditions"] = ",".join(m.conditions)
        r["l2_sources"] = ",".join(m.l2_sources)
        r["pages"] = ",".join(map(str, m.pages))
        r["encodings"] = ",".join(sorted({page_enc.get(p, "?") for p in m.pages}))
        x = det.get(m.mention_id)
        r["det_boundary"] = x.boundary if x else None
        r["det_type_correct"] = x.type_correct if x else None
        r["det_covered_fraction"] = x.covered_fraction if x else None
        r["plan_coverage"] = cov.get(m.mention_id)
        a = att.get(m.mention_id)
        r["attr_stage"] = a.stage if a else None
        r["attr_category"] = a.category if a else None
        r["attr_confidence"] = a.confidence if a else None
        r["attr_detail"] = a.detail if a else None
        r["attr_tags"] = ",".join(a.tags) if a else None
        mentions.append(r)
    words = [dict(base, word_id=w.word_id, page=w.page, requirement=w.requirement, status=w.status.value,
                  visible_fraction=w.visible_fraction) for w in ev.words]
    comp = dict(base)
    if score.text:
        _flat("text.general.", score.text.general, comp)
        _flat("text.pii.", score.text.pii, comp)
        comp["text.recognisability"] = score.text.mention_recognisability["rate"]
        _flat("text.order.", score.text.reading_order, comp)
    if score.detection:
        _flat("det.token.", score.detection.token, comp)
        _flat("det.mention.", score.detection.mention_scores, comp)
        _flat("det.boundary.", score.detection.boundary_counts, comp)
        comp["det.entities_complete"] = score.detection.entities_complete["complete"]
        comp["det.entities_total"] = score.detection.entities_complete["total"]
    if score.resolution:
        comp["res.source"] = score.resolution.source
        for k in ("muc", "b_cubed", "ceaf_e", "lea", "pairwise"):
            comp[f"res.{k}.f1"] = getattr(score.resolution, k)["f1"]
        comp["res.false_merges"] = len(score.resolution.false_merges)
        comp["res.false_splits"] = len(score.resolution.false_splits)
    if score.localization:
        comp["loc.full_coverage_rate"] = score.localization.must_full_coverage_rate
        comp["loc.mean_coverage"] = score.localization.must_mean_coverage
        comp["loc.spill_area_pt2"] = score.localization.spill_area_pt2
        comp["loc.diagnosis"] = score.localization.document_diagnosis
    if score.fidelity:
        comp["fid.not_applied"] = score.fidelity.not_applied
        comp["fid.partial"] = score.fidelity.partial
    if score.pseudonyms:
        comp["pseud.evaluable"] = score.pseudonyms.evaluable
        _flat("pseud.", score.pseudonyms.checks, comp)
    findings = [dict(base, location=f.location, term=f.term, mention_ids=",".join(f.mention_ids),
                     entity_ids=",".join(f.entity_ids)) for f in ev.findings]
    return {"documents": [doc], "mentions": mentions, "words": words, "components": [comp], "findings": findings}


def write_tables(out_dir: Path, tables: dict[str, list[dict]]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in TABLES:
        rows = tables.get(name, [])
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        arrays = {}
        for k in keys:
            vals = [r.get(k) for r in rows]
            kinds = {type(v) for v in vals if v is not None}
            if len(kinds) > 1 and not kinds <= {int, float}:
                vals = [None if v is None else str(v) for v in vals]  # mixed column: store as text
            elif kinds == {int, float}:
                vals = [None if v is None else float(v) for v in vals]
            arrays[k] = pa.array(vals)
        table = pa.table(arrays)
        path = out_dir / f"{name}.parquet"
        pq.write_table(table, path)
        paths[name] = path
    return paths


def query(results_dir: Path, sql: str):
    """Run SQL over a run's tables (each table is available by name)."""
    import duckdb

    con = duckdb.connect()
    for name in TABLES:
        p = results_dir / "tables" / f"{name}.parquet"
        if p.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")
    return con.execute(sql).fetchall(), [d[0] for d in con.description]
