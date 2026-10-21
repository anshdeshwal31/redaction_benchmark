"""Regression suite: failures promoted to stable, GT-keyed cases (PLAN.md §13).

A case is a fingerprint over (dataset doc, GT mention, failure category, stage). It
holds ids and categories only, never document text, so the case list can be tracked.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "datasets" / "regression" / "cases.yaml"
from ..evaluate import SAFE_VALUES

SAFE = SAFE_VALUES  # a hidden but unreplaced mention is still a failure worth a regression case


def load_cases(path: Path = CASES) -> list[dict]:
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def promote(results: Path, run_id: str, doc_id: str, mention_id: str, note: str = "", path: Path = CASES) -> dict:
    rows = pq.read_table(results / "tables" / "mentions.parquet").to_pylist()
    row = next((r for r in rows if r["doc_id"] == doc_id and r["mention_id"] == mention_id and r["repeat"] == 0), None)
    if row is None:
        raise KeyError(f"{doc_id}/{mention_id} not in run {run_id}")
    if row["status"] in SAFE:
        raise ValueError(f"{doc_id}/{mention_id} is {row['status']} in run {run_id}; nothing to promote")
    fp = hashlib.sha256(f"{row['dataset']}|{doc_id}|{mention_id}|{row.get('attr_category')}|{row.get('attr_stage')}"
                        .encode()).hexdigest()[:16]
    cases = load_cases(path)
    if any(c["fingerprint"] == fp for c in cases):
        return next(c for c in cases if c["fingerprint"] == fp)
    case = {"fingerprint": fp, "dataset": row["dataset"], "doc_id": doc_id, "mention_id": mention_id,
            "type": row["type"], "status_at_promotion": row["status"], "category": row.get("attr_category"),
            "stage": row.get("attr_stage"), "origin_run": run_id, "candidate": row["candidate"],
            "created": _dt.date.today().isoformat(), "note": note}
    cases.append(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cases, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return case


def check(results: Path, path: Path = CASES) -> list[dict]:
    """Status of every regression case covered by a run: pass (now safe), fail, or not covered."""
    rows = pq.read_table(results / "tables" / "mentions.parquet").to_pylist()
    status = {(r["doc_id"], r["mention_id"]): r for r in rows if r["repeat"] == 0}
    out = []
    for c in load_cases(path):
        r = status.get((c["doc_id"], c["mention_id"]))
        if r is None:
            out.append({**c, "result": "not_covered"})
        else:
            out.append({**c, "result": "pass" if r["status"] in SAFE else "fail", "status_now": r["status"],
                        "category_now": r.get("attr_category")})
    return out
