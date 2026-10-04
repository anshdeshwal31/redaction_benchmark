"""Score a recorded run against GT and write the results store (PLAN.md §12, §14.3)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from ..dataset import load_manifest
from ..evaluate import EvalConfig
from ..gt.io import find_gt, load_policy
from ..metrics import InkIndex, compare_runs
from ..report.store import rows_for_score, write_tables
from ..schemas import CandidateManifest, PiiType
from ..score import score_document
from .artifacts import load_artifacts
from .runner import ROOT, load_run, runs_root


def results_root() -> Path:
    return Path(os.environ.get("RB_RESULTS_DIR", ROOT / "results"))


def _oracle_status(run_id: str) -> dict[str, dict[str, str]]:
    """doc id -> {mention id: status} from repeat 0 of an already-scored oracle run."""
    out = {}
    for p in (results_root() / run_id / "scores").glob("*-r0.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["doc_id"]] = {m["mention_id"]: m["status"] for m in d["evaluation"]["mentions"]}
    return out


def _no_output_rows(ident: dict, doc_id: str, repeat: int, gt, policy, reason: str) -> dict[str, list[dict]]:
    """A failed run leaves every required mention unprotected (strict view); reports also
    show the completed-only view, which excludes these rows."""
    from ..gt.policy import resolve

    res = resolve(gt, policy)
    entities = {e.id: e for e in gt.entities}
    must = [m for m in gt.mentions if res.mentions[m.id].requirement.value == "MUST"]
    mentions = [dict(ident, doc_id=doc_id, repeat=repeat, mention_id=m.id, type=m.type.value, subtype=m.subtype,
                     entity=m.entity, role=entities[m.entity].role.value if m.entity in entities else None,
                     requirement="MUST", pending=None, status="NO_OUTPUT", conditions=",".join(m.conditions),
                     pages=",".join(str(sg.page) for sg in m.segments), attr_stage="RUN",
                     attr_category="CANDIDATE_ERROR", attr_confidence="confirmed", attr_detail=reason, attr_tags="")
                for m in must]
    doc = dict(ident, doc_id=doc_id, repeat=repeat, candidate_error=reason, must_mentions=len(must),
               must_evaluated=0, clean_document=False, integrity_ok=False,
               entities_with_must_mentions=len({m.entity for m in must if m.entity}))
    return {"documents": [doc], "mentions": mentions}


def score_run(run_id: str, policy_path: str | Path, config: EvalConfig | None = None,
              oracle_runs: dict[str, str] | None = None) -> Path:
    cfg = config or EvalConfig()
    run = load_run(run_id)
    manifest = CandidateManifest.model_validate(run.candidate["manifest"])
    policy = load_policy(policy_path)
    ds = load_manifest(run.dataset["name"], run.dataset["version"])
    oracle = {mode: _oracle_status(rid) for mode, rid in (oracle_runs or {}).items()}
    out_root = results_root() / run_id
    (out_root / "scores").mkdir(parents=True, exist_ok=True)
    label_map = {k: PiiType(v) for k, v in manifest.label_map.items()} or None
    ident = {"run_id": run_id, "candidate": run.candidate["name"], "candidate_version": run.candidate["version"],
             "experiment": run.experiment.name, "mode": run.experiment.mode,
             "dataset": f"{run.dataset['name']}@{run.dataset['version']}"}
    tables: dict[str, list[dict]] = defaultdict(list)
    skipped = []
    by_doc: dict[str, list] = defaultdict(list)
    for item in run.items:
        by_doc[item.doc_id].append(item)
    for doc_id, items in by_doc.items():
        entry = ds.document(doc_id)
        gt = find_gt(doc_id, entry.sha256)
        if gt is None:
            skipped.append({"doc_id": doc_id, "reason": "no GT"})
            continue
        source = (runs_root() / run_id / items[0].workdir / "input.pdf").read_bytes()
        per_rep = []
        for item in sorted(items, key=lambda i: i.repeat):
            workdir = runs_root() / run_id / item.workdir / "out"
            out_pdf = workdir / "output.pdf"
            if not out_pdf.exists():
                reason = item.error or "no output.pdf"
                skipped.append({"doc_id": doc_id, "repeat": item.repeat, "reason": reason})
                for name, rows in _no_output_rows(ident, doc_id, item.repeat, gt, policy, reason).items():
                    tables[name].extend(rows)
                continue
            artifacts, _, errors = load_artifacts(workdir, manifest)
            artifacts.pop("output_pdf", None)
            per_rep.append((item, out_pdf.read_bytes(), artifacts, errors))
        unstable = None
        if len(per_rep) > 1:
            index = InkIndex.build(gt, source, 100)
            from ..evaluate import evaluate_output

            evs = [evaluate_output(source, out, gt, policy, cfg) for _, out, _, _ in per_rep]
            det = compare_runs(source, [o for _, o, _, _ in per_rep], [a for _, _, a, _ in per_rep], evs, index)
            unstable = set(det.unstable_mentions)
            tables["determinism"].append(dict(ident, doc_id=doc_id, runs=det.runs, d0_exact=det.d0_exact,
                                              d1=json.dumps(det.d1_structural), d2_exact=det.d2_visual_exact,
                                              d2_tolerant=det.d2_visual_tolerant, d2_min_mask_iou=det.d2_min_mask_iou,
                                              d3_identical=det.d3_decisions_identical, flip_rate=det.flip_rate,
                                              unstable=",".join(det.unstable_mentions)))
        for item, out, artifacts, errors in per_rep:
            score = score_document(source, out, gt, policy, artifacts, label_map=label_map, config=cfg,
                                   oracle={m: o.get(doc_id, {}) for m, o in oracle.items()} or None,
                                   unstable=unstable)
            d = score.to_dict()
            d["invalid_artifacts"] = errors
            (out_root / "scores" / f"{doc_id}-r{item.repeat}.json").write_text(
                json.dumps(d, indent=1, default=str), encoding="utf-8")
            rows_by_table = rows_for_score(ident, {"doc_id": doc_id, "repeat": item.repeat}, score, entry)
            if item.error:
                rows_by_table["documents"][0]["candidate_error"] = item.error
            for name, rows in rows_by_table.items():
                tables[name].extend(rows)
    write_tables(out_root / "tables", tables)
    (out_root / "scoring.json").write_text(json.dumps({
        "run_id": run_id, "eval_config": cfg.model_dump(mode="json"), "eval_digest": cfg.digest(),
        "policy": str(policy_path), "oracle_runs": oracle_runs or {}, "skipped": skipped}, indent=1), encoding="utf-8")
    return out_root
