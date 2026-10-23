"""Score a recorded run against GT and write the results store (PLAN.md §12, §14.3)."""

from __future__ import annotations

import hashlib
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
from .artifacts import load_artifacts, sha256_file
from .runner import ROOT, load_run, runs_root


def results_root() -> Path:
    return Path(os.environ.get("RB_RESULTS_DIR", ROOT / "results"))


def _no_output_reason(item, out_pdf: Path) -> str | None:
    """Why an item has no output to grade: none was written, or output.pdf appeared or changed after
    the runner recorded the item (for example a tool still writing after its timeout)."""
    if not out_pdf.exists():
        return item.error or "no output.pdf"
    if sha256_file(out_pdf) != item.artifacts.get("output_pdf"):
        late = "output.pdf was written or changed after the run recorded it"
        return f"{item.error}; {late}" if item.error else late
    return None


def _oracle_status(run_id: str) -> dict[str, dict[str, str]]:
    """doc id -> {mention id: status} from repeat 0 of an already-scored oracle run."""
    out = {}
    for p in (results_root() / run_id / "scores").glob("*-r0.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["doc_id"]] = {m["mention_id"]: m["status"] for m in d["evaluation"]["mentions"]}
    return out


def _release(workdir: Path, recorded_sha256: str | None = None):
    """The tool's own release verdict from run_info.json, if it reported one when the run recorded it."""
    from ..schemas import RunInfo

    path = workdir / "run_info.json"
    if not path.exists() or (recorded_sha256 is not None and sha256_file(path) != recorded_sha256):
        return None
    try:
        return RunInfo.model_validate(json.loads(path.read_text(encoding="utf-8"))).release
    except ValueError:
        return None  # an invalid run_info is already reported by load_artifacts


def self_check_verdict(released: bool, clean: bool) -> str:
    """Grade the tool's safety check against what the evaluator found in the output."""
    if released:
        return "correct_release" if clean else "unsafe_release"
    return "false_alarm" if clean else "correct_hold"


def matter_consistency(reps: dict[tuple[str, str, str], dict[str, set[str]]], ident: dict) -> list[dict]:
    """Client Q8: a person keeps one fake across all documents of a matter. One row per
    (matter, entity, name component) replaced in at least two documents."""
    rows = []
    for (matter, entity, component), by_doc in sorted(reps.items()):
        if len(by_doc) < 2:
            continue
        values = set().union(*by_doc.values())
        rows.append(dict(ident, matter=matter, entity=entity, component=component, documents=",".join(sorted(by_doc)),
                         distinct_replacements=len(values), consistent=len(values) == 1))
    return rows


def _gt_info(gt) -> dict:
    from ..gt.io import dump_json

    return {"gt_version": gt.gt_version, "gt_sha256": hashlib.sha256(dump_json(gt).encode("utf-8")).hexdigest()}


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
    gt_used: dict[str, dict] = {}
    matter_mismatch = []  # the tool is given the dataset's matter; the matter check groups by the GT's
    # (matter, entity, component) -> doc id -> replacement values, from repeat 0
    matter_reps: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    by_doc: dict[str, list] = defaultdict(list)
    for item in run.items:
        by_doc[item.doc_id].append(item)
    for doc_id, items in by_doc.items():
        entry = ds.document(doc_id)
        gt = find_gt(doc_id, entry.sha256)
        if gt is None:
            skipped.append({"doc_id": doc_id, "reason": "no GT"})
            continue
        gt_used[doc_id] = _gt_info(gt)
        if (entry.matter or None) != (gt.matter or None):  # also when only one of them has a matter
            matter_mismatch.append({"doc_id": doc_id, "dataset": entry.matter, "gt": gt.matter})
        source = (runs_root() / run_id / items[0].workdir / "input.pdf").read_bytes()
        per_rep = []
        for item in sorted(items, key=lambda i: i.repeat):
            workdir = runs_root() / run_id / item.workdir / "out"
            out_pdf = workdir / "output.pdf"
            reason = _no_output_reason(item, out_pdf)
            if reason:
                skipped.append({"doc_id": doc_id, "repeat": item.repeat, "reason": reason})
                no_out = _no_output_rows(ident, doc_id, item.repeat, gt, policy, reason)
                no_out["documents"][0].update(gt_used[doc_id])
                for name, rows in no_out.items():
                    tables[name].extend(rows)
                continue
            artifacts, hashes, errors = load_artifacts(workdir, manifest)
            artifacts.pop("output_pdf", None)
            for kind, digest in hashes.items():  # graded as recorded by the run, never as changed later
                if kind in artifacts and digest != item.artifacts.get(kind):
                    artifacts.pop(kind)
                    errors[kind] = f"{kind} was written or changed after the run recorded it"
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
            if item.repeat == 0 and gt.matter and score.pseudonyms is not None and score.pseudonyms.evaluable:
                for r in score.pseudonyms.replacements:
                    if r.entity and r.replacement and r.component != "title":
                        matter_reps[(gt.matter, r.entity, r.component)][doc_id].add(
                            r.replacement.strip(".,;:()").casefold())
            d = score.to_dict()
            d["invalid_artifacts"] = errors
            (out_root / "scores" / f"{doc_id}-r{item.repeat}.json").write_text(
                json.dumps(d, indent=1, default=str), encoding="utf-8")
            rows_by_table = rows_for_score(ident, {"doc_id": doc_id, "repeat": item.repeat}, score, entry)
            doc_row = rows_by_table["documents"][0]
            doc_row.update(gt_used[doc_id])
            if item.error:
                doc_row["candidate_error"] = item.error
            release = _release(runs_root() / run_id / item.workdir / "out", item.artifacts.get("run_info", ""))
            if release is not None:
                doc_row["self_check.released"] = release.released
                doc_row["self_check.checks_failed"] = ",".join(release.checks_failed)
                doc_row["self_check.verdict"] = self_check_verdict(release.released,
                                                                   bool(doc_row.get("clean_document")))
            for name, rows in rows_by_table.items():
                tables[name].extend(rows)
    tables["matters"] = matter_consistency(matter_reps, ident)
    write_tables(out_root / "tables", tables)
    consistent = sum(1 for r in tables["matters"] if r["consistent"])
    (out_root / "scoring.json").write_text(json.dumps({
        "run_id": run_id, "eval_config": cfg.model_dump(mode="json"), "eval_digest": cfg.digest(),
        "policy": str(policy_path), "oracle_runs": oracle_runs or {}, "gt": gt_used,
        "matter_consistency": {"compared": len(tables["matters"]), "consistent": consistent},
        "matter_mismatch": matter_mismatch,
        "skipped": skipped}, indent=1),
        encoding="utf-8")
    return out_root
