"""CLI commands for running, scoring, reporting, comparing and regression (registered by cli_extra)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = str(ROOT / "policies" / "au-medicolegal-v0.yaml")


def _cmd_gt_draft(args) -> int:
    from .dataset import data_dir, document_pdf, load_manifest
    from .gt.draft import draft_gt
    from .gt.io import write_model

    name, version = args.dataset.split("@")
    m = load_manifest(name, version)
    out = Path(args.out) if args.out else data_dir() / "gt-drafts"
    for d in m.documents:
        if args.docs and d.doc_id not in args.docs:
            continue
        gt, notes = draft_gt(document_pdf(d).read_bytes(), d.doc_id, d.matter)
        write_model(gt, out / f"{d.doc_id}.draft.json")
        words = sum(len(p.words) for p in gt.pages)
        print(f"{d.doc_id}: {words} words, {len(gt.mentions)} pre-labelled mentions -> {out}")
        for n in notes[:3]:
            print(f"   {n}")
        if len(notes) > 3:
            print(f"   ... {len(notes) - 3} more page(s) without words")
    return 0


def _cmd_gt_review(args) -> int:
    from .gt.draft import review_sheets
    from .gt.io import load_gt

    paths = review_sheets(load_gt(args.gt), Path(args.pdf).read_bytes(), Path(args.out))
    print(f"{len(paths)} review sheet(s) -> {args.out}")
    return 0


def _cmd_synth_dataset(args) -> int:
    from .synth.register import register_synthetic

    m = register_synthetic(args.version, args.variants)
    print(f"dataset synthetic@{m.version}: {len(m.documents)} documents; GT in gt/synthetic/")
    return 0


def _cmd_run(args) -> int:
    from .runner.runner import run_candidate, runs_root

    run = run_candidate(args.candidate, args.dataset, args.experiment, args.policy, docs=args.docs, run_id=args.run_id)
    failed = [i for i in run.items if i.error or i.invalid_artifacts]
    print(f"run {run.run_id}: {len(run.items)} item(s), {len(failed)} with errors -> {runs_root() / run.run_id}")
    for i in failed:
        print(f"  {i.doc_id}/r{i.repeat}: {i.error or ''} {i.invalid_artifacts or ''}")
    return 0


def _cmd_score(args) -> int:
    from .evaluate import EvalConfig
    from .runner.scoring import score_run

    oracle = dict(o.split("=", 1) for o in args.oracle or [])
    out = score_run(args.run_id, args.policy, EvalConfig(dpi=args.dpi), oracle_runs=oracle)
    from .report.store import query

    rows, cols = query(out, "SELECT doc_id, repeat, must_mentions, must_leak_rate, clean_document FROM documents "
                            "WHERE must_mentions IS NOT NULL ORDER BY doc_id, repeat")
    print(f"scored -> {out}")
    for r in rows:
        print("  " + "  ".join(f"{c}={v}" for c, v in zip(cols, r)))
    return 0


def _cmd_report(args) -> int:
    from .report.html import build_report
    from .runner.scoring import results_root

    out = build_report(args.run_id, results_root() / args.run_id, shareable=args.shareable)
    print(f"report -> {out}")
    return 0


def _cmd_compare(args) -> int:
    from .report.compare import compare
    from .runner.scoring import results_root

    r = compare(results_root() / args.run_a, results_root() / args.run_b)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"compared {r['mentions_compared']} required mentions: {r['counts']}")
    print(f"leak rate {r['leak_rate_a']} -> {r['leak_rate_b']} (95% CI of difference {r['leak_rate_diff_ci95']})")
    for t in r["transitions"][: args.limit]:
        print(f"  {t['kind']:16s} {t['doc_id']}/{t['mention_id']}: {t['status_a']} -> {t['status_b']} "
              f"({t['category_a']} -> {t['category_b']})")
    return 0


def _cmd_regress_promote(args) -> int:
    from .report.regression import promote
    from .runner.scoring import results_root

    c = promote(results_root() / args.run_id, args.run_id, args.doc_id, args.mention_id, args.note or "")
    print(f"regression case {c['fingerprint']}: {c['doc_id']}/{c['mention_id']} {c['category']}")
    return 0


def _cmd_regress_check(args) -> int:
    from .report.regression import check
    from .runner.scoring import results_root

    res = check(results_root() / args.run_id)
    for r in res:
        print(f"  {r['result']:12s} {r['doc_id']}/{r['mention_id']} ({r['category']}) {r.get('status_now', '')}")
    fails = sum(1 for r in res if r["result"] == "fail")
    print(f"{len(res)} case(s), {fails} failing")
    return 1 if fails else 0


def _cmd_candidate_check(args) -> int:
    from .runner.artifacts import load_artifacts
    from .runner.runner import load_candidate

    manifest = load_candidate(args.manifest)
    arts, hashes, errors = load_artifacts(Path(args.workdir), manifest)
    for k in hashes:
        print(f"  {'INVALID' if k in errors else 'ok':8s} {k}" + (f": {errors[k]}" if k in errors else ""))
    for k, e in errors.items():
        if k not in hashes:
            print(f"  MISSING  {k}: {e}")
    print("valid" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


def _cmd_query(args) -> int:
    from .report.store import query
    from .runner.scoring import results_root

    rows, cols = query(results_root() / args.run_id, args.sql)
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if v is None else str(v) for v in r))
    return 0


def register(sub) -> None:
    gt = sub.choices["gt"]._subparsers._group_actions[0]
    dr = gt.add_parser("draft", help="machine pre-annotation drafts (text layer / measurement OCR / regex)")
    dr.add_argument("--dataset", required=True, help="name@version")
    dr.add_argument("--docs", nargs="*")
    dr.add_argument("--out", help="default: $RB_DATA_DIR/gt-drafts (git-ignored)")
    dr.set_defaults(func=_cmd_gt_draft)
    rv = gt.add_parser("review", help="overlay review sheets for a GT or draft file")
    rv.add_argument("gt")
    rv.add_argument("--pdf", required=True)
    rv.add_argument("--out", required=True)
    rv.set_defaults(func=_cmd_gt_review)

    sy = sub.choices["synth"]._subparsers._group_actions[0]
    ds = sy.add_parser("dataset", help="register the synthetic fixtures as a dataset with GT")
    ds.add_argument("--version", required=True)
    ds.add_argument("--variants", nargs="*")
    ds.set_defaults(func=_cmd_synth_dataset)

    r = sub.add_parser("run", help="run a candidate over a dataset")
    r.add_argument("--candidate", required=True)
    r.add_argument("--dataset", required=True, help="name@version")
    r.add_argument("--experiment", default=str(ROOT / "experiments" / "e2e.yaml"))
    r.add_argument("--policy", default=DEFAULT_POLICY)
    r.add_argument("--docs", nargs="*")
    r.add_argument("--run-id")
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("score", help="score a recorded run against GT")
    s.add_argument("run_id")
    s.add_argument("--policy", default=DEFAULT_POLICY)
    s.add_argument("--dpi", type=float, default=200)
    s.add_argument("--oracle", action="append", help="MODE=RUN_ID of a scored oracle run (repeatable)")
    s.set_defaults(func=_cmd_score)

    rp = sub.add_parser("report", help="write the HTML report of a scored run")
    rp.add_argument("run_id")
    rp.add_argument("--shareable", action="store_true", help="mask GT spans in gallery crops")
    rp.set_defaults(func=_cmd_report)

    c = sub.add_parser("compare", help="mention-level comparison of two scored runs")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--json")
    c.add_argument("--limit", type=int, default=30)
    c.set_defaults(func=_cmd_compare)

    rg = sub.add_parser("regress", help="regression suite").add_subparsers(dest="cmd", required=True)
    pr = rg.add_parser("promote", help="turn a failure into a regression case")
    pr.add_argument("run_id")
    pr.add_argument("doc_id")
    pr.add_argument("mention_id")
    pr.add_argument("--note")
    pr.set_defaults(func=_cmd_regress_promote)
    ck = rg.add_parser("check", help="evaluate regression cases against a scored run")
    ck.add_argument("run_id")
    ck.set_defaults(func=_cmd_regress_check)

    cd = sub.add_parser("candidate", help="candidate tools").add_subparsers(dest="cmd", required=True)
    cc = cd.add_parser("check", help="validate a candidate's output directory against its manifest")
    cc.add_argument("workdir")
    cc.add_argument("--manifest", required=True)
    cc.set_defaults(func=_cmd_candidate_check)

    q = sub.add_parser("query", help="SQL over a scored run's tables (DuckDB)")
    q.add_argument("run_id")
    q.add_argument("sql")
    q.set_defaults(func=_cmd_query)
