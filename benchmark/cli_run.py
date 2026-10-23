"""CLI commands for running, scoring, reporting, comparing and regression (registered by cli_extra)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The client's decisions (Q1-Q16). The draft (au-medicolegal-v0.yaml) stays for re-scoring old runs.
DEFAULT_POLICY = str(ROOT / "policies" / "au-medicolegal-v1.yaml")


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


def _cmd_gt_split_addresses(args) -> int:
    """Split ADDRESS mentions into street line + locality (GT 0.2). Prints counts only: the
    review list with the actual text goes next to the GT, in the git-ignored data directory."""
    import shutil
    from collections import Counter

    from .gt.address_split import split_addresses
    from .gt.io import load_gt, write_model
    from .gt.validate import validate_gt

    failed = 0
    for path in map(Path, args.gt):
        gt = load_gt(path)
        new, notes = split_addresses(gt, version=args.version)
        errors = [i for i in validate_gt(new) if i.severity == "error"]
        counts = Counter(f"{n.outcome}{'' if n.confident else ' (uncertain)'}" for n in notes)
        print(f"{gt.doc_id}: {gt.gt_version} -> {new.gt_version}; {dict(counts) or 'no addresses'}; "
              f"{len(errors)} validation error(s)")
        if errors:
            failed += 1
            continue
        if args.dry_run:
            continue
        if args.archive:
            dst = Path(args.archive) / gt.gt_version / path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
        write_model(new, path)
        mentions = {m.id: m for m in new.mentions}
        review = [dict(n.to_dict(), street=mentions[n.mention_id].text if n.outcome == "split" else None,
                       locality=(mentions.get(f"{n.mention_id}-loc") or mentions[n.mention_id]).text
                       if n.outcome != "unsplit" else None) for n in notes]
        path.with_name(path.name.replace(".gt.json", ".address-split.json")).write_text(
            json.dumps(review, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if failed else 0


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
    import sys

    from .runner.runner import CandidateEnvError, CommandValueError, run_candidate, runs_root

    try:
        run = run_candidate(args.candidate, args.dataset, args.experiment, args.policy, docs=args.docs,
                            run_id=args.run_id)
    except (CandidateEnvError, CommandValueError) as exc:
        print(f"rb run: {exc}", file=sys.stderr)
        return 2
    failed = [i for i in run.items if i.error or i.invalid_artifacts]
    unread = "" if run.policy.given_to_tool else ", which the tool does not read"
    print(f"run {run.run_id} (policy {run.policy.name}@{run.policy.version}{unread}): {len(run.items)} item(s), "
          f"{len(failed)} with errors -> {runs_root() / run.run_id}")
    for i in failed:
        print(f"  {i.doc_id}/r{i.repeat}: {i.error or ''} {i.invalid_artifacts or ''}")
    return 0


def _score_policy(run_id: str, given: str | None) -> str:
    """--policy if given; else the policy the run was made with, if recorded and still there; else the default."""
    if given:
        return given
    from .runner.runner import load_run

    recorded = load_run(run_id).policy
    return recorded.path if recorded and Path(recorded.path).exists() else DEFAULT_POLICY


def _cmd_score(args) -> int:
    import sys

    from .evaluate import EvalConfig
    from .runner.scoring import score_run

    oracle = dict(o.split("=", 1) for o in args.oracle or [])
    out = score_run(args.run_id, _score_policy(args.run_id, args.policy), EvalConfig(dpi=args.dpi),
                    oracle_runs=oracle)
    from .report.store import query

    rows, cols = query(out, "SELECT doc_id, repeat, must_mentions, must_leak_rate, clean_document FROM documents "
                            "WHERE must_mentions IS NOT NULL ORDER BY doc_id, repeat")
    scoring = json.loads((out / "scoring.json").read_text(encoding="utf-8"))
    print(f"scored under policy {scoring['policy_id']} -> {out}")
    if scoring.get("policy_warning"):
        print(f"WARNING: {scoring['policy_warning']}", file=sys.stderr)
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


def _cmd_gt_agree(args) -> int:
    from .gt.agreement import agreement
    from .gt.io import load_gt

    r = agreement(load_gt(args.a), load_gt(args.b))
    if args.json:
        Path(args.json).write_text(json.dumps(r.to_dict(), indent=1), encoding="utf-8")
    print(f"words: {r.words_matched} matched of {r.words_a} / {r.words_b} (median IoU {r.word_iou_median}, "
          f"same text {r.word_text_agreement})")
    print(f"mentions: {r.mentions_a} / {r.mentions_b}; strict F1 {r.strict_f1}; tolerant F1 {r.tolerant_f1}")
    print(f"token kappa {r.token_kappa}; classified kappa {r.classified_kappa}; entity-link B3 {r.entity_link_b3}")
    for k, v in r.disagreements.items():
        print(f"  {k}: {v if not isinstance(v, list) else v[:20]}")
    for gate, ok in r.gates.items():
        print(f"  gate {gate}: {'PASS' if ok else 'FAIL'}")
    return 0 if r.passed else 1


def _cmd_selfcheck(args) -> int:
    from .dataset import document_pdf, load_manifest
    from .evaluate import EvalConfig
    from .gt.io import find_gt, load_gt, load_policy
    from .selfcheck import selfcheck

    name, version = args.dataset.split("@")
    m = load_manifest(name, version)
    failed = 0
    report = {}
    for d in m.documents:
        if args.docs and d.doc_id not in args.docs:
            continue
        gt = load_gt(Path(args.gt_dir) / f"{d.doc_id}{args.gt_suffix}") if args.gt_dir else find_gt(d.doc_id, d.sha256)
        if gt is None:
            print(f"{d.doc_id}: no GT, skipped")
            continue
        res = selfcheck(document_pdf(d).read_bytes(), gt, load_policy(args.policy), EvalConfig(dpi=args.dpi),
                        only=args.only)
        report[d.doc_id] = [r.to_dict() for r in res]
        for r in res:
            mark = {True: "PASS", False: "FAIL", None: "N/A "}[r.passed]
            failed += r.passed is False
            print(f"{d.doc_id:12s} {r.name:17s} {mark} {r.seconds:6.1f}s  {r.observed}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{failed} failing check(s)")
    return 1 if failed else 0


def _cmd_synth_ladder(args) -> int:
    from .synth.register import register_ladder

    m = register_ladder(args.factor, args.version, args.source, args.docs)
    print(f"dataset {m.name}@{m.version}: {len(m.documents)} documents "
          + ", ".join(f"{d.doc_id}" for d in m.documents))
    return 0


def _cmd_board(args) -> int:
    from .report.board import build_board
    from .runner.scoring import results_root

    out = Path(args.out) if args.out else results_root() / f"board-{args.name}.html"
    print(f"board -> {build_board(args.run_ids, out, title=args.title)}")
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
    sa = gt.add_parser("split-addresses", help="split ADDRESS mentions into street line and locality (GT 0.2)")
    sa.add_argument("gt", nargs="+", help="GT files, rewritten in place")
    sa.add_argument("--version", required=True, help="new gt_version, e.g. 0.2-ai")
    sa.add_argument("--archive", help="copy each original to <archive>/<old gt_version>/ first; keep it off "
                                      "the GT search path so a document never has two GT files")
    sa.add_argument("--dry-run", action="store_true")
    sa.set_defaults(func=_cmd_gt_split_addresses)
    ag = gt.add_parser("agree", help="inter-annotator agreement between two GT files of one document")
    ag.add_argument("a")
    ag.add_argument("b")
    ag.add_argument("--json")
    ag.set_defaults(func=_cmd_gt_agree)

    sc = sub.add_parser("selfcheck", help="run the fault mocks on documents with GT and check the evaluator's verdicts")
    sc.add_argument("--dataset", required=True, help="name@version")
    sc.add_argument("--docs", nargs="*")
    sc.add_argument("--gt-dir", help="read GT from <dir>/<doc_id><suffix> instead of the GT search path")
    sc.add_argument("--gt-suffix", default=".gt.json")
    sc.add_argument("--only", nargs="*")
    sc.add_argument("--dpi", type=float, default=150)
    sc.add_argument("--policy", default=DEFAULT_POLICY)
    sc.add_argument("--json")
    sc.set_defaults(func=_cmd_selfcheck)

    bd = sub.add_parser("board", help="comparison board over several scored runs")
    bd.add_argument("run_ids", nargs="+")
    bd.add_argument("--name", default="latest")
    bd.add_argument("--title", default="Candidate comparison")
    bd.add_argument("--out")
    bd.set_defaults(func=_cmd_board)

    sy = sub.choices["synth"]._subparsers._group_actions[0]
    ds = sy.add_parser("dataset", help="register the synthetic fixtures as a dataset with GT")
    ds.add_argument("--version", required=True)
    ds.add_argument("--variants", nargs="*")
    ds.set_defaults(func=_cmd_synth_dataset)
    ld = sy.add_parser("ladder", help="register a single-factor degradation ladder of documents with GT")
    ld.add_argument("--factor", required=True, choices=["dpi", "skew", "jpeg", "noise", "blur"])
    ld.add_argument("--version", required=True)
    ld.add_argument("--source", required=True, help="source dataset name@version")
    ld.add_argument("--docs", nargs="*")
    ld.set_defaults(func=_cmd_synth_ladder)

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
    s.add_argument("--policy", help="default: the policy the run was made with (recorded in run.json), else the "
                                    "client's policy; scoring warns when it differs from the run's")
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
