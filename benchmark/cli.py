"""Command-line interface: ``rb <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cmd_schemas_export(args) -> int:
    from .schemas import EXPORTED

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stale = []
    for filename, model in EXPORTED.items():
        text = json.dumps(model.model_json_schema(by_alias=True), indent=1, ensure_ascii=False) + "\n"
        path = out / filename
        if args.check:
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != json.loads(text):
                stale.append(filename)
        else:
            path.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print("stale schemas: " + ", ".join(stale))
            return 1
        print("schemas are current")
        return 0
    print(f"wrote {len(EXPORTED)} schemas to {out}")
    return 0


def _cmd_gt_validate(args) -> int:
    from .gt import load_gt, load_policy, pending_rules, resolve, validate_gt

    gt = load_gt(args.gt)
    issues = validate_gt(gt, source_pdf=args.source)
    for issue in issues:
        print(issue)
    errors = sum(1 for i in issues if i.severity == "error")
    print(f"{errors} error(s), {len(issues) - errors} warning(s)")
    if args.policy:
        policy = load_policy(args.policy)
        res = resolve(gt, policy)
        counts = Counter(r.requirement.value for r in res.mentions.values())
        pend = Counter(r.pending for r in res.mentions.values() if r.pending)
        print(f"policy {res.policy}: mentions by requirement {dict(counts)}")
        if pend:
            print("mentions depending on open questions: " + ", ".join(f"{q}={n}" for q, n in sorted(pend.items())))
        if pending_rules(policy):
            print("pending rules: " + ", ".join(f"{r.id}({r.pending})" for r in pending_rules(policy)))
    return 1 if errors else 0


def _cmd_synth_write(args) -> int:
    from .gt import write_model
    from .synth import ALL_VARIANTS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.variants or sorted(ALL_VARIANTS)
    for name in names:
        fx = ALL_VARIANTS[name]()
        (out / f"{fx.name}.pdf").write_bytes(fx.pdf)
        write_model(fx.gt, out / f"{fx.name}.gt.json")
        print(f"{fx.name}: {len(fx.pdf)} bytes, sha256 {fx.sha256[:12]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rb", description="Redaction benchmark")
    sub = p.add_subparsers(dest="group", required=True)

    sch = sub.add_parser("schemas", help="JSON Schema export").add_subparsers(dest="cmd", required=True)
    ex = sch.add_parser("export", help="write JSON Schemas for every contract")
    ex.add_argument("--out", default=str(ROOT / "schemas"))
    ex.add_argument("--check", action="store_true", help="exit 1 if the committed schemas are stale")
    ex.set_defaults(func=_cmd_schemas_export)

    gt = sub.add_parser("gt", help="ground-truth tools").add_subparsers(dest="cmd", required=True)
    va = gt.add_parser("validate", help="validate a GT file")
    va.add_argument("gt")
    va.add_argument("--source", help="source PDF to cross-check hash, page sizes and rotation")
    va.add_argument("--policy", help="policy profile to summarise requirements")
    va.set_defaults(func=_cmd_gt_validate)

    sy = sub.add_parser("synth", help="synthetic documents").add_subparsers(dest="cmd", required=True)
    wr = sy.add_parser("write", help="write synthetic fixture PDFs and their GT")
    wr.add_argument("out")
    wr.add_argument("--variants", nargs="*")
    wr.set_defaults(func=_cmd_synth_write)

    from . import cli_extra

    cli_extra.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
