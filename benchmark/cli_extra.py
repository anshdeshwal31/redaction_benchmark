"""CLI commands for dataset management and forensics (registered by cli.py)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


def _cmd_inspect(args) -> int:
    from .forensics import inspect_pdf

    inv = inspect_pdf(args.pdf, include_values=args.show_values)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(inv.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    print(f"sha256 {inv.sha256[:16]}…  pages {inv.page_count}  pdf {inv.pdf_version}  producer {inv.producer!r}")
    print(f"metadata keys {sorted(inv.metadata)}  xmp {inv.has_xmp}  outline {inv.outline_entries}  "
          f"attachments {inv.embedded_files}  fields {len(inv.form_fields)}  js {inv.has_javascript}  "
          f"ocg {inv.optional_content_groups}  struct-tree {inv.struct_tree} (alt texts {inv.alt_text_count})")
    print(f"revisions: %%EOF markers {inv.eof_markers}, mupdf versions {inv.mupdf_versions}, linearized "
          f"{inv.linearized}; unreferenced objects {inv.unreferenced_objects} ({inv.unreferenced_streams} streams)")
    if args.show_values:
        for k, v in inv.metadata.items():
            print(f"  info {k}: {v!r}")
    for p in inv.pages:
        masks = sum(1 for im in p.images if im.is_mask or im.bpc == 1)
        print(f"  p{p.index + 1}: {p.display_size[0]:.0f}x{p.display_size[1]:.0f} rot {p.rotation} | text "
              f"{p.visible_chars} visible / {p.invisible_chars} invisible | images {len(p.images)} "
              f"({masks} bilevel) | paths {p.vector_paths} | annots {len(p.annotations)} | thumb {p.has_thumbnail}")
    return 0


def _parse_pairs(items: list[str] | None) -> dict[str, str]:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def _cmd_dataset_ingest(args) -> int:
    from .dataset import ingest
    from .schemas import Datasheet

    sources = {k: Path(v) for k, v in _parse_pairs(args.doc).items()}
    for k, p in sources.items():
        if not p.exists():
            raise SystemExit(f"{k}: {p} not found")
    sheets = {}
    if args.datasheets:
        raw = yaml.safe_load(Path(args.datasheets).read_text(encoding="utf-8")) or {}
        sheets = {k: Datasheet.model_validate(v) for k, v in raw.items()}
    m = ingest(args.name, args.version, sources, split=args.split, matters=_parse_pairs(args.matter),
               datasheets=sheets)
    print(f"dataset {m.name}@{m.version}: {len(m.documents)} documents")
    return _print_manifest(m)


def _print_manifest(m) -> int:
    for d in m.documents:
        enc = Counter(c.encoding.value for c in d.page_conditions)
        skews = [c.skew_deg for c in d.page_conditions if c.skew_deg is not None]
        axes = Counter(c.text_axis for c in d.page_conditions if c.text_axis)
        dpis = sorted({c.full_page_image_dpi for c in d.page_conditions if c.full_page_image_dpi})
        sk = f"skew {min(skews):+.2f}..{max(skews):+.2f}°" if skews else "skew n/a"
        print(f"  {d.doc_id:5s} {d.page_count:3d}p  {dict(enc)}  dpi {dpis or '-'}  {sk}  axes {dict(axes) or '-'}  "
              f"matter {d.matter}  sha {d.sha256[:10]}…")
    return 0


def _cmd_dataset_validate(args) -> int:
    from .dataset import load_manifest, validate_manifest

    m = load_manifest(args.name, args.version)
    problems = validate_manifest(m)
    for p in problems:
        print(p)
    print(f"{m.name}@{m.version}: {len(m.documents)} documents, {len(problems)} problem(s)")
    return 1 if problems else 0


def _cmd_dataset_show(args) -> int:
    from .dataset import load_manifest

    m = load_manifest(args.name, args.version)
    print(f"dataset {m.name}@{m.version} (created {m.created}, measured with {m.measured_with})")
    return _print_manifest(m)


def register(sub) -> None:
    ins = sub.add_parser("inspect", help="inventory every content surface of a PDF")
    ins.add_argument("pdf")
    ins.add_argument("--json", help="write the full inventory as JSON")
    ins.add_argument("--show-values", action="store_true",
                     help="include metadata/outline/field values (may contain PII)")
    ins.set_defaults(func=_cmd_inspect)

    ds = sub.add_parser("dataset", help="dataset registry").add_subparsers(dest="cmd", required=True)
    ing = ds.add_parser("ingest", help="store PDFs and write an immutable manifest")
    ing.add_argument("--name", required=True)
    ing.add_argument("--version", required=True)
    ing.add_argument("--split", default="dev", choices=["dev", "test", "regression", "synthetic"])
    ing.add_argument("--doc", action="append", required=True, help="DOC_ID=path/to.pdf (repeatable)")
    ing.add_argument("--matter", action="append", help="DOC_ID=MATTER (repeatable)")
    ing.add_argument("--datasheets", help="YAML mapping DOC_ID -> datasheet fields")
    ing.set_defaults(func=_cmd_dataset_ingest)
    for name, func, help_ in (("validate", _cmd_dataset_validate, "check store files against a manifest"),
                              ("show", _cmd_dataset_show, "summarise a manifest")):
        c = ds.add_parser(name, help=help_)
        c.add_argument("--name", required=True)
        c.add_argument("--version", required=True)
        c.set_defaults(func=func)

    from . import cli_run

    cli_run.register(sub)
