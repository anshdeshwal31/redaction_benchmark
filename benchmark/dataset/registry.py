"""Dataset registry: content-addressed PDF store + versioned manifests (PLAN.md §7.6, §8).

PDFs live under ``$RB_DATA_DIR/store/sha256/`` (default ``<repo>/data``, git-ignored).
Manifests under ``datasets/<name>/<version>/manifest.json`` hold hashes and measured
conditions only; original filenames (which can contain names) are kept in a git-ignored
sidecar under ``$RB_DATA_DIR/ingest/``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from pathlib import Path

import pymupdf

from .. import __version__
from ..forensics import inspect_pdf
from ..gt.io import read_model, write_model
from ..schemas import Datasheet, DatasetManifest, DocumentEntry
from .conditions import page_conditions

ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("RB_DATA_DIR", ROOT / "data"))


def store_path(sha256: str) -> Path:
    return data_dir() / "store" / "sha256" / sha256[:2] / f"{sha256}.pdf"


def datasets_dir() -> Path:
    return Path(os.environ.get("RB_DATASETS_DIR", ROOT / "datasets"))


def manifest_path(name: str, version: str) -> Path:
    return datasets_dir() / name / version / "manifest.json"


def store_pdf(src: Path) -> str:
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    dst = store_path(digest)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return digest


def describe_document(path: Path, doc_id: str, split: str, matter: str | None,
                      datasheet: Datasheet | None = None) -> DocumentEntry:
    inv = inspect_pdf(path)
    doc = pymupdf.open(str(path))
    conditions = [page_conditions(p, doc[p.index]) for p in inv.pages]
    return DocumentEntry(
        doc_id=doc_id,
        sha256=inv.sha256,
        size_bytes=inv.size_bytes,
        page_count=inv.page_count,
        split=split,
        matter=matter,
        pdf_version=inv.pdf_version,
        producer=inv.producer,
        metadata_keys=sorted(inv.metadata),
        has_xmp=inv.has_xmp,
        encrypted=inv.encrypted,
        revisions=max(1, inv.mupdf_versions),
        page_conditions=conditions,
        datasheet=datasheet or Datasheet(),
    )


def ingest(
    name: str,
    version: str,
    sources: dict[str, Path],
    split: str = "dev",
    matters: dict[str, str] | None = None,
    datasheets: dict[str, Datasheet] | None = None,
    derived: dict | None = None,
) -> DatasetManifest:
    entries = []
    for doc_id, src in sources.items():
        store_pdf(src)
        entry = describe_document(src, doc_id, split, (matters or {}).get(doc_id), (datasheets or {}).get(doc_id))
        if derived and doc_id in derived:
            entry = entry.model_copy(update={"derived_from": derived[doc_id]})
        entries.append(entry)
    manifest = DatasetManifest(
        name=name,
        version=version,
        created=_dt.date.today().isoformat(),
        measured_with={"redaction_benchmark": __version__, "pymupdf": pymupdf.VersionBind},
        documents=entries,
    )
    out = manifest_path(name, version)
    if out.exists():
        raise FileExistsError(f"{out} exists; dataset releases are immutable — bump the version")
    write_model(manifest, out)
    sidecar = data_dir() / "ingest" / f"{name}-{version}.sources.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({k: str(v) for k, v in sources.items()}, indent=1), encoding="utf-8")
    return manifest


def load_manifest(name: str, version: str) -> DatasetManifest:
    return read_model(DatasetManifest, manifest_path(name, version))


def document_pdf(entry: DocumentEntry) -> Path:
    return store_path(entry.sha256)


def validate_manifest(manifest: DatasetManifest) -> list[str]:
    problems = []
    ids = [d.doc_id for d in manifest.documents]
    if len(set(ids)) != len(ids):
        problems.append("duplicate doc_id")
    for d in manifest.documents:
        p = document_pdf(d)
        if not p.exists():
            problems.append(f"{d.doc_id}: missing from store ({p})")
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != d.sha256:
            problems.append(f"{d.doc_id}: store file hash mismatch")
        if pymupdf.open(str(p)).page_count != d.page_count:
            problems.append(f"{d.doc_id}: page count mismatch")
        if len(d.page_conditions) != d.page_count:
            problems.append(f"{d.doc_id}: conditions for {len(d.page_conditions)} of {d.page_count} pages")
    return problems
