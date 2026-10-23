"""Register the synthetic fixtures as a dataset with tracked GT (fictional content only)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..dataset import ingest
from ..gt.io import gt_dirs, write_model
from ..schemas import DatasetManifest, Datasheet
from .fixtures import ALL_VARIANTS

ROOT = Path(__file__).resolve().parents[2]


def register_synthetic(version: str, variants: list[str] | None = None) -> DatasetManifest:
    names = variants or sorted(ALL_VARIANTS)
    sources, sheets, matters = {}, {}, {}
    gt_dir = gt_dirs()[0] / "synthetic"
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            fx = ALL_VARIANTS[name]()
            doc_id = f"syn-{name.replace('_', '-')}"
            path = Path(tmp) / f"{doc_id}.pdf"
            path.write_bytes(fx.pdf)
            sources[doc_id] = path
            # The tool is given the dataset's matter ({matter}) and the matter check groups by the GT's,
            # so the two are kept equal: a fixture without a matter has none in the dataset either.
            if fx.gt.matter:
                matters[doc_id] = fx.gt.matter
            sheets[doc_id] = Datasheet(document_type=f"synthetic medico-legal ({name})",
                                       provenance="benchmark.synth (fictional)", authorisation="not required")
            write_model(fx.gt.model_copy(update={"doc_id": doc_id}), gt_dir / f"{doc_id}.gt.json")
        return ingest("synthetic", version, sources, split="synthetic", matters=matters, datasheets=sheets)


def register_ladder(factor: str, version: str, source_dataset: str, docs: list[str] | None = None) -> DatasetManifest:
    """Register one single-factor degradation ladder of documents that have GT.

    Each step is a separate document (``<doc_id>~<factor><value>``) whose manifest entry
    records ``derived_from``, so reports can plot any metric against the factor. GT of
    variants of synthetic documents is tracked under gt/ladders; variants of real documents
    contain PII and go under the data directory.
    """
    from ..dataset import data_dir, document_pdf, load_manifest
    from ..gt.io import find_gt
    from ..schemas import DerivedFrom
    from .twins import LADDERS, degrade

    name, ver = source_dataset.split("@")
    src_ds = load_manifest(name, ver)
    sources, sheets, matters, derived = {}, {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for entry in src_ds.documents:
            if docs and entry.doc_id not in docs:
                continue
            gt = find_gt(entry.doc_id, entry.sha256)
            if gt is None:
                continue
            tracked = entry.split == "synthetic"
            gt_dir = (gt_dirs()[0] if tracked else data_dir() / "gt") / "ladders" / factor
            pdf = document_pdf(entry).read_bytes()
            for params in LADDERS[factor]:
                value = float(next(iter(params.values())))
                fx = degrade(pdf, gt, **params)
                doc_id = f"{entry.doc_id}~{factor}{value:g}"
                path = Path(tmp) / f"{doc_id}.pdf"
                path.write_bytes(fx.pdf)
                sources[doc_id] = path
                if fx.gt.matter:  # the step's GT keeps its source GT's matter; the tool is given the same
                    matters[doc_id] = fx.gt.matter
                sheets[doc_id] = Datasheet(document_type=f"{factor} ladder step {value:g} of {entry.doc_id}",
                                           provenance=f"benchmark.synth.twins.degrade from {source_dataset}",
                                           authorisation=entry.datasheet.authorisation)
                derived[doc_id] = DerivedFrom(doc_id=entry.doc_id, dataset=source_dataset, factor=factor, value=value)
                write_model(fx.gt.model_copy(update={"doc_id": doc_id}), gt_dir / f"{doc_id}.gt.json")
        if not sources:
            raise ValueError(f"no documents with GT in {source_dataset}")
        split = "synthetic" if all(e.split == "synthetic" for e in src_ds.documents
                                   if not docs or e.doc_id in docs) else "dev"
        return ingest(f"ladder-{factor}", version, sources, split=split, matters=matters, datasheets=sheets,
                      derived=derived)
