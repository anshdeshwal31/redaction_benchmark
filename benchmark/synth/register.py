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
            matters[doc_id] = fx.gt.matter or "MX"
            sheets[doc_id] = Datasheet(document_type=f"synthetic medico-legal ({name})",
                                       provenance="benchmark.synth (fictional)", authorisation="not required")
            write_model(fx.gt.model_copy(update={"doc_id": doc_id}), gt_dir / f"{doc_id}.gt.json")
        return ingest("synthetic", version, sources, split="synthetic", matters=matters, datasheets=sheets)
