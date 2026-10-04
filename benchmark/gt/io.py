"""Reading and writing GT documents, policies and other schema-backed files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ..schemas import GTDocument, PolicyProfile

M = TypeVar("M", bound=BaseModel)


def dump_json(model: BaseModel) -> str:
    """Deterministic JSON: field order from the schema, no trailing whitespace."""
    data = model.model_dump(mode="json", by_alias=True, exclude_defaults=False)
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def write_model(model: BaseModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_json(model), encoding="utf-8")


def read_model(cls: type[M], path: str | Path) -> M:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    return cls.model_validate(data)


def load_gt(path: str | Path) -> GTDocument:
    return read_model(GTDocument, path)


def gt_dirs() -> list[Path]:
    """Where GT files live: $RB_GT_DIRS (os.pathsep-separated) or <repo>/gt and $RB_DATA_DIR/gt.

    Real-document GT contains PII and belongs under the git-ignored data directory until
    the data-governance decision (PLAN.md §16 Q12); synthetic GT is tracked under gt/.
    """
    import os

    env = os.environ.get("RB_GT_DIRS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    root = Path(__file__).resolve().parents[2]
    data = Path(os.environ.get("RB_DATA_DIR", root / "data"))
    return [root / "gt", data / "gt"]


def find_gt(doc_id: str | None = None, sha256: str | None = None) -> GTDocument | None:
    """First GT file whose doc_id and/or source hash match."""
    for d in gt_dirs():
        if not d.exists():
            continue
        for path in sorted(d.rglob("*.gt.json")):
            try:
                head = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if doc_id is not None and head.get("doc_id") != doc_id:
                continue
            if sha256 is not None and head.get("source_sha256") != sha256:
                continue
            return GTDocument.model_validate(head)
    return None


def load_policy(path: str | Path) -> PolicyProfile:
    return read_model(PolicyProfile, path)
