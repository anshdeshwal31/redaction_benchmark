"""Loading and validating candidate artifacts (``rb candidate check``, PLAN.md §9).

Malformed artifacts are rejected with a reason; they are never silently repaired.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
from pydantic import ValidationError

from ..schemas import (
    ARTIFACT_FILENAMES,
    ArtifactKind,
    CandidateManifest,
    EntitiesArtifact,
    MentionsArtifact,
    PlanArtifact,
    PseudonymsArtifact,
    RunInfo,
    TextArtifact,
)

MODELS = {
    ArtifactKind.TEXT: TextArtifact,
    ArtifactKind.MENTIONS: MentionsArtifact,
    ArtifactKind.ENTITIES: EntitiesArtifact,
    ArtifactKind.PLAN: PlanArtifact,
    ArtifactKind.PSEUDONYMS: PseudonymsArtifact,
    ArtifactKind.RUN_INFO: RunInfo,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifacts(workdir: Path, manifest: CandidateManifest | None = None) -> tuple[dict, dict, dict]:
    """Return (artifacts by kind name, sha256 by kind name, errors by kind name).

    Without a manifest every known file present is loaded. With one, files the manifest
    promises but does not deliver are errors, and so are mismatched coordinate frames.
    """
    artifacts, hashes, errors = {}, {}, {}
    expected = set(manifest.provides) if manifest else set(ArtifactKind)
    for kind in ArtifactKind:
        path = workdir / ARTIFACT_FILENAMES[kind]
        if manifest and kind not in expected and kind is not ArtifactKind.RUN_INFO:
            continue  # not declared: never scored, so a tool cannot be judged on undeclared claims
        if not path.exists():
            if manifest and kind in expected:
                errors[kind.value] = f"{path.name} promised by the manifest but missing"
            continue
        hashes[kind.value] = sha256_file(path)
        if kind is ArtifactKind.OUTPUT_PDF:
            try:
                doc = pymupdf.open(str(path))
                if doc.page_count == 0:
                    errors[kind.value] = "output.pdf has no pages"
            except Exception as exc:  # noqa: BLE001
                errors[kind.value] = f"output.pdf does not open: {exc}"
            continue
        try:
            model = MODELS[kind].model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (ValidationError, ValueError) as exc:
            errors[kind.value] = f"{path.name} invalid: {str(exc).splitlines()[0]}"
            continue
        frame = getattr(model, "frame", None)
        if manifest and frame is not None and frame != manifest.frame:
            errors[kind.value] = f"{path.name} frame {frame.value} differs from manifest frame {manifest.frame.value}"
            continue
        if kind is not ArtifactKind.RUN_INFO:
            artifacts[kind.value] = model
    if manifest and manifest.label_map and "mentions" in artifacts:
        unknown = sorted({m.label for m in artifacts["mentions"].mentions} - set(manifest.label_map))
        if unknown:
            errors["mentions"] = f"labels not in the manifest label_map: {unknown}"
    return artifacts, hashes, errors
