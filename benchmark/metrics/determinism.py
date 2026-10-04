"""Determinism across repeated runs (PLAN.md §6).

Levels, reported separately:
- D0 exact: byte-identical output files (informational only)
- D1 structural: canonicalised stage artifacts identical (mentions projected to GT
  words, clusters as partitions, plan regions rasterised, pseudonym map, OCR tokens)
- D2 visual: rendered output pages identical (exact and anti-aliasing tolerant)
- D3 decision: identical per-mention verdicts; unstable mentions listed (flip rate)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import pymupdf

from ..evaluate import DocumentEvaluation
from ..schemas import EntitiesArtifact, MentionsArtifact, PlanArtifact, PseudonymsArtifact, TextArtifact
from .align import InkIndex
from .detection import project_mentions


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def canonical_text(art: TextArtifact, index: InkIndex) -> str:
    toks = []
    for p in art.pages:
        for t in p.tokens:
            poly = np.round(index.to_display(p.index, t.poly, art.frame, art.dpi), 1).tolist()
            toks.append((p.index, " ".join(t.text.split()), poly))
    return _digest(sorted(toks))


def canonical_mentions(art: MentionsArtifact, index: InkIndex, text: TextArtifact | None = None) -> tuple[str, dict]:
    proj = project_mentions(art, index, text=text)
    sig = {p.id: (p.label, tuple(sorted(p.words))) for p in proj}
    return _digest(sorted(sig.values())), sig


def canonical_clusters(art: EntitiesArtifact, mention_sig: dict) -> str:
    """Clusters as a partition of mention signatures: invariant to ids and ordering."""
    parts = [sorted(str(mention_sig.get(m, m)) for m in c.mentions) for c in art.clusters]
    return _digest(sorted(parts))


def canonical_plan(art: PlanArtifact, index: InkIndex) -> str:
    masks = {}
    for r in art.regions:
        poly = index.to_display(r.page, r.poly, art.frame, art.dpi)
        m = index.region_mask(r.page, [poly])
        masks[r.page] = masks.get(r.page, np.zeros_like(m)) | m
    replacements = sorted((r.page, r.replacement or "") for r in art.regions if r.replacement)
    return _digest({str(k): hashlib.sha256(np.packbits(v).tobytes()).hexdigest() for k, v in sorted(masks.items())}
                   | {"replacements": replacements})


def canonical_pseudonyms(art: PseudonymsArtifact) -> str:
    return _digest(sorted((e.original.casefold(), e.replacement) for e in art.entries))


@dataclass
class DeterminismResult:
    runs: int
    d0_exact: bool
    d1_structural: dict  # artifact -> identical?
    d2_visual_exact: bool
    d2_visual_tolerant: bool
    d2_min_mask_iou: float | None
    d3_decisions_identical: bool
    flip_rate: float | None
    unstable_mentions: list[str]
    metric_spread: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _render(pdf: bytes, dpi: float = 72) -> list[np.ndarray]:
    doc = pymupdf.open("pdf", pdf)
    s = dpi / 72.0
    out = []
    for page in doc:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
        out.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy())
    return out


def compare_runs(
    source: bytes,
    outputs: list[bytes],
    artifacts: list[dict],
    evaluations: list[DocumentEvaluation],
    index: InkIndex,
) -> DeterminismResult:
    n = len(outputs)
    d0 = len({hashlib.sha256(o).hexdigest() for o in outputs}) == 1

    # -- D1 --------------------------------------------------------------------------------
    d1: dict[str, bool] = {}
    kinds = set().union(*(a.keys() for a in artifacts)) if artifacts else set()
    sigs_per_run = []
    for a in artifacts:
        sig = {}
        if "mentions" in a:
            sig["mentions"], mention_sig = canonical_mentions(a["mentions"], index, a.get("text"))
            if "entities" in a:
                sig["entities"] = canonical_clusters(a["entities"], mention_sig)
        if "text" in a:
            sig["text"] = canonical_text(a["text"], index)
        if "plan" in a:
            sig["plan"] = canonical_plan(a["plan"], index)
        if "pseudonyms" in a:
            sig["pseudonyms"] = canonical_pseudonyms(a["pseudonyms"])
        sigs_per_run.append(sig)
    for k in sorted(kinds):
        values = {s.get(k) for s in sigs_per_run}
        d1[k] = len(values) == 1 and None not in values

    # -- D2 --------------------------------------------------------------------------------
    renders = [_render(o) for o in outputs]
    src = _render(source)
    exact = all(len(r) == len(renders[0]) and all(np.array_equal(a, b) for a, b in zip(r, renders[0]))
                for r in renders)

    def tolerant_equal(a, b):
        if a.shape != b.shape:
            return False
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        return (diff > 8).mean() <= 1e-4

    tolerant = all(len(r) == len(renders[0]) and all(tolerant_equal(a, b) for a, b in zip(r, renders[0]))
                   for r in renders)
    ious = []
    masks = []
    for r in renders:
        masks.append([np.abs(a.astype(np.int16) - s.astype(np.int16)) > 40 if a.shape == s.shape else None
                      for a, s in zip(r, src)])
    for k in range(1, n):
        for a, b in zip(masks[0], masks[k]):
            if a is None or b is None:
                continue
            uni = (a | b).sum()
            ious.append(float((a & b).sum() / uni) if uni else 1.0)

    # -- D3 --------------------------------------------------------------------------------
    statuses: dict[str, set[str]] = {}
    for ev in evaluations:
        for m in ev.mentions:
            statuses.setdefault(m.mention_id, set()).add(m.status.value)
    unstable = sorted(m for m, s in statuses.items() if len(s) > 1)
    spread = {}
    for key in ("must_leak_rate", "collateral_rate", "effective_token_recall"):
        vals = [ev.summary.get(key) for ev in evaluations if ev.summary.get(key) is not None]
        if vals:
            spread[key] = {"min": min(vals), "max": max(vals)}
    return DeterminismResult(
        runs=n, d0_exact=d0, d1_structural=d1, d2_visual_exact=exact, d2_visual_tolerant=tolerant,
        d2_min_mask_iou=round(min(ious), 4) if ious else None, d3_decisions_identical=not unstable,
        flip_rate=round(len(unstable) / len(statuses), 4) if statuses else None, unstable_mentions=unstable,
        metric_spread=spread)
