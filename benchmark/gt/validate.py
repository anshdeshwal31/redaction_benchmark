"""GT validators (PLAN.md §7.5): integrity errors and annotation-quality warnings."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry import area, as_poly
from ..schemas import GTDocument, NameComponent, PiiType
from .address_split import ADDRESS_LOCALITY, LOCALITY_SUBTYPE, PLACE_SUBTYPES


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def validate_gt(gt: GTDocument, source_pdf: str | Path | None = None) -> list[Issue]:
    issues: list[Issue] = []

    def err(code, msg):
        issues.append(Issue("error", code, msg))

    def warn(code, msg):
        issues.append(Issue("warning", code, msg))

    # -- identifiers ---------------------------------------------------------
    all_ids = (
        [w.id for p in gt.pages for w in p.words]
        + [r.id for p in gt.pages for r in p.regions]
        + [m.id for m in gt.mentions]
        + [e.id for e in gt.entities]
        + [c.id for c in gt.classified]
        + [s.id for s in gt.surface]
    )
    for ident, n in Counter(all_ids).items():
        if n > 1:
            err("duplicate-id", f"id {ident!r} used {n} times")
    page_idx = [p.index for p in gt.pages]
    if len(set(page_idx)) != len(page_idx):
        err("duplicate-page", "page indices are not unique")

    # -- geometry ------------------------------------------------------------
    for p in gt.pages:
        for obj in [*p.words, *p.regions]:
            poly = as_poly(obj.poly)
            if area(poly) <= 0.01:
                err("degenerate-polygon", f"{obj.id} on page {p.index} has ~zero area")
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            if lo[0] < -1 or lo[1] < -1 or hi[0] > p.width_pt + 1 or hi[1] > p.height_pt + 1:
                err("polygon-outside-page", f"{obj.id} on page {p.index} extends beyond the page")
        orders = [w.order for w in p.words]
        if len(set(orders)) != len(orders):
            warn("duplicate-order", f"page {p.index}: reading order values repeat")

    words = gt.word_index()
    regions = gt.region_index()
    entities = {e.id: e for e in gt.entities}
    mentions = {m.id: m for m in gt.mentions}

    def check_segments(owner: str, segments) -> list[str]:
        parts: list[str] = []
        for seg in segments:
            for wid in seg.words:
                if wid not in words:
                    err("unknown-word", f"{owner} references missing word {wid}")
                    continue
                pg, w = words[wid]
                if pg != seg.page:
                    err("wrong-page", f"{owner}: word {wid} is on page {pg}, segment says {seg.page}")
                if seg.char_start is not None:
                    if seg.char_end > len(w.text):
                        err("bad-char-range", f"{owner}: char range beyond word {wid}")
                    parts.append(w.text[seg.char_start : seg.char_end])
                else:
                    parts.append(w.text)
            for rid in seg.regions:
                if rid not in regions:
                    err("unknown-region", f"{owner} references missing region {rid}")
                elif regions[rid][0] != seg.page:
                    err("wrong-page", f"{owner}: region {rid} is on another page")
        return parts

    # -- mentions ------------------------------------------------------------
    for m in gt.mentions:
        parts = check_segments(f"mention {m.id}", m.segments)
        has_words = any(seg.words for seg in m.segments)
        if has_words and _norm_space(" ".join(parts)) != _norm_space(m.text):
            err("text-mismatch", f"mention {m.id}: text {m.text!r} != words {' '.join(parts)!r}")
        if m.entity is not None and m.entity not in entities:
            err("unknown-entity", f"mention {m.id} references missing entity {m.entity}")
        seg_words = {wid for seg in m.segments for wid in seg.words}
        mention_pages = {seg.page for seg in m.segments}
        for wid in m.optional_words:
            if wid not in words:
                err("unknown-word", f"mention {m.id}: optional word {wid} missing")
            elif wid in seg_words:
                err("optional-overlap", f"mention {m.id}: optional word {wid} is also required")
            elif words[wid][0] not in mention_pages:
                warn("optional-far", f"mention {m.id}: optional word {wid} on a page without the mention")
        for comp, wids in m.components.items():
            for wid in wids:
                if wid not in seg_words:
                    err("component-outside", f"mention {m.id}: component {comp.value} word {wid} not in mention")
        if m.type is PiiType.PERSON_NAME and has_words and not m.components:
            warn("no-components", f"person mention {m.id} has no name components (needed for pseudonym checks)")
        if m.type is PiiType.LOCATION and m.subtype == LOCALITY_SUBTYPE and ADDRESS_LOCALITY not in m.conditions:
            # The policy decides by subtype: a place on its own labelled `locality` would be replaced like
            # an address's locality (policy 1.1.0) instead of kept.
            warn("locality-outside-address",
                 f"mention {m.id}: subtype '{LOCALITY_SUBTYPE}' is only for the locality part of an address "
                 f"(condition {ADDRESS_LOCALITY}); a place mentioned on its own takes one of "
                 f"{', '.join(PLACE_SUBTYPES)}")
        if m.parent is not None:
            parent = mentions.get(m.parent)
            if parent is None:
                err("unknown-parent", f"mention {m.id}: parent {m.parent} missing")
            else:
                parent_words = {wid for seg in parent.segments for wid in seg.words}
                if not seg_words <= parent_words:
                    err("parent-not-superset", f"mention {m.id} is not contained in parent {m.parent}")

    for c in gt.classified:
        check_segments(f"classified {c.id}", c.segments)

    # -- entities -------------------------------------------------------------
    mentioned = {m.entity for m in gt.mentions if m.entity} | {s.entity for s in gt.surface if s.entity}
    for e in gt.entities:
        if e.id not in mentioned:
            warn("orphan-entity", f"entity {e.id} has no mention")
        g = e.attributes.get("gender")
        if g is not None and not g.evidence:
            warn("attribute-no-evidence", f"entity {e.id}: gender has no evidence")
    for rel in gt.relations:
        for end in (rel.source, rel.target):
            if end not in entities:
                err("unknown-entity", f"relation {rel.type} references missing entity {end}")

    # -- annotation-quality heuristics ---------------------------------------
    covered = {wid for m in gt.mentions for seg in m.segments for wid in seg.words}
    covered |= {wid for m in gt.mentions for wid in m.optional_words}
    classified_words = {wid for c in gt.classified for seg in c.segments for wid in seg.words}

    def key(t: str) -> str:
        return re.sub(r"[^\w-]", "", t).casefold()

    by_text: dict[str, list[str]] = {}
    for wid, (_, w) in words.items():
        by_text.setdefault(key(w.text), []).append(wid)

    name_forms: set[str] = set()
    surnames: set[str] = set()
    for m in gt.mentions:
        if m.type is not PiiType.PERSON_NAME:
            continue
        for seg in m.segments:
            for wid in seg.words:
                k = key(words[wid][1].text) if wid in words else ""
                if len(k) >= 3:
                    name_forms.add(k)
        for wid in m.components.get(NameComponent.SURNAME, []):
            if wid in words:
                surnames.add(key(words[wid][1].text))

    for form in sorted(name_forms):
        for wid in by_text.get(form, []):
            if wid in covered or wid in classified_words:
                continue
            text = words[wid][1].text
            if form in surnames:
                warn(
                    "homograph-unclassified",
                    f"word {wid} ({text!r}) equals an annotated surname but is neither a mention nor classified",
                )
            elif text[:1].isupper():
                warn("unannotated-repeat", f"word {wid} ({text!r}) matches an annotated name but is not annotated")

    # -- cross-check against the source PDF ------------------------------------
    if source_pdf is not None:
        issues.extend(_check_source(gt, Path(source_pdf)))
    return issues


def _check_source(gt: GTDocument, path: Path) -> list[Issue]:
    import hashlib

    from ..geometry import load_page_geometries

    issues: list[Issue] = []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != gt.source_sha256:
        issues.append(Issue("error", "source-hash", f"source sha256 {digest} != GT {gt.source_sha256}"))
    geoms = load_page_geometries(path)
    for p in gt.pages:
        if p.index >= len(geoms):
            issues.append(Issue("error", "source-pages", f"GT page {p.index} beyond source page count {len(geoms)}"))
            continue
        g = geoms[p.index]
        w, h = g.display_size
        if not (np.isclose(w, p.width_pt, atol=0.5) and np.isclose(h, p.height_pt, atol=0.5)):
            issues.append(Issue("error", "source-size", f"page {p.index}: GT size {(p.width_pt, p.height_pt)} != {(w, h)}"))
        if g.rotation != p.rotation:
            issues.append(Issue("error", "source-rotation", f"page {p.index}: GT rotation {p.rotation} != {g.rotation}"))
    return issues
