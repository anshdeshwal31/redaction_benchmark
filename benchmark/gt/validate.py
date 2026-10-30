"""GT validators (PLAN.md §7.5): integrity errors and annotation-quality warnings."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry import area, as_poly
from ..schemas import GTDocument, GTMention, NameComponent, PiiType
from .address_split import (ADDRESS_LOCALITY, LOCALITY_SUBTYPE, PLACE_SUBTYPES, address_roles, address_unit_mentions,
                            read_end, unit_words)

# A place written with its postcode stays a place on its own only right after a label or phrase naming
# its kind (owner ruling 2 of 2026-10-22); every other one is the locality of an address.
PLACE_KINDS = "birth, injury, accident or incident, declaration"
PLACE_KIND_WORDS = {"birth", "born", "birthplace", "injury", "injuries", "injured", "accident", "incident",
                    "declaration", "declared"}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
           "dec"}
_PLACE_KIND_WINDOW = 3  # words before the place on its line, dates not counted ('Date of accident: 12 June 2023,')


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

    issues.extend(_address_issues(gt))

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


def _bare(text: str) -> str:
    return text.strip(string.punctuation + "‘’“”").casefold()


def _address_issues(gt: GTDocument) -> list[Issue]:
    """Review aids for address parts and places on their own (owner rulings 2 and 3 of 2026-10-22)."""
    issues: list[Issue] = []
    reading = {p.index: sorted(p.words, key=lambda w: w.order) for p in gt.pages}
    at = {w.id: (page, k) for page, ws in reading.items() for k, w in enumerate(ws)}

    def ends(m: GTMention) -> tuple[str | None, str | None]:
        wids = [wid for seg in m.segments for wid in seg.words if wid in at]
        return (wids[0], wids[-1]) if wids else (None, None)

    def after_place_kind(wid: str) -> bool:
        """A label or phrase naming the place's kind within a few words before it on its line, or on
        the line above it in reading order."""
        page, k = at[wid]
        ws = reading[page]
        j, counted = k - 1, 0
        while j >= 0 and ws[j].line == ws[k].line and counted < _PLACE_KIND_WINDOW:
            word = _bare(ws[j].text)
            if word in PLACE_KIND_WORDS:
                return True
            counted += not (any(ch.isdigit() for ch in word) or word in _MONTHS)
            j -= 1
        j = k - 1
        while j >= 0 and ws[j].line == ws[k].line:
            j -= 1
        above = ws[j].line if j >= 0 else None
        while j >= 0 and ws[j].line == above:
            if _bare(ws[j].text) in PLACE_KIND_WORDS:
                return True
            j -= 1
        return False

    def line_box(page: int, line: str | None) -> np.ndarray:
        pts = np.vstack([as_poly(w.poly) for w in reading[page] if w.line == line])
        return np.concatenate([pts.min(axis=0), pts.max(axis=0)])

    def right_after_or_below(street: GTMention, place: GTMention) -> bool:
        last, first = ends(street)[1], ends(place)[0]
        if last is None or first is None or at[last][0] != at[first][0]:
            return False
        page, (ka, kb) = at[last][0], (at[last][1], at[first][1])
        if kb == ka + 1:
            return True
        ws = reading[page]
        if kb > 0 and ws[kb - 1].line == ws[kb].line:  # the place does not start its line
            return False
        s, p = line_box(page, ws[ka].line), line_box(page, ws[kb].line)
        height = s[3] - s[1]
        return s[1] < p[1] and p[1] - s[3] <= 1.5 * height and min(s[2], p[2]) > max(s[0], p[0])

    for street, locality in address_unit_mentions(gt):
        if street is None or locality is not None or "address_street_line" not in street.conditions:
            continue
        if any(wid not in at for seg in street.segments for wid in seg.words):
            continue  # an unknown word, reported as an error
        kinds = {r.role for r in address_roles(unit_words(gt, street, None))}
        if kinds and (kinds <= {"dx", "dx_number", "dx_exchange"} or kinds <= {"box_kind", "box_number", "box_extra"}):
            continue  # a DX line, or a box line with no place
        issues.append(Issue("warning", "street-line-without-locality",
                            f"mention {street.id}: a street line (condition address_street_line) with no locality "
                            f"'{street.id}-loc' of the same entity; the place written with it is its locality "
                            f"(LOCATION/{LOCALITY_SUBTYPE}, condition {ADDRESS_LOCALITY}), or drop the condition when "
                            f"no place is written with it. A place right after a label or phrase naming {PLACE_KINDS} "
                            "stays a place on its own only when no street line comes with it"))

    words = gt.word_index()
    streets = [m for m in gt.mentions if m.type is PiiType.ADDRESS and m.entity is not None]
    for m in gt.mentions:
        if m.type is not PiiType.LOCATION or m.subtype not in PLACE_SUBTYPES:
            continue
        first = ends(m)[0]
        texts = [words[wid][1].text[seg.char_start:seg.char_end] if seg.char_start is not None else words[wid][1].text
                 for seg in m.segments for wid in seg.words if wid in words]
        end = read_end(texts)
        if first is not None and end.state and end.postcode and not after_place_kind(first):
            issues.append(Issue("warning", "place-with-postcode-not-after-place-kind",
                                f"mention {m.id}: a place written with a state and postcode is the locality of an "
                                f"address (LOCATION/{LOCALITY_SUBTYPE}, condition {ADDRESS_LOCALITY}) unless a label "
                                f"or phrase naming {PLACE_KINDS} comes right before it; check its subtype "
                                f"'{m.subtype}'"))
        for street in streets:
            if street.entity == m.entity and right_after_or_below(street, m):
                issues.append(Issue("warning", "place-after-street-line",
                                    f"mention {m.id}: a place right after or below the street line {street.id} of the "
                                    f"same entity is that address's locality (LOCATION/{LOCALITY_SUBTYPE} "
                                    f"'{street.id}-loc', condition {ADDRESS_LOCALITY}); a label or phrase naming "
                                    f"{PLACE_KINDS} keeps only a place with no street line"))
                break
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
