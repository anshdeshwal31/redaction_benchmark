"""Document-level leakage: PII outside the page content (PLAN.md §5.3).

Searches metadata values, XMP, outline titles, annotation contents, form fields,
attachment names, structure-tree alternate text and unreferenced streams for terms
taken from required GT mentions, and checks GT surface items at their locations.

Findings contain the matched PII term, so result files are sensitive by default.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pikepdf
import pymupdf

from ..forensics.inspect import _reachable_objgens, _struct_alt_texts
from ..gt.policy import Effective, Resolution, resolve_fact
from ..schemas import GTDocument, PolicyProfile, Requirement

_MIN_TERM = 3


@dataclass
class DocFinding:
    location: str
    term: str
    mention_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class SurfaceOutcome:
    surface_id: str
    location: str
    requirement: Requirement
    pending: str | None
    leaked: bool
    found_at: list[str]


def document_strings(output: bytes) -> list[tuple[str, str]]:
    """(location, text) for every document-level string a reader could recover."""
    out: list[tuple[str, str]] = []
    doc = pymupdf.open("pdf", output)
    for k, v in (doc.metadata or {}).items():
        if v and k not in ("format", "encryption"):
            out.append((f"info:{k}", str(v)))
    xmp = doc.get_xml_metadata() or ""
    if xmp.strip():
        out.append(("xmp", xmp))
    for entry in doc.get_toc(simple=True):
        out.append(("outline", entry[1]))
    for page in doc:
        for an in page.annots() or []:
            for key in ("content", "title", "subject"):
                if an.info.get(key):
                    out.append((f"annotation:{key}", an.info[key]))
        for w in page.widgets() or []:
            if w.field_value not in (None, ""):
                out.append(("form_field", str(w.field_value)))
    for name in doc.embfile_names():
        out.append(("attachment_name", name))
    with pikepdf.open(io.BytesIO(output)) as pk:
        for alt in _struct_alt_texts(pk):
            out.append(("alt_text", alt))
        reachable = _reachable_objgens(pk)
        for obj in pk.objects:
            if isinstance(obj, pikepdf.Stream) and obj.is_indirect and obj.objgen not in reachable:
                try:
                    out.append(("unreferenced_stream", obj.read_bytes().decode("latin-1")))
                except Exception:  # noqa: BLE001 - undecodable stream: nothing searchable
                    continue
    return out


def _search_terms(gt: GTDocument, res: Resolution) -> dict[str, tuple[set[str], set[str]]]:
    """Distinctive terms of required mentions -> (mention ids, entity ids).

    A token is distinctive when it never appears as an ordinary or protected word
    elsewhere in the document (so a surname that is also an English word is skipped).
    """
    words = gt.word_index()
    common = {w.text.casefold().strip(".,;:()") for wid, (_, w) in words.items()
              if res.words.get(wid) in (Effective.ORDINARY, Effective.MUST_NOT)}
    terms: dict[str, tuple[set[str], set[str]]] = {}
    for m in gt.mentions:
        if res.mentions[m.id].requirement is not Requirement.MUST:
            continue
        candidates = [m.text] if len(m.text) >= _MIN_TERM else []
        for seg in m.segments:
            for wid in seg.words:
                t = words[wid][1].text
                if seg.char_start is not None:
                    t = t[seg.char_start:seg.char_end]
                candidates.append(t.strip(".,;:()"))
        for t in candidates:
            key = t.casefold()
            if len(key) < _MIN_TERM or key in common:
                continue
            mids, eids = terms.setdefault(key, (set(), set()))
            mids.add(m.id)
            if m.entity:
                eids.add(m.entity)
    return terms


def _contains(haystack: str, term: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", haystack, flags=re.IGNORECASE) is not None


def document_findings(
    output: bytes, gt: GTDocument, res: Resolution, policy: PolicyProfile
) -> tuple[list[DocFinding], list[SurfaceOutcome]]:
    strings = document_strings(output)
    findings: list[DocFinding] = []
    for term, (mids, eids) in sorted(_search_terms(gt, res).items()):
        for location, text in strings:
            if _contains(text, term):
                findings.append(DocFinding(location, term, sorted(mids), sorted(eids)))
    surfaces: list[SurfaceOutcome] = []
    for item in gt.surface:
        r = resolve_fact(gt, policy, item.type, None, item.entity)
        found = sorted({loc for loc, text in strings if _contains(text, item.text)})
        surfaces.append(SurfaceOutcome(item.id, item.location, r.requirement, r.pending, bool(found), found))
    return findings, surfaces
