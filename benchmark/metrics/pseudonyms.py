"""Pseudonymisation / synthetic replacement checks (PLAN.md §3.8).

Replacement text is read from the *output* at each GT name word (text layer; image-only
outputs need the measurement OCR, which is reported as not evaluable until installed),
then compared component by component using the GT name structure.
"""

from __future__ import annotations

import string
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import pymupdf
from rapidfuzz import fuzz

from ..evaluate.layers import page_chars, text_in_words
from ..geometry import load_page_geometries
from ..gt.policy import Resolution
from ..schemas import GTDocument, NameComponent, PiiType, Requirement

_STRIP = string.punctuation + "‘’“”"
_FEMALE_TITLES = {"ms", "mrs", "miss"}
_MALE_TITLES = {"mr"}


def _core(s: str) -> str:
    return s.strip(_STRIP)


def _style(s: str) -> str:
    c = _core(s)
    if len(c) > 1 and c.isupper():
        return "upper"
    if c[:1].isupper():
        return "title"
    return "lower"


@dataclass
class ComponentReplacement:
    mention_id: str
    entity: str | None
    word_id: str
    component: str
    original: str
    replacement: str | None


@dataclass
class PseudonymResult:
    evaluable: bool
    reason: str | None
    replacements: list[ComponentReplacement]
    checks: dict
    issues: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_replacements(gt: GTDocument, res: Resolution, output: bytes) -> tuple[list[ComponentReplacement], bool]:
    import os
    import tempfile

    doc = pymupdf.open("pdf", output)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(output)
        path = f.name
    try:
        geoms = load_page_geometries(path)
    finally:
        os.unlink(path)
    words = gt.word_index()
    wanted: dict[int, set[str]] = defaultdict(set)
    for m in gt.mentions:
        if m.type is PiiType.PERSON_NAME and res.mentions[m.id].requirement is Requirement.MUST:
            for seg in m.segments:
                wanted[seg.page].update(seg.words)
            for wid in m.optional_words:
                wanted[words[wid][0]].add(wid)
    texts: dict[str, str] = {}
    any_text = False
    for pg, wids in wanted.items():
        if pg >= doc.page_count:
            continue
        centres, chars = page_chars(doc[pg], geoms[pg])
        any_text |= bool(chars)
        page = gt.page(pg)
        texts.update(text_in_words(page, centres, chars, wids))
    out = []
    for m in gt.mentions:
        if m.type is not PiiType.PERSON_NAME or res.mentions[m.id].requirement is not Requirement.MUST:
            continue
        for comp, wids in m.components.items():
            for wid in wids:
                text = words[wid][1].text
                seg = next((s for s in m.segments if wid in s.words), None)
                if seg is not None and seg.char_start is not None:
                    text = text[seg.char_start:seg.char_end]
                out.append(ComponentReplacement(m.id, m.entity, wid, comp.value, text, texts.get(wid)))
        for wid in m.optional_words:
            out.append(ComponentReplacement(m.id, m.entity, wid, "title", words[wid][1].text, texts.get(wid)))
    return out, any_text


def evaluate_pseudonyms(gt: GTDocument, res: Resolution, output: bytes) -> PseudonymResult:
    reps, any_text = _read_replacements(gt, res, output)
    names = [r for r in reps if r.component != "title"]
    if not any_text:
        return PseudonymResult(False, "output has no text layer; replacement text needs the measurement OCR",
                               reps, {})
    entities = {e.id: e for e in gt.entities}
    real_values = {_core(r.original).casefold() for r in names}
    survived, missing, similar, style_lost = [], [], [], []
    value_map: dict[tuple, set[str]] = defaultdict(set)
    entity_map: dict[tuple, set[str]] = defaultdict(set)
    replacement_owner: dict[tuple, set[str]] = defaultdict(set)
    for r in names:
        if r.replacement is None or not _core(r.replacement):
            missing.append(r.word_id)
            continue
        o, n = _core(r.original), _core(r.replacement)
        if o.casefold() in n.casefold():
            survived.append(r.word_id)
            continue
        if fuzz.ratio(o.casefold(), n.casefold()) >= 80:
            similar.append((r.original, r.replacement))
        if _style(r.original) != _style(r.replacement):
            style_lost.append(r.word_id)
        ent = r.entity or r.mention_id
        value_map[(ent, r.component, o.casefold())].add(n.casefold())
        entity_map[(ent, r.component)].add(n.casefold())
        replacement_owner[(r.component, n.casefold())].add(o.casefold())
    value_inconsistent = {f"{e}:{c}:{v}": sorted(s) for (e, c, v), s in value_map.items() if len(s) > 1}
    entity_inconsistent = {f"{e}:{c}": sorted(s) for (e, c), s in entity_map.items() if len(s) > 1}
    collisions = {f"{c}:{n}": sorted(o) for (c, n), o in replacement_owner.items() if len(o) > 1}
    real_name = sorted({n for (c, n) in replacement_owner if n in real_values})
    gender_mismatch = []
    for r in reps:
        if r.component != "title" or not r.replacement:
            continue
        title = _core(r.replacement).casefold()
        e = entities.get(r.entity or "")
        g = e.attributes["gender"].value if e and "gender" in e.attributes else None
        if g and ((g == "F" and title in _MALE_TITLES) or (g == "M" and title in _FEMALE_TITLES)):
            gender_mismatch.append(r.mention_id)
    replaced = len(names) - len(missing) - len(survived)
    checks = {
        "name_components": len(names),
        "replaced": replaced,
        "survived": len(survived),
        "no_replacement_text": len(missing),
        "value_consistency_rate": round(1 - len(value_inconsistent) / len(value_map), 4) if value_map else None,
        "entity_consistency_rate": round(1 - len(entity_inconsistent) / len(entity_map), 4) if entity_map else None,
        "collisions": len(collisions),
        "collides_with_real_name": len(real_name),
        "too_similar": len(similar),
        "case_style_lost": len(style_lost),
        "honorific_gender_mismatch": len(gender_mismatch),
    }
    issues = {"survived": survived, "value_inconsistent": value_inconsistent,
              "entity_inconsistent": entity_inconsistent, "collisions": collisions,
              "collides_with_real_name": real_name, "too_similar": similar,
              "honorific_gender_mismatch": gender_mismatch}
    return PseudonymResult(True, None, reps, checks, issues)


def replacements_by_mention(result: PseudonymResult) -> dict[str, dict[str, str]]:
    """mention -> {component: replacement value}, for inferring resolution from pseudonyms.

    Only mentions carrying a given name or surname are returned: a nickname-only or
    initials-only mention shares no component with the others, so its linkage is not
    observable from the output and must not be scored as a split.
    """
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for r in result.replacements:
        if r.component != "title" and r.replacement and _core(r.replacement):
            out[r.mention_id][r.component] = _core(r.replacement)
    return {m: c for m, c in out.items() if {"given", "surname"} & set(c)}
