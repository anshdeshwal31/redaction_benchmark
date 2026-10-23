"""Pseudonymisation / synthetic replacement checks (PLAN.md §3.8).

Replacement text is read from the *output* at each GT name word: from the text layer, or,
on a page with no text layer, from the measurement OCR (benchmark.ocr). With no OCR
installed such a page is reported as not evaluable. The replacements are then compared
component by component using the GT name structure.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np
import pymupdf
from rapidfuzz import fuzz

from .. import ocr
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
    other: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


OTHER_TYPES = (PiiType.PHONE, PiiType.EMAIL, PiiType.ADDRESS, PiiType.ID_GOVERNMENT)
LOCALITY = "locality"  # the suburb, state and postcode of a split address (GT 0.2)


def _is_other(m) -> bool:
    """Contact details and identifiers whose replacement format is checked, and the locality
    of a split address, which is where its state and postcode are."""
    return m.type in OTHER_TYPES or (m.type is PiiType.LOCATION and (m.subtype or "") == LOCALITY)


# Australia Post postcode ranges by state/territory.
POSTCODES = {
    "NSW": [(1000, 2599), (2619, 2899), (2921, 2999)],
    "ACT": [(200, 299), (2600, 2618), (2900, 2920)],
    "VIC": [(3000, 3999), (8000, 8999)],
    "QLD": [(4000, 4999), (9000, 9999)],
    "SA": [(5000, 5999)],
    "WA": [(6000, 6999)],
    "TAS": [(7000, 7999)],
    "NT": [(800, 999)],
}
_PHONE = re.compile(r"^(0[2-478]\d{8}|1[38]00\d{6}|13\d{4})$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_STATE_POSTCODE = re.compile(r"\b(NSW|ACT|VIC|QLD|SA|WA|TAS|NT)\b[\s,]*(\d{4})\b", re.I)


def phone_valid(text: str) -> bool:
    digits = re.sub(r"[^\d+]", "", text)
    if digits.startswith("+61"):
        digits = "0" + digits[3:]
    return bool(_PHONE.match(digits))


def email_valid(text: str) -> bool:
    return bool(_EMAIL.match(text.strip(".,;:()")))


def postcode_state_valid(text: str) -> bool | None:
    m = _STATE_POSTCODE.search(text)
    if not m:
        return None
    pc = int(m.group(2))
    return any(lo <= pc <= hi for lo, hi in POSTCODES[m.group(1).upper()])


def medicare_valid(text: str) -> bool:
    """Medicare card number: 10 (or 11 with IRN) digits, first digit 2-6, check digit =
    weighted sum (1,3,7,9,1,3,7,9) of the first 8 digits mod 10."""
    d = re.sub(r"\D", "", text)
    if len(d) not in (10, 11) or d[0] not in "23456":
        return False
    return sum(int(x) * w for x, w in zip(d[:8], (1, 3, 7, 9, 1, 3, 7, 9))) % 10 == int(d[8])


@dataclass
class OtherReplacement:
    mention_id: str
    entity: str | None
    type: str
    subtype: str | None
    original: str
    replacement: str | None


def _ocr_chars(output: bytes, page_index: int, notes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Character centres (display_pt) of an image-only output page, read by the measurement OCR.

    OCR returns words, not characters: each word's characters are spread evenly along its
    reading direction, which is enough to assign them to the GT word they fall in.
    """
    try:
        words = ocr.ocr_page(output, page_index)
    except RuntimeError as e:
        notes.append(f"page {page_index}: measurement OCR failed: {e}")
        return np.zeros((0, 2)), []
    if words is None:
        notes.append(f"page {page_index}: no text layer and no measurement OCR installed")
        return np.zeros((0, 2)), []
    pts, chars = [], []
    for w in words:
        x0, y0, x1, y1 = w.bbox
        quad = np.asarray(w.poly, float) if w.poly else np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        n = len(w.text)
        for k, ch in enumerate(w.text):
            t = (k + 0.5) / n
            a = quad[0] + (quad[1] - quad[0]) * t
            b = quad[3] + (quad[2] - quad[3]) * t
            pts.append((a + b) / 2)
            chars.append(ch)
    return (np.array(pts) if pts else np.zeros((0, 2))), chars


def _read_replacements(gt: GTDocument, res: Resolution, output: bytes) -> tuple[list[ComponentReplacement], bool]:
    reps, _, any_text, _ = _read_all(gt, res, output)
    return reps, any_text


def _read_all(gt: GTDocument, res: Resolution, output: bytes):
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
        if (m.type is PiiType.PERSON_NAME or _is_other(m)) and res.mentions[m.id].requirement is Requirement.MUST:
            for seg in m.segments:
                wanted[seg.page].update(seg.words)
            for wid in m.optional_words:
                wanted[words[wid][0]].add(wid)
    texts: dict[str, str] = {}
    any_text = False
    notes: list[str] = []
    for pg, wids in wanted.items():
        if pg >= doc.page_count:
            continue
        centres, chars = page_chars(doc[pg], geoms[pg])
        if not chars:  # image-only page: read the replacement text with the measurement OCR
            centres, chars = _ocr_chars(output, pg, notes)
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
    other = []
    for m in gt.mentions:
        if not _is_other(m) or res.mentions[m.id].requirement is not Requirement.MUST:
            continue
        got = [texts.get(wid) for seg in m.segments for wid in seg.words]
        other.append(OtherReplacement(m.id, m.entity, m.type.value, m.subtype, m.text,
                                      " ".join(t for t in got if t) or None))
    return out, other, any_text, notes


def evaluate_pseudonyms(gt: GTDocument, res: Resolution, output: bytes) -> PseudonymResult:
    reps, other, any_text, notes = _read_all(gt, res, output)
    names = [r for r in reps if r.component != "title"]
    if not any_text:
        why = "; ".join(notes) or "output has no text layer and the measurement OCR found no text"
        return PseudonymResult(False, f"{why}; replacement text cannot be read", reps, {})
    entities = {e.id: e for e in gt.entities}
    real_values = {_core(r.original).casefold() for r in names}
    survived, missing, similar, style_lost = [], [], [], []
    value_map: dict[tuple, set[str]] = defaultdict(set)
    entity_map: dict[tuple, set[str]] = defaultdict(set)
    # (component, fake) -> the (person, original value) pairs it replaced
    replacement_owner: dict[tuple, set[tuple[str, str]]] = defaultdict(set)
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
        replacement_owner[(r.component, n.casefold())].add((ent, o.casefold()))
    value_inconsistent = {f"{e}:{c}:{v}": sorted(s) for (e, c, v), s in value_map.items() if len(s) > 1}
    entity_inconsistent = {f"{e}:{c}": sorted(s) for (e, c), s in entity_map.items() if len(s) > 1}
    collisions = _collisions(replacement_owner)
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
    checks_other, invalid = _validity(other, names)
    checks.update(checks_other)
    issues["invalid"] = invalid
    return PseudonymResult(True, None, reps, checks, issues, other)


def _collisions(owners: dict[tuple[str, str], set[tuple[str, str]]]) -> dict[str, list[str]]:
    """Fakes that stand for two different people with different names: one value (per name
    component) given to two people who share none of the names it replaced. One person written two
    ways, or two people who share a name (a married couple's surname), may share a fake, also when
    both happen at once (McDonald and Macdonald for him, McDonald for her).
    ``owners``: (component, fake) -> {(person, original)}; the result is keyed "component:fake"."""
    out = {}
    for (component, fake), pairs in owners.items():
        names: dict[str, set[str]] = defaultdict(set)
        for person, original in pairs:
            names[person].add(original)
        if any(not (a & b) for a, b in combinations([names[p] for p in sorted(names)], 2)):
            out[f"{component}:{fake}"] = sorted({o for _, o in pairs})
    return out


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def _validity(other: list[OtherReplacement], names: list[ComponentReplacement]) -> tuple[dict, dict]:
    """Format validity of replaced contact details and identifiers. Only mentions whose
    output text differs from the original are checked (removed ones have nothing to check)."""
    new_names: dict[str, set[str]] = defaultdict(set)
    old_names: dict[str, set[str]] = defaultdict(set)
    for r in names:
        if r.component in ("given", "surname", "nickname") and r.entity:
            old_names[r.entity].add(_core(r.original).casefold())
            if r.replacement and _core(r.replacement):
                new_names[r.entity].add(_core(r.replacement).casefold())
    phone, email, derived, postcode, medicare = [], [], [], [], []
    invalid: dict[str, str] = {}
    replaced = 0
    for o in other:
        new = (o.replacement or "").strip()
        if not new or _core(o.original).casefold() in new.casefold():
            continue
        replaced += 1
        if o.type == PiiType.PHONE.value:
            ok = phone_valid(new)
            phone.append(ok)
            if not ok:
                invalid[o.mention_id] = "phone format"
        elif o.type == PiiType.EMAIL.value:
            ok = email_valid(new)
            email.append(ok)
            if not ok:
                invalid[o.mention_id] = "email syntax"
            elif o.entity and new_names.get(o.entity):
                local = new.split("@")[0].casefold()
                ok_d = any(n in local for n in new_names[o.entity]) and                     not any(n in local for n in old_names[o.entity])
                derived.append(ok_d)
                if not ok_d:
                    invalid[o.mention_id] = "email not derived from the new name"
        elif o.type in (PiiType.ADDRESS.value, PiiType.LOCATION.value):  # an unsplit address, or a locality
            ok = postcode_state_valid(new)
            if ok is not None:
                postcode.append(ok)
                if not ok:
                    invalid[o.mention_id] = "postcode does not belong to the state"
        elif o.type == PiiType.ID_GOVERNMENT.value and (o.subtype or "") == "medicare":
            ok = medicare_valid(new)
            medicare.append(ok)
            if not ok:
                invalid[o.mention_id] = "medicare checksum"
    return {"other_replaced": replaced, "phone_valid_rate": _rate(phone), "email_valid_rate": _rate(email),
            "email_from_new_name_rate": _rate(derived), "postcode_state_valid_rate": _rate(postcode),
            "medicare_checksum_valid_rate": _rate(medicare), "invalid_replacements": len(invalid)}, invalid


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
