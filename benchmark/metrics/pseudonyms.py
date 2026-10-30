"""Pseudonymisation / synthetic replacement checks (PLAN.md §3.8).

Replacement text is read from the *output* at each GT name word: from the text layer, or,
on a page with no text layer, from the measurement OCR (benchmark.ocr). With no OCR
installed such a page is reported as not evaluable. The replacements are then compared
component by component using the GT name structure.

An address is read per unit (eval-0.4, E6): the unit's read-back
(:func:`benchmark.evaluate.addresses.unit_readback`, both parts in reading order at their displayed
place, between the kept words around them), split into its street line and locality by the GT end
reader. Its checks: the postcode belongs to the state (any written form of the state), the postcode keeps
no original digit after the state's shared prefix, no fake word repeats an original address word of the
document in any role, and one address keeps one fake, within the document and across the matter. An
address is known by its identity (number, street name, canonical type, suburb and state; or box kind,
number and suburb; unit and level beneath), never by its GT entity, and the identity is stored hashed.
These checks are reported and never gate a document.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations

import hashlib

import numpy as np
import pymupdf
from rapidfuzz import fuzz

from .. import ocr
from ..evaluate.addresses import AddressUnit, Token, ocr_tokens, page_lines, page_tokens, unit_readback, units_of
from ..evaluate.config import EvalConfig
from ..evaluate.layers import PageText, find_at_place, page_chars, page_text, text_in_words
from ..geometry import as_poly, load_page_geometries
from ..gt.address_split import POSTCODES  # Australia Post postcode ranges by state/territory
from ..gt.address_split import (
    STATE_CODES,
    WHOLE,
    UnitWord,
    address_roles,
    canonical_street_type,
    postcode_in_state,
    read_end,
)
from ..gt.policy import Effective, Resolution
from ..gt.shared_roles import PREFIX
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
    # eval-0.4: one entry per address unit with a replaced part: {"unit", "identity" (hashed canonical
    # original), "fake" (hashed canonical fake, or None)}; matters compare them across documents.
    addresses: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


OTHER_TYPES = (PiiType.PHONE, PiiType.EMAIL, PiiType.ADDRESS, PiiType.ID_GOVERNMENT)
LOCALITY = "locality"  # the suburb, state and postcode of a split address (GT 0.2)


def _is_other(m) -> bool:
    """Contact details and identifiers whose replacement format is checked, and the locality
    of a split address, which is where its state and postcode are."""
    return m.type in OTHER_TYPES or (m.type is PiiType.LOCATION and (m.subtype or "") == LOCALITY)


_PHONE = re.compile(r"^(0[2-478]\d{8}|1[38]00\d{6}|13\d{4})$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def phone_valid(text: str) -> bool:
    digits = re.sub(r"[^\d+]", "", text)
    if digits.startswith("+61"):
        digits = "0" + digits[3:]
    return bool(_PHONE.match(digits))


def email_valid(text: str) -> bool:
    return bool(_EMAIL.match(text.strip(".,;:()")))


def _end(text: str):
    """The GT end reader on a fake's words: (state code, postcode) or None for each."""
    toks = text.split()
    while toks and not any(ch.isalnum() for ch in toks[-1]):
        toks.pop()
    r = read_end(toks, overseas=True)
    pc = None
    if r.postcode:
        pc = "".join(ch for ch in toks[r.postcode[-1]] if ch.isdigit())[-4:]
    return r.state_code, (pc if pc and len(pc) == 4 else None)


def postcode_state_valid(text: str) -> bool | None:
    """Does the postcode at the end belong to the state before it? The state in any written form: a code
    ('NSW', 'Qld', 'N.S.W.'), spelled out ('New South Wales'), or glued to its postcode ('QLD4132'). None
    when the text does not end in a state and a postcode."""
    state, pc = _end(text)
    if not state or not pc:
        return None
    return postcode_in_state(state, pc)


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
    unit: str | None = None  # the address unit of an address part (eval-0.4)
    part: str | None = None  # "street", "locality" or "whole"


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
    reps, _, any_text, _, _ = _read_all(gt, res, output)
    return reps, any_text


def _read_all(gt: GTDocument, res: Resolution, output: bytes, evaluation=None, config: EvalConfig | None = None):
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
    parts, idents = _address_parts(gt, res, doc, geoms, output, evaluation, config or EvalConfig(), notes)
    any_text |= any(text is not None for text, _, _ in parts.values())
    out, other = _entries(gt, res, texts, parts)
    return out, other, any_text, notes, idents


def _entries(gt: GTDocument, res: Resolution, texts: dict[str, str | None],
             parts: dict[str, tuple[str | None, str, str]]) -> tuple[list[ComponentReplacement], list[OtherReplacement]]:
    """The name components (and titles) of every required name, and every other required replacement, with the
    text read at each GT word (``texts``) or, for an address part, per unit (``parts``)."""
    words = gt.word_index()
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
        if m.id in parts:
            text, unit, part = parts[m.id]
            other.append(OtherReplacement(m.id, m.entity, m.type.value, m.subtype, m.text, text, unit, part))
            continue
        got = [texts.get(wid) for seg in m.segments for wid in seg.words]
        other.append(OtherReplacement(m.id, m.entity, m.type.value, m.subtype, m.text,
                                      " ".join(t for t in got if t) or None))
    return out, other


# -- addresses read per unit (eval-0.4, E6) -----------------------------------------------------------------

LOCALITY_ROLES = {"suburb", "state", "postcode", "state+postcode", "country", "capital_city_phrase"}
# Roles whose words a realistic fake keeps (the case table's structure roles), plus the street type a
# fake may share (owner, 2026-10-22): never an identifying word for the reuse check.
STRUCTURE_ROLES = {"designator", "box_kind", "corner", "via", "dx", "state", "country", "capital_city_phrase",
                   "street_type"}
_IDENTITY_MAIN = ("number", "building", "street_name", "street_type", "box_kind", "box_number", "dx", "dx_number",
                  "suburb", "state")
_IDENTITY_BENEATH = ("unit_number", "level_number")


def _norm(t: str) -> str:
    return "".join(ch for ch in t.casefold() if ch.isalnum() or ch in "/-")


def _role_words(texts: list[str]) -> list[tuple[str, str]]:
    """(text, role) of a fake's words, read by the GT role reader as one unsplit address."""
    roles = address_roles([UnitWord(f"t{k}", t, 0, WHOLE, 0, None) for k, t in enumerate(texts)])
    return [(t, r.role) for t, r in zip(texts, roles)]


def address_identity(words: list[tuple[str, str]]) -> str:
    """The canonical form of an address from its (text, role) words: number, building, street name, canonical
    street type, box kind and number, DX, suburb and state, then unit and level beneath; case folded,
    punctuation stripped, a glued state split from its postcode, a '5/3' number split into unit and number."""
    parts: dict[str, list[str]] = defaultdict(list)
    state: list[str] = []
    for text, role in words:
        n = _norm(text)
        if not n:
            continue
        if role == "street_type":
            parts[role].append(canonical_street_type(text) or n)
        elif role == "state":
            state.append(text.strip(_STRIP).casefold().replace(".", ""))
        elif role == "state+postcode":
            letters = "".join(ch for ch in n if ch.isalpha())
            parts["state"].append(STATE_CODES.get(letters, letters.upper()))
        elif role == "unit_number+number" and "/" in n:
            unit, num = n.split("/", 1)
            parts["unit_number"].append(unit)
            parts["number"].append(num)
        elif role in _IDENTITY_MAIN or role in _IDENTITY_BENEATH:
            parts[role].append(n)
    if state:  # a state in any written form, doubled or spelled out over several words: its code
        whole = " ".join(state)
        parts["state"].append(STATE_CODES.get(whole) or STATE_CODES.get(state[0]) or whole.upper())
    main = "|".join(" ".join(parts[r]) for r in _IDENTITY_MAIN)
    beneath = "|".join(" ".join(parts[r]) for r in _IDENTITY_BENEATH)
    return f"{main}#{beneath}"


def _key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _kept_places(gt: GTDocument, res: Resolution, page_texts: dict[int, PageText], evaluation, cfg: EvalConfig
                 ) -> tuple[dict[str, np.ndarray], dict[int, tuple[float, float]]]:
    """Kept words at their displayed place (the anchors of a unit's read-back) and the page shifts in pt: from
    the evaluation's word statuses (PRESERVED in place, MOVED at their found place), or, with no evaluation,
    the kept words found in the output's text at their GT place."""
    kept = (Effective.ORDINARY.value, Effective.MUST_NOT.value)
    places: dict[str, np.ndarray] = {}
    shifts: dict[int, tuple[float, float]] = {}
    if evaluation is not None:
        s = cfg.dpi / 72.0
        for p in evaluation.pages:
            if p.registration_shift_px:
                shifts[p.index] = (p.registration_shift_px[0] / s, p.registration_shift_px[1] / s)
        words = gt.word_index()
        for wo in evaluation.words:
            if wo.requirement not in kept or wo.word_id not in words:
                continue
            poly = as_poly(words[wo.word_id][1].poly).astype(float)
            if wo.status.value == "PRESERVED":
                places[wo.word_id] = poly
            elif wo.status.value == "MOVED" and wo.scale is not None:
                c = poly.mean(axis=0)
                places[wo.word_id] = wo.scale * (poly - c) + c + np.array([wo.dx_pt or 0.0, wo.dy_pt or 0.0])
        return places, shifts
    for page in gt.pages:
        pt = page_texts.get(page.index)
        if not pt:
            continue
        for w in page.words:
            if res.words.get(w.id, Effective.ORDINARY).value in kept and any(ch.isalnum() for ch in w.text) and \
                    find_at_place(w.poly, w.text, pt, (0.0, 0.0), None, cfg.searchable_band,
                                  cfg.searchable_margin, cfg.text_ratio) is not None:
                places[w.id] = as_poly(w.poly)
    return places, shifts


def _address_parts(gt: GTDocument, res: Resolution, doc, geoms, output: bytes, evaluation, cfg: EvalConfig,
                   notes: list[str]) -> tuple[dict[str, tuple[str | None, str, str]], list[dict]]:
    """Mention id -> (its fake as read per unit, unit key, part) for every address part a MUST rule replaces,
    and per unit its hashed identity and hashed canonical fake."""
    units = [u for u in units_of(gt) if any(res.mentions[m.id].requirement is Requirement.MUST for m in u.mentions)]
    if not units:
        return {}, []
    page_texts: dict[int, PageText] = {}
    tokens: dict[int, list[Token]] = {}
    for pg in sorted({p for u in units for p, _ in u.lines}):
        if pg >= doc.page_count:
            continue
        gp = gt.page(pg)
        ow, oh = geoms[pg].display_size
        scale = (gp.width_pt / ow, gp.height_pt / oh) if ow and oh else (1.0, 1.0)
        page_texts[pg] = page_text(doc[pg], geoms[pg], scale)
        if len(page_texts[pg]):
            tokens[pg] = page_tokens(page_texts[pg], pg)
            continue
        try:  # image-only page: the measurement OCR, as for names
            words = ocr.ocr_page(output, pg)
        except RuntimeError as e:
            notes.append(f"page {pg}: measurement OCR failed: {e}")
            words = None
        tokens[pg] = ocr_tokens(words, pg, scale)
    places, shifts = _kept_places(gt, res, page_texts, evaluation, cfg)
    lines = page_lines(gt, {})
    words = gt.word_index()
    parts: dict[str, tuple[str | None, str, str]] = {}
    idents: list[dict] = []
    for u in units:
        rb = unit_readback(u, gt, tokens, places, shifts, {}, cfg, lines)
        idents.append(_unit_parts(u, [t.text for t in rb], res, words, parts))
    return parts, idents


def _unit_parts(u: AddressUnit, texts: list[str], res: Resolution, words: dict,
                parts: dict[str, tuple[str | None, str, str]]) -> dict:
    """One unit's read-back (``texts``, in reading order) cut into its street line and locality where the GT
    role reader puts the locality's first word: each required part's fake goes into ``parts``. Returns the
    unit's hashed identity and hashed canonical fake."""
    roles = _role_words(texts) if texts else []
    cut = next((k for k, (_, r) in enumerate(roles) if r in LOCALITY_ROLES), len(roles))
    street_txt = " ".join(texts[:cut]) or None
    loc_txt = " ".join(texts[cut:]) or None
    for m in u.mentions:
        if res.mentions[m.id].requirement is not Requirement.MUST:
            continue
        if u.street is not None and u.locality is not None:
            part = "street" if m is u.street else "locality"
            parts[m.id] = (street_txt if part == "street" else loc_txt, u.key, part)
        else:
            parts[m.id] = (" ".join(texts) or None, u.key, "locality" if m is u.locality else "whole")
    original = [(words[w][1].text, u.roles[w]) for w in u.word_ids if w in u.roles]
    return {"unit": u.key, "identity": _key(address_identity(original)),
            "fake": _key(address_identity(roles)) if roles else None}


def _prefix(original: str) -> int:
    """How many leading digits of an original postcode a fake may keep: those matching its state's prefix
    ('2' NSW and ACT, '3' VIC, '4' QLD, '5' SA, '6' WA, '7' TAS, '08' NT), from the left."""
    state, pc = _end(original)
    pre = PREFIX.get(state or "", "")
    if not pc:
        return 0
    k = 0
    while k < len(pre) and pc[k] == pre[k]:
        k += 1
    return k


def postcode_keeps_digits(original: str, fake: str) -> bool | None:
    """Does a fake postcode keep an original digit at its position after the shared prefix? None when either
    has no state and postcode at its end."""
    _, o = _end(original)
    _, f = _end(fake)
    if not o or not f:
        return None
    k = _prefix(original)
    return any(a == b for a, b in zip(o[k:], f[k:]))


def reuses_original(fake: str, words: set[str], numbers: set[str]) -> bool:
    """Does a fake word equal, begin with or (for words of 3 or more letters) contain an identifying word of
    an address of the document, or a number of the fake equal one of its numbers (whole digit runs)?"""
    for tok in fake.split():
        n = _norm(tok)
        if not n:
            continue
        if any(run in numbers for run in re.findall(r"\d+", n)):
            return True
        letters = "".join(ch for ch in n if ch.isalpha())
        for w in words:
            if letters == w or (len(w) >= 3 and (letters.startswith(w) or w in letters)):
                return True
    return False


def address_words(gt: GTDocument) -> tuple[set[str], set[str]]:
    """The identifying words (letters, case folded) and numbers (whole digit runs) of every address unit of
    the document, in any role: every unit word outside the structure roles."""
    words = gt.word_index()
    letters, numbers = set(), set()
    for u in units_of(gt):
        for w in u.word_ids:
            role = u.roles.get(w)
            if role is None or role in STRUCTURE_ROLES:
                continue
            n = _norm(words[w][1].text)
            numbers.update(re.findall(r"\d+", n))
            if role in ("state+postcode", "postcode"):
                continue
            a = "".join(ch for ch in n if ch.isalpha())
            if a:
                letters.add(a)
    return letters, numbers


def evaluate_pseudonyms(gt: GTDocument, res: Resolution, output: bytes, evaluation=None,
                        config: EvalConfig | None = None) -> PseudonymResult:
    """``evaluation``: the output's evaluation, whose word statuses place the kept words that bound an address
    unit's read-back (without it, the kept words found in the text at their GT place)."""
    return _checks(gt, *_read_all(gt, res, output, evaluation, config))


def evaluate_text_pseudonyms(gt: GTDocument, res: Resolution, texts: dict[str, str | None],
                             units: dict[str, list[str]]) -> PseudonymResult:
    """The same checks on a tool's redacted text (eval-0.4, the text-level path): ``texts`` is the text the
    diff puts at each GT word (benchmark.evaluate.textlevel.word_replacements), ``units`` each address unit's
    new text as whitespace tokens (its read-back)."""
    words = gt.word_index()
    parts: dict[str, tuple[str | None, str, str]] = {}
    idents = [_unit_parts(u, units.get(u.key, []), res, words, parts) for u in units_of(gt)
              if any(res.mentions[m.id].requirement is Requirement.MUST for m in u.mentions)]
    reps, other = _entries(gt, res, texts, parts)
    return _checks(gt, reps, other, True, [], idents)


def _checks(gt: GTDocument, reps: list[ComponentReplacement], other: list[OtherReplacement], any_text: bool,
            notes: list[str], idents: list[dict]) -> PseudonymResult:
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
    checks_other, invalid = _validity(other, names, *address_words(gt))
    checks.update(checks_other)
    issues["invalid"] = invalid
    # One address, one fake within the document: units of the same identity read the same canonical fake.
    by_identity: dict[str, set] = defaultdict(set)
    count: dict[str, int] = defaultdict(int)
    for a in idents:
        count[a["identity"]] += 1
        if a["fake"]:
            by_identity[a["identity"]].add(a["fake"])
    repeated = [k for k, n in count.items() if n > 1]
    checks["address_value_consistency_rate"] = (
        round(sum(1 for k in repeated if len(by_identity[k]) <= 1) / len(repeated), 4) if repeated else None)
    issues["address_inconsistent"] = sorted(k for k in repeated if len(by_identity[k]) > 1)
    return PseudonymResult(True, None, reps, checks, issues, other, idents)


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


def _validity(other: list[OtherReplacement], names: list[ComponentReplacement], addr_words: set[str] = frozenset(),
              addr_numbers: set[str] = frozenset()) -> tuple[dict, dict]:
    """Format validity of replaced contact details and identifiers. Only mentions whose
    output text differs from the original are checked (removed ones have nothing to check).
    An address part also keeps no postcode digit after the state's prefix and repeats no identifying
    word or number of the document's addresses (``addr_words``, ``addr_numbers``) in any role."""
    new_names: dict[str, set[str]] = defaultdict(set)
    old_names: dict[str, set[str]] = defaultdict(set)
    for r in names:
        if r.component in ("given", "surname", "nickname") and r.entity:
            old_names[r.entity].add(_core(r.original).casefold())
            if r.replacement and _core(r.replacement):
                new_names[r.entity].add(_core(r.replacement).casefold())
    phone, email, derived, postcode, medicare = [], [], [], [], []
    kept_digits = reused = 0
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
        elif o.type in (PiiType.ADDRESS.value, PiiType.LOCATION.value):  # an address part, or a locality
            ok = postcode_state_valid(new)
            if ok is not None:
                postcode.append(ok)
                if not ok:
                    invalid[o.mention_id] = "postcode does not belong to the state"
            if postcode_keeps_digits(o.original, new):
                kept_digits += 1
                invalid.setdefault(o.mention_id, "postcode keeps original digits")
            if reuses_original(new, addr_words, addr_numbers):
                reused += 1
                invalid.setdefault(o.mention_id, "fake repeats an original address word")
        elif o.type == PiiType.ID_GOVERNMENT.value and (o.subtype or "") == "medicare":
            ok = medicare_valid(new)
            medicare.append(ok)
            if not ok:
                invalid[o.mention_id] = "medicare checksum"
    return {"other_replaced": replaced, "phone_valid_rate": _rate(phone), "email_valid_rate": _rate(email),
            "email_from_new_name_rate": _rate(derived), "postcode_state_valid_rate": _rate(postcode),
            "medicare_checksum_valid_rate": _rate(medicare), "postcode_shared_digits": kept_digits,
            "fake_reuses_original": reused, "invalid_replacements": len(invalid)}, invalid


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
