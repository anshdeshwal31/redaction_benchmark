"""Shared structure: the words and characters of a mention that a realistic fake keeps (eval-0.4;
docs/metrics.md, "Shared words").

The project owner decided on 2026-10-22 that a realistic fake may share a word with the original at the
same place (the state when the fake suburb is in the same state, a street-type word, the 'PO Box' words, a
title), and on 2026-10-29 that a fake organisation may keep its trailing legal form, a fake may keep an
email domain and an identifier's letter prefix, and a fake phone number its national structure prefix.
Those parts are read from GT alone, by their position in the mention, never from a tool's output or
report, and only for the roles the policy enables for the mention's type and subtype
(``PolicyProfile.shared_roles``; policy v0 enables none):

- ADDRESS (street line or unsplit) and LOCATION/locality, through :func:`address_split.address_roles`:
  box_kind, designator, corner, via, dx, country, capital_city_phrase; street_type only as the street's
  last type word after a name word that is neither a number nor 'The', never in a corner form (two
  streets) and never a building word; the state in any written form at the end, always with the
  postcode's prefix digits as a PAIR (the original's leading digits matching the state's prefix, '2'
  NSW and ACT, '3' VIC, '4' QLD, '5' SA, '6' WA, '7' TAS, '08' NT, from the left); a glued 'NSW2750'
  shares its letters and prefix digits; a postcode with no state word shares the prefix of the state
  its Australia Post range gives, as plain characters (no pair).
- ORGANISATION legal_form: the trailing Pty, Ltd, Limited, Proprietary, Inc or Incorporated after a name word.
- EMAIL email_domain: '@' and the domain after a non-empty local part.
- ID_* id_prefix: 1-4 capitals with their separator at the start of the first word when a digit follows in
  the word, or a first word of 1-4 capitals followed by a word with a digit.
- PHONE phone_prefix: a mobile's '04' (10 digits), '+61', the '13' of a 6-digit number, '1300' and '1800'
  (10 digits), a bracketed area code '(0X)'.

A word with shared characters has an identifying slice (``needle_cut``): an email's local part, an
identifier's or a phone number's other digits. The evaluator searches the text layer with that slice
only and looks for the slice on its own when it is not in place, so a changed domain or prefix cannot
hide a kept local part or kept digits. A mention whose every letter and digit would be shared gets
nothing (it is judged in full).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import GTDocument, GTMention, PiiType, PolicyProfile
from ..schemas.policy import SHARED_ROLE_NAMES
from .address_split import (
    LOCALITY_PART,
    POSTCODES,
    STATE_CODES,
    STREET_PART,
    WHOLE,
    address_roles,
    address_unit_mentions,
    unit_words,
)

ROLE_NAMES = SHARED_ROLE_NAMES
STRIP = ".,;:()"
# The postcode prefix of each state, matched from the left on the original postcode.
PREFIX = {"NSW": "2", "ACT": "2", "VIC": "3", "QLD": "4", "SA": "5", "WA": "6", "TAS": "7", "NT": "08"}
ADDRESS_WORD_ROLES = {"box_kind", "designator", "corner", "via", "dx", "country", "capital_city_phrase",
                      "street_type", "state"}
LEGAL_FORMS = {"pty", "ltd", "limited", "proprietary", "inc", "incorporated"}


@dataclass
class SharedUnits:
    """The shared parts of one mention: whole words (word id -> role), characters (word id, index),
    the state/postcode pairs (state word id, postcode word id, prefix character indices) and each
    word's identifying slice [a, b) when it has shared characters."""

    full: dict[str, str] = field(default_factory=dict)
    chars: set[tuple[str, int]] = field(default_factory=set)
    pairs: list[tuple[str, str, tuple[int, ...]]] = field(default_factory=list)
    needle_cut: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.full or self.chars)

    def shared(self, unit: tuple[str, int]) -> bool:
        return unit[0] in self.full or unit in self.chars

    def without_pair(self, state: str, postcode: str, ks) -> None:
        """The pair failed its check: the state and the prefix digits are identifying again."""
        self.full.pop(state, None)
        self.chars -= {(postcode, k) for k in ks}
        if state == postcode:  # a glued 'NSW2750': its letters too
            self.chars -= {u for u in self.chars if u[0] == state}


def role_type(pii_type: PiiType | str) -> str:
    """The key of :data:`ROLE_NAMES` for a mention type (every ID_* type shares the ID roles)."""
    t = getattr(pii_type, "value", str(pii_type))
    return "ID" if t.startswith("ID_") else t


def enabled_roles(policy: PolicyProfile, m: GTMention) -> set[str]:
    out: set[str] = set()
    for entry in policy.shared_roles:
        if m.type in entry.types and (entry.subtypes is None or m.subtype in entry.subtypes):
            out |= set(entry.roles)
    return out & set(ROLE_NAMES.get(role_type(m.type), ()))


def document_shared_units(gt: GTDocument, policy: PolicyProfile) -> dict[str, SharedUnits]:
    """Mention id -> its shared parts under the policy (mentions with none are left out)."""
    if not policy.shared_roles:
        return {}
    words = gt.word_index()
    unit_of: dict[str, tuple] = {}
    for street, locality in address_unit_mentions(gt):
        for m in (street, locality):
            if m is not None:
                unit_of[m.id] = (street, locality)
    out: dict[str, SharedUnits] = {}
    for m in gt.mentions:
        roles = enabled_roles(policy, m)
        if not roles:
            continue
        if m.id in unit_of:
            su = _address(m, gt, unit_of[m.id], roles)
        else:
            su = _other(m, words, roles)
        if su and not _all_shared(m, words, su):
            out[m.id] = su
    return out


def shared_units(m: GTMention, gt: GTDocument, policy: PolicyProfile) -> SharedUnits:
    """The shared parts of one mention under the policy (empty when it has none)."""
    return document_shared_units(gt, policy).get(m.id, SharedUnits())


def _mention_words(m: GTMention) -> list[str]:
    return [wid for seg in m.segments for wid in seg.words]


def _all_shared(m: GTMention, words, su: SharedUnits) -> bool:
    for seg in m.segments:
        for wid in seg.words:
            text = words[wid][1].text
            lo, hi = (seg.char_start, seg.char_end) if seg.char_start is not None else (0, len(text))
            if wid in su.full:
                continue
            if any(text[k].isalnum() and (wid, k) not in su.chars for k in range(lo, hi)):
                return False
    return True


def _lead(t: str) -> int:
    return len(t) - len(t.lstrip(STRIP))


def _prefix(state: str | None, postcode: str) -> int:
    """How many leading digits of the original postcode match the state's prefix."""
    pre = PREFIX.get(state or "", "")
    k = 0
    while k < len(pre) and k < len(postcode) and postcode[k] == pre[k]:
        k += 1
    return k


def _prefix_without_state(postcode: str) -> int:
    """The prefix of a postcode with no state word: the states whose ranges hold it must agree."""
    if not re.fullmatch(r"\d{4}", postcode):
        return 0
    n = {_prefix(s, postcode) for s, ranges in POSTCODES.items()
         if any(lo <= int(postcode) <= hi for lo, hi in ranges)}
    return n.pop() if len(n) == 1 else 0


# -- addresses ---------------------------------------------------------------------------------------------


def _address(m: GTMention, gt: GTDocument, unit: tuple, roles: set[str]) -> SharedUnits:
    street, locality = unit
    uws = unit_words(gt, street, locality)
    wroles = address_roles(uws)
    if len(wroles) != len(uws):  # the reader gives one role per word; anything else shares nothing
        return SharedUnits()
    words = gt.word_index()
    mine = {STREET_PART, WHOLE} if (street is not None and m.id == street.id) else {LOCALITY_PART}
    su = SharedUnits()
    items = [(uw, r) for uw, r in zip(uws, wroles)]
    street_items = [(uw, r) for uw, r in items if uw.part in (STREET_PART, WHOLE)]

    def put(uw, role: str) -> None:
        text = words[uw.wid][1].text
        if uw.offset == 0 and len(uw.text) == len(text):
            su.full[uw.wid] = role
        else:
            su.chars.update((uw.wid, uw.offset + k) for k, ch in enumerate(uw.text) if ch.isalnum())

    # the street type: the street's last type word, after a name word that is neither a number nor
    # 'The', in a street line with no corner form
    typed = [k for k, (_, r) in enumerate(street_items) if r.role == "street_type"]
    street_type_ok = None
    if typed and not any(r.role == "corner" for _, r in street_items):
        k = typed[-1]
        prev = street_items[k - 1] if k > 0 else None
        if (prev is not None and prev[1].role == "street_name" and not any(ch.isdigit() for ch in prev[0].text)
                and prev[0].text.strip(STRIP).casefold() != "the"):
            street_type_ok = street_items[k][0]

    loc_items = [(uw, r) for uw, r in items if uw.part == LOCALITY_PART or (uw.part == WHOLE and r.role in (
        "state", "postcode", "state+postcode", "suburb", "country", "capital_city_phrase"))]
    state_code = _state_code([uw for uw, r in loc_items if r.role == "state"])
    for idx, (uw, r) in enumerate(items):
        if uw.part not in mine:
            continue
        if r.role == "street_type":
            if "street_type" in roles and uw is street_type_ok:
                put(uw, "street_type")
        elif r.role == "state":
            if "state" in roles and "pc_prefix" in roles:
                put(uw, "state")
                pc = next((u2 for u2, r2 in items[idx + 1:] if r2.role == "postcode"), None)
                if pc is not None:
                    core = pc.text.strip(STRIP)
                    a = pc.offset + _lead(pc.text)
                    ks = tuple(a + j for j in range(_prefix(state_code, core)))
                    su.chars.update((pc.wid, k) for k in ks)
                    su.pairs.append((uw.wid, pc.wid, ks))
        elif r.role == "state+postcode":
            if "state" in roles and "pc_prefix" in roles:
                (a0, a1, _), (b0, b1, _) = r.parts[:2]
                text = words[uw.wid][1].text
                su.chars.update((uw.wid, k) for k in range(a0, a1) if text[k].isalnum())
                code = STATE_CODES.get(text[a0:a1].replace(".", "").casefold())
                ks = tuple(b0 + j for j in range(_prefix(code, text[b0:b1])))
                su.chars.update((uw.wid, k) for k in ks)
                su.pairs.append((uw.wid, uw.wid, ks))
        elif r.role == "postcode":
            if "pc_prefix" in roles and state_code is None:
                core = uw.text.strip(STRIP)
                a = uw.offset + _lead(uw.text)
                su.chars.update((uw.wid, a + j) for j in range(_prefix_without_state(core)))
        elif r.role in ADDRESS_WORD_ROLES and r.role in roles:
            put(uw, r.role)
        for a0, a1, role in r.parts:  # a hotel-style 'B02/53' keeps its level code
            if role == "designator" and "designator" in roles and r.role != "state+postcode":
                su.chars.update((uw.wid, k) for k in range(a0, a1))
    return su


def _state_code(state_words) -> str | None:
    if not state_words:
        return None
    first = [state_words[0]]
    for uw in state_words[1:]:  # a multi-word state ('New South Wales'); a doubled state reads its first
        joined = " ".join(w.text.strip(STRIP).replace(".", "").casefold() for w in first + [uw])
        if any(code_name.startswith(joined) for code_name in STATE_CODES):
            first.append(uw)
        else:
            break
    return STATE_CODES.get(" ".join(w.text.strip(STRIP).replace(".", "").casefold() for w in first))


# -- organisations, emails, identifiers, phone numbers ---------------------------------------------------


def _other(m: GTMention, words, roles: set[str]) -> SharedUnits:
    su = SharedUnits()
    wids = _mention_words(m)
    if not wids:
        return su
    texts = [words[w][1].text for w in wids]
    norm = [t.strip(STRIP).replace(".", "").casefold() for t in texts]
    last = len(wids) - 1
    if "legal_form" in roles:
        k, run = last, []
        while k >= 1 and norm[k] in LEGAL_FORMS:
            run.append(wids[k])
            k -= 1
        if run and norm[k] not in LEGAL_FORMS:  # only after a name word ('Pty Ltd' alone is judged in full)
            su.full.update({w: "legal_form" for w in run})
    if "email_domain" in roles:
        for wid, x in zip(wids, texts):
            at, end = x.find("@"), len(x.rstrip(STRIP))
            if at > _lead(x) and end > at + 1:
                su.chars.update((wid, k) for k in range(at, end))
                su.needle_cut[wid] = (_lead(x), at)
    if "id_prefix" in roles:
        x0 = texts[0]
        mm = re.match(r"([A-Z]{1,4})([-/.:]?)(?=\d)", x0[_lead(x0):])
        if mm:
            a = _lead(x0)
            b = a + len(mm.group(0))
            su.chars.update((wids[0], k) for k in range(a, b))
            su.needle_cut[wids[0]] = (b, len(x0.rstrip(STRIP)))
        elif last >= 1 and re.fullmatch(r"[A-Z]{1,4}[-/.:]?", x0.strip(STRIP)) and any(ch.isdigit() for ch in texts[1]):
            su.full[wids[0]] = "id_prefix"
    if "phone_prefix" in roles:
        x0 = texts[0]
        a = _lead(x0)
        body = x0[a:]
        end = len(x0.rstrip(STRIP))
        digits = sum(ch.isdigit() for x in texts for ch in x)
        k = 0
        if body.startswith("+61"):
            k = 3
        elif re.fullmatch(r"\(0\d\)", x0.strip()):
            k = end - a
        elif body.startswith("04") and digits == 10:
            k = 2
        elif body[:4] in ("1300", "1800") and digits == 10:
            k = 4
        elif body.startswith("13") and digits == 6:
            k = 2
        k = min(k, end - a)
        if k and a + k >= end:
            su.full[wids[0]] = "phone_prefix"
        elif k:
            su.chars.update((wids[0], a + j) for j in range(k))
            su.needle_cut[wids[0]] = (a + k, end)
    return su
