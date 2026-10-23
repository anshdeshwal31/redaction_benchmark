"""Split ADDRESS mentions into the street line and the locality (GT 0.2).

The client keeps suburbs, towns, states and postcodes but replaces street lines, so one
ADDRESS fact becomes two mentions of the same entity:

- the street line (unit, number, street, PO box), still ``ADDRESS`` with its subtype;
- the locality (suburb, state, postcode, country), ``LOCATION`` with subtype ``locality`` and the
  condition ``address_locality``. That subtype is reserved for this part of an address: a place
  mentioned on its own takes ``suburb``, ``city``, ``region``, ``state``, ``country``, ``postcode`` or
  ``place_of_declaration`` (``rb gt validate`` warns otherwise). Policy 1.1.0 replaces the locality
  too (deviation ``whole-address``): the project owner decided on 2026-10-22 that every tool replaces
  whole addresses. A check of the splits does not end that; only the owner can.

Australian addresses end in ``<suburb> <state> <postcode>``; the locality is found from
the end, and its start is the nearest boundary before the state: a street-type word, a
comma, a PO-box number, or a change of line or form box. Splits the rules cannot place
with confidence are flagged for the human GT review; they are never silently guessed
without a flag.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..schemas import GTDocument, GTMention, PiiType, Segment

STATES = {"qld", "nsw", "vic", "tas", "sa", "wa", "nt", "act", "queensland", "victoria", "tasmania"}
MULTI_WORD_STATES = [("new", "south", "wales"), ("south", "australia"), ("western", "australia"),
                     ("northern", "territory"), ("australian", "capital", "territory")]
COUNTRIES = {"australia"}
STREET_TYPES = {
    "st", "street", "rd", "road", "ave", "av", "avenue", "dr", "drive", "ct", "court", "pl", "place", "cres",
    "crescent", "pde", "parade", "hwy", "highway", "tce", "terrace", "bvd", "blvd", "boulevard", "ln", "lane",
    "way", "cl", "close", "cct", "circuit", "esp", "esplanade", "gr", "grove", "sq", "square", "mews", "row",
    "walk", "track", "trk", "loop", "rise", "vista", "view", "gdns", "gardens", "mall", "arcade", "crt",
}
PO_WORDS = {"box", "bag", "locked", "gpo", "po", "p.o.", "rmb", "rsd"}
POSTCODE = re.compile(r"^\d{4}$")
LOCALITY_SUBTYPE = "locality"
ADDRESS_LOCALITY = "address_locality"  # condition of the locality part of a split address
PLACE_SUBTYPES = ("suburb", "city", "region", "state", "country", "postcode", "place_of_declaration")


@dataclass
class SplitNote:
    mention_id: str
    outcome: str  # split | locality_only | unsplit
    confident: bool
    reason: str
    street_words: int = 0
    locality_words: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Tok:
    seg: int
    page: int
    wid: str
    text: str
    line: str | None


def _norm(t: str) -> str:
    return t.strip(".,;:()").casefold()


def _tokens(m: GTMention, words) -> list[_Tok]:
    out = []
    for si, seg in enumerate(m.segments):
        for wid in seg.words:
            w = words[wid][1]
            text = w.text[seg.char_start:seg.char_end] if seg.char_start is not None else w.text
            out.append(_Tok(si, seg.page, wid, text, w.line))
    return out


def _state_start(norm: list[str], end: int) -> int | None:
    """Index where a state name ending at ``end`` (inclusive) starts."""
    if norm[end] in STATES:
        return end
    for parts in MULTI_WORD_STATES:
        k = len(parts)
        if end - k + 1 >= 0 and tuple(norm[end - k + 1:end + 1]) == parts:
            return end - k + 1
    return None


def _locality_start(toks: list[_Tok]) -> tuple[int | None, bool, str]:
    """Index of the first locality token, whether the rules are confident, and why."""
    norm = [_norm(t.text) for t in toks]
    i = len(toks) - 1
    while i >= 0 and norm[i] in COUNTRIES:
        i -= 1
    has_postcode = i >= 0 and bool(POSTCODE.match(norm[i]))
    if has_postcode:
        i -= 1
    s = _state_start(norm, i) if i >= 0 else None
    if s is None and not has_postcode:
        return None, False, "no state or postcode at the end"
    anchor = s if s is not None else i + 1  # first token of state, else the postcode
    # Walk back to the nearest street-line boundary, leaving at least one suburb word: a
    # suburb can itself end in a street-type word ("Kelvin Grove") or carry a comma
    # ("Brisbane, QLD"). The nearest boundary wins: forms keep a whole suburb in one box, and
    # a line break before the suburb is nearer than any comma in the street line.
    for j in range(anchor - 2, -1, -1):
        prev, cur = toks[j], toks[j + 1]
        if norm[j] in STREET_TYPES or prev.text.rstrip().endswith(","):
            return j + 1, True, "after street type or comma"
        if norm[j].isdigit() and j > 0 and norm[j - 1] in PO_WORDS:
            return j + 1, True, "after a PO box number"
        if prev.seg != cur.seg or prev.line != cur.line:
            return j + 1, True, "after a line or box change"
    if anchor <= 1:
        return 0, True, "locality only"
    if anchor == 2:
        return 0, False, "two words before the state and no boundary; assumed a two-word suburb"
    return anchor - 1, False, "no boundary before the suburb; assumed a one-word suburb"


def _segments(toks: list[_Tok], m: GTMention) -> list[Segment]:
    """Segments of one part. A sub-word segment has a single word, so it is never cut."""
    segs: list[Segment] = []
    for si in sorted({t.seg for t in toks}):
        src = m.segments[si]
        wids = [t.wid for t in toks if t.seg == si]
        whole = wids == src.words
        segs.append(Segment(page=src.page, words=wids, char_start=src.char_start if whole else None,
                            char_end=src.char_end if whole else None))
    return segs


def _text(toks: list[_Tok]) -> str:
    return " ".join(t.text for t in toks)


def split_addresses(gt: GTDocument, version: str | None = None) -> tuple[GTDocument, list[SplitNote]]:
    words = gt.word_index()
    notes: list[SplitNote] = []
    out: list[GTMention] = []
    parent_of_part: dict[str, tuple[str, set[str], str, set[str]]] = {}
    for m in gt.mentions:
        if m.type is not PiiType.ADDRESS or any(seg.regions for seg in m.segments):
            out.append(m)
            continue
        toks = _tokens(m, words)
        k, confident, reason = _locality_start(toks)
        if k is None:
            notes.append(SplitNote(m.id, "unsplit", False, reason, len(toks), 0))
            out.append(m)
            continue
        street, loc = toks[:k], toks[k:]
        loc_conditions = sorted(set(m.conditions) | {ADDRESS_LOCALITY})
        if not street:
            out.append(m.model_copy(update={"type": PiiType.LOCATION, "subtype": LOCALITY_SUBTYPE,
                                            "conditions": loc_conditions}))
            notes.append(SplitNote(m.id, "locality_only", confident, reason, 0, len(loc)))
            continue
        if not loc:
            out.append(m)
            notes.append(SplitNote(m.id, "unsplit", False, "locality start at the end", len(street), 0))
            continue
        street_m = m.model_copy(update={"segments": _segments(street, m), "text": _text(street),
                                        "conditions": sorted(set(m.conditions) | {"address_street_line"})})
        loc_m = GTMention(id=f"{m.id}-loc", type=PiiType.LOCATION, subtype=LOCALITY_SUBTYPE, entity=m.entity,
                          text=_text(loc), segments=_segments(loc, m), conditions=loc_conditions)
        out.extend([street_m, loc_m])
        parent_of_part[m.id] = (street_m.id, {t.wid for t in street}, loc_m.id, {t.wid for t in loc})
        notes.append(SplitNote(m.id, "split", confident, reason, len(street), len(loc)))
    # Nested mentions follow the part that contains them.
    fixed = []
    for m in out:
        if m.parent in parent_of_part:
            sid, swords, lid, lwords = parent_of_part[m.parent]
            mine = {wid for seg in m.segments for wid in seg.words}
            m = m.model_copy(update={"parent": sid if mine <= swords else lid if mine <= lwords else None})
        fixed.append(m)
    update = {"mentions": fixed}
    if version:
        update["gt_version"] = version
    return gt.model_copy(update=update), notes
