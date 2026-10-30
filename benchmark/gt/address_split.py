"""Split ADDRESS mentions into the street line and the locality (GT 0.2).

The client keeps suburbs, towns, states and postcodes but replaces street lines, so one
ADDRESS fact becomes two mentions of the same entity:

- the street line (unit, number, street, PO box), still ``ADDRESS`` with its subtype;
- the locality (suburb, state, postcode, country), ``LOCATION`` with subtype ``locality`` and the
  condition ``address_locality``. That subtype is reserved for this part of an address: a place
  mentioned on its own takes one of ``PLACE_SUBTYPES`` (``rb gt validate`` warns otherwise), such as
  ``suburb``, ``place_of_birth`` after a label naming the place's kind, or ``street`` for a street
  named in running text with no house number. Policy 1.1.0 and later replace the
  locality too (deviation ``whole-address``): the project owner decided on 2026-10-22 that tools 1
  and 2 replace whole addresses, and the scores hold every tool scored under the policy to that rule.
  A check of the splits does not end that; only the owner can.

Australian addresses end in ``<suburb> <state> <postcode>``; the locality is found from
the end (``read_end``), and its start is a known locality name that holds a street-type word
('St Marys', 'Lane Cove'), else the nearest boundary before the state: a street-type word, a
comma, a PO-box number, or a change of line or form box. A DX line stays whole. Splits the
rules cannot place with confidence are flagged for the human GT review; they are never
silently guessed without a flag. The fictional case table ``docs/address_cases.json`` is the
splitter's test data.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..schemas import GTDocument, GTMention, PiiType, Segment

# Every written form of a state as _norm leaves it ('N.S.W.' reads 'n.s.w').
STATES = {"qld", "nsw", "vic", "tas", "sa", "wa", "nt", "act", "queensland", "victoria", "tasmania",
          "n.s.w", "s.a", "w.a", "n.t", "a.c.t"}
MULTI_WORD_STATES = [("new", "south", "wales"), ("south", "australia"), ("western", "australia"),
                     ("northern", "territory"), ("australian", "capital", "territory")]
# Written forms (dots removed) -> the state's code.
STATE_CODES = {
    "nsw": "NSW", "new south wales": "NSW", "vic": "VIC", "victoria": "VIC", "qld": "QLD", "queensland": "QLD",
    "sa": "SA", "south australia": "SA", "wa": "WA", "western australia": "WA", "tas": "TAS", "tasmania": "TAS",
    "nt": "NT", "northern territory": "NT", "act": "ACT", "australian capital territory": "ACT",
}
SPELLED_STATES = {"queensland", "victoria", "tasmania", *(" ".join(p) for p in MULTI_WORD_STATES)}
COUNTRIES = {"australia"}
STREET_TYPES = {
    "st", "street", "rd", "road", "ave", "av", "avenue", "dr", "drive", "ct", "court", "pl", "place", "cres",
    "crescent", "pde", "parade", "hwy", "highway", "tce", "terrace", "bvd", "blvd", "boulevard", "ln", "lane",
    "way", "cl", "close", "cct", "circuit", "esp", "esplanade", "gr", "grove", "sq", "square", "mews", "row",
    "walk", "track", "trk", "loop", "rise", "vista", "view", "gdns", "gardens", "mall", "arcade", "crt",
    # plural types name the streets of a corner ('Cnr Smith & Jones Streets')
    "streets", "roads", "sts", "rds", "avenues", "aves", "lanes", "drives", "parades", "terraces",
}
# Words of a postal delivery type: a word with a digit after one of them is the box number.
PO_WORDS = {"box", "bag", "locked", "gpo", "po", "p.o", "rmb", "r.m.b", "rsd", "cmb", "pmb", "private", "reply",
            "paid", "care"}
PO_WORDS_UPPER = {"ms", "rms", "cma", "cpa"}  # box kinds only in upper case: 'MS 123', never 'Ms' or 'ms'
# Australia Post's postal delivery types, dots removed ('P.O. Box' reads 'po box'). 'Box' alone is
# never one ('Box Hill'). MS, RMS, CMA and CPA count only in upper case.
BOX_KINDS = [("post", "office", "box"), ("locked", "mail", "bag"), ("private", "mail", "bag"), ("po", "box"),
             ("gpo", "box"), ("locked", "bag"), ("private", "bag"), ("reply", "paid"), ("care", "po"), ("rmb",),
             ("rsd",), ("cmb",), ("pmb",), ("ms",), ("rms",), ("cma",), ("cpa",)]
CORNER_WORDS = {"cnr", "crn", "corner", "&"}
# Countries that end an overseas address (the owner's ruling of 2026-10-22: the country word stays).
# Australian states are matched first, so 'New South Wales' never reads as 'Wales'.
FOREIGN_COUNTRIES = ("new zealand", "united kingdom", "uk", "england", "scotland", "wales", "ireland",
                     "united states", "usa", "canada", "singapore", "hong kong", "india", "china", "japan",
                     "philippines", "indonesia", "malaysia", "fiji", "papua new guinea", "south africa")
_AFTER_POSTCODE_OR_COMMA = {("england",), ("scotland",), ("wales",)}  # only after a postcode or a comma
CAPITAL_CITY_PHRASE = ("in", "your", "capital", "city")  # Australia Post's form for national agencies
_UK_OUTWARD = re.compile(r"^[a-z]{1,2}\d[a-z\d]?$")
_UK_INWARD = re.compile(r"^\d[a-z]{2}$")
DIRECTIONALS = {"north", "south", "east", "west", "nth", "sth"}
POSTCODE = re.compile(r"^\d{4}$")
GLUED = re.compile(r"^(n\.?s\.?w|vic|qld|tas|s\.?a|w\.?a|n\.?t|a\.?c\.?t)\.?(\d{4})$")  # 'NSW2750', 'N.S.W.2750'
LOCALITY_SUBTYPE = "locality"
ADDRESS_LOCALITY = "address_locality"  # condition of the locality part of a split address
# Subtypes of a place on its own. The place kinds name a place written right after a label or phrase
# naming birth, injury, accident or incident, declaration (owner ruling 2): the place stays, postcode
# included. 'street' is a street named in running text with no house number (owner ruling 3).
PLACE_SUBTYPES = ("suburb", "city", "region", "state", "country", "postcode", "place_of_declaration", "place_of_birth",
                  "place_of_injury", "place_of_accident", "place_of_incident", "street")

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

# Localities that hold a street-type word before their last word, by state: the splitter would
# otherwise end the street line inside them ('7 King Street, St Marys'). From the ABS Suburbs and
# Localities (SAL) 2021 list (CC BY 4.0), as stored in suburb list v2 of the address case table.
TYPE_WORD_LOCALITIES = {
    "NSW": ("Crescent Head", "Lane Cove", "Lane Cove North", "Lane Cove West", "North St Marys", "St Albans",
            "St Andrews", "St Clair", "St Fillans", "St George", "St Georges Basin", "St Helens Park",
            "St Huberts Island", "St Ives", "St Ives Chase", "St Johns Park", "St Leonards", "St Marys",
            "St Peters", "Terrace Creek", "Way Way"),
    "QLD": ("Mount St John", "St Agnes", "St Aubyn", "St George", "St Helens", "St Helens Beach", "St Kilda",
            "St Lawrence", "St Lucia", "St Mary", "St Ruth"),
    "SA": ("Avenue Range", "Square Mile", "St Agnes", "St Clair", "St Georges", "St Ives", "St Johns", "St Kilda",
           "St Kitts", "St Marys", "St Morris", "St Peters"),
    "TAS": ("Lake St Clair", "St Helens", "St Leonards", "St Marys"),
    "VIC": ("St Albans", "St Albans Park", "St Andrews", "St Andrews Beach", "St Arnaud", "St Arnaud East",
            "St Arnaud North", "St Clair", "St Germains", "St Helena", "St Helens", "St Helens Plains", "St James",
            "St Kilda", "St Kilda East", "St Kilda West", "St Leonards"),
    "WA": ("Peppermint Grove Beach", "St George Ranges", "St James", "St Ronans"),
}


def _name_words(names) -> list[tuple[str, ...]]:
    """Locality names as folded word tuples, longest first."""
    return sorted({tuple(n.casefold().split()) for n in names}, key=lambda t: (-len(t), t))


_TYPE_WORD_NAMES = {s: _name_words(ns) for s, ns in TYPE_WORD_LOCALITIES.items()}
_TYPE_WORD_NAMES_ANY = _name_words(n for ns in TYPE_WORD_LOCALITIES.values() for n in ns)
_FOREIGN_COUNTRIES = _name_words(FOREIGN_COUNTRIES)


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
class EndReading:
    """The end of an address as token indexes (``read_end``)."""

    country: list[int] = field(default_factory=list)
    postcode: list[int] = field(default_factory=list)
    state: list[int] = field(default_factory=list)
    state_code: str | None = None  # 'NSW', 'VIC', ...
    glued: int | None = None  # a state and its postcode in one word ('NSW2750')
    capital_city: list[int] = field(default_factory=list)  # 'IN YOUR CAPITAL CITY' in place of a locality
    overseas: bool = False  # the address ends in a foreign country
    anchor: int | None = None  # the first word the locality is found from; None when nothing is read


@dataclass
class _Tok:
    seg: int
    page: int
    wid: str
    text: str
    line: str | None


def _norm(t: str) -> str:
    return t.strip(".,;:()").casefold()


def _fold(t: str) -> str:
    """_norm without dots: 'P.O.' and 'PO' both read 'po'."""
    return _norm(t).replace(".", "")


def _has_digit(t: str) -> bool:
    return any(ch.isdigit() for ch in t)


def postcode_in_state(state: str, postcode: str) -> bool:
    """Whether a 4-digit postcode lies in the state's Australia Post ranges."""
    pc = int(postcode)
    return any(lo <= pc <= hi for lo, hi in POSTCODES.get(state, []))


def _state_at(norm: list[str], end: int, postcode: str | None) -> tuple[int | None, str | None]:
    """Start and code of a state written as the word(s) ending at ``end``. A spelled-out state
    before a postcode must own that postcode; with no postcode, 'Victoria' after 'Mount' or 'Port'
    is part of the suburb (Mount Victoria NSW, Port Victoria SA: the only two ABS localities that
    end in a state word)."""
    for parts in MULTI_WORD_STATES:
        k = len(parts)
        if end - k + 1 >= 0 and tuple(norm[end - k + 1:end + 1]) == parts:
            code = STATE_CODES[" ".join(parts)]
            if postcode is None or postcode_in_state(code, postcode):
                return end - k + 1, code
            return None, None
    word = norm[end]
    if word not in STATES:
        return None, None
    code = STATE_CODES[word.replace(".", "")]
    if word in SPELLED_STATES:
        if postcode is not None and not postcode_in_state(code, postcode):
            return None, None
        if postcode is None and word == "victoria" and end > 0 and norm[end - 1] in ("mount", "mt", "port", "pt"):
            return None, None
    return end, code


def _foreign_country_at(texts: list[str], end: int) -> int | None:
    """Start of a foreign country name from the closed list ending at ``end``."""
    folded = [_fold(t) for t in texts]
    for name in _FOREIGN_COUNTRIES:
        k = end - len(name) + 1
        if k < 0 or tuple(folded[k:end + 1]) != name:
            continue
        if name in _AFTER_POSTCODE_OR_COMMA and not (k > 0 and (_has_digit(texts[k - 1])
                                                                or texts[k - 1].rstrip().endswith(","))):
            continue
        return k
    return None


def read_end(texts: list[str], overseas: bool = False) -> EndReading:
    """Read the country, postcode and state at the end of an address's words.

    The state may be written in any form: a code ('NSW', 'Qld', 'N.S.W.'), spelled out ('Victoria',
    'New South Wales'), doubled ('Sydney NSW NSW 2000': both words) or glued to its postcode
    ('NSW2750': state and postcode). 'Australia' after 'South' or 'Western' is part of the state.
    ``anchor`` is the state's first word, else the postcode; for a final 'Mount Victoria' or 'Port
    Victoria' with no postcode it is that 'Victoria', so the locality is still found although no
    state is read.

    With ``overseas`` (the role reader; the splitter leaves these to the GT review) it also reads
    'IN YOUR CAPITAL CITY' standing in for a locality, and a foreign country from a closed list at
    the end with the postcode before it, a 4-digit one or a UK one ('OX1 4AA'). No Australian state
    is read before a foreign country.
    """
    norm = [_norm(t) for t in texts]
    r = EndReading()
    i = len(norm) - 1
    if overseas and len(texts) >= 4 and tuple(_fold(t) for t in texts[-4:]) == CAPITAL_CITY_PHRASE:
        r.capital_city = list(range(len(texts) - 4, len(texts)))
        r.anchor = r.capital_city[0]
        return r
    while i >= 0 and norm[i] in COUNTRIES and not (norm[i] == "australia" and i > 0
                                                   and norm[i - 1] in ("south", "western")):
        r.country.insert(0, i)
        i -= 1
    if i < 0:
        return r
    if overseas and not r.country and _state_at(norm, i, None)[0] is None:
        k = _foreign_country_at(texts, i)
        if k is not None:
            r.country, r.overseas = list(range(k, i + 1)), True
            i = k - 1
            if i >= 0 and POSTCODE.match(norm[i]):
                r.postcode = [i]
            elif i >= 1 and _UK_OUTWARD.match(norm[i - 1]) and _UK_INWARD.match(norm[i]):
                r.postcode = [i - 1, i]
            r.anchor = (r.postcode or r.country)[0]
            return r
    glued = GLUED.match(norm[i])
    if glued:
        r.state, r.postcode, r.glued, r.anchor = [i], [i], i, i
        r.state_code = STATE_CODES[glued.group(1).replace(".", "")]
        return r
    if POSTCODE.match(norm[i]):
        r.postcode = [i]
        i -= 1
    if i >= 0:
        s, code = _state_at(norm, i, norm[r.postcode[0]] if r.postcode else None)
        if s is not None:
            if s > 0 and norm[s - 1] in STATES and norm[s - 1] not in SPELLED_STATES \
                    and STATE_CODES[norm[s - 1].replace(".", "")] == code:
                s -= 1  # a doubled state: both words are the state as written
            r.state, r.state_code, r.anchor = list(range(s, i + 1)), code, s
            return r
    if r.postcode:
        r.anchor = r.postcode[0]
    elif i >= 1 and norm[i] == "victoria" and norm[i - 1] in ("mount", "mt", "port", "pt"):
        r.anchor = i
    return r


def _tokens(m: GTMention, words) -> list[_Tok]:
    out = []
    for si, seg in enumerate(m.segments):
        for wid in seg.words:
            w = words[wid][1]
            text = w.text[seg.char_start:seg.char_end] if seg.char_start is not None else w.text
            out.append(_Tok(si, seg.page, wid, text, w.line))
    return out


def _is_box_word(tok: _Tok) -> bool:
    n = _norm(tok.text)
    return n in PO_WORDS or (n in PO_WORDS_UPPER and tok.text.strip(".,;:()").isupper())


def _box_kind_at(toks: list[_Tok], i: int) -> int:
    """Length of the postal delivery type starting at ``i`` (0 when there is none)."""
    folded = [_fold(t.text) for t in toks[i:i + 3]]
    for kind in BOX_KINDS:
        if tuple(folded[:len(kind)]) == kind:
            if kind[0] in PO_WORDS_UPPER and not toks[i].text.strip(".,;:()").isupper():
                continue
            return len(kind)
    return 0


def _is_street_type(toks: list[_Tok], j: int) -> bool:
    """A street-type word; 'St' is never one as the first word, right after a comma or right after a
    number ('St Marys', '7 King Street, St Ives', '100 St Georges Terrace')."""
    n = _norm(toks[j].text)
    if n not in STREET_TYPES:
        return False
    if n == "st" and (j == 0 or toks[j - 1].text.rstrip().endswith(",") or _has_digit(toks[j - 1].text)):
        return False
    return True


def _has_street_line_word(toks: list[_Tok]) -> bool:
    """Whether words hold a digit, a street type, a corner word, a postal delivery type (DX
    included) or a 'via' after a property name: a sign of a street line before the state."""
    for j, t in enumerate(toks):
        if _has_digit(t.text) or _is_street_type(toks, j) or _norm(t.text) in CORNER_WORDS:
            return True
        if _box_kind_at(toks, j) or _fold(t.text) == "dx" or (j > 0 and _fold(t.text) == "via"):
            return True
    return False


def _type_word_locality(toks: list[_Tok], anchor: int, state: str | None) -> int | None:
    """Start of a known locality name holding a street-type word that ends just before ``anchor``."""
    folded = [_fold(t.text) for t in toks]
    for name in _TYPE_WORD_NAMES.get(state, []) if state else _TYPE_WORD_NAMES_ANY:
        s = anchor - len(name)
        if s >= 0 and tuple(folded[s:anchor]) == name \
                and len({(t.seg, t.line) for t in toks[s:anchor]}) == 1:
            return s
    return None


def _boundary(toks: list[_Tok], overseas: bool = False) -> tuple[int | None, bool, str]:
    r = read_end([t.text for t in toks], overseas)
    if r.anchor is None:
        return None, False, "no state or postcode at the end"
    anchor = r.anchor  # first token of state, else the postcode
    if r.capital_city:
        return anchor, True, "'IN YOUR CAPITAL CITY' stands for the locality"
    k = _type_word_locality(toks, anchor, r.state_code)
    if k is not None:
        return k, True, "a locality name that holds a street-type word"
    # Walk back to the nearest street-line boundary, leaving at least one suburb word: a
    # suburb can itself end in a street-type word ("Kelvin Grove") or carry a comma
    # ("Brisbane, QLD"). The nearest boundary wins: forms keep a whole suburb in one box, and
    # a line break before the suburb is nearer than any comma in the street line.
    for j in range(anchor - 2, -1, -1):
        prev, cur = toks[j], toks[j + 1]
        if _is_street_type(toks, j) or prev.text.rstrip().endswith(","):
            return j + 1, True, "after street type or comma"
        if _has_digit(prev.text) and j > 0 and _is_box_word(toks[j - 1]):
            return j + 1, True, "after a PO box number"
        if prev.seg != cur.seg or prev.line != cur.line:
            return j + 1, True, "after a line or box change"
    if anchor <= 1:
        return 0, True, "locality only"
    if not _has_street_line_word(toks[:anchor]):
        return 0, False, "no number, street type, corner or box word before the state; read as a place on its own"
    if anchor == 2:
        return 0, False, "two words before the state and no boundary; assumed a two-word suburb"
    return anchor - 1, False, "no boundary before the suburb; assumed a one-word suburb"


def _locality_start(toks: list[_Tok]) -> tuple[int | None, bool, str]:
    """Index of the first locality token, whether the rules are confident, and why."""
    k, confident, reason = _boundary(toks)
    if k and k < len(toks) and _fold(toks[k].text) in DIRECTIONALS and _norm(toks[k - 1].text) in STREET_TYPES \
            and not toks[k - 1].text.rstrip().endswith(","):
        # '12 Smith Street North Ryde NSW 2112': 'North' is the street's suffix when 2112 is Ryde's
        # postcode and part of the suburb when it is North Ryde's; only a gazetteer can tell.
        return k, False, f"{reason}; a directional after the street type may be the street's suffix"
    return k, confident, reason


def _is_dx_line(toks: list[_Tok]) -> bool:
    return len(toks) >= 2 and _fold(toks[0].text) == "dx" and _has_digit(toks[1].text)


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
        if _is_dx_line(toks):
            # A DX line is a postal address in itself: its exchange is not a locality.
            notes.append(SplitNote(m.id, "unsplit", True, "a DX line stays whole", len(toks), 0))
            out.append(m)
            continue
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


# -- roles of an address unit's words (eval-0.4) --------------------------------------------------
#
# One reader for the shared words (E2), the address units' token survival (E3), the fakes' read-back
# (E6) and the per-mention record of shared words (E7). It reads GT words only, in their place in the
# unit, never word text on its own and never a tool's output or report. Roles use the case table's
# vocabulary (docs/address_cases.md).

ROLE_NAMES = ("building", "designator", "unit_number", "level_number", "number", "unit_number+number",
              "street_name", "street_type", "street_suffix", "corner", "box_kind", "box_number", "box_extra", "via",
              "dx", "dx_number", "dx_exchange", "suburb", "state", "postcode", "state+postcode", "country",
              "capital_city_phrase")
# Street types (AS 4590 and AS/NZS 4819 abbreviations) -> the canonical type.
STREET_TYPE_NAMES = {
    "st": "street", "street": "street", "sts": "street", "streets": "street", "rd": "road", "road": "road",
    "rds": "road", "roads": "road", "ave": "avenue", "av": "avenue", "avenue": "avenue", "aves": "avenue",
    "avenues": "avenue", "dr": "drive", "drive": "drive", "drives": "drive", "ct": "court", "crt": "court",
    "court": "court", "pl": "place", "place": "place", "cres": "crescent", "cr": "crescent", "crescent": "crescent",
    "pde": "parade", "parade": "parade", "parades": "parade", "hwy": "highway", "highway": "highway",
    "tce": "terrace", "terrace": "terrace", "terraces": "terrace", "bvd": "boulevard", "blvd": "boulevard",
    "boulevard": "boulevard", "ln": "lane", "lane": "lane", "lanes": "lane", "way": "way", "cl": "close",
    "close": "close", "cct": "circuit", "circuit": "circuit", "esp": "esplanade", "esplanade": "esplanade",
    "gr": "grove", "grove": "grove", "sq": "square", "square": "square", "mews": "mews", "row": "row",
    "walk": "walk", "track": "track", "trk": "track", "loop": "loop", "rise": "rise", "vista": "vista",
    "view": "view", "gdns": "gardens", "gardens": "gardens", "mall": "mall", "arcade": "arcade",
}
STREET_SUFFIXES = DIRECTIONALS | {"n", "s", "e", "w"}
# Australia Post's unit and floor types (the table's designators) before their number or letter.
UNIT_DESIGNATORS = {"unit", "u", "apartment", "apt", "flat", "f", "factory", "fy", "house", "hse", "office", "off",
                    "penthouse", "pths", "room", "rm", "shop", "studio", "stu", "suite", "se", "ste", "townhouse",
                    "tnhs", "villa", "warehouse", "we", "lot"}
LEVEL_DESIGNATORS = {"level", "l", "lvl", "floor", "fl"}
FLOOR_WORDS = {"ground", "mezzanine", "basement", "lg", "ug"}  # floors that need no number
BUILDING_DESIGNATORS = {"building", "block", "tower", "wing"}  # before a letter or short number only
# The last word of a building or property name standing in a street line with no house number.
BUILDING_WORDS = {"station", "homestead", "house", "tower", "towers", "chambers", "plaza", "centre", "center",
                  "building", "complex"}
_LEVEL_CODES = {"b", "g", "l", "lg", "ug", "m"}  # a level-type code before a hotel-style unit number
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_SLASH = re.compile(r"^\d+[A-Za-z]?/\d+[A-Za-z]?$")  # '5/3': unit and house number
_HOTEL = re.compile(r"^([A-Za-z]{1,2})(\d+[A-Za-z]?/\d+[A-Za-z]?)$")  # 'B02/53'
_STRIP = ".,;:()"
_NUMBERS = ("number", "unit_number", "unit_number+number", "level_number", "designator")
STREET_PART, LOCALITY_PART, WHOLE = "street", "locality", "whole"


@dataclass(frozen=True)
class UnitWord:
    """One GT word of an address unit, as its mention holds it."""

    wid: str
    text: str  # a sub-word segment's characters, else the whole word
    offset: int  # where ``text`` starts in the GT word's text
    part: str  # STREET_PART, LOCALITY_PART, or WHOLE for an unsplit ADDRESS
    seg: int  # segment number within the unit
    line: str | None


@dataclass(frozen=True)
class WordRole:
    wid: str
    role: str
    # Sub-word roles as (start, end, role), offsets in the GT word's text: a glued 'NSW2750' is
    # state + postcode, a hotel-style 'B02/53' keeps its level code 'B' as a designator.
    parts: tuple[tuple[int, int, str], ...] = ()


def canonical_street_type(text: str) -> str | None:
    """'St' -> 'street', 'Cres' and 'Cr' -> 'crescent'; None for a word that is not a street type."""
    return STREET_TYPE_NAMES.get(_fold(text))


def address_unit_mentions(gt: GTDocument) -> list[tuple[GTMention | None, GTMention | None]]:
    """The address units of a GT document as (street line, locality) mentions: a street line (ADDRESS,
    condition ``address_street_line``) with its ``<id>-loc`` LOCATION/locality of the same entity, a
    locality-only LOCATION/locality as (None, locality), and an unsplit ADDRESS as (address, None)."""
    by_id = {m.id: m for m in gt.mentions}
    partner: dict[str, GTMention] = {}
    for m in gt.mentions:
        loc = by_id.get(f"{m.id}-loc")
        if (m.type is PiiType.ADDRESS and "address_street_line" in m.conditions and loc is not None
                and loc.type is PiiType.LOCATION and loc.subtype == LOCALITY_SUBTYPE and loc.entity == m.entity):
            partner[m.id] = loc
    paired = {loc.id for loc in partner.values()}
    out: list[tuple[GTMention | None, GTMention | None]] = []
    for m in gt.mentions:
        if m.type is PiiType.ADDRESS:
            out.append((m, partner.get(m.id)))
        elif m.type is PiiType.LOCATION and m.subtype == LOCALITY_SUBTYPE and m.id not in paired:
            out.append((None, m))
    return out


def unit_words(gt: GTDocument, street: GTMention | None, locality: GTMention | None) -> list[UnitWord]:
    """The words of one address unit in reading order: the street line's, then the locality's."""
    words = gt.word_index()
    out: list[UnitWord] = []
    seg_no = 0
    for part, m in ((STREET_PART if locality is not None else WHOLE, street), (LOCALITY_PART, locality)):
        if m is None:
            continue
        for seg in m.segments:
            for wid in seg.words:
                w = words[wid][1]
                cut = seg.char_start is not None
                out.append(UnitWord(wid, w.text[seg.char_start:seg.char_end] if cut else w.text,
                                    seg.char_start if cut else 0, part, seg_no, w.line))
            seg_no += 1
    return out


def address_roles(words: list[UnitWord]) -> list[WordRole]:
    """The role of every word of one address unit (``unit_words``), in the case table's vocabulary.

    A split unit's street line and locality are read with the GT's own boundary. An unsplit ADDRESS
    is cut where the splitter would cut it (foreign countries and 'IN YOUR CAPITAL CITY' included);
    a DX line, or one with no place found at its end, is all street line.
    """
    street = [w for w in words if w.part == STREET_PART]
    locality = [w for w in words if w.part == LOCALITY_PART]
    whole = [w for w in words if w.part == WHOLE]
    alone = False
    if whole:
        toks = [_Tok(w.seg, 0, w.wid, w.text, w.line) for w in whole]
        k = None if _is_dx_line(toks) else _boundary(toks, overseas=True)[0]
        if k is None:
            street, alone = whole, True
        else:
            street, locality = whole[:k], whole[k:]
    return _street_roles(street, alone) + _locality_roles(locality)


def mention_address_roles(gt: GTDocument) -> dict[str, list[WordRole]]:
    """Roles of the words of every address mention (street line, locality, unsplit ADDRESS), keyed by
    mention id; each part is read together with its partner."""
    out: dict[str, list[WordRole]] = {}
    for street, locality in address_unit_mentions(gt):
        roles = address_roles(unit_words(gt, street, locality))
        for m in (street, locality):
            if m is not None:
                mine = {wid for seg in m.segments for wid in seg.words}
                out[m.id] = [r for r in roles if r.wid in mine]
    return out


def _core_span(w: UnitWord) -> tuple[str, int]:
    """The word without outer punctuation, and where it starts in the GT word's text."""
    lead = len(w.text) - len(w.text.lstrip(_STRIP))
    return w.text.strip(_STRIP), w.offset + lead


def _locality_roles(ws: list[UnitWord]) -> list[WordRole]:
    r = read_end([w.text for w in ws], overseas=True)
    out = []
    for k, w in enumerate(ws):
        if k == r.glued:
            core, a = _core_span(w)
            d = next(i for i, ch in enumerate(core) if ch.isdigit())
            out.append(WordRole(w.wid, "state+postcode", ((a, a + d, "state"), (a + d, a + len(core), "postcode"))))
            continue
        role = ("capital_city_phrase" if k in r.capital_city else "country" if k in r.country
                else "postcode" if k in r.postcode else "state" if k in r.state else "suburb")
        out.append(WordRole(w.wid, role))
    return out


def _street_roles(ws: list[UnitWord], alone: bool) -> list[WordRole]:
    """Roles of a street line's words. ``alone``: an unsplit ADDRESS with no place found at its end,
    where name words after a street type are read as its suburb."""
    n = len(ws)
    texts = [w.text for w in ws]
    fold = [_fold(t) for t in texts]
    core = [t.strip(_STRIP) for t in texts]
    comma = [t.rstrip().endswith(",") for t in texts]
    role: list[str | None] = [None] * n
    parts: dict[int, tuple[tuple[int, int, str], ...]] = {}
    toks = [_Tok(w.seg, 0, w.wid, w.text, w.line) for w in ws]
    if _is_dx_line(toks):
        return [WordRole(w.wid, r) for w, r in zip(ws, ["dx", "dx_number", *["dx_exchange"] * (n - 2)])]

    # Postal delivery types before their number ('PO Box 123', 'RMB 4600'); Care PO, CMA, CPA and CMB
    # may stand without one.
    i = 0
    while i < n:
        k = _box_kind_at(toks, i)
        if k and i + k < n and _has_digit(texts[i + k]):
            role[i:i + k + 1] = ["box_kind"] * k + ["box_number"]
            i += k + 1
            continue
        if k and fold[i] in ("care", "cma", "cpa", "cmb"):
            role[i:i + k] = ["box_kind"] * k
            i += k
            continue
        i += 1

    # Designators and the number or letter they take.
    k = 0
    while k < n:
        f = fold[k]
        nxt = k + 1 if k + 1 < n and role[k + 1] is None else None
        if role[k] is not None:
            k += 1
            continue
        if _ORDINAL.match(f) and nxt is not None and fold[nxt] in LEVEL_DESIGNATORS:  # '1st Floor'
            role[k], role[nxt] = "level_number", "designator"
            k = nxt + 1
            continue
        span = [k, k + 1] if f in ("lower", "upper") and nxt is not None and fold[nxt] == "ground" else \
            [k] if f in FLOOR_WORDS else []
        if span:  # 'Ground Floor', 'Lower Ground,', 'Mezzanine, 45 ...': no number
            e = span[-1] + 1
            if e < n and role[e] is None and fold[e] in ("floor", "fl", "level"):
                span.append(e)
            if len(span) > 1 + (f in ("lower", "upper")) or span[0] == 0 or comma[span[-1]]:
                for x in span:
                    role[x] = "designator"
                k = span[-1] + 1
                continue
        if nxt is not None and not comma[k]:
            nt = core[nxt]
            letter = len(nt) == 1 and nt.isalpha()
            if f in UNIT_DESIGNATORS | LEVEL_DESIGNATORS and (_has_digit(nt) or (letter and len(f) > 1)):
                role[k] = "designator"
                role[nxt] = ("level_number" if f in LEVEL_DESIGNATORS
                             else "unit_number+number" if _SLASH.match(nt) else "unit_number")
                k = nxt + 1
                continue
            # 'Building C', 'Tower 2'; never after a name word ('Westpoint Tower')
            after_name = k > 0 and role[k - 1] is None and not comma[k - 1] and not _has_digit(texts[k - 1])
            if f in BUILDING_DESIGNATORS and not after_name and (letter or (nt.isdigit() and len(nt) <= 3)):
                role[k], role[nxt] = "designator", "unit_number"
                k = nxt + 1
                continue
        k += 1

    # Numbers: a level-type code before a hotel-style number stays ('B02/53'); a number right before
    # another is the unit's (form boxes: '5' '12').
    for k in range(n):
        if role[k] is not None or not _has_digit(texts[k]):
            continue
        hotel = _HOTEL.match(core[k])
        if hotel and hotel.group(1).casefold() in _LEVEL_CODES:
            _, a = _core_span(ws[k])
            c = len(hotel.group(1))
            role[k] = "unit_number+number"
            parts[k] = ((a, a + c, "designator"), (a + c, a + len(core[k]), "unit_number+number"))
        elif _SLASH.match(core[k]):
            role[k] = "unit_number+number"
        elif k + 1 < n and role[k + 1] is None and _has_digit(texts[k + 1]):
            role[k] = "unit_number"
        else:
            role[k] = "number"

    # Corner forms: 'Cnr', 'Crn' and '&' anywhere; 'Corner' first after any designator, with its 'of';
    # 'and' only inside a corner form.
    corner_form = False
    for k in range(n):
        if role[k] is None and (fold[k] in ("cnr", "crn") or core[k] == "&"):
            role[k], corner_form = "corner", True
    lead = next((k for k in range(n) if role[k] not in ("designator", "unit_number", "level_number")), None)
    if lead is not None and role[lead] is None and fold[lead] == "corner":
        role[lead], corner_form = "corner", True
        if lead + 1 < n and role[lead + 1] is None and fold[lead + 1] == "of":
            role[lead + 1] = "corner"
    if corner_form:
        for k in range(n):
            if role[k] is None and fold[k] == "and":
                role[k] = "corner"
    for k in range(1, n):  # 'Kilbride Station via Berrima'
        if role[k] is None and fold[k] == "via" and role[k - 1] is None and not _has_digit(texts[k - 1]):
            role[k] = "via"

    # Name words, in runs cut by structure words, commas and line or box changes.
    k = 0
    house = any(r in ("number", "unit_number+number") for r in role)
    while k < n:
        if role[k] is not None:
            k += 1
            continue
        s = k
        while not comma[k] and k + 1 < n and role[k + 1] is None and (ws[k + 1].seg, ws[k + 1].line) == \
                (ws[k].seg, ws[k].line):
            k += 1
        e = k + 1
        before, after = (role[s - 1] if s > 0 else None), (role[e] if e < n else None)
        named = before == "street_name"  # the street's name ends the run before ('12 Smith' / 'Street, ...')
        t = e - 1
        if e - s >= (2 if named else 3) and fold[t] in STREET_SUFFIXES and fold[t - 1] in STREET_TYPE_NAMES:
            t -= 1  # 'Smith Street North'
        typed = [x for x in range(s + (not named), e) if fold[x] in STREET_TYPE_NAMES]
        if alone and before in ("street_type", "street_suffix", "suburb"):
            names = ["suburb"] * (e - s)
        elif after in _NUMBERS:  # a name before the number or a designator ('Westpoint Tower, 45 ...')
            names = ["building"] * (e - s)
        elif (t > s or named) and fold[t] in STREET_TYPE_NAMES:
            names = ["street_name"] * (t - s) + ["street_type"] + ["street_suffix"] * (e - 1 - t)
        elif alone and typed:  # '123 George Street Sydney' with no state or postcode: the suburb follows
            names = ["street_name"] * (typed[-1] - s) + ["street_type"] + ["suburb"] * (e - 1 - typed[-1])
        elif before in ("box_number", "box_extra"):  # 'PO Box 123 City East'
            names = ["box_extra"] * (e - s)
        elif not house and fold[e - 1] in BUILDING_WORDS:  # 'Shop 4 Westfield Plaza', 'Kilbride Station via'
            names = ["building"] * (e - s)
        else:
            names = ["street_name"] * (e - s)
        role[s:e] = names
        k = e
    return [WordRole(w.wid, r, parts.get(k, ())) for k, (w, r) in enumerate(zip(ws, role))]
