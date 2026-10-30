"""Addresses judged as units (eval-0.4; docs/metrics.md, "Address units").

An address unit is a street line (ADDRESS, condition ``address_street_line``) read with its ``<id>-loc``
LOCATION/locality of the same entity, a locality-only LOCATION/locality, or an unsplit ADDRESS
(:func:`benchmark.gt.address_split.address_unit_mentions`). The evaluator judges each part of a unit
with three rules from the address measurement:

- E1, replacement per part. The unit's read-back (:func:`unit_readback`) is the output's text-layer
  tokens on the unit's lines, between the nearest kept words before and after the unit at their
  displayed place. A street line is replaced when an identifying word has new text in the read-back at
  its place or new ink over most of its bands; a locality, when the page has text there, exactly when
  the read-back ends in a locality shape (:func:`locality_shape`): a suburb word, plus a state and a
  4-digit postcode where the original had them.
- E3, token survival in the text layer (:func:`token_survivals`): an identifying word of the unit
  survives when the output's whole tokens equal to it in the unit's region outnumber the input words
  equal to it there that the output keeps.
- E4, short words: a word of at most ``address_short_chars`` letters or digits counts in a channel only
  when all its characters do there, and a fully visible identifying word makes its part at least
  PARTIAL (in the evaluator).

The measurement OCR never decides here: on a page with no text layer it may only add a review flag.
Nothing in this module prints or logs document text.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..geometry import as_poly
from ..gt.address_split import (
    address_roles,
    canonical_street_type,
    address_unit_mentions,
    read_end,
    unit_words,
)
from ..schemas import GTDocument, GTMention
from .config import EvalConfig
from .layers import PageText

STRIP = ".,;:()"
# Roles of the locality part that the locality shape asks the fake to have again.
_STATE_ROLES = ("state", "state+postcode")
_POSTCODE_ROLES = ("postcode", "state+postcode")


@dataclass
class AddressUnit:
    """One address unit: its mentions, its words in reading order with their roles, and its GT lines."""

    key: str  # the street line's mention id, else the locality's
    street: GTMention | None
    locality: GTMention | None
    roles: dict[str, str]  # word id -> role (the case table's vocabulary)
    word_ids: list[str]  # in reading order
    lines: list[tuple[int, str]] = field(default_factory=list)  # (page, GT line id) in reading order

    @property
    def mentions(self) -> list[GTMention]:
        return [m for m in (self.street, self.locality) if m is not None]


@dataclass(frozen=True)
class Token:
    """A whole token of the output's text (or a word the measurement OCR read) on a page."""

    page: int
    index: int  # token number in the page's PageText; -1 for an OCR word
    text: str
    centre: tuple[float, float]  # display pt of the source page


def _line_key(w) -> str:
    return w.line or f"word:{w.id}"


def units_of(gt: GTDocument) -> list[AddressUnit]:
    """The address units of a GT document, with their roles from the GT role reader."""
    words = gt.word_index()
    out = []
    for street, locality in address_unit_mentions(gt):
        uw = unit_words(gt, street, locality)
        roles = {r.wid: r.role for r in address_roles(uw)}
        ids: list[str] = []
        lines: list[tuple[int, str]] = []
        for w in uw:
            if w.wid not in ids:
                ids.append(w.wid)
            pg, gw = words[w.wid]
            key = (pg, _line_key(gw))
            if key not in lines:
                lines.append(key)
        out.append(AddressUnit((street or locality).id, street, locality, roles, ids, lines))
    return out


def page_tokens(pt: PageText, page: int) -> list[Token]:
    """The whole tokens of a page's text with their centres."""
    out = []
    for t, ids in enumerate(pt.tokens):
        if not ids:
            continue
        c = pt.mids[ids].mean(axis=0)
        out.append(Token(page, t, "".join(pt[k].c for k in ids), (float(c[0]), float(c[1]))))
    return out


def ocr_tokens(words, page: int, scale: tuple[float, float] = (1.0, 1.0)) -> list[Token]:
    """Words read by the measurement OCR as tokens (review flags only, never a verdict)."""
    out = []
    for w in words or []:
        x0, y0, x1, y1 = w.bbox
        out.append(Token(page, -1, w.text, ((x0 + x1) / 2 * scale[0], (y0 + y1) / 2 * scale[1])))
    return out


def _axes(angle: float) -> tuple[np.ndarray, np.ndarray]:
    t = math.radians(angle)
    return np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])


def _height(w) -> float:
    p = as_poly(w.poly)
    return float(np.hypot(*(p[3] - p[0])))


def _centre(w) -> np.ndarray:
    return as_poly(w.poly).mean(axis=0)


@dataclass
class Line:
    page: int
    key: str
    words: list  # GT words of the line, in order
    u: np.ndarray
    v: np.ndarray
    across: float  # the line's centre across it, in the frame (u, v)
    h: float  # line height


def page_lines(gt: GTDocument, line_h: dict[str, float]) -> dict[int, list[Line]]:
    """Every GT line of every page, in reading order, in the frame of its first word."""
    out: dict[int, list[Line]] = {}
    for p in gt.pages:
        groups: dict[str, list] = defaultdict(list)
        for w in p.words:
            groups[_line_key(w)].append(w)
        lines = []
        for key, ws in groups.items():
            ws.sort(key=lambda w: w.order)
            u, v = _axes(ws[0].angle)
            across = float(np.median([_centre(w) @ v for w in ws]))
            h = line_h.get(ws[0].id) or float(np.median([_height(w) for w in ws]))
            lines.append(Line(p.index, key, ws, u, v, across, h))
        lines.sort(key=lambda ln: ln.words[0].order)
        out[p.index] = lines
    return out


def _interval(poly, u: np.ndarray) -> tuple[float, float]:
    a = as_poly(poly) @ u
    return float(a.min()), float(a.max())


def unit_readback(unit: AddressUnit, gt: GTDocument, tokens: dict[int, list[Token]],
                  kept_place: dict[str, np.ndarray], shift_pt: dict[int, tuple[float, float]],
                  line_h: dict[str, float], cfg: EvalConfig, lines: dict[int, list[Line]] | None = None
                  ) -> list[Token]:
    """The output's tokens on the unit's lines, in reading order: centre within ``searchable_band`` x H of
    a unit line across it (plus the page's registration shift), and along it between the nearest kept
    words before and after the unit on that line at their displayed place (``kept_place``: kept words
    PRESERVED in place or MOVED), or within ``moved_along`` x H of the unit where a side has none."""
    lines = lines if lines is not None else page_lines(gt, line_h)
    mine = set(unit.word_ids)
    picked: dict[tuple[int, int], tuple[int, float, Token]] = {}
    for li, (pg, key) in enumerate(unit.lines):
        line = next((ln for ln in lines.get(pg, []) if ln.key == key), None)
        if line is None or not tokens.get(pg):
            continue
        own = [w for w in line.words if w.id in mine]
        a0 = min(_interval(w.poly, line.u)[0] for w in own)
        a1 = max(_interval(w.poly, line.u)[1] for w in own)
        before = [w for w in line.words if w.id not in mine and w.id in kept_place and _centre(w) @ line.u < a0]
        after = [w for w in line.words if w.id not in mine and w.id in kept_place and _centre(w) @ line.u > a1]
        start = max((_interval(kept_place[w.id], line.u)[1] for w in before), default=a0 - cfg.moved_along * line.h)
        end = min((_interval(kept_place[w.id], line.u)[0] for w in after), default=a1 + cfg.moved_along * line.h)
        if end < start:
            start, end = a0, a1
        shift = np.asarray(shift_pt.get(pg, (0.0, 0.0)), float)
        for t in tokens[pg]:
            c = np.asarray(t.centre, float) - shift
            along, across = float(c @ line.u), float(c @ line.v)
            if abs(across - line.across) <= cfg.searchable_band * line.h and start <= along <= end:
                k = (pg, t.index if t.index >= 0 else id(t))
                if k not in picked:
                    picked[k] = (li, along, t)
    return [t for _, _, t in sorted(picked.values(), key=lambda x: (x[0], x[1]))]


def _suburb_word(text: str, comma_ok: bool) -> bool:
    """A word that can end a suburb: letters and no digit; a street type closed by a comma ('Road,') only
    where the original suburb was closed by one too ('Glenelg, South Australia')."""
    core = text.strip(STRIP)
    if not any(ch.isalpha() for ch in core) or any(ch.isdigit() for ch in core):
        return False
    return comma_ok or not (text.rstrip().endswith(",") and canonical_street_type(core) is not None)


def locality_shape(texts: list[str], roles: list[str], comma_ok: bool = False) -> tuple[bool, str]:
    """Does a read-back end in a locality like the original's? Read from the end with the GT end reader
    (abbreviated, dotted, spelled-out, doubled and glued states, a 4-digit postcode, a country, 'IN YOUR
    CAPITAL CITY'): a suburb word right before the state (or the postcode), plus a state where the
    original locality had one and a postcode where it had one. ``roles`` are the original locality
    words' roles; ``comma_ok`` says the original suburb ended with a comma. Returns (holds, the parts found
    joined by '+', or 'none')."""
    had_state = any(r in _STATE_ROLES for r in roles)
    had_postcode = any(r in _POSTCODE_ROLES for r in roles)
    had_suburb = "suburb" in roles
    r = read_end(texts, overseas=True) if texts else None
    if r is None:
        return False, "none"
    if r.anchor is not None:
        suburb = r.anchor >= 1 and not r.capital_city and _suburb_word(texts[r.anchor - 1], comma_ok)
    else:
        suburb = not had_state and not had_postcode and any(_suburb_word(t, comma_ok) for t in texts)
    parts = [name for name, ok in (("suburb", suburb), ("state", bool(r.state)), ("postcode", bool(r.postcode)),
                                   ("country", bool(r.country)), ("capital_city", bool(r.capital_city))) if ok]
    if not had_suburb and not had_state and not had_postcode:
        holds = bool(r.capital_city) or not roles
    else:
        holds = suburb and (bool(r.state) or not had_state) and (bool(r.postcode) or not had_postcode)
    return holds, "+".join(parts) or "none"


def locality_roles(unit: AddressUnit, gt: GTDocument) -> tuple[list[str], bool]:
    """Roles of the unit's locality words (of the whole unit when it is unsplit), and whether its last suburb
    word ends with a comma."""
    words = gt.word_index()
    ids = {wid for seg in unit.locality.segments for wid in seg.words} if unit.locality is not None \
        else set(unit.word_ids)
    mine = [w for w in unit.word_ids if w in ids and w in unit.roles]
    suburb = [w for w in mine if unit.roles[w] == "suburb"]
    return [unit.roles[w] for w in mine], bool(suburb) and words[suburb[-1]][1].text.rstrip().endswith(",")


@dataclass
class Region:
    """E3's region of a unit on one page: the unit's GT lines and ``address_adjacent_lines`` GT lines before
    and after them in the same block, across from the first line's band (+-0.5 H) to the last one's, and the
    block's extent along the lines widened 0.25 H. Where a side has fewer adjacent lines, it reaches as many
    line pitches beyond the unit, so a line pushed into a gap or below the block's last line is inside."""

    page: int
    u: np.ndarray
    v: np.ndarray
    lo_across: float
    hi_across: float
    lo: float
    hi: float

    def contains(self, centre, shift=(0.0, 0.0)) -> bool:
        c = np.asarray(centre, float) - np.asarray(shift, float)
        along, across = float(c @ self.u), float(c @ self.v)
        return self.lo <= along <= self.hi and self.lo_across <= across <= self.hi_across


def _block_of(line: Line) -> str | None:
    blocks = Counter(w.block for w in line.words)
    return blocks.most_common(1)[0][0]


def unit_regions(unit: AddressUnit, lines: dict[int, list[Line]], cfg: EvalConfig) -> list[Region]:
    out = []
    for pg in sorted({p for p, _ in unit.lines}):
        on_page = lines.get(pg, [])
        keys = [k for p, k in unit.lines if p == pg]
        own = [ln for ln in on_page if ln.key in keys]
        if not own:
            continue
        blocks = {_block_of(ln) for ln in own}
        seq = [ln for ln in on_page if _block_of(ln) in blocks]
        idx = [i for i, ln in enumerate(seq) if ln.key in keys]
        k = cfg.address_adjacent_lines
        first, last = max(min(idx) - k, 0), min(max(idx) + k, len(seq) - 1)
        u, v = own[0].u, own[0].v
        along = [a for ln in seq for w in ln.words for a in _interval(w.poly, u)]
        h = own[0].h
        across = [float(np.median([_centre(w) @ v for w in ln.words])) for ln in seq]
        steps = np.diff(sorted(across))
        pitch = float(np.median(steps[steps > 0.5 * h])) if np.any(steps > 0.5 * h) else 1.4 * h
        lo_a = min(across[i] - 0.5 * seq[i].h for i in range(first, last + 1))
        hi_a = max(across[i] + 0.5 * seq[i].h for i in range(first, last + 1))
        lo_a -= max(0, k - (min(idx) - first)) * pitch
        hi_a += max(0, k - (last - max(idx))) * pitch
        out.append(Region(pg, u, v, lo_a, hi_a, min(along) - 0.25 * h, max(along) + 0.25 * h))
    return out


_JOINS = re.compile(r"[,;:()/|]+")


def same_token(needle: str, token: str) -> bool:
    """Is an output token equal to an address word? Case folded and stripped of punctuation: a number as a
    maximal digit run ('214' and '142750' do not hold '14'), anything else as the whole token or as a part
    of it between commas, colons, semicolons, brackets, slashes or bars (a GT word split between the street
    line and the locality, 'St,Penrith'), a word of letters also as the letters of a part glued to digits
    ('NSW2750'). A word inside an email or a web address ('example' in 'jane@example.com') is not one."""
    n = needle.strip(STRIP).casefold()
    if not n:
        return False
    t = token.casefold()
    if n.isdigit():
        return any(m.group() == n for m in re.finditer(r"\d+", t))
    if t.strip(STRIP) == n:
        return True
    for part in _JOINS.split(t):
        part = part.strip(STRIP)
        if part == n:
            return True
        if n.isalpha() and any(ch.isdigit() for ch in part) and \
                any(m.group() == n for m in re.finditer(r"[^\W\d_]+", part)):
            return True
    return False


def token_survivals(needles: dict[str, str], out_tokens: list[Token], in_words: list[str]) -> set[str]:
    """E3: the identifying words (word id -> needle) that survive: more output tokens equal to the word in
    the region than input words equal to it there that stay in the output (``in_words``, the allowance)."""
    out = set()
    for wid, n in needles.items():
        if not n.strip(STRIP):
            continue
        have = sum(1 for t in out_tokens if same_token(n, t.text))
        allow = sum(1 for t in in_words if same_token(n, t))
        if have > allow:
            out.add(wid)
    return out


def widened(poly, angle: float, d: float) -> np.ndarray:
    """A word polygon (TL, TR, BR, BL) grown by ``d`` pt at both ends along its reading direction."""
    p = as_poly(poly).astype(float).copy()
    u, _ = _axes(angle)
    p[[0, 3]] -= u * d
    p[[1, 2]] += u * d
    return p


def at_place(t: Token, poly, h: float, shift, cfg: EvalConfig) -> bool:
    """Is a token at a word's displayed place (the searchable tolerance)?"""
    p = as_poly(poly)
    u = p[1] - p[0]
    length = float(np.hypot(*u))
    u = u / length if length > 0 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    c = np.asarray(t.centre, float) - np.asarray(shift, float) - p.mean(axis=0)
    return abs(float(c @ v)) <= cfg.searchable_band * h and abs(float(c @ u)) <= length / 2 + cfg.searchable_margin * h


def is_new_text(original: str, got: str) -> bool:
    """A read-back token that is neither the original word nor a fragment of it."""
    a = "".join(ch for ch in original.casefold() if ch.isalnum())
    b = "".join(ch for ch in got.casefold() if ch.isalnum())
    return bool(b) and b != a and b not in a
