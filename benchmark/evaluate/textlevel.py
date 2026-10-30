"""Text-level scoring path (eval-0.4; docs/metrics.md, "Text-level evaluation"): a tool that emits redacted
text instead of a PDF is graded on that text alone.

The tool hands over ``redacted_text.json`` (:class:`benchmark.schemas.RedactedTextArtifact`): the text it
redacted, as it extracted it or as the benchmark injected it (``original_text``), and its redacted text.

1. GT words, in reading order (page, order), are aligned to the original text's whitespace tokens on
   case-folded tokens stripped of punctuation (difflib, autojunk off). When the original text holds one
   ``page_break`` per GT page break, each page is aligned and diffed on its own. A GT word with no token is
   NOT_EVALUATED, and so is every mention with such a word.
2. The original text is diffed against the redacted text (Levenshtein opcodes): each character of an aligned
   GT word is kept (equal) or changed.
3. Words get the PDF path's statuses and fractions on kept characters: PRESERVED, DAMAGED, REMOVED. MOVED
   cannot occur.
4. Mentions use the PDF path's units, shared roles and cut-offs. A character is visible when it is kept in
   place; the run rule of :func:`benchmark.evaluate.cells.unit_flags` applies per word over its characters,
   and E4 to address parts. A displaced original is an identifying whole token of the mention (a number as a
   maximal digit run) found in the redacted text more often than the original text holds it outside the MUST
   mentions plus the MUST occurrences kept in place (E3's count over the whole document); its characters are
   visible in every mention holding it. A mention is replaced when the diff puts new text with a letter or
   digit at it that is not a placeholder (a bracketed tag, a run of X, asterisks or block characters, a tag
   in capitals naming a PII type or a redaction, or one text equal across every replaced mention), and an
   address part's locality also needs E1's locality shape on the unit's new text. LEAK_RECOVERABLE and
   LEAK_LATENT cannot occur: the text is the whole output.
5. Detection reads the changed spans (runs of changed GT words, through shared words left as they were) as
   predictions; the pseudonym checks read the new text per GT word. Signatures, marks and images have no text:
   their mentions are left out.

Nothing here prints or logs document text.
"""

from __future__ import annotations

import bisect
import difflib
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from ..gt.address_split import STATE_CODES, postcode_in_state
from ..gt.policy import Effective, resolve
from ..gt.shared_roles import STRIP as SHARED_STRIP
from ..gt.shared_roles import SharedUnits, document_shared_units
from ..schemas import Action, GTDocument, PiiType, PolicyProfile, Requirement
from ..schemas.artifacts import RedactedTextArtifact
from .addresses import AddressUnit, locality_roles, locality_shape, same_token, units_of
from .cells import run_needed
from .config import EvalConfig
from .evaluator import _fraction, _searchable, _summarise
from .outcomes import (
    DocumentEvaluation,
    Integrity,
    MentionOutcome,
    MentionStatus,
    PageOutcome,
    WordOutcome,
    WordStatus,
)

EVIDENCE = "text"
# A redaction tag in capitals: a PII type, or a word for a redaction ('PERSON', 'ADDRESS_2', 'REDACTED').
_TAG_WORDS = ({p.value for p in PiiType} | {w for p in PiiType for w in p.value.split("_")}
              | {"REDACTED", "REMOVED", "WITHHELD", "HIDDEN", "MASKED", "ANONYMISED", "ANONYMIZED", "NAME", "PERSON",
                 "PII", "ID", "NUMBER", "ORGANIZATION"})
_BLOCKS = set("Xx*#█▇▆▅▄▃▂▁■□▓▒░▉▊▋▌▍▎▏-_.·•")
_BRACKETED = re.compile(r"^[\[<{(].*[^\W\d_].*[\]>})][.,;:]*$")


def _key(token: str) -> str:
    """A token as aligned: case folded, letters and digits only (punctuation alone stays as written)."""
    k = "".join(ch for ch in token.casefold() if ch.isalnum())
    return k or token


def is_placeholder(token: str) -> bool:
    """A redaction placeholder rather than a fake: a bracketed tag ('[REDACTED]', '<PERSON>', '{NAME_1}',
    '[Redacted]': letters in capitals, or only words for a PII type or a redaction), a run of X, asterisks
    or block characters, or a tag in capitals naming a PII type or a redaction ('PERSON_1', 'REDACTED'). A
    fake written in brackets is no tag: '(Ruth)', '[Dunbar]', '<ruth.dunbar@example.com>'."""
    t = token.strip(".,;:")
    if not t:
        return True
    if _BRACKETED.match(t):
        inside = t.rstrip(".,;:")[1:-1]
        words = re.sub(r"[^\W\d_]+", lambda m: " " + m.group() + " ", inside).split()
        letters = [w for w in words if w.isalpha()]
        if "@" not in inside and (inside.upper() == inside or all(w.upper() in _TAG_WORDS for w in letters)):
            return True
        return False
    if set(t) <= _BLOCKS:
        return True
    letters = re.sub(r"[\d_\-]+", " ", t).split()
    return t.upper() == t and bool(letters) and all(w in _TAG_WORDS for w in letters)


# -- alignment and diff --------------------------------------------------------------------------------


@dataclass
class TextDiff:
    """The original and redacted texts, the GT words' characters in the original text, and the diff."""

    original: str
    redacted: str
    # word id -> the position in ``original`` of each of its characters (None: not matched)
    chars: dict[str, list[int | None]] = field(default_factory=dict)
    kept: list[bool] = field(default_factory=list)  # per character of ``original``: kept in place by the diff
    new: list[bool] = field(default_factory=list)  # per character of ``redacted``: not an original character
    left: list[int] = field(default_factory=list)  # per boundary of ``original``: its position in ``redacted``
    right: list[int] = field(default_factory=list)  # ...after any text inserted there
    per_page: bool = False  # aligned and diffed page by page
    page_breaks: tuple[int, int] = (0, 0)  # page breaks in the original and redacted texts
    _tokens: list[tuple[int, int]] | None = field(default=None, repr=False)  # whitespace tokens of ``redacted``

    def aligned(self, wid: str) -> bool:
        return any(p is not None for p in self.chars.get(wid, []))

    def char_kept(self, wid: str) -> list[bool | None]:
        return [None if p is None else self.kept[p] for p in self.chars.get(wid, [])]

    def span(self, wid: str) -> tuple[int, int] | None:
        """The word's extent in ``original`` (first to last matched character)."""
        ps = [p for p in self.chars.get(wid, []) if p is not None]
        return (min(ps), max(ps) + 1) if ps else None

    def dest(self, a: int, b: int) -> tuple[int, int]:
        """The extent in ``redacted`` of the original extent [a, b), with text inserted at either end."""
        return self.left[a], max(self.left[a], self.right[b])


def _pages(text: str, brk: str) -> list[tuple[int, int]]:
    out, start = [], 0
    if brk:
        for m in re.finditer(re.escape(brk), text):
            out.append((start, m.start()))
            start = m.end()
    out.append((start, len(text)))
    return out


def _word_chars(word: str, token: str, offset: int) -> list[int | None]:
    """The position of each character of a GT word in its token (case folded, matched in order)."""
    if word == token:
        return [offset + k for k in range(len(word))]
    pos: list[int | None] = [None] * len(word)
    sm = difflib.SequenceMatcher(None, word.casefold(), token.casefold(), autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            pos[a + k] = offset + b + k
    return pos


def _diff_chars(d: TextDiff, oa: int, ob: int, ra: int, rb: int) -> None:
    """The character diff of original[oa:ob] against redacted[ra:rb], with every boundary of the original
    mapped into the redacted text: the leftmost and rightmost place of any edit touching it, so text
    inserted at a boundary lies between the two."""
    if oa >= ob and ra >= rb:
        d.left[oa] = min(d.left[oa], ra)
        d.right[oa] = max(d.right[oa], ra)
        return
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(d.original[oa:ob], d.redacted[ra:rb]):
        i1, i2, j1, j2 = i1 + oa, i2 + oa, j1 + ra, j2 + ra
        n = i2 - i1
        if tag == "equal":
            for k in range(n):
                d.kept[i1 + k] = True
                d.new[j1 + k] = False
        for k in range(n + 1):
            if tag == "equal":
                lo = hi = j1 + k
            else:
                j = j1 + (round(k * (j2 - j1) / n) if n else 0)
                lo, hi = (j1 if k == 0 else j), (j2 if k == n else j)
            d.left[i1 + k] = min(d.left[i1 + k], lo)
            d.right[i1 + k] = max(d.right[i1 + k], hi)


def align(gt: GTDocument, art: RedactedTextArtifact) -> TextDiff:
    """Align the GT words to the original text and diff the original text against the redacted text."""
    o, r = art.original_text, art.redacted_text
    unset = len(r) + 1
    d = TextDiff(o, r, kept=[False] * len(o), new=[True] * len(r), left=[unset] * (len(o) + 1),
                 right=[-1] * (len(o) + 1))
    po, pr = _pages(o, art.page_break), _pages(r, art.page_break)
    d.page_breaks = (len(po) - 1, len(pr) - 1)
    pages = sorted(gt.pages, key=lambda p: p.index)
    if len(po) == len(pages) == len(pr) and len(pages) > 1:
        d.per_page = True
        chunks = [([p], po[k], pr[k]) for k, p in enumerate(pages)]
    else:
        chunks = [(pages, (0, len(o)), (0, len(r)))]
    for gpages, (oa, ob), (ra, rb) in chunks:
        gws = [w for p in gpages for w in sorted(p.words, key=lambda w: w.order)]
        toks = [(m.start() + oa, m.group()) for m in re.finditer(r"\S+", o[oa:ob])]
        sm = difflib.SequenceMatcher(None, [_key(w.text) for w in gws], [_key(t) for _, t in toks], autojunk=False)
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                w, (off, tok) = gws[a + k], toks[b + k]
                d.chars[w.id] = _word_chars(w.text, tok, off)
        # The diff in two steps: whole tokens (line breaks are tokens) first, then characters only between
        # the tokens matched. A character diff of the whole text can merge a deleted mention with the next
        # line's new text into one edit, which then spread the fake over both (a deleted 'Peter Strong'
        # credited with the 'Ruth Dunbar' that replaced 'SMITHERS Jane' below it).
        ot = [(m.start() + oa, m.end() + oa) for m in re.finditer(r"\n|[^\s]+", o[oa:ob])]
        rt = [(m.start() + ra, m.end() + ra) for m in re.finditer(r"\n|[^\s]+", r[ra:rb])]
        tm = difflib.SequenceMatcher(None, [o[a:b] for a, b in ot], [r[a:b] for a, b in rt], autojunk=False)
        pi, pj = oa, ra
        for a, b, n in tm.get_matching_blocks():
            _diff_chars(d, pi, ot[a][0] if a < len(ot) else ob, pj, rt[b][0] if b < len(rt) else rb)
            if n:
                _diff_chars(d, ot[a][0], ot[a + n - 1][1], rt[b][0], rt[b + n - 1][1])
                pi, pj = ot[a + n - 1][1], rt[b + n - 1][1]
    # Boundaries no diff reached (page breaks between pages) take the place of the boundary before them.
    prev = 0
    for i in range(len(o) + 1):
        if d.right[i] < 0:
            d.left[i] = d.right[i] = prev
        prev = d.right[i]
    return d


# -- the evaluation ---------------------------------------------------------------------------------------


def _visible(kept: list[bool | None]) -> list[bool]:
    """Characters that count as visible: kept, in a run of kept characters as long as the PDF path's run
    rule asks for the word's length (1 for initials, 2 for short words, 3 from six characters)."""
    need = run_needed(len(kept))
    out = [False] * len(kept)
    k = 0
    while k < len(kept):
        if not kept[k]:
            k += 1
            continue
        e = k
        while e + 1 < len(kept) and kept[e + 1]:
            e += 1
        if e - k + 1 >= need:
            for i in range(k, e + 1):
                out[i] = True
        k = e + 1
    return out


def _token_has(needle: str, token: str) -> bool:
    """Is ``token`` (a whitespace token of a text) the identifying word ``needle``? As E3 compares tokens
    (:func:`benchmark.evaluate.addresses.same_token`), and an email's local part also as the part before '@'."""
    if same_token(needle, token):
        return True
    if "@" in token:
        local = token.split("@", 1)[0].strip(SHARED_STRIP + "<>\"'").casefold()
        return bool(local) and local == needle.strip(SHARED_STRIP).casefold()
    return False


@dataclass
class _Mention:
    """What the mention loop needs again after it: the new text and its placeholder test."""

    key: str  # the new tokens, letters and digits only, case folded ('' when none)
    placeholder: bool


def _tokens_over(d: TextDiff, a: int, b: int, new_only: bool = False) -> list[str]:
    """The whole tokens of ``redacted`` overlapping [a, b) (``new_only``: those holding a character not kept
    from the original there). Whole tokens: the diff can keep a letter of a fake that equals the original's
    ('d' of 'Removed' over 'Strong'), which would otherwise cut the token at the mention's edge."""
    if d._tokens is None:
        d._tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", d.redacted)]
    out = []
    for s, e in d._tokens[max(0, bisect.bisect_right(d._tokens, (a, len(d.redacted) + 1)) - 1):]:
        if s >= b:
            break
        if e > a and (not new_only or any(d.new[k] for k in range(max(s, a), min(e, b)))):
            out.append(d.redacted[s:e])
    return out


def _state_postcode_follows(d: TextDiff, s_wid: str, state: str) -> bool:
    """Is the kept state word followed, in the redacted text, by a 4-digit postcode of that state?"""
    sp = d.span(s_wid)
    if sp is None:
        return False
    _, b = d.dest(*sp)
    m = re.match(r"\s*(\S+)", d.redacted[b:])
    pc = m.group(1).strip(SHARED_STRIP) if m else ""
    code = STATE_CODES.get("".join(ch for ch in state.casefold() if ch.isalnum()))
    return len(pc) == 4 and pc.isdigit() and code is not None and postcode_in_state(code, pc)


def _mention_extent(d: TextDiff, wids: list[str]) -> tuple[int, int] | None:
    spans = [d.span(w) for w in wids if d.span(w) is not None]
    if not spans:
        return None
    return d.dest(min(s[0] for s in spans), max(s[1] for s in spans))


def score_text_document(gt: GTDocument, policy: PolicyProfile, art: RedactedTextArtifact,
                        config: EvalConfig | None = None, diff: TextDiff | None = None) -> DocumentEvaluation:
    """Evaluate a tool's redacted text against GT, with the PDF path's statuses, cut-offs and summary keys
    (``evidence`` 'text')."""
    cfg = config or EvalConfig()
    res = resolve(gt, policy)
    d = diff or align(gt, art)
    words = gt.word_index()
    entities = {e.id: e for e in gt.entities}
    visible: dict[str, list[bool]] = {}
    for page in gt.pages:
        for w in page.words:
            ck = d.char_kept(w.id)
            visible[w.id] = _visible([bool(c) for c in ck]) if ck else []

    # -- words --------------------------------------------------------------------------------------
    word_outcomes: list[WordOutcome] = []
    for page in gt.pages:
        for w in page.words:
            eff = res.words.get(w.id, Effective.ORDINARY)
            ks = [c for c in d.char_kept(w.id) if c is not None]
            if not ks:
                word_outcomes.append(WordOutcome(w.id, page.index, eff.value, None, WordStatus.NOT_EVALUATED))
                continue
            frac = _fraction(ks)
            st = (WordStatus.PRESERVED if frac >= cfg.word_preserved_fraction
                  else WordStatus.REMOVED if frac <= cfg.word_removed_fraction else WordStatus.DAMAGED)
            word_outcomes.append(WordOutcome(w.id, page.index, eff.value, round(frac, 4), st))

    # -- displaced originals: identifying whole tokens found more often than expected -----------------
    must_pos: set[int] = set()
    for m in gt.mentions:
        if res.mentions[m.id].requirement is Requirement.MUST:
            for seg in m.segments:
                for wid in seg.words:
                    must_pos.update(p for p in d.chars.get(wid, []) if p is not None)
    orig_tokens = [(mt.start(), mt.group()) for mt in re.finditer(r"\S+", d.original)]
    outside = [t for s, t in orig_tokens if not any(p in must_pos for p in range(s, s + len(t)))]
    red_tokens = [mt.group() for mt in re.finditer(r"\S+", d.redacted)]
    must_kept: list[str] = [words[wid][1].text for m in gt.mentions
                            if res.mentions[m.id].requirement is Requirement.MUST
                            for seg in m.segments for wid in seg.words
                            if visible.get(wid) and all(visible[wid])]
    displaced_cache: dict[str, bool] = {}

    def displaced(needle: str) -> bool:
        n = needle.strip(SHARED_STRIP)
        if not any(ch.isalnum() for ch in n):
            return False
        if n not in displaced_cache:
            have = sum(1 for t in red_tokens if _token_has(n, t))
            allow = sum(1 for t in outside if _token_has(n, t)) + sum(1 for t in must_kept if _token_has(n, t))
            displaced_cache[n] = have > allow
        return displaced_cache[n]

    # -- mentions ------------------------------------------------------------------------------------
    shared = document_shared_units(gt, policy)
    unit_of: dict[str, AddressUnit] = {m.id: u for u in units_of(gt) for m in u.mentions}
    outcomes: list[MentionOutcome] = []
    news: dict[str, _Mention] = {}
    excused: set[str] = set()
    pending: list[tuple] = []
    for m in gt.mentions:
        wids = [wid for seg in m.segments for wid in seg.words]
        if not wids:
            continue  # a signature, mark or image: no text to judge
        r = res.mentions[m.id]
        ent = entities.get(m.entity) if m.entity else None
        su = shared.get(m.id, SharedUnits())
        pair_log: list[list] = []
        for s_wid, p_wid, ks in list(su.pairs):
            # The pair holds when the state is not seen, or is seen with the postcode's prefix digits kept
            # (each digit on its own, with no run rule, as the PDF path checks them).
            state_seen = any(visible.get(s_wid, [])[k] for k in range(len(visible.get(s_wid, [])))
                             if (s_wid in su.full or (s_wid, k) in su.chars) and k not in ks)
            pk = d.char_kept(p_wid)
            if ks:
                ok = not state_seen or all(k < len(pk) and pk[k] for k in ks)
            else:
                # No prefix digit to keep (NSW 1xxx, VIC 8xxx, QLD 9xxx, ACT 02xx): the kept state must be
                # followed by a postcode of that state, as the PDF path reads it from the text layer.
                ok = state_seen and _state_postcode_follows(d, s_wid, words[s_wid][1].text)
            pair_log.append([s_wid, ok])
            if not ok:
                su.without_pair(s_wid, p_wid, ks)
        excused.update(su.full)
        units: list[tuple[str, int]] = []
        for seg in m.segments:
            for wid in seg.words:
                n = len(words[wid][1].text)
                lo, hi = (seg.char_start, seg.char_end) if seg.char_start is not None else (0, n)
                units.extend((wid, k) for k in range(lo, hi))
        shared_units = [u for u in units if su.shared(u)]
        ident_words = [wid for wid in dict.fromkeys(wids) if wid not in su.full]
        unaligned = [wid for wid in ident_words if not d.aligned(wid)]
        units = [u for u in units if not su.shared(u) and d.chars.get(u[0]) and u[1] < len(d.chars[u[0]])
                 and d.chars[u[0]][u[1]] is not None]
        unit = unit_of.get(m.id)
        # Identifying words and their needles (a word with shared characters by its identifying slice).
        needles: dict[str, str] = {}
        for seg in m.segments:
            for wid in seg.words:
                if wid in su.full:
                    continue
                t = words[wid][1].text
                if seg.char_start is not None:
                    t = t[seg.char_start:seg.char_end]
                if wid in su.needle_cut:
                    a, b = su.needle_cut[wid]
                    t = words[wid][1].text[a:b]
                needles[wid] = t.strip(SHARED_STRIP)
        moved = {wid for wid, t in needles.items() if t and displaced(t)}
        # E4: an address part's word of at most address_short_chars letters or digits counts only whole.
        short = set()
        if unit is not None:
            short = {wid for wid in needles if sum(ch.isalnum() for ch in words[wid][1].text) <= cfg.address_short_chars}

        def seen(u) -> bool:
            return u[0] in moved or bool(visible[u[0]][u[1]])

        flags = [seen(u) for u in units]
        for wid in short:
            idx = [k for k, u in enumerate(units) if u[0] == wid]
            if not all(flags[k] for k in idx):
                for k in idx:
                    flags[k] = False
        f_vis = _fraction(flags)
        f_disp = _fraction([u[0] in moved and not visible[u[0]][u[1]] for u in units])
        word_seen = False
        if unit is not None:
            by_word: dict[str, list[bool]] = defaultdict(list)
            for u, v in zip(units, flags):
                by_word[u[0]].append(v)
            word_seen = any(vs and all(vs) for vs in by_word.values())
        ext = _mention_extent(d, ident_words or wids)
        new = _tokens_over(d, *ext, new_only=True) if ext else []
        key = "".join(ch for t in new for ch in t.casefold() if ch.isalnum())
        news[m.id] = _Mention(key, bool(new) and all(is_placeholder(t) for t in new))
        pending.append((m, r, ent, su, pair_log, units, unaligned, f_vis, f_disp, word_seen, unit, shared_units))

    # One text equal across every replaced required mention is a placeholder too ('Removed', 'Name'), once it
    # stands for two different originals (one person's fake used twice is no placeholder).
    replaced_must = [m for m, r, *_ in pending if r.requirement is Requirement.MUST and news[m.id].key]
    same_everywhere = (len({news[m.id].key for m in replaced_must}) == 1
                       and len({_key(m.text or "") for m in replaced_must}) >= 2)

    for m, r, ent, su, pair_log, units, unaligned, f_vis, f_disp, word_seen, unit, shared_units in pending:
        nm = news[m.id]
        replaced = bool(nm.key) and not nm.placeholder and not (same_everywhere and r.requirement is Requirement.MUST)
        shape = None
        if replaced and unit is not None and m.type is PiiType.LOCATION:
            # E1: the locality part needs a locality on the unit's new text.
            ext = _mention_extent(d, unit.word_ids)
            texts = _tokens_over(d, *ext) if ext else []
            state_kept = any(unit.roles.get(w) in ("state", "state+postcode") for w in su.full)
            holds, shape = locality_shape(texts, *locality_roles(unit, gt), state_kept=state_kept)
            replaced = holds
        pages = sorted({seg.page for seg in m.segments})
        if unaligned or not units:
            status = MentionStatus.NOT_EVALUATED
        elif f_vis >= cfg.leak_min_residual:
            status = MentionStatus.LEAK_VISIBLE
        elif f_vis > cfg.safe_max_residual or word_seen:
            status = MentionStatus.PARTIAL
        elif replaced:
            status = MentionStatus.SAFE_REPLACED
        elif r.action is Action.REPLACE:
            status = MentionStatus.NOT_REPLACED
        else:
            status = MentionStatus.SAFE_REMOVED
        sv = [bool(visible.get(w, [])[k]) for w, k in shared_units if k < len(visible.get(w, []))]
        outcomes.append(MentionOutcome(
            mention_id=m.id, type=m.type.value, subtype=m.subtype, entity=m.entity,
            role=ent.role.value if ent else None, requirement=r.requirement.value, pending=r.pending,
            pages=pages, conditions=list(m.conditions), cells=len(units), visible_residual=round(f_vis, 4),
            worst_config=EVIDENCE if f_vis > 0 else None, latent_residual=0.0, text_residual=0.0,
            image_residual=0.0, image_latent_residual=0.0, vector_residual=0.0, revision_residual=0.0,
            replaced=replaced, l2_sources=[], status=status, displaced_residual=round(f_disp, 4),
            shared_words=sorted(su.full), shared_chars=sorted([w, k] for w, k in su.chars),
            shared_visible=round(_fraction(sv), 4), shared_pairs=pair_log,
            address_unit=unit.key if unit is not None else None, locality_shape=shape))

    # -- pages, integrity, summary ---------------------------------------------------------------------
    page_outcomes = [PageOutcome(p.index, any(d.aligned(w.id) for w in p.words), False, None, 0.0, 0.0, 0.0, [],
                                 False) for p in gt.pages]
    warnings = []
    if d.page_breaks[0] and not d.per_page:
        warnings.append(f"page breaks: original {d.page_breaks[0]}, redacted {d.page_breaks[1]}, GT pages "
                        f"{len(gt.pages)}; aligned as one text")
    integrity = Integrity(True, d.per_page or not d.page_breaks[0], [], warnings)
    word_text = {w.id: w.text for p in gt.pages for w in p.words}
    _, text_info = _searchable(gt, [], {}, {}, {}, cfg)  # n/a: the text is the whole output
    summary = _summarise(outcomes, word_outcomes, [], [], integrity, page_outcomes, word_text=word_text,
                         searchable={}, cfg=cfg, excused=excused)
    summary.update(text_info)
    return DocumentEvaluation(
        doc_id=gt.doc_id, eval_version=cfg.version, eval_digest=cfg.digest(), policy=res.policy,
        source_sha256=hashlib.sha256(art.original_text.encode("utf-8")).hexdigest(),
        output_sha256=hashlib.sha256(art.redacted_text.encode("utf-8")).hexdigest(),
        render_configs=[EVIDENCE], integrity=integrity, mentions=outcomes, words=word_outcomes,
        pages=page_outcomes, findings=[], surfaces=[], revisions_checked=0, summary=summary, evidence=EVIDENCE)


# -- detection and pseudonyms from the diff ----------------------------------------------------------------


def changed_spans(gt: GTDocument, d: TextDiff, ev: DocumentEvaluation, art: RedactedTextArtifact,
                  label_map: dict | None = None):
    """The tool's redactions as predictions: runs of consecutive changed GT words in reading order on a page,
    carried through shared words the evaluation excused and left as they were. A prediction's label comes
    from the tool's mappings when one's original equals the run's original text, its type from the label map."""
    from ..metrics.detection import ProjectedMention

    excused = {w for m in ev.mentions for w in m.shared_words}
    claims = {_key(x.original): x.label for x in art.mappings}
    # A run ends at a GT line change unless the next word shares a mention or an address unit with the
    # last: two required mentions with only a line break between them are two redactions.
    groups: dict[str, set] = defaultdict(set)
    for m in gt.mentions:
        for seg in m.segments:
            for wid in seg.words:
                groups[wid].add(m.id)
    for u in units_of(gt):
        for wid in u.word_ids:
            groups[wid].add(("unit", u.key))
    preds = []
    for page in gt.pages:
        run: list = []
        held: list = []

        def close():
            if run:
                text = _key(" ".join(w.text for w in run))
                label = claims.get(text, "changed")
                typ = (label_map or {}).get(label)
                preds.append(ProjectedMention(f"t{len(preds)}", label, getattr(typ, "value", typ),
                                              {(page.index, w.id) for w in run}))

        for w in sorted(page.words, key=lambda w: w.order):
            ks = d.char_kept(w.id)
            if not ks or all(c is None for c in ks):
                close()
                run, held = [], []
                continue
            if not all(c for c in ks if c is not None):
                last = (run + held)[-1] if run else None
                if last is not None and last.line != w.line and not groups[last.id] & groups[w.id]:
                    close()
                    run, held = [], []
                run += held + [w]
                held = []
            elif run and w.id in excused:
                held.append(w)
            else:
                close()
                run, held = [], []
        close()
    return preds


def word_replacements(gt: GTDocument, d: TextDiff) -> dict[str, str | None]:
    """The text at each GT word of a mention in the redacted text: a mention's new text split on white space
    when it has one token per word, else cut where the diff puts each word's start."""
    out: dict[str, str | None] = {}
    words = gt.word_index()
    for m in gt.mentions:
        groups = [[wid for seg in m.segments for wid in seg.words], list(m.optional_words)]
        for wids in groups:
            wids = [w for w in dict.fromkeys(wids) if d.span(w) is not None]
            if not wids:
                continue
            ext = _mention_extent(d, wids)
            toks = _tokens_over(d, *ext)
            if len(toks) == len(wids):
                out.update({w: t for w, t in zip(wids, toks)})
                continue
            starts = [d.left[d.span(w)[0]] for w in wids] + [ext[1]]
            for k, w in enumerate(wids):
                out[w] = d.redacted[starts[k]:max(starts[k], starts[k + 1])].strip() or None
    for wid in list(out):
        if wid not in words:
            out.pop(wid)
    return out


def unit_texts(gt: GTDocument, d: TextDiff) -> dict[str, list[str]]:
    """Address unit key -> the whitespace tokens of the redacted text over the unit (its read-back)."""
    out = {}
    for u in units_of(gt):
        ext = _mention_extent(d, u.word_ids)
        out[u.key] = _tokens_over(d, *ext) if ext else []
    return out
