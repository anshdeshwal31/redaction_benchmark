"""Mock candidates: GT-driven redactors with controlled faults (PLAN.md §15 phase 3).

A mock knows the ground truth and fails only in the ways it is told to, so every
fault the evaluator must detect can be produced on demand and the evaluator's
verdicts can be checked against the injected cause ("mutation testing" for the
benchmark).

Mocks run the same stages as the reference model (PLAN.md §3.1) and emit the
artifact of each stage, so faults propagate the way they would in a real tool:

    [T] text      GT words, optionally corrupted or dropped        -> text.json
    [D] detect    MUST mentions restricted to recognised words     -> mentions.json
    [R] resolve   clusters from GT entities, optionally merged/split -> entities.json
    [L] localize  regions from detected words + geometric faults   -> plan.json
    [P] replace   pseudonyms keyed on *predicted* clusters         -> pseudonyms.json
                  (or on GT entities, one fake per person across a matter)
    [A] apply     true redaction / overlay / rasterise / ...       -> output.pdf

A mock follows the policy it is given: it targets the mentions that policy requires. With
``replace_all``, ``stand_ins`` and ``redacted_text_layer`` it can also do everything the client's
rules ask for (a fake of the right kind for every required item, a drawn stand-in for every
signature or image, a searchable text layer), which makes it the ceiling a real tool is
compared against.

Fakes follow the address case table's constraints (docs/address_cases.md): no fake word equals,
begins with or holds an identifying word of an address of the document, every digit of a number
changes at its position, and the shared structure the policy enables stays as written (the state
with a same-state postcode keeping its prefix, a street type, box and unit words, an organisation's
legal form, an email domain, an identifier's prefix); a mobile keeps its '04'. The eval-0.4
self-checks draw fakes the way tools do: in the original's font, size and place, over or beside
the original, re-typeset lines, and addresses faked as units (``fakes_in_place`` and the options
after it).
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np
import pikepdf
import pymupdf
import shapely
from rapidfuzz import fuzz

from ..geometry import Frame, PageGeometry, as_poly, bbox, dilate, geometry_from_pikepdf_page, rect_poly, to_shapely
from ..gt.address_split import STATES, STREET_TYPES, address_unit_mentions, read_end
from ..gt.policy import resolve
from ..gt.shared_roles import PREFIX, SharedUnits, document_shared_units
from ..metrics.pseudonyms import POSTCODES
from ..schemas import (
    Cluster,
    EntitiesArtifact,
    GTDocument,
    GTPage,
    MentionReplacement,
    MentionsArtifact,
    NameComponent,
    PiiType,
    PlanArtifact,
    PlanRegion,
    PolicyProfile,
    PredMention,
    PseudonymEntry,
    PseudonymsArtifact,
    RegionAnchor,
    Requirement,
    TextArtifact,
    TextPage,
    TextToken,
)

Method = Literal["true", "rasterize", "overlay", "annotation", "translucent", "blur", "null", "corrupt"]


@dataclass(frozen=True)
class MockSpec:
    name: str
    method: Method = "true"
    include_may: bool = False
    # [T] text acquisition faults
    ocr_char_error: float = 0.0  # per-character substitution probability
    ocr_drop_types: tuple[str, ...] = ()  # PII types whose words the "OCR" fails to read
    ocr_drop_conditions: tuple[str, ...] = ()  # mention conditions whose words go unread
    ocr_box_jitter_pt: float = 0.0
    # [D] detection faults
    drop_types: tuple[str, ...] = ()
    drop_conditions: tuple[str, ...] = ()
    drop_fraction: float = 0.0
    boundary: Literal["none", "shrink_last", "extend_title", "extend_next"] = "none"
    type_swap: tuple[tuple[str, str], ...] = ()
    redact_protected: bool = False  # false positives on protected spans
    # [R] resolution faults
    merge_same_given: bool = False
    split_aliases: bool = False
    split_conditions: tuple[str, ...] = ()  # mentions with these conditions get a cluster of their own
    # [L] localization faults
    shift_pt: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    y_flip: bool = False
    page_offset: int = 0
    rotation_unaware: bool = False
    cropbox_unaware: bool = False
    partial: float = 1.0  # fraction of each word covered, from its start
    dilate_pt: float = 0.0
    # [A] application faults
    incremental: bool = False
    keep_text_layer: bool = False  # rasterise, then re-add the original words invisibly
    leak_metadata: bool = False  # writes the claimant's name into /Author
    # [P] replacement
    pseudonyms: Literal["none", "consistent", "inconsistent", "gender_flip", "collision"] = "none"
    replace_other: Literal["none", "valid", "invalid"] = "none"  # phones, emails, addresses, Medicare
    replace_all: bool = False  # every detected required mention gets a valid fake of its kind
    pseudonym_key: Literal["cluster", "entity"] = "cluster"  # entity: fakes from a hash of the GT entity id
    # [A] application options
    stand_ins: bool = False  # signatures, marks and images: paper-white plus drawn pen strokes, not a box
    redacted_text_layer: bool = False  # rasterize: fakes set as visible text, kept words as invisible text
    raster_dpi: float = 150
    # Fakes drawn the way tools draw them (eval-0.4 self-checks), each in the original's font (a base-14
    # stand-in of its family) and size, at its word's origin, unless said otherwise:
    fakes_in_place: bool = False  # true redaction, each fake left-aligned where its word started
    retypeset: bool = False  # true redaction, then every line holding a fake re-typeset at 1.02-1.04x from
    #                          its first word, kept words moving with it, one text object per word
    retypeset_keep: bool = False  # the same with the originals re-typeset instead of fakes (a leak)
    overprint: bool = False  # the fake drawn over the original, which stays
    veil: float = 0.0  # a white box of this opacity over the original, the fake on top
    shift_white: float = 0.0  # a white box shifted this share of the word's height across the line, the
    #                           fake inside it
    text_next_line: bool = False  # rasterize: the text layer one line pitch below the words it stands for
    address_whole: bool = False  # one flowing fake per address unit, per line, from its first word
    keeps_locality_moved: Literal["", "line", "next_line"] = ""  # a fake street line of another length with
    #   the original locality kept after it on the line, or the original locality pushed to the next line
    erases_locality: bool = False  # a fake street line, the locality erased with no fake
    seed: int | None = 0  # None = fresh randomness on every run (nondeterministic tool)


@dataclass
class MockOutput:
    pdf: bytes
    plan: PlanArtifact
    text: TextArtifact | None = None
    mentions: MentionsArtifact | None = None
    entities: EntitiesArtifact | None = None
    pseudonyms: PseudonymsArtifact | None = None
    targeted: list[str] = field(default_factory=list)

    def artifacts(self) -> dict[str, object]:
        out = {"plan": self.plan, "text": self.text, "mentions": self.mentions, "entities": self.entities,
               "pseudonyms": self.pseudonyms}
        return {k: v for k, v in out.items() if v is not None}


# -- names for pseudonyms (fictional) -----------------------------------------------------

_GIVEN = {"F": ["Anna", "Clare", "Ruth", "Grace", "Nora", "Ivy", "Mabel"],
          "M": ["Owen", "Hugh", "Felix", "Ralph", "Ned", "Abel", "Silas"]}
_SURNAMES = ["Baxter", "Fenwick", "Holloway", "Pryce", "Quill", "Ashdown", "Merrow", "Tolland"]
_NICK = {"F": ["Annie", "Nell", "Tess"], "M": ["Ollie", "Ned", "Fitz"]}
# Larger pools for entity-keyed fakes: a hash picks the value, so a document with many people
# needs room to keep every person's fake distinct.
_GIVEN_KEYED = {"F": _GIVEN["F"] + ["Hazel", "Edith", "Iris", "Leah", "Maeve", "Olive", "Rosa", "Thea", "Vera", "Wren",
                                    "Elsie", "Faye", "Greta", "Joan", "Lena", "Mira", "Tessa"],
                "M": _GIVEN["M"] + ["Arlo", "Bram", "Cyril", "Dean", "Ezra", "Floyd", "Gus", "Ivan", "Jude", "Kurt",
                                    "Lyle", "Miles", "Otto", "Rex", "Seth", "Tobias", "Wade"]}
_SURNAMES_KEYED = _SURNAMES + ["Aldine", "Brackley", "Carrow", "Dunmore", "Ellery", "Farrant", "Garside", "Hargrave",
                               "Ingleby", "Jessop", "Kettering", "Larkworth", "Marlowe", "Northam", "Orchard",
                               "Pellow", "Radcliffe", "Sefton", "Thackeray", "Upton", "Varley", "Wexford", "Yarrow",
                               "Ainsley", "Blackwood", "Cresswell", "Darrow", "Elmhurst", "Fairbairn", "Kendrick"]
_NICK_KEYED = {"F": _NICK["F"] + ["Bea", "Kit", "Molly", "Dot"], "M": _NICK["M"] + ["Alf", "Kip", "Theo", "Bert"]}
_PUNCT = ".,;:()"


def _core(text: str) -> str:
    return text.strip(_PUNCT)


def _style(template: str, value: str) -> str:
    core = template.strip(_PUNCT)
    lead = template[: len(template) - len(template.lstrip(_PUNCT))]
    trail = template[len(template.rstrip(_PUNCT)):]
    if core.isupper() and len(core) > 1:
        value = value.upper()
    return lead + value + trail


def _stable(*parts: object) -> int:
    """A hash that is the same in every run and on every machine (Python's own hash() is not)."""
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()[:8], "big")


def _alnum(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _unlike(original: str, fake: str) -> bool:
    """The fake shares no recognisable text with the original: neither contains the other, and
    they are less alike than the evaluator's text-layer and too-similar tests allow."""
    o, f = _alnum(original), _alnum(fake)
    return not o or not f or (o not in f and f not in o and fuzz.ratio(o, f) < 80)


def _gender(value: str | None) -> str:
    """The name pool for a GT gender attribute, "F" or "M". GT writes the attribute freely ("M",
    "male", ...); any other value, or none, takes the "F" pool."""
    return "M" if (value or "").strip().casefold() in ("m", "male", "man") else "F"


def _keyed_pool(component: NameComponent, gender: str | None) -> list[str]:
    if component in (NameComponent.GIVEN, NameComponent.MIDDLE):
        return _GIVEN_KEYED[_gender(gender)]
    if component is NameComponent.NICKNAME:
        return _NICK_KEYED[_gender(gender)]
    return _SURNAMES_KEYED


def _candidates(pool: list[str], start: int):
    """Fake values to try, from ``start``: the pool's own, then double names made of two of them
    ("Ashdown-Baxter"), so a matter with more people than the pool still gets a fake per person."""
    n = len(pool)
    for k in range(n):
        yield pool[(start + k) % n]
    for k in range(n):
        for j in range(1, n):
            yield f"{pool[(start + k) % n]}-{pool[(start + k + j) % n]}"


def _free(v: str, slot: tuple[str, str], originals, real_names: frozenset[str]) -> bool:
    """A fake value may be used: it resembles none of the person's originals, puts no letter over the same
    or a look-alike letter of one of them when drawn where it was (eval-0.4: that glyph would be the
    original's), and it is no real name (equal, or alike for names of three letters or more)."""
    return (all(_unlike(o, v) and not _same_place(_core(o), v) for o in originals)
            and all(r != slot[1] and (len(r) < 3 or _unlike(r, v)) for r in real_names))


def _matter_people(gts: list[GTDocument]
                   ) -> tuple[dict[tuple[str, str], set[str]], dict[str, str], frozenset[str]]:
    """What a matter-wide assignment of fakes needs from the matter's documents: every (person, name
    component) with its original spellings, each person's name pool by gender, and every real name
    word."""
    originals: dict[tuple[str, str], set[str]] = defaultdict(set)
    gender: dict[str, str] = {}
    real: set[str] = set()
    for gt in sorted(gts, key=lambda g: g.doc_id):
        words = gt.word_index()
        for e in sorted(gt.entities, key=lambda e: e.id):
            if "gender" in e.attributes:
                gender.setdefault(e.id, _gender(e.attributes["gender"].value))
        for m in gt.mentions:
            if m.type is PiiType.EMAIL and m.entity:  # an email's fake is built from its person's fake name
                for c in (NameComponent.GIVEN, NameComponent.SURNAME):
                    originals.setdefault((m.entity, c.value), set())
            if m.type is not PiiType.PERSON_NAME:
                continue
            texts = {wid: words[wid][1].text[seg.char_start:seg.char_end] if seg.char_start is not None
                     else words[wid][1].text for seg in m.segments for wid in seg.words}
            real.update(_core(t).casefold() for t in texts.values())
            if not m.entity:
                continue
            comp_of = {wid: c for c, wids in m.components.items() for wid in wids}
            unlabelled = [wid for wid in texts if wid not in comp_of]
            for k, wid in enumerate(unlabelled):
                comp_of[wid] = _name_component(k, len(unlabelled), texts[wid])
            for wid, c in comp_of.items():
                if c is not NameComponent.INITIALS and wid in texts:
                    originals[(m.entity, c.value)].add(texts[wid])
    return originals, gender, frozenset(real)


def _assign(originals: dict[tuple[str, str], set[str]], gender: dict[str, str], real: frozenset[str],
            salt: str) -> dict[tuple[str, str], str]:
    """One fake per (person, name component) for a whole matter (``_matter_people``). Keys are taken
    in a fixed order and each starts from a hash of (matter, person, component), moving on while the
    value is another person's, a real name of the matter or like one of the person's own spellings
    (``_candidates``: double names once the pool is used up). So a person's fake depends only on the
    matter, never on which people a document names or in what order."""
    owner: set[tuple[str, str]] = set()
    out: dict[tuple[str, str], str] = {}
    for entity, comp in sorted(originals):
        pool = _keyed_pool(NameComponent(comp), gender.get(entity))
        start = _stable(salt, entity, comp) % len(pool)
        value = next((v for v in _candidates(pool, start) if (comp, v.casefold()) not in owner
                      and _free(v, (comp, v.casefold()), originals[(entity, comp)], real)), None)
        if value is None:
            raise ValueError(f"no fake {comp} left that no other person of the matter has")
        owner.add((comp, value.casefold()))
        out[(entity, comp)] = value
    return out


class _Pseudonymiser:
    """Maps (cluster, component) to a replacement value; keyed on *predicted* clusters.

    With ``keyed`` the clusters are GT entities and the values come from ``_assign`` over the
    matter's documents (``matter_gts``; the salt is the matter), so a person gets the same fake in
    every document of a matter and two people of the matter never share one. A key the assignment
    does not know (a name with no entity) gets a value of its own the same way, first come."""

    def __init__(self, gender_of: dict[str, str], mode: str, rng: random.Random, keyed: bool = False,
                 salt: str = "", real_names: frozenset[str] = frozenset(),
                 assigned: dict[tuple[str, str], str] | None = None):
        self.gender_of, self.mode, self.rng = gender_of, mode, rng
        self.keyed, self.salt, self.real_names = keyed, salt, real_names
        self.table: dict[tuple[str, str], str] = dict(assigned or {})
        # With fakes drawn in place: a value whose letters would sit over the same or a look-alike letter
        # of the original is passed over (that glyph would be the original's).
        self.in_place = False
        # (component, value) -> key that has it
        self.owner: dict[tuple[str, str], tuple[str, str]] = {(c, v.casefold()): (e, c)
                                                              for (e, c), v in self.table.items()}
        self.counter = 0

    def value(self, cluster: str, component: NameComponent, original: str) -> str:
        self.counter += 1
        comp = component.value if self.mode != "inconsistent" else f"{component.value}{self.counter}"
        key = ("all" if self.mode == "collision" else cluster, comp)
        if key not in self.table:
            g = _gender(self.gender_of.get(cluster))
            if self.mode == "gender_flip":
                g = "M" if g == "F" else "F"
            if self.keyed:
                v = self._keyed(key, component, original, g)
            else:
                i = len(self.table)
                if component in (NameComponent.GIVEN, NameComponent.MIDDLE):
                    pool = _GIVEN[g]
                elif component is NameComponent.NICKNAME:
                    pool = _NICK[g]
                elif component is NameComponent.INITIALS:
                    pool = None
                else:
                    pool = _SURNAMES
                if pool is None:
                    v = "".join(self.rng.choice("BCDFGHKLMNPRST") for _ in range(max(1, len(original))))
                else:
                    v = pool[i % len(pool)]
                    if self.in_place:
                        v = next((x for x in _candidates(pool, i % len(pool))
                                  if not _same_place(_core(original), x) and _fits(_core(original), x)), v)
            self.table[key] = v
        return self.table[key]

    def _keyed(self, key: tuple[str, str], component: NameComponent, original: str, gender: str) -> str:
        h = _stable(self.salt, *key)
        if component is NameComponent.INITIALS:
            return _fake_chars(_core(original), h)
        pool = _keyed_pool(component, gender)
        for v in _candidates(pool, h % len(pool)):
            slot = (component.value, v.casefold())
            if self.owner.get(slot, key) == key and _free(v, slot, (original,), self.real_names):
                self.owner[slot] = key
                return v
        raise ValueError(f"no fake {component.value} left that no other person has")


# -- stages -------------------------------------------------------------------------------------


@dataclass
class _Det:
    """A detected mention: which GT mention it came from and the words it covers."""

    id: str
    gt_mention: str | None
    label: str
    parts: list[tuple[int, str, tuple[int, int] | None]]  # (page, word id, char span)


def _rng(spec: MockSpec) -> random.Random:
    return random.Random(spec.seed if spec.seed is not None else int.from_bytes(os.urandom(8), "little"))


def _text_stage(gt: GTDocument, spec: MockSpec, rng: random.Random) -> tuple[TextArtifact, set[str], dict[str, str]]:
    """GT words as an OCR result, with the configured corruption. Returns the artifact,
    the ids of words that were read at all, and GT word id -> token id."""
    unread: set[str] = set()
    for m in gt.mentions:
        if m.type.value in spec.ocr_drop_types or set(m.conditions) & set(spec.ocr_drop_conditions):
            unread.update(wid for seg in m.segments for wid in seg.words)
    pages, token_of, read = [], {}, set()
    for p in gt.pages:
        tokens = []
        for w in p.words:
            if w.id in unread:
                continue
            text = w.text
            if spec.ocr_char_error:
                text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") if rng.random() < spec.ocr_char_error else c
                               for c in text)
            poly = as_poly(w.poly)
            if spec.ocr_box_jitter_pt:
                poly = poly + np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)]) * spec.ocr_box_jitter_pt
            tid = f"t{p.index}_{len(tokens)}"
            tokens.append(TextToken(id=tid, text=text, poly=poly.tolist(), order=len(tokens), line=w.line))
            token_of[w.id] = tid
            read.add(w.id)
        pages.append(TextPage(index=p.index, tokens=tokens))
    return TextArtifact(source="ocr", pages=pages), read, token_of


def _detect_stage(gt: GTDocument, policy: PolicyProfile, spec: MockSpec, read: set[str],
                  rng: random.Random) -> list[_Det]:
    res = resolve(gt, policy)
    words = gt.word_index()
    order = {w.id: (p.index, w.order) for p in gt.pages for w in p.words}
    by_order = {(p.index, w.order): w.id for p in gt.pages for w in p.words}
    swap = dict(spec.type_swap)
    wanted = {Requirement.MUST} | ({Requirement.MAY} if spec.include_may else set())
    dets: list[_Det] = []
    for m in gt.mentions:
        if res.mentions[m.id].requirement not in wanted:
            continue
        if m.type.value in spec.drop_types or set(m.conditions) & set(spec.drop_conditions):
            continue
        if spec.drop_fraction and rng.random() < spec.drop_fraction:
            continue
        parts = [(seg.page, wid, (seg.char_start, seg.char_end) if seg.char_start is not None else None)
                 for seg in m.segments for wid in seg.words if wid in read]
        regions_only = [seg for seg in m.segments if seg.regions and not seg.words]
        if not parts and not regions_only:
            continue  # nothing of it was read: an OCR-caused miss
        if spec.boundary == "shrink_last" and len(parts) > 1:
            parts = parts[:-1]
        elif spec.boundary == "extend_title":
            parts = [(words[w][0], w, None) for w in m.optional_words if w in read] + parts
        elif spec.boundary == "extend_next" and parts:
            pg, last, _ = parts[-1]
            nxt = by_order.get((pg, order[last][1] + 1))
            if nxt and nxt in read and words[nxt][1].line == words[last][1].line:
                parts = parts + [(pg, nxt, None)]
        dets.append(_Det(id=f"d{len(dets)}", gt_mention=m.id, label=swap.get(m.type.value, m.type.value),
                         parts=parts))
    if spec.redact_protected:
        for c in gt.classified:
            parts = [(seg.page, wid, None) for seg in c.segments for wid in seg.words if wid in read]
            if parts:
                dets.append(_Det(id=f"d{len(dets)}", gt_mention=None, label="PERSON_NAME", parts=parts))
    return dets


def _resolve_stage(gt: GTDocument, dets: list[_Det], spec: MockSpec) -> tuple[EntitiesArtifact, dict[str, str]]:
    mentions = {m.id: m for m in gt.mentions}
    words = gt.word_index()
    cluster_of: dict[str, str] = {}
    for d in dets:
        m = mentions.get(d.gt_mention) if d.gt_mention else None
        key = (m.entity or m.id) if m else d.id
        if m and spec.split_aliases and m.components and set(m.components) & {
                NameComponent.NICKNAME, NameComponent.INITIALS, NameComponent.FORMER_SURNAME}:
            key = f"{key}-alias-{m.id}"
        hit = sorted(set(m.conditions) & set(spec.split_conditions)) if m else []
        if hit:
            key = f"{key}-split-{hit[0]}"
        cluster_of[d.id] = key
    if spec.merge_same_given:
        given_owner: dict[str, str] = {}
        for d in dets:
            m = mentions.get(d.gt_mention) if d.gt_mention else None
            for wid in (m.components.get(NameComponent.GIVEN, []) if m else []):
                g = words[wid][1].text.strip(".,;:()").casefold()
                owner = given_owner.setdefault(g, cluster_of[d.id])
                old = cluster_of[d.id]
                for k, v in cluster_of.items():
                    if v == old:
                        cluster_of[k] = owner
    clusters: dict[str, list[str]] = {}
    for d in dets:
        clusters.setdefault(cluster_of[d.id], []).append(d.id)
    rename = {k: f"c{i}" for i, k in enumerate(clusters)}
    art = EntitiesArtifact(clusters=[Cluster(id=rename[k], mentions=v) for k, v in clusters.items()])
    return art, {d: rename[c] for d, c in cluster_of.items()}


def _word_region(word, seg_start, seg_end, partial: float) -> np.ndarray:
    if word.char_polys and seg_start is not None:
        pts = np.vstack([as_poly(p) for p in word.char_polys[seg_start:seg_end]])
        poly = rect_poly(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    else:
        poly = as_poly(word.poly)
    if partial < 1.0:
        x0, y0, x1, y1 = bbox(poly)
        a = word.angle % 360
        if a == 0:
            poly = rect_poly(x0, y0, x0 + (x1 - x0) * partial, y1)
        elif a == 180:
            poly = rect_poly(x1 - (x1 - x0) * partial, y0, x1, y1)
        elif a == 270:
            poly = rect_poly(x0, y1 - (y1 - y0) * partial, x1, y1)
        else:
            poly = rect_poly(x0, y0, x1, y0 + (y1 - y0) * partial)
    return poly


OTHER_VALUES = {"PHONE", "EMAIL", "ADDRESS", "ID_GOVERNMENT"}


def _is_other(m) -> bool:
    """Contact details and identifiers the replace_other mocks fake, and an address's locality."""
    return m.type.value in OTHER_VALUES or (m.type is PiiType.LOCATION and (m.subtype or "") == "locality")


def _medicare(first8: str, valid: bool, irn: str = "2") -> str:
    check = sum(int(x) * w for x, w in zip(first8, (1, 3, 7, 9, 1, 3, 7, 9))) % 10
    if not valid:
        check = (check + 1) % 10
    return f"{first8[:4]} {first8[4:]}{check} {irn}"


def _other_value(m, kind: str, pseud: "_Pseudonymiser", cluster: str) -> str:
    """A replacement for a non-name mention; ``invalid`` breaks the format on purpose.
    Values are chosen so no two-character run lines up with the fixture's originals."""
    valid = kind == "valid"
    t = m.type.value
    if t == "PHONE":
        return "(02) 5550 1234" if valid else "12 34"
    if t == "EMAIL":
        given = pseud.value(cluster, NameComponent.GIVEN, "").casefold()
        surname = pseud.value(cluster, NameComponent.SURNAME, "").casefold()
        return f"{given}.{surname}@mail.test" if valid else f"{given}.{surname} at mail"
    if t == "ADDRESS" and "address_street_line" in m.conditions:  # the street line of a split address
        return "27 Banksia Road,"
    if t == "LOCATION":  # its locality
        return "Oakvale NSW " + ("2150" if valid else "4150")
    if t == "ADDRESS":
        return "27 Banksia Road, Oakvale NSW " + ("2150" if valid else "4150")
    if t == "ID_GOVERNMENT" and (m.subtype or "") == "medicare":
        return _medicare("59867304", valid)
    return ""


def _distribute(text: str, n: int) -> list[str]:
    """Spread replacement tokens over n word regions (extra tokens go to the last one)."""
    toks = text.split()
    if n <= 0:
        return []
    if len(toks) <= n:
        return toks + [""] * (n - len(toks))
    return toks[: n - 1] + [" ".join(toks[n - 1:])]


# -- fakes of every other kind (replace_all) -----------------------------------------------------
# One fake per word, so each word region gets its own text. Lists are fictional.

_FAKE_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
# Full and dotted names the GT split recognises, so the fake never picks the original's state.
_STATE_NAMES = {"queensland": "qld", "victoria": "vic", "tasmania": "tas", "n.s.w": "nsw", "s.a": "sa", "w.a": "wa",
                "n.t": "nt", "a.c.t": "act"}
_SUBURBS = ["Oakvale", "Brushmere", "Tallowby", "Karribrook", "Wattlecombe", "Bimbadeen", "Fernlea", "Ironmere"]
_STREETS = ["Banksia", "Coolabah", "Grevillea", "Jacaranda", "Kurrajong", "Waratah", "Melaleuca", "Bottlebrush"]
_FAKE_STREET_TYPES = ["Road", "Street", "Avenue", "Parade", "Crescent", "Lane", "Close", "Terrace"]
_ORGS = ["Bellbird Freight Group", "Northgate Supply Co", "Kestrel Joinery Partners", "Mallee Print Works",
         "Harbourline Services", "Quandong Health Group"]
_ORG_FILL = ["Holdings", "Trust", "Partners", "Services", "Group", "Co"]
_URL_HOSTS = ["bellbird-freight.test", "northgate-supply.test", "kestrel-joinery.test"]
_FOLDERS = ["Matters", "Files", "Archive"]
_FILE_NAMES = ["letter", "notes", "report", "summary"]


def _fake_chars(text: str, h: int) -> str:
    """Format-preserving fake: every digit and letter changes (none keeps its value at its
    position), case and every other character stay; a number never gains a leading zero."""
    rng = random.Random(h)
    out = []
    for i, c in enumerate(text):
        if c.isdigit():
            lead = i == 0 or not text[i - 1].isdigit()
            out.append(rng.choice([d for d in string.digits if d != c and not (lead and d == "0")]))
        elif c.isalpha():
            letters = string.ascii_uppercase if c.isupper() else string.ascii_lowercase
            out.append(rng.choice([x for x in letters if x != c]))
        else:
            out.append(c)
    return "".join(out)


def _fake_id(word: str, h: int) -> str:
    """A format-preserving fake for one word, redrawn until it shares no recognisable text."""
    core = _core(word)
    fake = _fake_chars(core, h)
    for attempt in range(1, 8):
        if _unlike(core, fake):
            break
        fake = _fake_chars(core, h + attempt)
    return _style(word, fake)


def _fit(tokens: list[str], n: int) -> list[str]:
    """Exactly ``n`` non-empty tokens: extra ones join the last, missing ones come from splitting
    the longest, so no word region is left without text."""
    toks = [t for t in tokens if t] or ["-"]
    if n <= 0:
        return []
    if len(toks) > n:
        toks = toks[: n - 1] + [" ".join(toks[n - 1:])]
    while len(toks) < n:
        i = max(range(len(toks)), key=lambda k: len(toks[k]))
        t = toks[i]
        toks[i:i + 1] = [t[: len(t) // 2], t[len(t) // 2:]] if len(t) > 1 else [t, t]
    return toks


def _postcode(state: str, original: str, rng: random.Random, keep: int = 0) -> str:
    """A postcode inside the state's Australia Post ranges, keeping the original's first ``keep``
    digits (the state's shared prefix) and differing from it at every later position."""
    pc = original
    for _ in range(400):
        lo, hi = rng.choice(POSTCODES[state])
        pc = f"{rng.randint(lo, hi):04d}"
        if pc[:keep] == original[:keep] and all(a != b for a, b in zip(pc[keep:], original[keep:])):
            break
    return pc


# Letters (and tabular digits) that pass for one another in the same font at the same place: the identity
# gate tells most other pairs apart (docs/metrics.md, "Character identity").
_ALIKE = ("iljI1|!", "38", "56", "68")


def _xs(text: str) -> list[tuple[float, float]]:
    out, x = [], 0.0
    for c in text:
        w = _helv().text_length(c, fontsize=1)
        out.append((x, x + w))
        x += w
    return out


def _same_place(original: str, fake: str, kept: frozenset[int] = frozenset()) -> bool:
    """Would a letter or digit of a fake drawn at the original's origin (in a font of Helvetica's widths) sit
    over the same or a look-alike character of the original? ``kept``: indices kept on purpose (shared
    structure)."""
    a, b = _xs(original), _xs(fake)
    for i, c in enumerate(fake):
        if i in kept or not c.isalnum():
            continue
        for j, o in enumerate(original):
            lo, hi = max(a[j][0], b[i][0]), min(a[j][1], b[i][1])
            if hi - lo <= 0.3 * min(a[j][1] - a[j][0], b[i][1] - b[i][0]):
                continue
            if c.casefold() == o.casefold() or any(c in g and o in g for g in _ALIKE):
                return True
    return False


def _fits(original: str, fake: str) -> bool:
    """Is a fake no wider than the original word (in a font of Helvetica's widths), give or take 0.05 em?"""
    return _helv().text_length(fake, fontsize=1) <= _helv().text_length(original, fontsize=1) + 0.05


def _drawable(originals: list[str], fakes: list[str], in_place: bool) -> bool:
    """Fakes a careful tool draws: no changed word holds an original word of its mention (text is also read
    as a substring), and, drawn at the original's origin and size (``in_place``), none is wider than its
    original or puts a character over the same or a look-alike one. Words kept on purpose (shared
    structure) are the original."""
    kept = {_alnum(o) for o, f in zip(originals, fakes) if f == o}
    words = [x for x in (_alnum(o) for o in originals) if x and x not in kept]
    for o, f in zip(originals, fakes):
        if not f or f == o:
            continue
        if any(x in _alnum(f) for x in words):
            return False
        same = frozenset(i for i in range(min(len(o), len(f))) if o[i] == f[i])
        if in_place and (not _fits(o, f) or _same_place(o, f, same)):
            return False
    return True


def _banned(fake: str, banned: frozenset[str], numbers: frozenset[str]) -> bool:
    """Does a fake word equal, begin with or (3 letters or more) hold an identifying address word of the
    document, or a number of it equal one of its identifying numbers (whole digit runs)?"""
    n = _alnum(fake)
    if any(r in numbers for r in re.findall(r"\d+", n)):
        return True
    letters = "".join(ch for ch in n if ch.isalpha())
    return any(letters == w or (len(w) >= 3 and (letters.startswith(w) or w in letters)) for w in banned)


def _place_tokens(originals: list[str], h: int, street: bool, su: SharedUnits | None = None,
                  wids: list[str] | None = None, banned: frozenset[str] = frozenset(),
                  numbers: frozenset[str] = frozenset()) -> list[str]:
    """A street line (``street``) or a locality, word by word: numbers keep their shape, a street
    type stays a street type, a state is followed by a postcode valid for the fake state. The shared
    structure (``su``: the policy's shared roles for this mention, by word id ``wids``) stays as
    written; a shared state keeps a same-state fake: the postcode keeps the state's prefix and changes
    every later digit. No fake word repeats an identifying address word or number (``banned``,
    ``numbers``)."""
    rng = random.Random(h)
    norm = [_core(w).casefold() for w in originals]
    wids = wids or [""] * len(originals)
    su = su or SharedUnits()
    kept_state = next((_STATE_NAMES.get(n, n) for n, w in zip(norm, wids) if n in STATES and w in su.full), None)
    real_states = {_STATE_NAMES.get(n, n) for n in norm}
    state = kept_state.upper() if kept_state else rng.choice([s for s in _FAKE_STATES
                                                              if s.casefold() not in real_states])
    names = _STREETS if street else _SUBURBS
    out = []
    for k, (word, n, wid) in enumerate(zip(originals, norm, wids)):
        if wid in su.full:
            out.append(word)
        elif any((wid, i) in su.chars for i in range(len(word))) and re.search(r"\d", n):
            # characters shared inside a word: a glued 'NSW2750' or a postcode's prefix; the rest changes
            keep = sorted(i for i in range(len(word)) if (wid, i) in su.chars)
            digits = "".join(ch for ch in word if ch.isdigit())
            lead = len(word) - len(digits)
            code = read_end([word]).state_code or state  # a glued word names its own state
            if re.fullmatch(r"\d{4}", digits) and code in POSTCODES:
                pc = _postcode(code, digits, rng, keep=sum(1 for i in keep if i >= lead))
                out.append(word[:lead] + pc)
            else:
                fake = _fake_chars(word, rng.getrandbits(32))
                out.append("".join(word[i] if i in keep else fake[i] for i in range(len(word))))
        elif n in STATES:
            out.append(_style(word, state))
        elif re.fullmatch(r"\d{4}", n) and k > 0 and norm[k - 1] in STATES:
            out.append(_style(word, _postcode(state, n, rng)))
        elif not n or any(ch.isdigit() for ch in n):
            fake = _fake_id(word, rng.getrandbits(32))
            for _ in range(8):
                if not _banned(fake, banned, numbers):
                    break
                fake = _fake_id(word, rng.getrandbits(32))
            out.append(fake)
        elif n in STREET_TYPES:
            out.append(_style(word, rng.choice([t for t in _FAKE_STREET_TYPES if _unlike(n, t)
                                                and not _banned(t, banned, frozenset())] or _FAKE_STREET_TYPES)))
        else:
            out.append(_style(word, rng.choice([x for x in names
                                                if _unlike(n, x) and not _banned(x, banned, frozenset())] or names)))
    return out


def _address_words(gt: GTDocument) -> tuple[frozenset[str], frozenset[str]]:
    """The identifying words (letters, case folded) and numbers of every address unit of the document, any
    role (the case table's structure roles and street types left out)."""
    from ..gt.address_split import address_roles, unit_words

    structure = {"designator", "box_kind", "corner", "via", "dx", "state", "country", "capital_city_phrase",
                 "street_type"}
    words = gt.word_index()
    letters, numbers = set(), set()
    for street, loc in address_unit_mentions(gt):
        for r in address_roles(unit_words(gt, street, loc)):
            if r.role in structure:
                continue
            n = _alnum(words[r.wid][1].text)
            numbers.update(re.findall(r"\d+", n))
            a = "".join(ch for ch in n if ch.isalpha())
            if a and r.role not in ("postcode", "state+postcode"):
                letters.add(a)
    return frozenset(letters), frozenset(numbers)


def _org_tokens(originals: list[str], h: int, kept: frozenset[int] = frozenset()) -> list[str]:
    """An organisation's fake word by word; the words at ``kept`` (a shared legal form) stay as written."""
    rng = random.Random(h)
    n = len(originals) - len(kept)
    for base in rng.sample(_ORGS, len(_ORGS)):
        toks = base.split()[:n]
        toks += [_ORG_FILL[k % len(_ORG_FILL)] for k in range(n - len(toks))]
        it = iter(toks)
        full = [originals[k] if k in kept else next(it) for k in range(len(originals))]
        if all(_unlike(o, t) for k, (o, t) in enumerate(zip(originals, full)) if k not in kept):
            return [o if k in kept else _style(o, t) for k, (o, t) in enumerate(zip(originals, full))]
    return [o if k in kept else _fake_id(o, h + k) for k, o in enumerate(originals)]


def _url_tokens(originals: list[str], h: int) -> list[str]:
    rng = random.Random(h)
    first = _core(originals[0]).casefold()
    scheme = next((s for s in ("https://", "http://") if first.startswith(s)), "")
    url = f"{scheme}{'www.' if 'www.' in first else ''}{rng.choice(_URL_HOSTS)}"
    return [_style(originals[0], url)] + [_fake_id(o, h + k) for k, o in enumerate(originals[1:], 1)]


def _path_tokens(originals: list[str], h: int) -> list[str]:
    rng = random.Random(h)
    first = _core(originals[0])
    ext = os.path.splitext(first.replace("\\", "/"))[1]
    path = first
    for _ in range(8):
        name = f"{rng.choice(_FILE_NAMES)}{ext}"
        folder = rng.choice(_FOLDERS)
        path = f"C:\\{folder}\\{name}" if "\\" in first else f"/{folder.lower()}/{name}" if "/" in first else name
        if _unlike(first, path):
            break
    return [_style(originals[0], path)] + [_fake_id(o, h + k) for k, o in enumerate(originals[1:], 1)]


_LANDLINE_AREAS = "2378"


def _other_digit(rng: random.Random, old: str, lead: bool = False) -> str:
    return rng.choice([d for d in string.digits if d != old and not (lead and d == "0")])


def _fill_digits(originals: list[str], digits: str, rng: random.Random) -> list[str]:
    """The original words with their digits replaced, in order, by ``digits`` and every letter by
    another; spaces, brackets and other punctuation stay where they were."""
    it = iter(digits)
    out = []
    for word in originals:
        chars = []
        for c in word:
            if c.isdigit():
                chars.append(next(it))
            elif c.isalpha():
                letters = string.ascii_uppercase if c.isupper() else string.ascii_lowercase
                chars.append(rng.choice([x for x in letters if x != c]))
            else:
                chars.append(c)
        out.append("".join(chars))
    return out


def _phone_digits(digits: str, rng: random.Random, kind: str) -> str | None:
    """A fake Australian number of the same length as the national-form ``digits``, every digit after
    the kind's prefix changed: a mobile stays 04, a landline gets another area code, 1300 and 1800
    swap, 13 stays 13, a local number keeps no prefix. ``kind="landline"`` forces a landline."""
    n = len(digits)
    if kind == "landline" and n == 10:
        head = "0" + rng.choice([a for a in _LANDLINE_AREAS if a != digits[1]])
    elif n == 10 and digits.startswith("04"):
        head = "04"
    elif n == 10 and digits[0] == "0" and digits[1] in _LANDLINE_AREAS:
        head = "0" + rng.choice([a for a in _LANDLINE_AREAS if a != digits[1]])
    elif n == 10 and digits[:4] in ("1300", "1800"):
        head = "1800" if digits[:4] == "1300" else "1300"
    elif n == 6 and digits.startswith("13"):
        head = "13"
    elif n == 8:
        head = ""
    else:
        return None
    rest = digits[len(head):]
    return head + "".join(_other_digit(rng, c, lead=not head and k == 0) for k, c in enumerate(rest))


def _phone_groups(number: str) -> list[str]:
    """A national-form number in its usual groups: (0X) XXXX XXXX, 04XX XXX XXX, 1300 XXX XXX,
    13 XX XX or XXXX XXXX."""
    if len(number) == 10 and number[0] == "0" and number[1] != "4":
        return [f"({number[:2]})", number[2:6], number[6:]]
    if len(number) == 10:
        return [number[:4], number[4:7], number[7:]]
    if len(number) == 6:
        return [number[:2], number[2:4], number[4:]]
    return [number[:4], number[4:]] if len(number) == 8 else [number]


def _fit_groups(groups: list[str], n: int) -> list[str]:
    """``_fit`` for the groups of a phone number: a bracketed area code is never split."""
    toks = list(groups)
    while len(toks) < n:
        split = [k for k, t in enumerate(toks) if not t.startswith("(") and len(t) > 1]
        if not split:
            break
        k = max(split, key=lambda i: len(toks[i]))
        toks[k:k + 1] = [toks[k][: len(toks[k]) // 2], toks[k][len(toks[k]) // 2:]]
    return _fit(toks, n)


def _fake_phone(originals: list[str], h: int, keep_mobile: bool = True) -> list[str]:
    """A phone number, its own for each person and number, written in the original's word layout, of
    the same kind where that layout allows it. No fake word may share recognisable text with its
    original word, since the evaluator reads such a word as the original left in place; so a layout
    that gives the kind's prefix a word of its own ("04 1234 5678") cannot keep it, and the number
    becomes a landline. A +61 number is written in national form over the same words, its bracketed
    area code in one word ("+61 (0)4..." included). ``keep_mobile`` False: a mobile becomes a landline
    too (its '04' drawn where it was is the original's, unless the policy shares the prefix)."""
    joined = " ".join(originals)
    digits = "".join(c for c in joined if c.isdigit())
    international = joined.lstrip().startswith("+")
    national = digits
    if international and digits.startswith("61"):
        national = digits[2:] if digits.startswith("610") else "0" + digits[2:]  # "+61 (0)4..." has its 0
    for attempt in range(16):
        rng = random.Random(h + attempt)
        mobile = national.startswith("04") and len(national) == 10
        new = _phone_digits(national, rng, "same" if attempt < 8 and (keep_mobile or not mobile) else "landline")
        if new is None:
            break
        if international or len(new) != len(digits):  # +61: written in national form over the same words
            toks = _fit_groups(_phone_groups(new), len(originals))
        else:
            toks = _fill_digits(originals, new, rng)
        if all(_unlike(o, f) for o, f in zip(originals, toks)):
            return toks
    return [_fake_id(o, h + k) for k, o in enumerate(originals)]


def _fake_medicare(originals: list[str], h: int) -> list[str]:
    """A Medicare number with a valid check digit and another issue number (and reference number,
    when written), every other digit changed too, in the original's word layout."""
    digits = "".join(c for c in "".join(originals) if c.isdigit())
    if len(digits) in (10, 11):
        for attempt in range(8):
            rng = random.Random(h + attempt)
            body = rng.choice([d for d in "23456" if d != digits[0]]) + "".join(
                _other_digit(rng, c) for c in digits[1:8])
            new = body + str(sum(int(x) * w for x, w in zip(body, (1, 3, 7, 9, 1, 3, 7, 9))) % 10)
            new += "".join(rng.choice([d for d in "123456789" if d != c]) for c in digits[9:])
            toks = _fill_digits(originals, new, rng)
            if all(_unlike(o, f) for o, f in zip(originals, toks)):
                return toks
    return [_fake_id(o, h + k) for k, o in enumerate(originals)]


def _fake_tokens(m, originals: list[str], kind: str, pseud: "_Pseudonymiser", key: str,
                 su: SharedUnits | None = None, wids: list[str] | None = None,
                 address_words: tuple[frozenset[str], frozenset[str]] = (frozenset(), frozenset()),
                 attempt: int = 0, keep_mobile: bool = True) -> list[str]:
    """One fake per word of a required mention that is not a name (replace_all), of the same kind:
    a phone or Medicare number of its own for each person and value, an email from the person's
    fake name; any other identifier keeps its format with every letter and digit changed. The shared
    structure the policy enables (``su``, by word id ``wids``) stays as written: an address's state,
    street type and box words, a legal form, an email domain, an identifier's prefix."""
    t, sub = m.type.value, m.subtype or ""
    h = _stable(pseud.salt, key, t, " ".join(originals), *([attempt] if attempt else []))
    su = su or SharedUnits()
    wids = wids or [""] * len(originals)
    if kind == "valid" and t == "PHONE":
        return _fake_phone(originals, h, keep_mobile)
    if kind == "valid" and t == "ID_GOVERNMENT" and sub == "medicare":
        return _fake_medicare(originals, h)
    if kind == "valid" and t == "EMAIL" and su.needle_cut and len(originals) == 1:
        a, b = su.needle_cut.get(wids[0], (0, len(originals[0])))
        local = _other_value(m, kind, pseud, key).split("@")[0]
        return [originals[0][:a] + local + originals[0][b:]]
    if t in ("PHONE", "EMAIL") or (t == "ID_GOVERNMENT" and sub == "medicare"):
        return _fit(_other_value(m, kind, pseud, key).split(), len(originals))
    if t in ("ADDRESS", "LOCATION"):  # a street line; a locality, suburb, town or other place
        return _place_tokens(originals, h, street=t == "ADDRESS", su=su, wids=wids, banned=address_words[0],
                             numbers=address_words[1])
    if t == "ORGANISATION":
        return _org_tokens(originals, h, frozenset(k for k, w in enumerate(wids) if w in su.full))
    if su.chars or su.full:  # an identifier's shared prefix stays; every other letter and digit changes
        out = []
        for k, (o, w) in enumerate(zip(originals, wids)):
            if w in su.full:
                out.append(o)
                continue
            fake = _fake_id(o, h + k)
            out.append("".join(o[i] if (w, i) in su.chars else fake[i] if i < len(fake) else "" for i in range(len(o))))
        return out
    if t == "URL":
        return _url_tokens(originals, h)
    if t == "FILE_PATH":
        return _path_tokens(originals, h)
    return [_fake_id(o, h + k) for k, o in enumerate(originals)]


def _name_component(k: int, n: int, word: str) -> NameComponent:
    """Component of a name word the GT does not label (e.g. machine drafts), by its position."""
    core = _core(word).replace(".", "")
    if core.isupper() and len(core) <= 2:
        return NameComponent.INITIALS
    if n == 1 or k == n - 1:
        return NameComponent.SURNAME
    return NameComponent.GIVEN if k == 0 else NameComponent.MIDDLE


def _stand_in_strokes(poly, h: int) -> list[np.ndarray]:
    """Pen strokes for a stand-in signature or mark inside a region (display_pt): a looping
    line and an underline, unlike the original's strokes."""
    x0, y0, x1, y1 = bbox(as_poly(poly))
    w, ht = x1 - x0, y1 - y0
    rng = random.Random(h)
    loops, phase = rng.uniform(4.0, 6.0), rng.uniform(0.0, 2 * math.pi)
    t = np.linspace(0.0, 1.0, 96)
    xs = x0 + w * (0.15 + 0.7 * t) + 0.04 * w * np.cos(2 * math.pi * loops * t + phase)
    ys = y0 + ht * (0.45 - 0.1 * t) + 0.18 * ht * np.sin(2 * math.pi * loops * t + phase)
    under = np.array([[x0 + 0.2 * w, y0 + 0.78 * ht], [x0 + 0.8 * w, y0 + 0.72 * ht]])
    return [np.column_stack([xs, ys]), under]


def _plan_stage(gt: GTDocument, dets: list[_Det], cluster_of: dict[str, str], spec: MockSpec,
                rng: random.Random, matter_gts: list[GTDocument] | None = None,
                policy: PolicyProfile | None = None) -> tuple[list[PlanRegion], dict[str, str], dict]:
    """The plan regions, the fakes by detection and word (``"<det>:<word>"``), and each region's word and
    character span (region id -> (word id, span))."""
    words = gt.word_index()
    regions_idx = gt.region_index()
    mentions = {m.id: m for m in gt.mentions}
    entities = {e.id: e for e in gt.entities}
    by_entity = spec.pseudonym_key == "entity"

    def key_of(d: _Det, m) -> str:
        """Whose fake this is: the predicted cluster, or with pseudonym_key "entity" the GT entity."""
        return (m.entity or m.id) if by_entity and m is not None else cluster_of[d.id]

    gender_of = {}
    for d in dets:
        m = mentions.get(d.gt_mention) if d.gt_mention else None
        e = entities.get(m.entity) if m and m.entity else None
        if e and "gender" in e.attributes:
            gender_of.setdefault(key_of(d, m), _gender(e.attributes["gender"].value))
    names = spec.pseudonyms != "none" or spec.replace_all
    other_kind = spec.replace_other if spec.replace_other != "none" else ("valid" if spec.replace_all else "none")
    mode = spec.pseudonyms if spec.pseudonyms != "none" else "consistent"
    real_names = frozenset(_core(words[wid][1].text).casefold() for m in gt.mentions if m.type is PiiType.PERSON_NAME
                           for seg in m.segments for wid in seg.words)
    assigned = None
    if by_entity and mode == "consistent":
        # Every fake comes from the people of the whole matter (this document included), so it is the
        # same in each document of the matter.
        same = {g.doc_id: g for g in (matter_gts or []) if gt.matter and g.matter == gt.matter}
        same[gt.doc_id] = gt
        people = _matter_people(list(same.values()))
        assigned = _assign(*people, salt=gt.matter or "")
        real_names |= people[2]
    pseud = _Pseudonymiser(gender_of, mode, rng, keyed=by_entity, salt=gt.matter or "", real_names=real_names,
                           assigned=assigned) if (names or other_kind != "none") else None
    careful = spec.fakes_in_place or spec.retypeset or spec.redacted_text_layer
    # A fake mobile keeps its '04' where the policy shares the phone prefix (owner, 2026-10-29), in the in-place
    # self-check, which shows what the prefix does without it, and in the ceiling, which draws a fake it may not
    # line up with the original centred and smaller; a re-typesetting mock makes it a landline otherwise.
    keep_mobile = spec.fakes_in_place or spec.redacted_text_layer or any(
        PiiType.PHONE in e.types and "phone_prefix" in e.roles for e in (policy.shared_roles if policy else []))
    if pseud is not None:  # names drawn where the original was (re-typeset lines start where they did)
        pseud.in_place = spec.fakes_in_place or spec.retypeset
    shared = document_shared_units(gt, policy) if policy is not None else {}
    addr_words = _address_words(gt)
    plan: list[PlanRegion] = []
    replacements: dict[str, str] = {}
    word_of: dict[str, tuple[str, tuple[int, int] | None]] = {}
    for d in dets:
        m = mentions.get(d.gt_mention) if d.gt_mention else None
        comp_of = {wid: c for c, wids in (m.components.items() if m else []) for wid in wids}
        texts = [words[wid][1].text if span is None else words[wid][1].text[span[0]:span[1]]
                 for _, wid, span in d.parts]
        other = None
        if m is not None and m.type is not PiiType.PERSON_NAME and spec.replace_all:
            wids = [wid for _, wid, _ in d.parts]
            su = shared.get(m.id)
            first = None
            for attempt in range(24):
                other = _fake_tokens(m, texts, other_kind, pseud, key_of(d, m), su, wids, addr_words, attempt,
                                     keep_mobile or not careful)
                first = first or other
                # A fake drawn at the original's origin and size (in place, or the ceiling's text) must not
                # show the original's glyphs, nor hold one of its words: draw another.
                if not careful or _drawable(texts, other, spec.fakes_in_place or spec.redacted_text_layer):
                    break
            else:
                other = first
        elif spec.replace_other != "none" and m is not None and _is_other(m):
            other = _distribute(_other_value(m, spec.replace_other, pseud, cluster_of[d.id]), len(d.parts))
        if m is not None and m.type is PiiType.PERSON_NAME and spec.replace_all:
            unlabelled = [(wid, text) for (_, wid, _), text in zip(d.parts, texts) if wid not in comp_of]
            for k, (wid, text) in enumerate(unlabelled):
                comp_of[wid] = _name_component(k, len(unlabelled), text)
        for k, (page, wid, span) in enumerate(d.parts):
            w = words[wid][1]
            if other is not None and other[k] and other[k] == texts[k] and m is not None and (
                    wid in shared.get(m.id, SharedUnits()).full):
                continue  # shared structure the fake keeps as written: left where it is
            poly = _word_region(w, span[0] if span else None, span[1] if span else None, spec.partial)
            repl = None
            if other is not None and other[k]:
                repl = other[k]
                replacements[f"{d.id}:{wid}"] = repl
            elif names and m is not None and m.type is PiiType.PERSON_NAME and wid in comp_of:
                original = texts[k]
                repl = _style(original, pseud.value(key_of(d, m), comp_of[wid], original))
                replacements[f"{d.id}:{wid}"] = repl
            plan.append(PlanRegion(id=f"{d.id}-{k}", page=page, poly=poly.tolist(),
                                   action="replace" if repl else "remove", replacement=repl, mentions=[d.id]))
            word_of[f"{d.id}-{k}"] = (wid, span)
        if m is not None and spec.pseudonyms == "gender_flip":
            # A gender-flipping pseudonymiser also swaps the honorific in front of the name.
            for k2, wid in enumerate(m.optional_words):
                title = words[wid][1].text.strip(".").casefold()
                flipped = {"ms": "Mr", "mrs": "Mr", "miss": "Mr", "mr": "Ms"}.get(title)
                if flipped:
                    plan.append(PlanRegion(id=f"{d.id}-t{k2}", page=words[wid][0], poly=words[wid][1].poly,
                                           action="replace", replacement=flipped, mentions=[d.id]))
        if m is not None:
            for seg in m.segments:
                for rid in seg.regions:
                    # With stand_ins a signature, mark or image is replaced by drawn strokes (no text).
                    plan.append(PlanRegion(id=f"{d.id}-r{rid}", page=seg.page, poly=regions_idx[rid][1].poly,
                                           action="replace" if spec.stand_ins else "remove", mentions=[d.id]))
    return plan, replacements, word_of


def _mentions_artifact(gt: GTDocument, dets: list[_Det], token_of: dict[str, str] | None) -> MentionsArtifact:
    words = gt.word_index()
    out = []
    for d in dets:
        regions = []
        for page, wid, span in d.parts:
            w = words[wid][1]
            regions.append(RegionAnchor(page=page, poly=_word_region(w, span[0] if span else None,
                                                                     span[1] if span else None, 1.0).tolist()))
        if not regions:
            continue
        tokens = [token_of[wid] for _, wid, _ in d.parts if token_of and wid in token_of] or None
        text = " ".join(words[wid][1].text for _, wid, _ in d.parts)
        out.append(PredMention(id=d.id, label=d.label, text=text, regions=regions, tokens=tokens))
    return MentionsArtifact(mentions=out)



def _covered_words(gt: GTDocument, polys_by_page: dict[int, list], min_fraction: float = 0.5) -> set[str]:
    """GT words whose area is at least ``min_fraction`` covered by the given display_pt polygons."""
    from shapely import union_all

    from ..geometry import to_shapely

    out = set()
    for page, polys in polys_by_page.items():
        try:
            gp = gt.page(page)
        except KeyError:
            continue
        cover = union_all([to_shapely(as_poly(p)) for p in polys])
        for w in gp.words:
            wp = to_shapely(as_poly(w.poly))
            if wp.area > 0 and wp.intersection(cover).area / wp.area >= min_fraction:
                out.add(w.id)
    return out


def _read_from_text(gt: GTDocument, text: TextArtifact) -> set[str]:
    """GT words an injected text artifact actually read: covered by a token whose text
    resembles the word (so missed and garbled words stay unread downstream)."""
    from rapidfuzz import fuzz
    from shapely.strtree import STRtree

    from ..geometry import to_shapely

    read = set()
    for tp in text.pages:
        try:
            gp = gt.page(tp.index)
        except KeyError:
            continue
        if not tp.tokens:
            continue
        tpolys = [to_shapely(as_poly(t.poly)) for t in tp.tokens]
        tree = STRtree(tpolys)
        for w in gp.words:
            wp = to_shapely(as_poly(w.poly))
            hits = [i for i in tree.query(wp) if wp.area and tpolys[i].intersection(wp).area / wp.area >= 0.5]
            joined = "".join(tp.tokens[i].text for i in hits)
            if hits and fuzz.ratio(joined.casefold(), w.text.casefold()) >= 80:
                read.add(w.id)
    return read


def _dets_from_injected(gt: GTDocument, dets: list["_Det"], art: MentionsArtifact) -> list["_Det"]:
    """Use injected mentions as the detector would: keep exactly the GT words they cover
    (inheriting boundary errors), and turn covered words outside any GT mention into
    false-positive detections."""
    polys: dict[int, list] = {}
    for m in art.mentions:
        for r in m.regions or []:
            polys.setdefault(r.page, []).append(r.poly)
    covered = _covered_words(gt, polys)
    out, used = [], set()
    for d in dets:
        parts = [p for p in d.parts if p[1] in covered]
        if parts:
            out.append(_Det(id=d.id, gt_mention=d.gt_mention, label=d.label, parts=parts))
            used.update(p[1] for p in parts)
    words = gt.word_index()
    extra = sorted(covered - used, key=lambda w: (words[w][0], words[w][1].order))
    for wid in extra:
        out.append(_Det(id=f"x{len(out)}", gt_mention=None, label="PERSON_NAME", parts=[(words[wid][0], wid, None)]))
    return out

def target_regions(gt: GTDocument, policy: PolicyProfile, spec: MockSpec,
                   matter_gts: list[GTDocument] | None = None) -> tuple[list[PlanRegion], list[str], dict]:
    """Plan regions for a spec (without applying them). Kept for callers that only need the plan."""
    rng = _rng(spec)
    _, read, _ = _text_stage(gt, spec, rng)
    dets = _detect_stage(gt, policy, spec, read, rng)
    _, cluster_of = _resolve_stage(gt, dets, spec)
    plan, replacements, _ = _plan_stage(gt, dets, cluster_of, spec, rng, matter_gts, policy)
    return plan, [d.gt_mention for d in dets if d.gt_mention], replacements


def _apply_geometry_faults(plan: list[PlanRegion], gt: GTDocument, geoms: list[PageGeometry],
                           spec: MockSpec) -> list[PlanRegion]:
    out = []
    n_pages = len(gt.pages)
    for r in plan:
        poly = as_poly(r.poly)
        page = r.page
        g = geoms[page]
        if spec.dilate_pt:
            poly = dilate(poly, spec.dilate_pt)
        if spec.scale != 1.0:
            poly = poly * spec.scale
        if spec.shift_pt != (0.0, 0.0):
            poly = poly + np.array(spec.shift_pt)
        if spec.y_flip:
            poly = np.column_stack([poly[:, 0], g.display_size[1] - poly[:, 1]])
        if spec.rotation_unaware and g.rotation:
            # Coordinates computed as if the page were unrotated.
            poly = g.transform(poly, Frame.UNROTATED_PT, Frame.DISPLAY_PT)
        if spec.cropbox_unaware:
            # Coordinates relative to the MediaBox instead of the CropBox.
            poly = poly + np.array([g.mediabox[0] - g.cropbox[0], g.cropbox[3] - g.mediabox[3]])
        if spec.page_offset:
            page = min(max(page + spec.page_offset, 0), n_pages - 1)
        out.append(r.model_copy(update={"poly": [tuple(pt) for pt in poly.tolist()], "page": page}))
    return out


# -- application methods --------------------------------------------------------------------


def _geoms(pdf: bytes) -> list[PageGeometry]:
    with pikepdf.open(io.BytesIO(pdf)) as p:
        return [geometry_from_pikepdf_page(pg) for pg in p.pages]


def _scrub(doc: pymupdf.Document) -> None:
    doc.set_metadata({})
    doc.del_xml_metadata()


def _unrotated_rect(geom: PageGeometry, poly) -> pymupdf.Rect:
    pts = geom.transform(as_poly(poly), Frame.DISPLAY_PT, Frame.UNROTATED_PT)
    return pymupdf.Rect(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def _true_redact(source: bytes, plan: list[PlanRegion], geoms, stand_ins: frozenset[str] = frozenset()
                 ) -> pymupdf.Document:
    doc = pymupdf.open("pdf", source)
    for r in plan:
        rect = _unrotated_rect(geoms[r.page], r.poly)
        page = doc[r.page]
        if r.replacement:
            unit = pymupdf.get_text_length(r.replacement, fontname="helv", fontsize=1) or 1.0
            size = max(3.0, min(11.0, rect.height * 0.75, rect.width * 0.95 / unit))
            page.add_redact_annot(rect, text=r.replacement, fontname="helv", fontsize=size, fill=(1, 1, 1),
                                  text_color=(0, 0, 0), align=pymupdf.TEXT_ALIGN_LEFT)
        else:
            page.add_redact_annot(rect, fill=(1, 1, 1) if r.id in stand_ins else (0, 0, 0))
    for page in doc:
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    for r in plan:
        if r.id in stand_ins:
            for stroke in _stand_in_strokes(r.poly, _stable(r.id)):
                pts = geoms[r.page].transform(stroke, Frame.DISPLAY_PT, Frame.UNROTATED_PT)
                doc[r.page].draw_polyline([tuple(p) for p in pts.tolist()], color=(0, 0, 0), width=1.0)
    return doc


def _polys_user(plan, geoms, page_index):
    return [geoms[page_index].transform(as_poly(r.poly), Frame.DISPLAY_PT, Frame.PDF_USER)
            for r in plan if r.page == page_index]


def _overlay(source: bytes, plan: list[PlanRegion], geoms, opacity: float = 1.0) -> bytes:
    pdf = pikepdf.open(io.BytesIO(source))
    for i, page in enumerate(pdf.pages):
        polys = _polys_user(plan, geoms, i)
        if not polys:
            continue
        ops = ["q"]
        if opacity < 1.0:
            gs = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.ExtGState, ca=opacity, CA=opacity))
            res = page.obj.get("/Resources", pikepdf.Dictionary())
            ext = res.get("/ExtGState", pikepdf.Dictionary())
            ext["/GSrb"] = gs
            res["/ExtGState"] = ext
            page.obj.Resources = res
            ops.append("/GSrb gs")
        ops.append("0 g")
        for p in polys:
            ops.append(f"{p[0, 0]:.3f} {p[0, 1]:.3f} m " + " ".join(f"{x:.3f} {y:.3f} l" for x, y in p[1:]) + " h f")
        ops.append("Q")
        page.contents_coalesce()
        body = page.obj.Contents.read_bytes()
        page.obj.Contents = pdf.make_stream(b"q\n" + body + b"\nQ\n" + "\n".join(ops).encode())
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


def _annotation(source: bytes, plan: list[PlanRegion], geoms) -> pymupdf.Document:
    doc = pymupdf.open("pdf", source)
    for r in plan:
        a = doc[r.page].add_rect_annot(_unrotated_rect(geoms[r.page], r.poly))
        a.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
        a.update()
    return doc


def _rasterize(source: bytes, plan: list[PlanRegion], geoms, spec: MockSpec, blur: bool = False,
               keep_text: bool = False, gt: GTDocument | None = None,
               stand_ins: frozenset[str] = frozenset(), aligned: frozenset[str] = frozenset()) -> pymupdf.Document:
    src = pymupdf.open("pdf", source)
    out = pymupdf.open()
    s = spec.raster_dpi / 72.0
    # With a redacted text layer a replaced word is erased to paper and its fake set as text.
    text_layer = spec.redacted_text_layer and gt is not None
    for i, page in enumerate(src):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()
        for r in plan:
            if r.page != i:
                continue
            pts = np.round(as_poly(r.poly) * s).astype(np.int32)
            if blur:
                x0, y0 = max(pts[:, 0].min(), 0), max(pts[:, 1].min(), 0)
                x1, y1 = min(pts[:, 0].max(), img.shape[1]), min(pts[:, 1].max(), img.shape[0])
                if x1 > x0 and y1 > y0:
                    img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (5, 5), 1.2)
            elif r.id in stand_ins or (text_layer and r.replacement):
                # Paper white a little beyond the region, so no edge of the original stays.
                cv2.fillPoly(img, [np.round(dilate(as_poly(r.poly), 0.5) * s).astype(np.int32)], 255)
                if r.id in stand_ins:
                    for stroke in _stand_in_strokes(r.poly, _stable(r.id)):
                        cv2.polylines(img, [np.round(stroke * s).astype(np.int32)], False, 0, max(1, round(s)),
                                      cv2.LINE_AA)
            else:
                cv2.fillPoly(img, [pts], 0)
        w, h = geoms[i].display_size
        newp = out.new_page(width=w, height=h)
        ok, png = cv2.imencode(".png", img)
        newp.insert_image(newp.rect, stream=png.tobytes())
        if keep_text:
            for wd in page.get_text("words"):
                newp.insert_text((wd[0], wd[3] - 2), wd[4], fontsize=max(4, (wd[3] - wd[1]) * 0.8), render_mode=3)
        if text_layer:
            _redacted_text_layer(newp, gt, i, [r for r in plan if r.page == i],
                                 _line_pitch(gt.page(i)) if spec.text_next_line else 0.0, aligned)
    return out


# -- the text layer of a rasterised page -----------------------------------------------------------

_HELV_LINE = 1.075 + 0.299  # Helvetica ascender - descender: line height per point of size
_HELV_MID = (1.075 - 0.299) / 2  # centre of a character box above the baseline, per point of size


@lru_cache(maxsize=1)
def _helv() -> pymupdf.Font:
    return pymupdf.Font("helv")


def _angle_at(page: GTPage, poly) -> float:
    """Reading direction (a multiple of 90) of the GT word a region overlaps most."""
    shape = to_shapely(as_poly(poly))
    x0, y0, x1, y1 = shape.bounds
    best, angle = 0.0, 0.0
    for w in page.words:
        wx0, wy0, wx1, wy1 = bbox(as_poly(w.poly))
        if wx1 < x0 or wx0 > x1 or wy1 < y0 or wy0 > y1:
            continue
        a = shape.intersection(to_shapely(as_poly(w.poly))).area
        if a > best:
            best, angle = a, w.angle
    return round(angle / 90) * 90 % 360


def _place(poly, text: str, angle: float, scale: float | None) -> tuple[np.ndarray, float] | None:
    """Baseline start and font size that put every character's centre inside ``poly``: the text
    reads along ``angle`` (clockwise, a multiple of 90), is centred, ``scale`` of the polygon's line
    height and at most 90% of its length. ``scale`` None: from the polygon's start at the line's size
    (Helvetica's proportions), smaller only where the text would run past the polygon's end."""
    p = as_poly(poly)
    a = math.radians(angle)
    d = np.array([math.cos(a), math.sin(a)])  # reading direction (display_pt, y down)
    u = np.array([math.sin(a), -math.cos(a)])  # up the glyphs
    along, across = p @ d, p @ u
    length, height = float(np.ptp(along)), float(np.ptp(across))
    unit = _helv().text_length(text, fontsize=1)
    if not text.strip() or unit <= 0 or length <= 0 or height <= 0:
        return None
    if scale is None:
        size = min(height / _HELV_LINE, length / unit)
        start = d * along.min() + u * (across.min() + across.max()) / 2
        return start - u * (_HELV_MID * size), size
    size = min(height / _HELV_LINE * scale, 0.9 * length / unit)
    centre = d * (along.min() + along.max()) / 2 + u * (across.min() + across.max()) / 2
    return centre - d * (unit * size / 2) - u * (_HELV_MID * size), size


def _across(poly, angle: float, d: float) -> np.ndarray:
    """A polygon moved ``d`` pt across the reading direction ``angle`` (down the page for angle 0)."""
    a = math.radians(angle)
    return as_poly(poly) + d * np.array([-math.sin(a), math.cos(a)])


def _line_pitch(page: GTPage) -> float:
    """The page's line pitch: the median distance between consecutive GT lines, across them."""
    lines: dict = defaultdict(list)
    for w in page.words:
        a = math.radians(w.angle)
        lines[w.line or w.id].append(float(as_poly(w.poly).mean(axis=0) @ np.array([-math.sin(a), math.cos(a)])))
    centres = sorted(float(np.median(v)) for v in lines.values())
    steps = [b - a for a, b in zip(centres, centres[1:]) if b - a > 2.0]
    return float(np.median(steps)) if steps else 14.0


def _redacted_text_layer(page: pymupdf.Page, gt: GTDocument, index: int, plan: list[PlanRegion],
                         offset: float = 0.0, aligned: frozenset[str] = frozenset()) -> None:
    """What an OCR'd scan carries: invisible text (render mode 3) for every GT word left on the
    page, at its place, and each replacement as visible text in its erased region: from the region's
    start at the original size where it may line up with the original (``aligned``), else centred and
    a little smaller."""
    try:
        gp = gt.page(index)
    except KeyError:
        return
    removed = _covered_words(gt, {index: [r.poly for r in plan]}) if plan else set()
    items = [(w.poly, w.text, round(w.angle / 90) * 90 % 360, 3, 1.0) for w in gp.words if w.id not in removed]
    # Fakes from the word's start at the original size (eval-0.4: the character identity gate tells a fake
    # from the original glyphs; eval-0.3.1 needed them centred and a little smaller than the line).
    items += [(r.poly, r.replacement, _angle_at(gp, r.poly), 0, None if r.id in aligned else 0.8)
              for r in plan if r.replacement]
    if offset:  # the text layer drawn one line pitch below the words it stands for
        items = [(_across(poly, angle, offset), text, angle, mode, scale) for poly, text, angle, mode, scale in items]
    font = _helv()
    flat = {0: pymupdf.TextWriter(page.rect), 3: pymupdf.TextWriter(page.rect)}
    used = set()
    for poly, text, angle, mode, scale in items:
        placed = _place(poly, text, angle, scale)
        if placed is None:
            continue
        start, size = placed
        if angle == 0:
            flat[mode].append(tuple(start), text, font=font, fontsize=size)
            used.add(mode)
            continue
        tw = pymupdf.TextWriter(page.rect)
        tw.append(tuple(start), text, font=font, fontsize=size)
        tw.write_text(page, render_mode=mode, color=(0, 0, 0),
                      morph=(pymupdf.Point(*start), pymupdf.Matrix(-angle)))
    for mode in sorted(used):
        flat[mode].write_text(page, render_mode=mode, color=(0, 0, 0))


def _incremental(source: bytes, plan: list[PlanRegion], geoms) -> bytes:
    """True redaction appended as an incremental update: the original revision stays in the file."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "doc.pdf")
        with open(path, "wb") as f:
            f.write(source)
        doc = pymupdf.open(path)
        for r in plan:
            doc[r.page].add_redact_annot(_unrotated_rect(geoms[r.page], r.poly), fill=(0, 0, 0))
        for page in doc:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
        doc.saveIncr()
        doc.close()
        with open(path, "rb") as f:
            return f.read()


# -- fakes drawn the way tools draw them (eval-0.4 self-checks) -----------------------------------------------


def _draws_fakes(spec: MockSpec) -> bool:
    return bool(spec.fakes_in_place or spec.retypeset or spec.retypeset_keep or spec.overprint or spec.veil
                or spec.shift_white or spec.address_whole or spec.keeps_locality_moved or spec.erases_locality)


_BASE14 = (("courier", "cour", "cobo"), ("times", "tiro", "tibo"), ("helvetica", "helv", "hebo"),
           ("arial", "helv", "hebo"))


@lru_cache(maxsize=None)
def _font(name: str) -> pymupdf.Font:
    return pymupdf.Font(name)


def _base14(span_font: str) -> str:
    """A base-14 stand-in of a text span's font family (Helvetica when it is none of them)."""
    f = span_font.casefold()
    bold = "bold" in f
    for key, regular, heavy in _BASE14:
        if key in f:
            return heavy if bold else regular
    return "hebo" if bold else "helv"


@dataclass(frozen=True)
class _Style:
    origin: tuple[float, float]  # where the word's baseline starts, display pt
    size: float
    angle: float  # reading direction, degrees clockwise in display pt
    font: str  # a base-14 PyMuPDF font name

    def at(self, along: float = 0.0, across: float = 0.0) -> tuple[float, float]:
        a = math.radians(self.angle)
        return (self.origin[0] + along * math.cos(a) - across * math.sin(a),
                self.origin[1] + along * math.sin(a) + across * math.cos(a))


def _styles(src: pymupdf.Document, gt: GTDocument, geoms) -> dict[str, _Style]:
    """Each GT word's font, size and baseline origin: from the source's text layer, else from its box (size
    = height / 1.374, baseline 0.21 of the height above its bottom, Helvetica)."""
    out: dict[str, _Style] = {}
    for page in gt.pages:
        if page.index >= src.page_count:
            continue
        geom = geoms[page.index]
        chars = [((x0 + x1) / 2, (y0 + y1) / 2, ch["origin"][0], ch["origin"][1], sp["size"], sp["font"])
                 for b in src[page.index].get_text("rawdict")["blocks"] for ln in b.get("lines", [])
                 for sp in ln["spans"] for ch in sp["chars"] for x0, y0, x1, y1 in [ch["bbox"]]
                 if not ch["c"].isspace()]
        if chars:
            a = np.array([c[:4] for c in chars], float)
            cen = geom.transform(a[:, 0:2], Frame.UNROTATED_PT, Frame.DISPLAY_PT)
            org = geom.transform(a[:, 2:4], Frame.UNROTATED_PT, Frame.DISPLAY_PT)
        for w in page.words:
            angle = round(w.angle / 90) * 90 % 360
            p = as_poly(w.poly)
            style = None
            if chars:
                inside = shapely.contains_xy(to_shapely(dilate(p, 0.5)), cen[:, 0], cen[:, 1])
                if inside.any():
                    idx = np.flatnonzero(inside)
                    u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
                    first = idx[int(np.argmin(cen[idx] @ u))]
                    style = _Style((float(org[first][0]), float(org[first][1])),
                                   float(np.median([chars[i][4] for i in idx])), angle, _base14(chars[first][5]))
            if style is None:
                h = float(np.hypot(*(p[3] - p[0])))
                base = p[3] + 0.21 * (p[0] - p[3])
                style = _Style((float(base[0]), float(base[1])), h / _HELV_LINE, angle, "helv")
            out[w.id] = style
    return out


def _write(page: pymupdf.Page, geom: PageGeometry, origin, angle: float, text: str, size: float, font: str) -> None:
    """``text`` from a baseline origin (display pt) along ``angle``, in a page of any rotation."""
    if not text:
        return
    a = math.radians(angle)
    pts = geom.transform(np.array([origin, (origin[0] + math.cos(a), origin[1] + math.sin(a))], float),
                         Frame.DISPLAY_PT, Frame.UNROTATED_PT)
    ang = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    tw = pymupdf.TextWriter(page.rect)
    tw.append(tuple(pts[0]), text, font=_font(font), fontsize=size)
    if abs(ang) < 0.01:
        tw.write_text(page, color=(0, 0, 0))
    else:
        tw.write_text(page, color=(0, 0, 0), morph=(pymupdf.Point(*pts[0]), pymupdf.Matrix(-ang)))


def _over_original(ws, texts: list[str], styles: dict, fake_of: dict, s: float) -> bool:
    """Would a line re-typeset at ``s`` from its first word put a letter of a fake over the same or a look-alike
    letter of the original it replaces (along the line, characters overlapping by more than 30%)?"""
    first = styles[ws[0].id]
    a = math.radians(first.angle)
    u = np.array([math.cos(a), math.sin(a)])
    x = 0.0
    for w, text in zip(ws, texts):
        st = styles[w.id]
        font = _font(st.font)
        if fake_of.get(w.id) is not None:
            o0 = float((np.asarray(st.origin) - np.asarray(first.origin)) @ u)
            orig = [(o0 + font.text_length(w.text[:j], fontsize=st.size),
                     o0 + font.text_length(w.text[:j + 1], fontsize=st.size)) for j in range(len(w.text))]
            for i, c in enumerate(text):
                if not c.isalnum():
                    continue
                lo = x + font.text_length(text[:i], fontsize=st.size * s)
                hi = x + font.text_length(text[:i + 1], fontsize=st.size * s)
                for (p, q), o in zip(orig, w.text):
                    if min(hi, q) - max(lo, p) > 0.3 * min(hi - lo, q - p) and (
                            c.casefold() == o.casefold() or any(c in g and o in g for g in _ALIKE)):
                        return True
        x += font.text_length(text + " ", fontsize=st.size * s)
    return False


def _draw_fakes(source: bytes, plan: list[PlanRegion], word_of: dict, geoms, gt: GTDocument, spec: MockSpec,
                stand_ins: frozenset[str] = frozenset()) -> pymupdf.Document:
    """Fakes drawn the way tools draw them, each in its word's font and size from its origin: in place after
    true redaction (``fakes_in_place``), with every line holding one re-typeset (``retypeset``, or the
    originals with ``retypeset_keep``), over the original (``overprint``), on a translucent white box
    (``veil``) or in a white box shifted across the line (``shift_white``); addresses as units
    (``address_whole``, ``keeps_locality_moved``, ``erases_locality``). Regions with no fake are removed as
    with true redaction."""
    src = pymupdf.open("pdf", source)
    styles = _styles(src, gt, geoms)
    words = gt.word_index()
    doc = pymupdf.open("pdf", source)
    erase: list[tuple[int, object, tuple]] = []  # (page, display polygon, fill) truly redacted
    boxes: list[tuple[int, object, float]] = []  # (page, display polygon, opacity) white boxes drawn
    draws: list[tuple[int, tuple, float, float, str, str]] = []  # (page, origin, angle, size, font, text)

    def text_of(r: PlanRegion) -> tuple[str, str, int]:
        wid, span = word_of[r.id]
        return wid, r.replacement, span[0] if span else 0

    def draw_at(wid: str, text: str, along: float = 0.0, across: float = 0.0, scale: float = 1.0) -> None:
        st = styles[wid]
        draws.append((words[wid][0], st.at(along, across), st.angle, st.size * scale, st.font, text))

    def width(wid: str, text: str, scale: float = 1.0) -> float:
        st = styles[wid]
        return _font(st.font).text_length(text, fontsize=st.size * scale)

    fakes = [r for r in plan if r.replacement and r.id in word_of]
    fake_of = {word_of[r.id][0]: r for r in fakes}
    done: set[str] = set()  # word ids an address mode has drawn already
    removed = {word_of[r.id][0] for r in plan if r.id in word_of and not r.replacement}
    white, black = (1, 1, 1), (0, 0, 0)
    for r in plan:
        if not (r.replacement and r.id in word_of) and not spec.retypeset_keep:
            # removed (a box), or a signature's stand-in (strokes after); a tool keeping its originals keeps these
            erase.append((r.page, r.poly, white if r.id in stand_ins else black))

    # -- addresses as units ----------------------------------------------------------------------------
    address = spec.address_whole or spec.keeps_locality_moved or spec.erases_locality
    for street, loc in (address_unit_mentions(gt) if address else []):
        s_ids = [w for seg in street.segments for w in seg.words] if street is not None else []
        l_ids = [w for seg in loc.segments for w in seg.words] if loc is not None else []
        # a part the tool does not replace (the policy does not require it) stays as it is
        s_ids = s_ids if any(w in fake_of or w in removed for w in s_ids) else []
        l_ids = l_ids if any(w in fake_of or w in removed for w in l_ids) else []
        if not any(w in fake_of for w in s_ids + l_ids):
            continue
        if spec.address_whole or (spec.keeps_locality_moved == "line" and s_ids):
            by_line: dict = {}
            for w in s_ids + l_ids:
                by_line.setdefault((words[w][0], words[w][1].line or w), []).append(w)
            for ws in by_line.values():
                keep_loc = spec.keeps_locality_moved == "line" and any(w in s_ids for w in ws)
                if spec.keeps_locality_moved == "line" and not keep_loc:
                    continue  # a locality on its own line stays as it is
                # a word with no fake is shared structure kept as written (or a word removed with none)
                text = " ".join(words[w][1].text if (keep_loc and w in l_ids) else
                                fake_of[w].replacement if w in fake_of else "" if w in removed else
                                words[w][1].text for w in ws)
                for w in ws:
                    erase.append((words[w][0], words[w][1].poly, white))
                    done.add(w)
                draw_at(ws[0], " ".join(text.split()))
            if spec.keeps_locality_moved == "line":
                done.update(l_ids)  # kept where they are, or moved along with the line
            continue
        if spec.keeps_locality_moved == "next_line" or spec.erases_locality:
            for w in l_ids:
                erase.append((words[w][0], words[w][1].poly, white))
                done.add(w)
            if spec.keeps_locality_moved == "next_line" and l_ids:
                draw_at(l_ids[0], " ".join(words[w][1].text for w in l_ids),
                        across=_line_pitch(gt.page(words[l_ids[0]][0])))

    # -- every other fake ------------------------------------------------------------------------------
    rest = [r for r in fakes if word_of[r.id][0] not in done]
    lines: dict = {}
    if spec.retypeset or spec.retypeset_keep:
        for r in rest:
            wid = word_of[r.id][0]
            lines.setdefault((words[wid][0], words[wid][1].line or wid), set()).add(wid)
        rng = random.Random(_stable("retypeset", gt.doc_id))
        for (pg, line), _ in sorted(lines.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            ws = sorted((w for w in gt.page(pg).words if (w.line or w.id) == line), key=lambda w: w.order)
            texts = [w.text if (fake_of.get(w.id) is None or spec.retypeset_keep) else fake_of[w.id].replacement
                     for w in ws]
            first = styles[ws[0].id]
            # A size in 1.02-1.04x whose fakes put no letter over the same or a look-alike letter of an
            # original they replace (that glyph would be the original's: a chance any re-typesetting tool runs).
            sizes = [rng.uniform(1.02, 1.04)] + [1.02 + 0.0025 * k for k in range(9)]
            s = next((x for x in sizes if spec.retypeset_keep or not _over_original(ws, texts, styles, fake_of, x)),
                     sizes[0])
            x = 0.0
            for w, text in zip(ws, texts):
                erase.append((pg, w.poly, white))
                if fake_of.get(w.id) is None and w.id in removed:
                    continue  # removed with no fake
                st = styles[w.id]
                draws.append((pg, first.at(x), first.angle, st.size * s, st.font, text))
                x += _font(st.font).text_length(text + " ", fontsize=st.size * s)
    else:
        for r in rest:
            wid, text, k0 = text_of(r)
            lead = width(wid, words[wid][1].text[:k0]) if k0 else 0.0
            if spec.overprint:
                draw_at(wid, text, lead)
            elif spec.veil:
                boxes.append((r.page, r.poly, spec.veil))
                draw_at(wid, text, lead)
            elif spec.shift_white:
                p = as_poly(r.poly)
                h = float(np.hypot(*(p[3] - p[0])))
                boxes.append((r.page, _across(p, styles[wid].angle, spec.shift_white * h), 1.0))
                draw_at(wid, text, lead, spec.shift_white * h)
            else:
                erase.append((r.page, r.poly, white))
                draw_at(wid, text, lead)

    for pg, poly, fill in erase:
        doc[pg].add_redact_annot(_unrotated_rect(geoms[pg], poly), fill=fill)
    for page in doc:
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
                              graphics=(pymupdf.PDF_REDACT_LINE_ART_NONE if spec.retypeset_keep
                                        else pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED),
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    for r in plan:
        if r.id in stand_ins:
            for stroke in _stand_in_strokes(r.poly, _stable(r.id)):
                pts = geoms[r.page].transform(stroke, Frame.DISPLAY_PT, Frame.UNROTATED_PT)
                doc[r.page].draw_polyline([tuple(p) for p in pts.tolist()], color=(0, 0, 0), width=1.0)
    for pg, poly, opacity in boxes:
        doc[pg].draw_rect(_unrotated_rect(geoms[pg], poly), color=None, fill=(1, 1, 1), fill_opacity=opacity)
    for pg, origin, angle, size, font, text in draws:
        _write(doc[pg], geoms[pg], origin, angle, text, size, font)
    if spec.retypeset or spec.retypeset_keep:
        # A scanned page has no text layer: the tool OCRs it, so the words it did not re-typeset get invisible
        # text at their place, as the ceiling's scans do.
        drawn = {w.id for p in gt.pages for w in p.words if (p.index, w.line or w.id) in lines}
        for p in gt.pages:
            if p.index < src.page_count and not src[p.index].get_text().strip():
                for w in p.words:
                    placed = _place(w.poly, w.text, round(w.angle / 90) * 90 % 360, 1.0) if w.id not in drawn else None
                    if placed is not None:
                        _write_invisible(doc[p.index], geoms[p.index], placed[0], round(w.angle / 90) * 90 % 360,
                                         w.text, placed[1])
    return doc


def _write_invisible(page: pymupdf.Page, geom: PageGeometry, origin, angle: float, text: str, size: float) -> None:
    """Invisible text (render mode 3) from a baseline origin (display pt) along ``angle``."""
    a = math.radians(angle)
    pts = geom.transform(np.array([origin, (origin[0] + math.cos(a), origin[1] + math.sin(a))], float),
                         Frame.DISPLAY_PT, Frame.UNROTATED_PT)
    ang = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    tw = pymupdf.TextWriter(page.rect)
    tw.append(tuple(pts[0]), text, font=_helv(), fontsize=size)
    if abs(ang) < 0.01:
        tw.write_text(page, render_mode=3, color=(0, 0, 0))
    else:
        tw.write_text(page, render_mode=3, color=(0, 0, 0), morph=(pymupdf.Point(*pts[0]), pymupdf.Matrix(-ang)))


def _aligned(plan: list[PlanRegion], word_of: dict, gt: GTDocument, policy: PolicyProfile) -> frozenset[str]:
    """Regions whose fake may be drawn from the word's start at the original size: every character it has in
    common with the original at the same index is shared structure the policy allows (a fake that keeps a
    mobile's '04' or a landline's leading '0' where the policy does not share it would show the original's
    glyphs there, so it is drawn centred and smaller, as before eval-0.4). A number keeping no shared digit is
    drawn that way too: same-font digit groups of a phone or card number look alike, and the moved search can
    take a neighbouring fake group for its original moved (measured on the fixtures: '485' for '456')."""
    shared = document_shared_units(gt, policy)
    words = gt.word_index()
    owner = {wid: m.id for m in gt.mentions for seg in m.segments for wid in seg.words}
    out = set()
    for r in plan:
        if not r.replacement or r.id not in word_of:
            continue
        wid, span = word_of[r.id]
        k0 = span[0] if span else 0
        text = words[wid][1].text[k0:span[1]] if span else words[wid][1].text
        su = shared.get(owner.get(wid, ""), SharedUnits())
        same = {i for i, (a, b) in enumerate(zip(text, r.replacement)) if a == b and a.isalnum()}
        number = not any(ch.isalpha() for ch in r.replacement)
        if all(wid in su.full or (wid, i + k0) in su.chars for i in same) and (same or not number):
            out.add(r.id)
    return frozenset(out)


def _clean(spec: MockSpec, **overrides) -> MockSpec:
    from dataclasses import replace

    return replace(spec, **overrides)


def _poly_key(page: int, poly) -> tuple:
    return (page, tuple(round(float(v), 3) for pt in poly for v in pt))


def _region_parts(plan: list[PlanRegion], gt: GTDocument) -> frozenset[str]:
    """Plan regions that are a GT region (signature, mark, image): they get a stand-in."""
    keys = {_poly_key(p.index, r.poly) for p in gt.pages for r in p.regions}
    return frozenset(r.id for r in plan if _poly_key(r.page, r.poly) in keys)


def _with_fakes(injected: list[PlanRegion], own: list[PlanRegion]) -> list[PlanRegion]:
    """Injected regions (oracle_regions) filled with the fakes the mock made for the same word."""
    fake = {_poly_key(r.page, r.poly): r.replacement for r in own if r.replacement}
    out = []
    for r in injected:
        text = fake.get(_poly_key(r.page, r.poly))
        out.append(r.model_copy(update={"action": "replace", "replacement": text}) if text else r)
    return out


def run_mock(source: bytes, gt: GTDocument, policy: PolicyProfile, spec: MockSpec,
             injected: dict | None = None, matter_gts: list[GTDocument] | None = None) -> MockOutput:
    """``injected`` holds oracle artifacts by kind ("text", "mentions", "entities",
    "regions"); an injected stage replaces the mock's own (faulty) version of it.
    ``matter_gts``: the GT of the other documents of the matter, for fakes keyed on GT entities
    (``pseudonym_key="entity"``) that stay the same across the matter."""
    injected = injected or {}
    rng = _rng(spec)
    geoms = _geoms(source)
    if "text" in injected or "mentions" in injected:
        spec_t = _clean(spec, ocr_char_error=0.0, ocr_drop_types=(), ocr_drop_conditions=(), ocr_box_jitter_pt=0.0)
    else:
        spec_t = spec
    text, read, token_of = _text_stage(gt, spec_t, rng)
    if "text" in injected:
        text = injected["text"]
        read = _read_from_text(gt, text)  # an upstream tool's misreadings propagate
    spec_d = _clean(spec_t, drop_types=(), drop_conditions=(), drop_fraction=0.0, boundary="none", type_swap=(),
                    redact_protected=False) if "mentions" in injected else spec_t
    dets = _detect_stage(gt, policy, spec_d, read, rng)
    if "mentions" in injected:
        dets = _dets_from_injected(gt, dets, injected["mentions"])
    if "entities" in injected:
        spec_r = _clean(spec_d, merge_same_given=False, split_aliases=False, split_conditions=())
    else:
        spec_r = spec_d
    entities, cluster_of = _resolve_stage(gt, dets, spec_r)
    plan, replacements, word_of = _plan_stage(gt, dets, cluster_of, spec_r, rng, matter_gts, policy)
    if "regions" in injected:
        own = plan
        plan = list(injected["regions"].regions)
        if spec.replace_all:  # a tool that replaces everything fills the given regions with its own fakes
            plan = _with_fakes(plan, own)
        else:
            replacements = {}
    stand_ins = _region_parts(plan, gt) if spec.stand_ins else frozenset()
    if "regions" not in injected:
        plan = _apply_geometry_faults(plan, gt, geoms, spec)
    pseud = None
    if spec.pseudonyms != "none" or spec.replace_all:
        pseud = PseudonymsArtifact(
            mentions=[MentionReplacement(mention=k, replacement=v) for k, v in replacements.items()],
            entries=[PseudonymEntry(cluster=cluster_of[k.split(":")[0]], original=k.split(":")[1], replacement=v)
                     for k, v in replacements.items()])
    m = spec.method
    if m == "null":
        data = source
    elif m == "corrupt":
        data = source[: len(source) // 3]
    elif spec.incremental:
        data = _incremental(source, plan, geoms)
    elif m in ("overlay", "translucent"):
        data = _overlay(source, plan, geoms, opacity=0.9 if m == "translucent" else 1.0)
    else:
        if m == "true" and _draws_fakes(spec):
            doc = _draw_fakes(source, plan, word_of, geoms, gt, spec, stand_ins)
        elif m == "true":
            doc = _true_redact(source, plan, geoms, stand_ins)
        elif m == "annotation":
            doc = _annotation(source, plan, geoms)
        elif m in ("rasterize", "blur"):
            doc = _rasterize(source, plan, geoms, spec, blur=(m == "blur"), keep_text=spec.keep_text_layer, gt=gt,
                             aligned=_aligned(plan, word_of, gt, policy),
                             stand_ins=stand_ins)
        else:
            raise ValueError(f"unknown method {m}")
        if m != "annotation":
            _scrub(doc)
        if spec.leak_metadata:
            claimant = next((e for e in gt.entities if e.role.value == "claimant"), None)
            doc.set_metadata({"author": claimant.canonical_name if claimant else "unknown"})
        data = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    return MockOutput(pdf=data, plan=PlanArtifact(regions=plan), text=text,
                      mentions=_mentions_artifact(gt, dets, token_of), entities=entities, pseudonyms=pseud,
                      targeted=[d.gt_mention for d in dets if d.gt_mention])
