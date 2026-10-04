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
    [A] apply     true redaction / overlay / rasterise / ...       -> output.pdf
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import pikepdf
import pymupdf

from ..geometry import Frame, PageGeometry, as_poly, bbox, dilate, geometry_from_pikepdf_page, rect_poly
from ..gt.policy import resolve
from ..schemas import (
    Cluster,
    EntitiesArtifact,
    GTDocument,
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
    raster_dpi: float = 150
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


def _style(template: str, value: str) -> str:
    core = template.strip(".,;:()")
    lead = template[: len(template) - len(template.lstrip(".,;:()"))]
    trail = template[len(template.rstrip(".,;:()")):]
    if core.isupper() and len(core) > 1:
        value = value.upper()
    return lead + value + trail


class _Pseudonymiser:
    """Maps (cluster, component) to a replacement value; keyed on *predicted* clusters."""

    def __init__(self, gender_of: dict[str, str], mode: str, rng: random.Random):
        self.gender_of, self.mode, self.rng = gender_of, mode, rng
        self.table: dict[tuple[str, str], str] = {}
        self.counter = 0

    def value(self, cluster: str, component: NameComponent, original: str) -> str:
        self.counter += 1
        comp = component.value if self.mode != "inconsistent" else f"{component.value}{self.counter}"
        key = ("all" if self.mode == "collision" else cluster, comp)
        if key not in self.table:
            g = self.gender_of.get(cluster, "F")
            if self.mode == "gender_flip":
                g = "M" if g == "F" else "F"
            i = len(self.table)
            if component in (NameComponent.GIVEN, NameComponent.MIDDLE):
                v = _GIVEN[g][i % len(_GIVEN[g])]
            elif component is NameComponent.NICKNAME:
                v = _NICK[g][i % len(_NICK[g])]
            elif component is NameComponent.INITIALS:
                v = "".join(self.rng.choice("BCDFGHKLMNPRST") for _ in range(max(1, len(original))))
            else:
                v = _SURNAMES[i % len(_SURNAMES)]
            self.table[key] = v
        return self.table[key]


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


def _plan_stage(gt: GTDocument, dets: list[_Det], cluster_of: dict[str, str], spec: MockSpec,
                rng: random.Random) -> tuple[list[PlanRegion], dict[str, str]]:
    words = gt.word_index()
    regions_idx = gt.region_index()
    mentions = {m.id: m for m in gt.mentions}
    entities = {e.id: e for e in gt.entities}
    gender_of = {}
    for d in dets:
        m = mentions.get(d.gt_mention) if d.gt_mention else None
        e = entities.get(m.entity) if m and m.entity else None
        if e and "gender" in e.attributes:
            gender_of.setdefault(cluster_of[d.id], e.attributes["gender"].value)
    mode = spec.pseudonyms if spec.pseudonyms != "none" else "consistent"
    pseud = _Pseudonymiser(gender_of, mode, rng) if (spec.pseudonyms != "none" or spec.replace_other != "none")         else None
    plan: list[PlanRegion] = []
    replacements: dict[str, str] = {}
    for d in dets:
        m = mentions.get(d.gt_mention) if d.gt_mention else None
        comp_of = {wid: c for c, wids in (m.components.items() if m else []) for wid in wids}
        other = None
        if spec.replace_other != "none" and m is not None and m.type.value in OTHER_VALUES:
            other = _distribute(_other_value(m, spec.replace_other, pseud, cluster_of[d.id]), len(d.parts))
        for k, (page, wid, span) in enumerate(d.parts):
            w = words[wid][1]
            poly = _word_region(w, span[0] if span else None, span[1] if span else None, spec.partial)
            repl = None
            if other is not None and other[k]:
                repl = other[k]
                replacements[f"{d.id}:{wid}"] = repl
            elif spec.pseudonyms != "none" and m is not None and m.type is PiiType.PERSON_NAME and wid in comp_of:
                original = w.text if span is None else w.text[span[0]:span[1]]
                repl = _style(original, pseud.value(cluster_of[d.id], comp_of[wid], original))
                replacements[f"{d.id}:{wid}"] = repl
            plan.append(PlanRegion(id=f"{d.id}-{k}", page=page, poly=poly.tolist(),
                                   action="replace" if repl else "remove", replacement=repl, mentions=[d.id]))
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
                    plan.append(PlanRegion(id=f"{d.id}-r{rid}", page=seg.page, poly=regions_idx[rid][1].poly,
                                           action="remove", mentions=[d.id]))
    return plan, replacements


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

def target_regions(gt: GTDocument, policy: PolicyProfile, spec: MockSpec) -> tuple[list[PlanRegion], list[str], dict]:
    """Plan regions for a spec (without applying them). Kept for callers that only need the plan."""
    rng = _rng(spec)
    _, read, _ = _text_stage(gt, spec, rng)
    dets = _detect_stage(gt, policy, spec, read, rng)
    _, cluster_of = _resolve_stage(gt, dets, spec)
    plan, replacements = _plan_stage(gt, dets, cluster_of, spec, rng)
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


def _true_redact(source: bytes, plan: list[PlanRegion], geoms) -> pymupdf.Document:
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
            page.add_redact_annot(rect, fill=(0, 0, 0))
    for page in doc:
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
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
               keep_text: bool = False) -> pymupdf.Document:
    src = pymupdf.open("pdf", source)
    out = pymupdf.open()
    s = spec.raster_dpi / 72.0
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
            else:
                cv2.fillPoly(img, [pts], 0)
        w, h = geoms[i].display_size
        newp = out.new_page(width=w, height=h)
        ok, png = cv2.imencode(".png", img)
        newp.insert_image(newp.rect, stream=png.tobytes())
        if keep_text:
            for wd in page.get_text("words"):
                newp.insert_text((wd[0], wd[3] - 2), wd[4], fontsize=max(4, (wd[3] - wd[1]) * 0.8), render_mode=3)
    return out


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


def _clean(spec: MockSpec, **overrides) -> MockSpec:
    from dataclasses import replace

    return replace(spec, **overrides)


def run_mock(source: bytes, gt: GTDocument, policy: PolicyProfile, spec: MockSpec,
             injected: dict | None = None) -> MockOutput:
    """``injected`` holds oracle artifacts by kind ("text", "mentions", "entities",
    "regions"); an injected stage replaces the mock's own (faulty) version of it."""
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
    plan, replacements = _plan_stage(gt, dets, cluster_of, spec_r, rng)
    if "regions" in injected:
        plan, replacements = list(injected["regions"].regions), {}
    else:
        plan = _apply_geometry_faults(plan, gt, geoms, spec)
    pseud = None
    if spec.pseudonyms != "none":
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
        if m == "true":
            doc = _true_redact(source, plan, geoms)
        elif m == "annotation":
            doc = _annotation(source, plan, geoms)
        elif m in ("rasterize", "blur"):
            doc = _rasterize(source, plan, geoms, spec, blur=(m == "blur"), keep_text=spec.keep_text_layer)
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
