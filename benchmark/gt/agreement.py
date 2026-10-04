"""Inter-annotator agreement between two GT files of the same document (PLAN.md §7.5).

Annotators may tokenise, box and number words differently, so B's words are first
matched to A's geometrically (one-to-one, polygon IoU >= 0.5, same page). Agreement is
then computed in A's word space:

- mentions: strict (same required words and type) and tolerant (required words equal up
  to either side's optional words, type ignored), as F1 = 2·matched / (|A| + |B|)
- tokens: Cohen's kappa over matched words, label = PII type of the mention covering it
  (or O), and over classified (protected) spans separately
- entity links: B³ over mentions matched in both, A's entities as key
- geometry: IoU distribution of matched words; regions matched the same way

Release gates (annotation guidelines): tolerant mention F1 >= 0.95 and entity-link B³ F1
>= 0.95.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..geometry import as_poly, to_shapely
from ..metrics.resolution import b_cubed
from ..schemas import GTDocument

GATE_MENTION_F1 = 0.95
GATE_LINK_F1 = 0.95


@dataclass
class AgreementResult:
    words_a: int
    words_b: int
    words_matched: int
    word_iou_median: float | None
    word_text_agreement: float | None
    regions_matched: dict
    mentions_a: int
    mentions_b: int
    strict_f1: float | None
    tolerant_f1: float | None
    token_kappa: float | None
    classified_kappa: float | None
    entity_link_b3: dict
    disagreements: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.gates.values()) if self.gates else False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def _iou(a, b) -> float:
    pa, pb = to_shapely(as_poly(a)), to_shapely(as_poly(b))
    inter = pa.intersection(pb).area
    union = pa.union(pb).area
    return inter / union if union > 0 else 0.0


def _match(items_a: list, items_b: list, threshold: float = 0.5) -> dict[str, tuple[str, float]]:
    """One-to-one geometric matching of (id, poly) lists. Returns b_id -> (a_id, iou)."""
    if not items_a or not items_b:
        return {}
    boxes_a = np.array([to_shapely(as_poly(p)).bounds for _, p in items_a])
    sim = np.zeros((len(items_b), len(items_a)))
    for j, (_, pb) in enumerate(items_b):
        x0, y0, x1, y1 = to_shapely(as_poly(pb)).bounds
        near = np.flatnonzero((boxes_a[:, 0] < x1) & (boxes_a[:, 2] > x0) & (boxes_a[:, 1] < y1) & (boxes_a[:, 3] > y0))
        for i in near:
            sim[j, i] = _iou(items_a[i][1], pb)
    rows, cols = linear_sum_assignment(-sim)
    return {items_b[r][0]: (items_a[c][0], float(sim[r, c])) for r, c in zip(rows, cols) if sim[r, c] >= threshold}


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe >= 1.0 else round((po - pe) / (1 - pe), 4)


def _f1(matched: int, na: int, nb: int) -> float | None:
    return round(2 * matched / (na + nb), 4) if na + nb else None


def agreement(a: GTDocument, b: GTDocument) -> AgreementResult:
    if a.source_sha256 != b.source_sha256:
        raise ValueError("the two GT files annotate different source documents")
    word_map: dict[str, str] = {}
    region_map: dict[str, str] = {}
    ious, same_text = [], []
    words_a = a.word_index()
    words_b = b.word_index()
    for pa in a.pages:
        try:
            pb = b.page(pa.index)
        except KeyError:
            continue
        m = _match([(w.id, w.poly) for w in pa.words], [(w.id, w.poly) for w in pb.words])
        for bid, (aid, iou) in m.items():
            word_map[bid] = aid
            ious.append(iou)
            same_text.append(words_a[aid][1].text == words_b[bid][1].text)
        for bid, (aid, _) in _match([(r.id, r.poly) for r in pa.regions], [(r.id, r.poly) for r in pb.regions]).items():
            region_map[bid] = aid

    def key(m, side_b: bool):
        req = set()
        for seg in m.segments:
            for w in seg.words:
                req.add(("w", word_map.get(w, f"B:{w}") if side_b else w,
                         (seg.char_start, seg.char_end)))
            for r in seg.regions:
                req.add(("r", region_map.get(r, f"B:{r}") if side_b else r, None))
        opt = {("w", word_map.get(w, f"B:{w}") if side_b else w, (None, None)) for w in m.optional_words}
        return frozenset(req), frozenset(opt)

    ka = {m.id: key(m, False) for m in a.mentions}
    kb = {m.id: key(m, True) for m in b.mentions}
    type_a = {m.id: m.type.value for m in a.mentions}
    type_b = {m.id: m.type.value for m in b.mentions}

    # strict: identical required set and type, one-to-one
    strict_pairs = {}
    by_key_a: dict = {}
    for mid, (req, _) in ka.items():
        by_key_a.setdefault((req, type_a[mid]), []).append(mid)
    for mid, (req, _) in kb.items():
        cands = by_key_a.get((req, type_b[mid]))
        if cands:
            strict_pairs[mid] = cands.pop()

    # tolerant: required sets equal up to optional words, any type, one-to-one
    def strip(req):
        return frozenset((k, ident, None) for k, ident, _ in req)

    tolerant_pairs = {}
    used = set()
    for mb, (rb, ob) in kb.items():
        for ma, (ra, oa) in ka.items():
            if ma in used:
                continue
            diff = strip(ra) ^ strip(rb)
            if diff and not diff <= strip(oa | ob):
                continue
            if not (strip(ra) & strip(rb)):
                continue
            tolerant_pairs[mb] = ma
            used.add(ma)
            break

    # token-level PII labels over matched words
    def labels(doc, side_b):
        out = {}
        for m in doc.mentions:
            for seg in m.segments:
                for w in seg.words:
                    wid = word_map.get(w) if side_b else w
                    if wid is not None:
                        out.setdefault(wid, m.type.value)
        return out

    def classes(doc, side_b):
        out = {}
        for c in doc.classified:
            for seg in c.segments:
                for w in seg.words:
                    wid = word_map.get(w) if side_b else w
                    if wid is not None:
                        out.setdefault(wid, c.cls.value)
        return out

    matched_a = set(word_map.values())
    la, lb = labels(a, False), labels(b, True)
    ca, cb = classes(a, False), classes(b, True)
    token_pairs = [(la.get(w, "O"), lb.get(w, "O")) for w in sorted(matched_a)]
    class_pairs = [(ca.get(w, "O"), cb.get(w, "O")) for w in sorted(matched_a)]

    # entity links over tolerant-matched mentions (singletons for missing entities)
    ent_a = {m.id: m.entity for m in a.mentions}
    ent_b = {m.id: m.entity for m in b.mentions}
    key_assign = {ma: ent_a[ma] or f"_a_{ma}" for mb, ma in tolerant_pairs.items()}
    resp_assign = {ma: ent_b[mb] or f"_b_{mb}" for mb, ma in tolerant_pairs.items()}
    link = b_cubed(key_assign, resp_assign)

    only_a = sorted(set(ka) - set(tolerant_pairs.values()))
    only_b = sorted(set(kb) - set(tolerant_pairs))
    type_diff = sorted(f"{ma}:{type_a[ma]}/{type_b[mb]}" for mb, ma in tolerant_pairs.items()
                       if type_a[ma] != type_b[mb])
    tolerant = _f1(len(tolerant_pairs), len(ka), len(kb))
    res = AgreementResult(
        words_a=len(words_a), words_b=len(words_b), words_matched=len(word_map),
        word_iou_median=round(float(np.median(ious)), 4) if ious else None,
        word_text_agreement=round(sum(same_text) / len(same_text), 4) if same_text else None,
        regions_matched={"a": sum(len(p.regions) for p in a.pages), "b": sum(len(p.regions) for p in b.pages),
                         "matched": len(region_map)},
        mentions_a=len(ka), mentions_b=len(kb),
        strict_f1=_f1(len(strict_pairs), len(ka), len(kb)),
        tolerant_f1=tolerant,
        token_kappa=_kappa(token_pairs),
        classified_kappa=_kappa(class_pairs) if (ca or cb) else None,
        entity_link_b3=link,
        disagreements={"only_in_a": only_a, "only_in_b": only_b, "type_differs": type_diff,
                       "unmatched_words_b": len(words_b) - len(word_map)},
    )
    res.gates = {f"tolerant_mention_f1>={GATE_MENTION_F1}": tolerant is not None and tolerant >= GATE_MENTION_F1,
                 f"entity_link_b3_f1>={GATE_LINK_F1}": (link["f1"] or 0) >= GATE_LINK_F1}
    return res
