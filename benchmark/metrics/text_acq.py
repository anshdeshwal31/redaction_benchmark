"""Text-acquisition (OCR / extraction) metrics, general and PII-weighted (PLAN.md §3.3).

Candidate tokens are assigned to the GT word they overlap most (>= half of the token's
area), so split tokens are concatenated and merged tokens count against the word they
mostly belong to. Recognition is scored on matched words; localisation, reading order
and orientation separately; and every metric again restricted to required PII.
"""

from __future__ import annotations

import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.stats import kendalltau
from shapely.strtree import STRtree

from ..geometry import Frame, as_poly, to_shapely
from ..gt.policy import Effective, Resolution
from ..schemas import GTDocument, Requirement, TextArtifact
from .align import InkIndex

_PUNCT = str.maketrans("", "", string.punctuation + "‘’“”–—")


def _norm(s: str, mode: str) -> str:
    if mode == "casefold":
        return s.casefold()
    if mode == "alnum":
        return s.translate(_PUNCT).casefold()
    return s


@dataclass
class WordRecognition:
    word_id: str
    page: int
    gt: str
    recognised: str | None
    distance: int
    iou: float | None
    covered: float  # share of the GT word polygon covered by its token boxes
    requirement: str
    source: str


@dataclass
class TextAcqResult:
    words: list[WordRecognition]
    general: dict
    pii: dict
    pii_by_type: dict
    by_source: dict
    reading_order: dict
    orientation: dict
    mention_recognisability: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _rates(items: list[WordRecognition]) -> dict:
    if not items:
        return {"words": 0}
    chars = sum(len(w.gt) for w in items) or 1
    out = {"words": len(items),
           "missed_rate": round(sum(1 for w in items if w.recognised is None) / len(items), 4),
           "box_iou_median": None, "covered_95_rate": None}
    for mode in ("raw", "casefold", "alnum"):
        dist = sum(Levenshtein.distance(_norm(w.gt, mode), _norm(w.recognised or "", mode)) for w in items)
        wrong = sum(1 for w in items if _norm(w.gt, mode) != _norm(w.recognised or "", mode))
        out[f"cer_{mode}"] = round(dist / chars, 4)
        out[f"wer_{mode}"] = round(wrong / len(items), 4)
    ious = [w.iou for w in items if w.iou is not None]
    if ious:
        out["box_iou_median"] = round(float(np.median(ious)), 4)
    out["covered_95_rate"] = round(sum(1 for w in items if w.covered >= 0.95) / len(items), 4)
    return out


def _gt_orientation(angles: list[float]) -> int | None:
    if not angles:
        return None
    a = Counter(int(round(x / 90.0)) * 90 % 360 for x in angles).most_common(1)[0][0]
    return a


def evaluate_text(art: TextArtifact, gt: GTDocument, res: Resolution, index: InkIndex) -> TextAcqResult:
    must_words = {wid for wid, eff in res.words.items() if eff is Effective.MUST}
    word_type: dict[str, str] = {}
    for m in gt.mentions:
        if res.mentions[m.id].requirement is Requirement.MUST:
            for seg in m.segments:
                for wid in seg.words:
                    word_type.setdefault(wid, m.type.value)
    pages = {p.index: p for p in art.pages}
    records: list[WordRecognition] = []
    hallucinated = total_tokens = 0
    taus, splices, orient_hits, orient_total = [], 0, 0, 0
    for page in gt.pages:
        cand = pages.get(page.index)
        tokens = cand.tokens if cand else []
        total_tokens += len(tokens)
        tpolys = [to_shapely(index.to_display(page.index, t.poly, art.frame, art.dpi)) for t in tokens]
        gpolys = [to_shapely(as_poly(w.poly)) for w in page.words]
        tree = STRtree(gpolys) if gpolys else None
        assigned: dict[int, list[int]] = defaultdict(list)
        for ti, tp in enumerate(tpolys):
            if tree is None or tp.area <= 0:
                hallucinated += 1
                continue
            best, best_frac = None, 0.0
            for gi in tree.query(tp):
                frac = tp.intersection(gpolys[gi]).area / tp.area
                if frac > best_frac:
                    best, best_frac = int(gi), frac
            if best is not None and best_frac >= 0.5:
                assigned[best].append(ti)
            elif tokens[ti].text.strip():
                hallucinated += 1
        order_pairs = []
        for gi, w in enumerate(page.words):
            tis = assigned.get(gi, [])
            angle = np.deg2rad(w.angle)
            u = np.array([np.cos(angle), np.sin(angle)])
            tis = sorted(tis, key=lambda ti: float(np.array(tpolys[ti].centroid.coords[0]) @ u))
            rec = "".join(tokens[ti].text for ti in tis) if tis else None
            iou = None
            if len(tis) == 1:
                inter = tpolys[tis[0]].intersection(gpolys[gi]).area
                uni = tpolys[tis[0]].union(gpolys[gi]).area
                iou = round(inter / uni, 4) if uni else None
            covered = 0.0
            if tis:
                from shapely import union_all

                covered = union_all([tpolys[ti] for ti in tis]).intersection(gpolys[gi]).area / max(gpolys[gi].area, 1e-9)
            records.append(WordRecognition(
                word_id=w.id, page=page.index, gt=w.text, recognised=rec,
                distance=Levenshtein.distance(w.text, rec or ""), iou=iou, covered=round(min(covered, 1.0), 4),
                requirement=res.words.get(w.id, Effective.ORDINARY).value, source=w.source.value))
            if tis:
                first = min(tokens[ti].order if tokens[ti].order is not None else ti for ti in tis)
                order_pairs.append((w.order, first, w.line))
        if len(order_pairs) >= 3:
            order_pairs.sort(key=lambda t: t[1])
            tau = kendalltau([p[0] for p in order_pairs], [p[1] for p in order_pairs]).statistic
            taus.append(float(tau))
            for a, b in zip(order_pairs, order_pairs[1:]):
                if a[2] != b[2] and abs(b[0] - a[0]) > 1:
                    splices += 1
        expected = _gt_orientation([w.angle for w in page.words])
        if cand is not None and cand.detected_rotation is not None and expected is not None:
            orient_total += 1
            orient_hits += int(cand.detected_rotation == expected)

    pii = [r for r in records if r.word_id in must_words]
    by_type: dict[str, list[WordRecognition]] = defaultdict(list)
    for r in pii:
        by_type[word_type.get(r.word_id, "?")].append(r)
    by_source: dict[str, list[WordRecognition]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)

    # mention recognisability: every required word recognised (ratio >= 80) and located
    rec_of = {r.word_id: r for r in records}
    recog = []
    for m in gt.mentions:
        if res.mentions[m.id].requirement is not Requirement.MUST:
            continue
        ws = [rec_of[wid] for seg in m.segments for wid in seg.words if wid in rec_of]
        if not ws:
            continue
        ok = all(w.recognised is not None and w.covered >= 0.5
                 and fuzz.ratio(_norm(w.gt, "alnum"), _norm(w.recognised, "alnum")) >= 80 for w in ws)
        recog.append((m.id, m.type.value, ok))
    general = _rates(records)
    general["hallucinated_rate"] = round(hallucinated / total_tokens, 4) if total_tokens else None
    return TextAcqResult(
        words=records,
        general=general,
        pii=_rates(pii),
        pii_by_type={t: _rates(v) for t, v in sorted(by_type.items())},
        by_source={s: _rates(v) for s, v in sorted(by_source.items())},
        reading_order={"kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
                       "line_splices": splices},
        orientation={"pages": orient_total, "correct": orient_hits},
        mention_recognisability={"rate": round(sum(1 for *_, ok in recog if ok) / len(recog), 4) if recog else None,
                                 "unrecognisable": [mid for mid, _, ok in recog if not ok]},
    )
