"""PII detection metrics in GT-word space (PLAN.md §3.4).

Matching schemes, all reported side by side:
- strict          exact required-word set and correct type
- exact_boundary  exact required-word set, type ignored
- tolerant        all required words, extra words only from the mention's optional set
- partial         any overlap

Boundary classes per GT mention: exact, over_benign (optional words only), over_harmful
(adjacent ordinary or protected words), merged (with another GT mention), split (covered
only by several predictions), under (some required words missed; the missing name
components are named), missed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from ..gt.policy import Effective, Resolution
from ..schemas import GTDocument, MentionsArtifact, PiiType, Requirement, TextArtifact
from .align import DEFAULT_THETA, InkIndex


@dataclass
class ProjectedMention:
    id: str
    label: str
    mapped_type: str | None
    words: set[tuple[int, str]]  # (page, GT word id)


@dataclass
class MentionMatch:
    mention_id: str
    type: str
    requirement: str
    entity: str | None
    conditions: list[str]
    detected: bool
    covered_fraction: float
    exact: bool
    tolerant: bool
    strict: bool
    boundary: str
    missing_components: list[str]
    best_pred: str | None
    pred_type: str | None
    type_correct: bool | None


@dataclass
class DetectionResult:
    matches: list[MentionMatch]
    predictions: int
    token: dict
    mention_scores: dict
    fp_classes: dict
    type_confusion: dict
    entities_complete: dict
    boundary_counts: dict
    unprojected_predictions: list[str] = field(default_factory=list)
    predicted_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def project_mentions(art: MentionsArtifact, index: InkIndex, label_map: dict[str, PiiType] | None = None,
                     text: TextArtifact | None = None, theta: float = DEFAULT_THETA) -> list[ProjectedMention]:
    """Turn predicted mentions into sets of GT words. Anchor priority: regions, tokens, span."""
    token_polys: dict[str, tuple[int, list]] = {}
    page_tokens: dict[int, list] = {}
    if text is not None:
        for p in text.pages:
            page_tokens[p.index] = p.tokens
            for t in p.tokens:
                token_polys[t.id] = (p.index, index.to_display(p.index, t.poly, text.frame, text.dpi))
    out = []
    for m in art.mentions:
        by_page: dict[int, list] = {}
        if m.regions:
            for r in m.regions:
                by_page.setdefault(r.page, []).append(index.to_display(r.page, r.poly, art.frame, art.dpi))
        elif m.tokens:
            for tid in m.tokens:
                if tid in token_polys:
                    pg, poly = token_polys[tid]
                    by_page.setdefault(pg, []).append(poly)
        elif m.span and art.page_texts and m.span.page in page_tokens:
            # Convention: page_texts[page] == " ".join(token texts in order) (docs/adapter_guide.md).
            pos = 0
            for t in page_tokens[m.span.page]:
                start, end = pos, pos + len(t.text)
                if start < m.span.end and end > m.span.start:
                    by_page.setdefault(m.span.page, []).append(token_polys[t.id][1])
                pos = end + 1
        words = {(pg, w) for pg, polys in by_page.items() for w in index.select(pg, polys, theta)}
        mapped = None
        if label_map is not None:
            mapped = label_map[m.label].value if m.label in label_map else None
        elif m.label in PiiType.__members__:
            mapped = m.label
        out.append(ProjectedMention(m.id, m.label, mapped, words))
    return out


def evaluate_detection(gt: GTDocument, res: Resolution, preds: list[ProjectedMention]) -> DetectionResult:
    words = gt.word_index()
    classified_cls = {wid: c.cls.value for c in gt.classified for seg in c.segments for wid in seg.words}
    req_sets: dict[str, set] = {}
    for m in gt.mentions:
        req_sets[m.id] = {(seg.page, wid) for seg in m.segments for wid in seg.words}
    word_owner: dict[tuple[int, str], list[str]] = {}
    for mid, ws in req_sets.items():
        for w in ws:
            word_owner.setdefault(w, []).append(mid)

    matches: list[MentionMatch] = []
    evaluated = [m for m in gt.mentions if res.mentions[m.id].requirement in (Requirement.MUST, Requirement.MAY)
                 and req_sets[m.id]]
    for m in evaluated:
        R = req_sets[m.id]
        O = {(words[w][0], w) for w in m.optional_words}
        covering = [p for p in preds if p.words & R]
        union = set().union(*(p.words for p in covering)) if covering else set()
        chars_total = sum(len(words[w][1].text) for _, w in R) or 1
        chars_cov = sum(len(words[w][1].text) for pg, w in R if (pg, w) in union)
        missing = []
        if not covering:
            boundary, best = "missed", None
        elif not R <= union:
            boundary = "under"
            best = max(covering, key=lambda p: len(p.words & R))
            missing_words = {w for _, w in R - union}
            missing = sorted({c.value for c, wids in m.components.items() if set(wids) & missing_words})
        else:
            single = [p for p in covering if R <= p.words]
            if not single:
                boundary, best = "split", max(covering, key=lambda p: len(p.words & R))
            else:
                best = min(single, key=lambda p: len(p.words))
                extra = best.words - R
                if not extra:
                    boundary = "exact"
                elif extra <= O:
                    boundary = "over_benign"
                elif any(o != m.id for w in extra for o in word_owner.get(w, [])):
                    boundary = "merged"
                else:
                    boundary = "over_harmful"
        exact = boundary == "exact"
        tolerant = boundary in ("exact", "over_benign")
        type_ok = None if best is None or best.mapped_type is None else best.mapped_type == m.type.value
        matches.append(MentionMatch(
            mention_id=m.id, type=m.type.value, requirement=res.mentions[m.id].requirement.value, entity=m.entity,
            conditions=list(m.conditions), detected=bool(covering), covered_fraction=round(chars_cov / chars_total, 4),
            exact=exact, tolerant=tolerant, strict=exact and bool(type_ok), boundary=boundary,
            missing_components=missing, best_pred=best.id if best else None,
            pred_type=best.mapped_type if best else None, type_correct=type_ok))

    # -- token level (MAY words are neither rewarded nor penalised) ------------------
    predicted = set().union(*(p.words for p in preds)) if preds else set()
    must_words = {(words[w][0], w) for w, eff in res.words.items() if eff is Effective.MUST}
    negative = {(words[w][0], w) for w, eff in res.words.items() if eff in (Effective.ORDINARY, Effective.MUST_NOT)}
    tp, fn, fp = must_words & predicted, must_words - predicted, negative & predicted
    precision = len(tp) / (len(tp) + len(fp)) if tp or fp else None
    recall = len(tp) / (len(tp) + len(fn)) if tp or fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    fp_classes = Counter(classified_cls.get(w, "ORDINARY") for _, w in fp)

    # -- mention level --------------------------------------------------------------------
    must_matches = [x for x in matches if x.requirement == Requirement.MUST.value]
    pii_words = set().union(*req_sets.values()) if req_sets else set()
    exact_sets = {frozenset(req_sets[x.mention_id]) for x in matches}

    def prec(ok):
        return round(sum(1 for p in preds if ok(p)) / len(preds), 4) if preds else None

    def rec(key):
        return round(sum(1 for x in must_matches if getattr(x, key)) / len(must_matches), 4) if must_matches else None

    mention_scores = {
        "partial": {"recall": rec("detected"), "precision": prec(lambda p: bool(p.words & pii_words))},
        "tolerant": {"recall": rec("tolerant")},
        "exact_boundary": {"recall": rec("exact"), "precision": prec(lambda p: frozenset(p.words) in exact_sets)},
        "strict": {"recall": rec("strict")},
    }

    # -- entities and types --------------------------------------------------------------
    by_entity: dict[str, list[bool]] = {}
    for x in must_matches:
        if x.entity:
            by_entity.setdefault(x.entity, []).append(x.tolerant or x.boundary in ("merged", "over_harmful", "split"))
    complete = {e: all(v) for e, v in by_entity.items()}
    confusion = Counter(f"{x.type}->{x.pred_type}" for x in matches if x.detected and x.pred_type is not None)
    return DetectionResult(
        matches=matches,
        predictions=len(preds),
        token={"tp": len(tp), "fn": len(fn), "fp": len(fp),
               "precision": round(precision, 4) if precision is not None else None,
               "recall": round(recall, 4) if recall is not None else None,
               "f1": round(f1, 4) if f1 is not None else None},
        mention_scores=mention_scores,
        fp_classes=dict(fp_classes),
        type_confusion=dict(confusion),
        entities_complete={"complete": sum(complete.values()), "total": len(complete),
                           "incomplete": sorted(e for e, ok in complete.items() if not ok)},
        boundary_counts=dict(Counter(x.boundary for x in must_matches)),
        unprojected_predictions=[p.id for p in preds if not p.words],
        predicted_words=sorted(w for _, w in predicted),
    )
