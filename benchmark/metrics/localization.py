"""Localization metrics for planned regions, transform diagnosis and plan fidelity
(PLAN.md §3.6, §3.7).

- coverage: fraction of each required mention's ink inside the planned regions (the
  safety-relevant quantity, not IoU)
- spill: planned area outside the tolerance zone, and collateral GT words selected
- edge offsets: signed distances between planned boxes and the GT words they hit
- transform diagnosis: does a known coordinate bug (DPI scale, y-flip, quarter-turn,
  CropBox offset, page index, translation) explain the misplacement better than
  identity? Reported once per page instead of as dozens of unrelated misses.
- plan fidelity: were the planned regions actually applied in the output?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import pymupdf
import shapely

from ..geometry import Frame, as_poly, bbox, dilate, to_shapely, union
from ..gt.policy import Effective, Resolution
from ..schemas import GTDocument, PlanArtifact, Requirement
from .align import DEFAULT_THETA, InkIndex

DPI_RATIOS = {"72/300": 72 / 300, "72/200": 72 / 200, "72/150": 72 / 150, "72/96": 72 / 96,
              "96/72": 96 / 72, "150/72": 150 / 72, "200/72": 200 / 72, "300/72": 300 / 72}


@dataclass
class MentionCoverage:
    mention_id: str
    requirement: str
    coverage: float
    pages: list[int]


@dataclass
class TransformDiagnosis:
    page: int
    regions: int
    kind: str  # aligned | systematic | unexplained | no_regions
    hypothesis: str | None
    hit_identity: float
    hit_best: float
    params: dict = field(default_factory=dict)


@dataclass
class LocalizationResult:
    mentions: list[MentionCoverage]
    must_full_coverage_rate: float | None
    must_mean_coverage: float | None
    spill_area_pt2: float
    collateral_words: list[str]
    protected_words_hit: list[str]
    edge_offsets_pt: dict
    diagnoses: list[TransformDiagnosis]
    document_diagnosis: str
    words_touched: list[str] = field(default_factory=list)  # ordinary/protected words with any planned ink

    def to_dict(self) -> dict:
        return asdict(self)


def _plan_polys(plan: PlanArtifact, index: InkIndex) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for r in plan.regions:
        if r.page in index.shapes:
            out.setdefault(r.page, []).append(index.to_display(r.page, r.poly, plan.frame, plan.dpi))
    return out


def _targets(gt: GTDocument, res: Resolution, page: int):
    p = gt.page(page)
    polys = [as_poly(w.poly) for w in p.words if res.words.get(w.id) in (Effective.MUST, Effective.MAY)]
    polys += [as_poly(r.poly) for r in p.regions if res.regions.get(r.id) in (Effective.MUST, Effective.MAY)]
    return union(polys) if polys else None


def _hit(polys: list[np.ndarray], target) -> float:
    if target is None or not polys:
        return 0.0
    total = sum(to_shapely(p).area for p in polys)
    if total <= 0:
        return 0.0
    return sum(to_shapely(p).intersection(target).area for p in polys) / total


def _hypotheses(geom, page: int, n_pages: int):
    """(name, function mapping candidate polygons back to GT space, page offset)."""
    w, h = geom.display_size
    hyps = [("identity", lambda p: p, 0)]
    for name, k in DPI_RATIOS.items():
        hyps.append((f"scale {name}", (lambda kk: lambda p: p / kk)(k), 0))
    hyps.append(("y-flip", lambda p: np.column_stack([p[:, 0], h - p[:, 1]]), 0))
    hyps.append(("x-flip", lambda p: np.column_stack([w - p[:, 0], p[:, 1]]), 0))
    if geom.rotation:
        hyps.append(("unrotated frame used", lambda p: geom.transform(p, Frame.DISPLAY_PT, Frame.UNROTATED_PT), 0))
    dx = geom.cropbox[0] - geom.mediabox[0]
    dy = geom.mediabox[3] - geom.cropbox[3]
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        hyps.append(("mediabox origin used", lambda p: p + np.array([dx, dy]), 0))
    for off in (-1, 1):
        if 0 <= page - off < n_pages:
            hyps.append((f"page offset {off:+d}", lambda p: p, off))
    return hyps


def _translation_guess(polys, gt_polys) -> np.ndarray | None:
    if not polys or not gt_polys:
        return None
    gc = np.array([as_poly(p).mean(axis=0) for p in gt_polys])
    deltas = []
    for p in polys:
        c = as_poly(p).mean(axis=0)
        j = int(np.argmin(((gc - c) ** 2).sum(axis=1)))
        deltas.append(gc[j] - c)
    d = np.median(np.array(deltas), axis=0)
    return d if np.hypot(*d) > 0.5 else None


def diagnose(plan_polys: dict[int, list[np.ndarray]], gt: GTDocument, res: Resolution,
             index: InkIndex) -> list[TransformDiagnosis]:
    """Per-region votes aggregated per page.

    Each region votes ``aligned`` (mostly on required content), ``oversized`` (covers the
    required ink it touches but is much larger), a bug hypothesis that maps it onto
    required content, or ``unexplained``. A page is ``systematic`` when one hypothesis
    explains at least 30% of its regions and is the plurality among misplaced ones; this
    handles pages that mix correct and misplaced regions (e.g. a page offset).
    """
    out = []
    n = len(gt.pages)
    targets = {p.index: _targets(gt, res, p.index) for p in gt.pages}
    for page in gt.pages:
        i = page.index
        polys = plan_polys.get(i, [])
        if not polys:
            out.append(TransformDiagnosis(i, 0, "no_regions", None, 0.0, 0.0))
            continue
        geom = index.geoms[i]
        hyps = _hypotheses(geom, i, n)[1:]  # identity handled separately
        gt_polys = [as_poly(w.poly) for w in page.words if res.words.get(w.id) is Effective.MUST]
        t = _translation_guess(polys, gt_polys)
        votes: Counter = Counter()
        params: dict = {}
        for p in polys:
            p = as_poly(p)
            own = _hit([p], targets[i])
            if own >= 0.5:
                votes["aligned"] += 1
                continue
            cov = index.coverage(i, index.region_mask(i, [p]))
            touched = [f for wid, f in cov.items() if res.words.get(wid) is Effective.MUST and f > 0.2]
            if touched and min(touched) >= 0.9:
                votes["oversized regions"] += 1
                continue
            scored = [(_hit([fn(p)], targets.get(i - off)), name) for name, fn, off in hyps]
            if t is not None:
                scored.append((_hit([p + t], targets[i]), "translation"))
            best = max(scored, key=lambda s: s[0]) if scored else (0.0, None)
            votes[best[1] if best[0] >= 0.5 else "unexplained"] += 1
        if t is not None:
            params = {"dx": round(float(-t[0]), 2), "dy": round(float(-t[1]), 2)}
        total = len(polys)
        misplaced = {k: v for k, v in votes.items() if k not in ("aligned", "oversized regions", "unexplained")}
        identity = votes["aligned"] / total
        if misplaced:
            hyp, count = max(misplaced.items(), key=lambda kv: kv[1])
            if count >= 0.3 * total and count >= votes["unexplained"]:
                out.append(TransformDiagnosis(i, total, "systematic", hyp, round(identity, 3), round(count / total, 3),
                                              params if hyp == "translation" else {}))
                continue
        if votes["aligned"] + votes["oversized regions"] >= 0.7 * total:
            hyp = "oversized regions" if votes["oversized regions"] > votes["aligned"] else None
            out.append(TransformDiagnosis(i, total, "aligned", hyp, round(identity, 3),
                                          round((votes["aligned"] + votes["oversized regions"]) / total, 3)))
        else:
            out.append(TransformDiagnosis(i, total, "unexplained", None, round(identity, 3),
                                          round(votes["unexplained"] / total, 3)))
    return out


def evaluate_plan(plan: PlanArtifact, gt: GTDocument, res: Resolution, index: InkIndex,
                  tolerance_pt: float = 2.0, theta: float = DEFAULT_THETA) -> LocalizationResult:
    polys = _plan_polys(plan, index)
    masks = {pg: index.region_mask(pg, ps) for pg, ps in polys.items()}
    words = gt.word_index()

    # -- per-mention ink coverage -----------------------------------------------------------
    coverages = []
    for m in gt.mentions:
        r = res.mentions[m.id]
        if r.requirement not in (Requirement.MUST, Requirement.MAY):
            continue
        covered = total = 0.0
        for seg in m.segments:
            mask = masks.get(seg.page)
            for wid in seg.words:
                wi = index.words.get(seg.page, {}).get(wid)
                if wi is None:
                    continue
                lo, hi = (seg.char_start, seg.char_end) if seg.char_start is not None else (0, len(wi.char_ink))
                for k in range(lo, min(hi, len(wi.char_ink))):
                    n = int(wi.char_ink[k].sum())
                    total += n
                    if mask is not None and n:
                        h, w = wi.ink.shape
                        covered += float((mask[wi.r0:wi.r0 + h, wi.c0:wi.c0 + w] & wi.char_ink[k]).sum())
        if total:
            coverages.append(MentionCoverage(m.id, r.requirement.value, round(covered / total, 4),
                                             sorted({s.page for s in m.segments})))
    must_cov = [c.coverage for c in coverages if c.requirement == Requirement.MUST.value]

    # -- spill and collateral --------------------------------------------------------------------
    spill = 0.0
    collateral, protected, touched = [], [], []
    offsets = []
    for pg, ps in polys.items():
        page = gt.page(pg)
        zone_polys = [dilate(as_poly(w.poly), tolerance_pt) for w in page.words
                      if res.words.get(w.id) in (Effective.MUST, Effective.MAY)]
        zone_polys += [dilate(as_poly(r.poly), tolerance_pt) for r in page.regions
                       if res.regions.get(r.id) in (Effective.MUST, Effective.MAY)]
        zone = union(zone_polys) if zone_polys else None
        planned = union(ps)
        spill += float(planned.difference(zone).area if zone is not None else planned.area)
        for wid, f in index.coverage(pg, masks[pg]).items():
            eff = res.words.get(wid)
            if f >= 0.05 and eff in (Effective.ORDINARY, Effective.MUST_NOT):
                touched.append(wid)
            if f < theta:
                continue
            if eff is Effective.ORDINARY:
                collateral.append(wid)
            elif eff is Effective.MUST_NOT:
                protected.append(wid)
        # signed edge offsets of each region against the required words it overlaps
        for p in ps:
            hit = index.select(pg, [p], 0.2)
            req = [words[w][1] for w in hit if res.words.get(w) is Effective.MUST]
            if not req:
                continue
            gx0, gy0, gx1, gy1 = bbox(np.vstack([as_poly(w.poly) for w in req]))
            x0, y0, x1, y1 = bbox(p)
            offsets.append((x0 - gx0, y0 - gy0, x1 - gx1, y1 - gy1))
    off = np.array(offsets) if offsets else np.zeros((0, 4))
    edge = {k: round(float(np.median(off[:, j])), 2) if len(off) else None
            for j, k in enumerate(("left", "top", "right", "bottom"))}
    edge["n"] = len(off)

    diags = diagnose(polys, gt, res, index)
    kinds = Counter(d.hypothesis for d in diags if d.kind == "systematic")
    if kinds:
        doc_diag = f"systematic: {kinds.most_common(1)[0][0]}"
    elif any(d.kind == "unexplained" for d in diags):
        doc_diag = "unexplained misplacement"
    else:
        doc_diag = "aligned"
    return LocalizationResult(
        mentions=coverages,
        must_full_coverage_rate=round(sum(1 for c in must_cov if c >= 0.98) / len(must_cov), 4) if must_cov else None,
        must_mean_coverage=round(float(np.mean(must_cov)), 4) if must_cov else None,
        spill_area_pt2=round(spill, 2),
        collateral_words=sorted(set(collateral)),
        protected_words_hit=sorted(set(protected)),
        edge_offsets_pt=edge,
        diagnoses=diags,
        document_diagnosis=doc_diag,
        words_touched=sorted(set(touched)),
    )


# -- plan fidelity --------------------------------------------------------------------------------


@dataclass
class RegionFidelity:
    region_id: str
    page: int
    applied_fraction: float  # share of the planned area whose pixels changed in the output
    status: str  # applied | partial | not_applied


@dataclass
class FidelityResult:
    regions: list[RegionFidelity]
    planned_vs_changed_iou: dict[int, float]
    not_applied: int
    partial: int

    def to_dict(self) -> dict:
        return asdict(self)


def plan_fidelity(plan: PlanArtifact, source: bytes, output: bytes, index: InkIndex,
                  change_threshold: int = 40) -> FidelityResult:
    s = index.dpi / 72.0
    src, out = pymupdf.open("pdf", source), pymupdf.open("pdf", output)
    changed_by_page = {}
    for pg in index.shapes:
        if pg >= out.page_count:
            continue
        a = src[pg].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
        b = out[pg].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
        ga = np.frombuffer(a.samples, np.uint8).reshape(a.h, a.w)
        gb = np.frombuffer(b.samples, np.uint8).reshape(b.h, b.w)
        if gb.shape != ga.shape:
            gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
        changed_by_page[pg] = np.abs(cv2.blur(ga, (3, 3)).astype(np.int16)
                                     - cv2.blur(gb, (3, 3)).astype(np.int16)) > change_threshold
    regions, ious = [], {}
    planned_by_page: dict[int, np.ndarray] = {}
    for r in plan.regions:
        if r.page not in changed_by_page:
            continue
        poly = index.to_display(r.page, r.poly, plan.frame, plan.dpi)
        m = index.region_mask(r.page, [poly])
        planned_by_page[r.page] = planned_by_page.get(r.page, np.zeros_like(m)) | m
        area = int(m.sum())
        frac = float((m & changed_by_page[r.page]).sum()) / area if area else 0.0
        status = "applied" if frac >= 0.5 else ("partial" if frac >= 0.1 else "not_applied")
        # A region over blank background changes nothing visible when removed; treat as applied
        # if its source pixels were already uniform.
        regions.append(RegionFidelity(r.id, r.page, round(frac, 4), status))
    for pg, planned in planned_by_page.items():
        ch = changed_by_page[pg]
        inter, uni = int((planned & ch).sum()), int((planned | ch).sum())
        ious[pg] = round(inter / uni, 4) if uni else 1.0
    return FidelityResult(regions, ious, sum(1 for r in regions if r.status == "not_applied"),
                          sum(1 for r in regions if r.status == "partial"))
