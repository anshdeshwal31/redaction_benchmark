"""Moved words and displaced originals (eval-0.4; docs/metrics.md, "Moved words and displaced originals").

A tool that redraws a line holding a fake (a longer or shorter fake moves the words after it, and the
line may be re-typeset a little larger or smaller) leaves the kept words of that line intact but not at
their GT place: the in-place test calls them DAMAGED or REMOVED. A word the in-place test does not find
is looked for near its place in the output, by its own original image:

- template: the original render cropped to the GT box, grown moved_margin_along x h along the line and
  moved_margin_across x h across it (h the word's height);
- window: moved_across x h across and moved_along x h along, clipped along the line to the nearest
  anchors on its GT line (words found in place whose whole-word image is there, TM_CCOEFF_NORMED within
  2 px >= moved_anchor) widened by moved_gap x h, so reading order is kept;
- scales moved_scale_min..moved_scale_max by moved_scale_step; up to moved_candidates positions per scale
  scoring at least moved_floor after non-maximum suppression, then suppression across scales;
- a candidate is valid when its score reaches moved_match and the band test at the found transform
  finds at least word_preserved_fraction of the word's characters (:meth:`PageCells.word_fraction_at`);
- one-to-one assignment per page, greedy over score - moved_lambda x |displacement| / h, no box
  overlapping a taken one by more than moved_overlap; identical texts of one line keep their order.

Two filters follow the assignment. A match at scale 1 within moved_zero_px of the GT place must pass the
character identity gate there (an altered word in place, 'far' for 'for', is not a moved 'for'). A kept
word must be found as a whole word: no ink in the strips beside the found box (a kept 'on' is not found
inside a fake 'Wilson'). MUST words are not filtered by the second rule: an original inside a longer word
is still a leak.

A MUST or MAY word found is a displaced original: the evaluator counts its characters visible. Words are
looked for on the primary render pair (the first render configuration) of each evaluated page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping

import cv2
import numpy as np

from ..geometry import as_poly, bbox
from ..schemas import GTDocument, GTPage
from .cells import PageCells, _paper
from .config import EvalConfig

# Positions of the score map looked at per scale before suppression (the prototype's constant).
TOP_POSITIONS = 1500
# Template matching within this many pixels of the GT place for anchors and the original's own image.
JITTER_PX = 2
# The whole-word strips cover the middle of the found box across the line: this share is left out at
# each side.
SIDE_ACROSS = 0.15


@dataclass
class Found:
    """A word found away from its GT place. The transform maps original pixels to pixels of the
    registered output render: output_px = s * original_px + (tx, ty)."""

    page: int
    s: float
    tx: float
    ty: float
    box: tuple[float, float, float, float]  # the GT box at the found transform (display pt)
    dx_pt: float  # displacement of the box centre (display pt)
    dy_pt: float
    score: float  # template score
    v: float  # share of the word's characters the band test finds at the transform
    requirement: str
    restyled: bool = False  # found re-typeset in a substitute font (find_restyled)
    font: str | None = None

    def poly(self, poly, scale: float) -> np.ndarray:
        """The word's polygon (display pt) at the found transform."""
        return self.s * np.asarray(poly, float).reshape(-1, 2) + np.array([self.tx, self.ty]) / scale


# -- geometry of a GT word ------------------------------------------------------------------------------


def word_frame(w) -> tuple[float, float, float]:
    """(height across the line, length along it, angle) of a GT word, from its polygon (pt)."""
    t = math.radians(w.angle)
    v = np.array([-math.sin(t), math.cos(t)])
    u = np.array([math.cos(t), math.sin(t)])
    p = as_poly(w.poly)
    return float((p @ v).max() - (p @ v).min()), float((p @ u).max() - (p @ u).min()), w.angle


def _axis(angle: float) -> tuple[np.ndarray, np.ndarray]:
    t = math.radians(angle)
    return np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])


def _interval(poly: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    p = poly @ u
    return float(p.min()), float(p.max())


def overlap(a, b) -> float:
    """Area of the intersection of boxes a and b (x0, y0, x1, y1) over the area of a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def _vertical(angle: float) -> bool:
    """Does the word read up or down the page?"""
    return abs(math.sin(math.radians(angle))) > 0.7


def scales(cfg: EvalConfig) -> list[float]:
    n = int(round((cfg.moved_scale_max - cfg.moved_scale_min) / cfg.moved_scale_step))
    return [round(cfg.moved_scale_min + cfg.moved_scale_step * k, 4) for k in range(n + 1)]


# -- template search -------------------------------------------------------------------------------------


def template(o: np.ndarray, w, cfg: EvalConfig) -> dict | None:
    """The word's original image: its GT box grown moved_margin_along x h along the line and
    moved_margin_across x h across, with the GT box in pixels."""
    scale = cfg.dpi / 72.0
    hpt, _, ang = word_frame(w)
    x0, y0, x1, y1 = (v * scale for v in bbox(as_poly(w.poly)))
    vertical = _vertical(ang)
    mh, mv = cfg.moved_margin_along * hpt * scale, cfg.moved_margin_across * hpt * scale
    if vertical:
        mh, mv = mv, mh
    hh, ww = o.shape
    c0, c1 = max(int(math.floor(x0 - mh)), 0), min(int(math.ceil(x1 + mh)), ww)
    r0, r1 = max(int(math.floor(y0 - mv)), 0), min(int(math.ceil(y1 + mv)), hh)
    if c1 - c0 < 3 or r1 - r0 < 3:
        return None
    t = o[r0:r1, c0:c1]
    if t.std() < 1.0:
        return None
    return dict(t=t, c0=c0, r0=r0, c1=c1, r1=r1, box=(x0, y0, x1, y1), hpt=hpt, ang=ang, vertical=vertical)


def inplace_score(o: np.ndarray, x: np.ndarray, w, cfg: EvalConfig, tpl: dict | None = None) -> float | None:
    """Best template score of the word's original image within JITTER_PX of its GT place, at scale 1."""
    tpl = tpl or template(o, w, cfg)
    if tpl is None:
        return None
    hh, ww = x.shape
    a0, b0 = max(tpl["c0"] - JITTER_PX, 0), max(tpl["r0"] - JITTER_PX, 0)
    a1, b1 = min(tpl["c1"] + JITTER_PX, ww), min(tpl["r1"] + JITTER_PX, hh)
    win = x[b0:b1, a0:a1]
    if win.shape[0] < tpl["t"].shape[0] or win.shape[1] < tpl["t"].shape[1]:
        return None
    res = cv2.matchTemplate(win.astype(np.float32), tpl["t"].astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return float(np.nan_to_num(res, nan=-1.0).max())


def candidates(o: np.ndarray, x: np.ndarray, w, cfg: EvalConfig, lo_u: float | None, hi_u: float | None,
               claimed: list, tpl: dict | None = None, n_keep: int | None = None) -> list[dict]:
    """Distinct positions of the template in the output near the word's place, best first: dicts with
    score, s, tx, ty (px), box (pt), dx_pt and dy_pt. lo_u/hi_u bound the found box along the reading
    direction (pt); a position whose box overlaps a claimed box (pt) by more than moved_overlap is left
    out."""
    tpl = tpl or template(o, w, cfg)
    if tpl is None:
        return []
    n_keep = n_keep or cfg.moved_candidates
    scale = cfg.dpi / 72.0
    t0, c0, r0, c1, r1 = tpl["t"], tpl["c0"], tpl["r0"], tpl["c1"], tpl["r1"]
    gx0, gy0, gx1, gy1 = tpl["box"]
    hpt, vertical = tpl["hpt"], tpl["vertical"]
    u, _ = _axis(tpl["ang"])
    hh, ww = x.shape
    across, along = cfg.moved_across * hpt * scale, cfg.moved_along * hpt * scale
    ex, ey = (across, along) if vertical else (along, across)
    found: list[dict] = []
    for s in scales(cfg):
        th, tw = max(3, int(round((r1 - r0) * s))), max(3, int(round((c1 - c0) * s)))
        t = cv2.resize(t0, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if t.std() < 1.0:
            continue
        cx, cy = (c0 + c1) / 2, (r0 + r1) / 2
        wx0, wx1 = max(int(math.floor(cx - tw / 2 - ex)), 0), min(int(math.ceil(cx + tw / 2 + ex)), ww)
        wy0, wy1 = max(int(math.floor(cy - th / 2 - ey)), 0), min(int(math.ceil(cy + th / 2 + ey)), hh)
        # the window is clipped to the gap between the anchors (axis-aligned reading directions)
        if not vertical and u[0] > 0:
            if lo_u is not None:
                wx0 = max(wx0, int(math.floor(lo_u * scale - (gx0 - c0) * s)) - 2)
            if hi_u is not None:
                wx1 = min(wx1, int(math.ceil(hi_u * scale + (c1 - gx1) * s)) + 2)
        elif vertical and u[1] > 0:
            if lo_u is not None:
                wy0 = max(wy0, int(math.floor(lo_u * scale - (gy0 - r0) * s)) - 2)
            if hi_u is not None:
                wy1 = min(wy1, int(math.ceil(hi_u * scale + (r1 - gy1) * s)) + 2)
        elif vertical and u[1] < 0:
            if lo_u is not None:
                wy1 = min(wy1, int(math.ceil(-lo_u * scale + (r1 - gy1) * s)) + 2)
            if hi_u is not None:
                wy0 = max(wy0, int(math.floor(-hi_u * scale - (gy0 - r0) * s)) - 2)
        if wx1 - wx0 < tw or wy1 - wy0 < th:
            continue
        res = cv2.matchTemplate(x[wy0:wy1, wx0:wx1].astype(np.float32), t.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        wpt = (wx0 / scale, wy0 / scale, wx1 / scale, wy1 / scale)
        near = [cb for cb in claimed if cb[0] < wpt[2] and cb[2] > wpt[0] and cb[1] < wpt[3] and cb[3] > wpt[1]]
        flat = res.ravel()
        kk = min(flat.size, TOP_POSITIONS)
        top = np.argpartition(-flat, kk - 1)[:kk]
        top = top[np.argsort(-flat[top], kind="stable")]
        got = 0
        for k in top:
            sc = float(flat[k])
            if sc < cfg.moved_floor or got >= n_keep:
                break
            ly, lx = divmod(int(k), res.shape[1])
            tx = wx0 + lx - s * c0 + 0.5 * s - 0.5
            ty = wy0 + ly - s * r0 + 0.5 * s - 0.5
            b = ((s * gx0 + tx) / scale, (s * gy0 + ty) / scale, (s * gx1 + tx) / scale, (s * gy1 + ty) / scale)
            if vertical:
                a0, a1 = (b[1], b[3]) if u[1] > 0 else (-b[3], -b[1])
            else:
                a0, a1 = (b[0], b[2]) if u[0] > 0 else (-b[2], -b[0])
            if (lo_u is not None and a0 < lo_u) or (hi_u is not None and a1 > hi_u):
                continue
            if any(overlap(b, cb) > cfg.moved_overlap for cb in near):
                continue
            if any(overlap(b, f["box"]) > cfg.moved_nms for f in found if f["s"] == s):
                continue
            found.append(dict(score=sc, s=float(s), tx=float(tx), ty=float(ty), box=b,
                              dx_pt=((s * (gx0 + gx1) / 2 + tx) - (gx0 + gx1) / 2) / scale,
                              dy_pt=((s * (gy0 + gy1) / 2 + ty) - (gy0 + gy1) / 2) / scale))
            got += 1
    # suppression across scales: the best of overlapping positions
    found.sort(key=lambda f: -f["score"])
    keep: list[dict] = []
    for f in found:
        if all(overlap(f["box"], g["box"]) <= cfg.moved_nms and overlap(g["box"], f["box"]) <= cfg.moved_nms
               for g in keep):
            keep.append(f)
        if len(keep) >= n_keep:
            break
    return keep


# -- anchors and windows ------------------------------------------------------------------------------------


def anchors_of(page: GTPage, o: np.ndarray, x: np.ndarray, inplace: Mapping[str, tuple[str, float | None]],
               cfg: EvalConfig, exclude=()) -> dict[str, tuple[float, float, float, float]]:
    """Word id -> GT box (pt) of the words judged PRESERVED in place whose whole-word image is really at
    their place (inplace_score >= moved_anchor). They bound their neighbours' windows and claim their
    ink."""
    out = {}
    for w in page.words:
        if inplace.get(w.id, ("",))[0] != "PRESERVED" or w.id in exclude:
            continue
        sc = inplace_score(o, x, w, cfg)
        if sc is not None and sc >= cfg.moved_anchor:
            out[w.id] = tuple(bbox(as_poly(w.poly)))
    return out


def page_lines(page: GTPage) -> dict[str, list]:
    lines: dict[str, list] = {}
    for w in page.words:
        lines.setdefault(w.line, []).append(w)
    for ws in lines.values():
        ws.sort(key=lambda w: w.order)
    return lines


def gap(w, lines: dict[str, list], anchors: Mapping, cfg: EvalConfig) -> tuple[float | None, float | None]:
    """The range (pt, along the reading direction) the word's found box must lie in: between the nearest
    anchors before and after it on its GT line, widened by moved_gap x h."""
    hpt, _, ang = word_frame(w)
    u, _ = _axis(ang)
    ws = lines[w.line]
    k = ws.index(w)
    left = next((q for q in reversed(ws[:k]) if q.id in anchors), None)
    right = next((q for q in ws[k + 1:] if q.id in anchors), None)
    lo = _interval(as_poly(left.poly), u)[1] - cfg.moved_gap * hpt if left is not None else None
    hi = _interval(as_poly(right.poly), u)[0] + cfg.moved_gap * hpt if right is not None else None
    return lo, hi


# -- assignment ----------------------------------------------------------------------------------------------


def assign(rows: list[dict], cfg: EvalConfig, taken_boxes=()) -> dict[str, dict]:
    """One candidate per word and one word per place: greedy over score - moved_lambda x |d| / h among the
    valid candidates (score >= moved_match, v >= word_preserved_fraction); no box overlapping a taken
    one by more than moved_overlap. rows: dicts with wid, h_pt and cands."""
    pairs = []
    for r in rows:
        for c in r["cands"]:
            if c["score"] >= cfg.moved_match and c.get("v") is not None and c["v"] >= cfg.word_preserved_fraction:
                d = math.hypot(c["dx_pt"], c["dy_pt"]) / max(r["h_pt"], 1e-6)
                pairs.append((c["score"] - cfg.moved_lambda * d, r["wid"], c))
    pairs.sort(key=lambda p: -p[0])
    taken: dict[str, dict] = {}
    boxes = list(taken_boxes)
    for _, wid, c in pairs:
        if wid in taken:
            continue
        if any(overlap(c["box"], b) > cfg.moved_overlap or overlap(b, c["box"]) > cfg.moved_overlap for b in boxes):
            continue
        taken[wid] = c
        boxes.append(c["box"])
    return taken


def order_fix(page: GTPage, got: dict[str, dict], cfg: EvalConfig) -> dict[str, dict]:
    """Words of one line with the same text keep their reading order: their matches are swapped when
    they cross. A swapped match is retargeted to the word that takes it (the same found box, the
    transform worked out from that word's GT box) and marked ``swapped``."""
    scale = cfg.dpi / 72.0
    by_line: dict[str, list] = {}
    for w in page.words:
        if w.id in got:
            by_line.setdefault(w.line, []).append(w)
    for ws in by_line.values():
        ws.sort(key=lambda w: w.order)
        u, _ = _axis(ws[0].angle)

        def pos(c):
            b = c["box"]
            return float(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) @ u)

        changed = True
        while changed:
            changed = False
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    a, b = ws[i], ws[j]
                    if a.text != b.text:
                        continue
                    if pos(got[a.id]) > pos(got[b.id]) + 1e-6:
                        got[a.id], got[b.id] = _retarget(got[b.id], b, a, scale), _retarget(got[a.id], a, b, scale)
                        changed = True
    return got


def _retarget(c: dict, w_from, w_to, scale: float) -> dict:
    """The match ``c`` of word ``w_from`` as a match of ``w_to`` (same text): the same found box and
    scale, the transform and displacement taken from ``w_to``'s GT box."""
    s = c["s"]
    f = bbox(as_poly(w_from.poly))
    g = bbox(as_poly(w_to.poly))
    tx = c["tx"] + s * (f[0] - g[0]) * scale
    ty = c["ty"] + s * (f[1] - g[1]) * scale
    return dict(c, tx=tx, ty=ty, swapped=True,
                dx_pt=((s * (g[0] + g[2]) / 2 * scale + tx) / scale - (g[0] + g[2]) / 2),
                dy_pt=((s * (g[1] + g[3]) / 2 * scale + ty) / scale - (g[1] + g[3]) / 2))


# -- filters ---------------------------------------------------------------------------------------------------


def inky_beside(x: np.ndarray, box, vertical: bool, cfg: EvalConfig, paper: float) -> bool:
    """Is there ink (darker than half the paper level) on more than moved_side_ink of the pixels of either
    strip beside the found box: moved_side_strip x h along the reading direction, the middle of the box
    across it?"""
    sc = cfg.dpi / 72.0
    x0, y0, x1, y1 = box
    a, b = cfg.moved_side_strip
    strips = []
    if not vertical:
        h = y1 - y0
        r0, r1 = int((y0 + SIDE_ACROSS * h) * sc), int(math.ceil((y1 - SIDE_ACROSS * h) * sc))
        for s0, s1 in ((x0 - b * h, x0 - a * h), (x1 + a * h, x1 + b * h)):
            c0, c1 = int(math.floor(s0 * sc)), int(math.ceil(s1 * sc))
            strips.append(x[max(r0, 0):max(r1, 1), max(c0, 0):max(c1, 1)])
    else:
        h = x1 - x0
        c0, c1 = int((x0 + SIDE_ACROSS * h) * sc), int(math.ceil((x1 - SIDE_ACROSS * h) * sc))
        for s0, s1 in ((y0 - b * h, y0 - a * h), (y1 + a * h, y1 + b * h)):
            r0, r1 = int(math.floor(s0 * sc)), int(math.ceil(s1 * sc))
            strips.append(x[max(r0, 0):max(r1, 1), max(c0, 0):max(c1, 1)])
    return any(st.size and float((st < paper / 2).mean()) > cfg.moved_side_ink for st in strips)


def _keep(w, c: dict, pc: PageCells, o: np.ndarray, x: np.ndarray, requirement: str, cfg: EvalConfig,
          strict_words, paper: float) -> bool:
    """The two filters after assignment: the identity gate for a match at the GT place, the whole-word
    check for a word that is not MUST."""
    scale = cfg.dpi / 72.0
    if abs(c["s"] - 1.0) < 1e-6 and math.hypot(c["dx_pt"], c["dy_pt"]) * scale <= cfg.moved_zero_px:
        v = pc.word_fraction_at(w.id, o, x, c["s"], c["tx"], c["ty"], cfg, gated=True, strict_words=strict_words)
        if v is None or v < cfg.word_preserved_fraction:
            return False
    if requirement != "MUST" and inky_beside(x, c["box"], _vertical(w.angle), cfg, paper):
        return False
    return True


# -- the search -----------------------------------------------------------------------------------------------


def find_moved(gt: GTDocument, requirement: Mapping[str, str], pcs: Mapping[int, PageCells],
               pairs: Mapping[int, tuple[np.ndarray, np.ndarray]], inplace: Mapping[str, tuple[str, float | None]],
               unreliable: set[int], cfg: EvalConfig, strict_words=None) -> tuple[dict[str, Found], dict]:
    """Words the in-place test judged DAMAGED or REMOVED (any requirement) found intact near their place.

    requirement: word id -> effective requirement value; inplace: word id -> (status, visible fraction)
    of the in-place test on the primary render. Returns (word id -> :class:`Found`, page -> anchors)."""
    out: dict[str, Found] = {}
    anchors_by_page: dict[int, dict] = {}
    for page in gt.pages:
        i = page.index
        if i in unreliable or i not in pairs:
            continue
        o, x = pairs[i]
        pc = pcs[i]
        anchors = anchors_of(page, o, x, inplace, cfg)
        anchors_by_page[i] = anchors
        lines = page_lines(page)
        words = {w.id: w for w in page.words}
        rows = []
        for w in page.words:
            if inplace.get(w.id, ("",))[0] not in ("DAMAGED", "REMOVED"):
                continue
            lo, hi = gap(w, lines, anchors, cfg)
            claimed = [b for wid, b in anchors.items() if wid != w.id]
            cands = candidates(o, x, w, cfg, lo, hi, claimed)
            for c in cands:
                if c["score"] >= cfg.moved_match:  # only a candidate that can be valid is verified
                    c["v"] = pc.word_fraction_at(w.id, o, x, c["s"], c["tx"], c["ty"], cfg)
            if cands:
                rows.append(dict(wid=w.id, h_pt=word_frame(w)[0], cands=cands))
        got = order_fix(page, assign(rows, cfg), cfg)
        paper = _paper(x)
        for wid, c in got.items():
            req = requirement.get(wid, "ORDINARY")
            if c.get("swapped"):
                c["v"] = pc.word_fraction_at(wid, o, x, c["s"], c["tx"], c["ty"], cfg)
                if c["v"] is None or c["v"] < cfg.word_preserved_fraction:
                    continue
            if not _keep(words[wid], c, pc, o, x, req, cfg, strict_words, paper):
                continue
            out[wid] = Found(i, c["s"], c["tx"], c["ty"], tuple(c["box"]), round(c["dx_pt"], 2), round(c["dy_pt"], 2),
                             round(c["score"], 4), round(float(c["v"]), 4), req)
    return out, anchors_by_page
