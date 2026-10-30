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

A word still DAMAGED or REMOVED is then looked for re-typeset in a substitute font (:func:`find_restyled`):
its GT text typeset in the output's embedded fonts and the base-14 families.
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
          strict_words, paper: float, gated: frozenset[str] = frozenset()) -> bool:
    """The filters after assignment: the identity gate for a match at the GT place, and anywhere for a word
    with shared characters (``gated``: a fake that keeps them, such as a re-typeset mobile keeping its '04',
    is not the original moved), the whole-word check for a word that is not MUST."""
    scale = cfg.dpi / 72.0
    at_place = abs(c["s"] - 1.0) < 1e-6 and math.hypot(c["dx_pt"], c["dy_pt"]) * scale <= cfg.moved_zero_px
    if at_place or w.id in gated:
        v = pc.word_fraction_at(w.id, o, x, c["s"], c["tx"], c["ty"], cfg, gated=True, strict_words=strict_words)
        if v is None or v < cfg.word_preserved_fraction:
            return False
    if requirement != "MUST" and inky_beside(x, c["box"], _vertical(w.angle), cfg, paper):
        return False
    return True


# -- the search -----------------------------------------------------------------------------------------------


def find_moved(gt: GTDocument, requirement: Mapping[str, str], pcs: Mapping[int, PageCells],
               pairs: Mapping[int, tuple[np.ndarray, np.ndarray]], inplace: Mapping[str, tuple[str, float | None]],
               unreliable: set[int], cfg: EvalConfig, strict_words=None, gated: frozenset[str] = frozenset()
               ) -> tuple[dict[str, Found], dict]:
    """Words the in-place test judged DAMAGED or REMOVED (any requirement) found intact near their place.

    requirement: word id -> effective requirement value; inplace: word id -> (status, visible fraction)
    of the in-place test on the primary render; gated: words with shared characters, found only where they
    pass the character identity gate. Returns (word id -> :class:`Found`, page -> anchors)."""
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
            if not _keep(words[wid], c, pc, o, x, req, cfg, strict_words, paper, gated):
                continue
            out[wid] = Found(i, c["s"], c["tx"], c["ty"], tuple(c["box"]), round(c["dx_pt"], 2), round(c["dy_pt"], 2),
                             round(c["score"], 4), round(float(c["v"]), 4), req)
    return out, anchors_by_page


# -- the restyled search (words re-typeset in a substitute font) ---------------------------------------------

# The base-14 families of restyle_fonts, as PyMuPDF draws them.
BASE14 = {"Helvetica": "helv", "Times-Roman": "tiro", "Courier": "cour", "Helvetica-Bold": "hebo"}
# Without a source text layer: font size and baseline from the GT box height h (Helvetica's proportions).
SIZE_PER_HEIGHT = 1.374
BASELINE_ABOVE_BOTTOM = 0.21


@dataclass
class _Font:
    name: str  # a restyle_fonts name, or 'embedded:<base font>'
    base14: str | None = None
    buffer: bytes | None = None
    glyphs: object = None  # pymupdf.Font of an embedded font, to know which characters it has

    def has(self, text: str) -> bool:
        if self.glyphs is None:
            return True
        return all(ch.isspace() or self.glyphs.has_glyph(ord(ch)) for ch in text)


def restyle_fonts(out_doc, page_index: int, cfg: EvalConfig, cache: dict) -> list[_Font]:
    """The fonts tried on one output page, in restyle_fonts order; 'embedded' stands for the fonts the
    output page embeds when it has a text layer (each once, by content)."""
    import pymupdf

    out: list[_Font] = []
    for name in cfg.restyle_fonts:
        if name != "embedded":
            if name in BASE14:
                out.append(_Font(name, base14=BASE14[name]))
            continue
        if out_doc is None or page_index >= out_doc.page_count:
            continue
        page = out_doc[page_index]
        if not page.get_text("text").strip():
            continue
        for xref, *_rest in page.get_fonts(full=True):
            if xref not in cache:
                cache[xref] = None
                try:
                    basename, ext, _, buf = out_doc.extract_font(xref)
                    if buf and ext not in ("n/a", ""):
                        cache[xref] = _Font(f"embedded:{basename}", buffer=buf,
                                            glyphs=pymupdf.Font(fontbuffer=buf))
                except Exception:  # noqa: BLE001 - a font PyMuPDF cannot load is not tried
                    cache[xref] = None
            f = cache[xref]
            if f is not None and all(g.buffer != f.buffer for g in out if g.buffer is not None):
                out.append(f)
    return out


def source_glyphs(src_page, geom) -> np.ndarray:
    """Every character of the source page's text layer: (centre x, centre y, font size, baseline y),
    display pt."""
    from ..geometry import Frame

    rows = []
    for b in src_page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                for ch in sp["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    rows.append(((x0 + x1) / 2, (y0 + y1) / 2, sp["size"], ch["origin"][0], ch["origin"][1]))
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows, float)
    c = geom.transform(a[:, 0:2], Frame.UNROTATED_PT, Frame.DISPLAY_PT)
    base = geom.transform(a[:, 3:5], Frame.UNROTATED_PT, Frame.DISPLAY_PT)
    return np.column_stack([c, a[:, 2], base[:, 1]])


def size_and_baseline(glyphs: np.ndarray, w) -> tuple[float, float]:
    """The source's font size and baseline at the word (its characters in the source text layer); without
    them, from the GT box height."""
    x0, y0, x1, y1 = bbox(as_poly(w.poly))
    if len(glyphs):
        m = (glyphs[:, 0] >= x0) & (glyphs[:, 0] <= x1) & (glyphs[:, 1] >= y0) & (glyphs[:, 1] <= y1)
        if m.any():
            return float(np.median(glyphs[m, 2])), float(np.median(glyphs[m, 3]))
    h = y1 - y0
    return h / SIZE_PER_HEIGHT, y1 - BASELINE_ABOVE_BOTTOM * h


def render_text(text: str, font: _Font, size: float, x0: float, baseline: float, page_w: float, page_h: float,
                cfg: EvalConfig) -> dict | None:
    """The GT text typeset in ``font`` at the source's size and baseline on a blank page of the source's
    size, as a template (the moved search's margins) with its word box and character boxes in pixels."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=page_w, height=page_h)
    if font.buffer is not None:
        page.insert_font(fontname="rbrestyle", fontbuffer=font.buffer)
        page.insert_text((x0, baseline), text, fontname="rbrestyle", fontsize=size)
    else:
        page.insert_text((x0, baseline), text, fontname=font.base14, fontsize=size)
    chars = [ch["bbox"] for b in page.get_text("rawdict")["blocks"] for ln in b.get("lines", [])
             for sp in ln["spans"] for ch in sp["chars"] if not ch["c"].isspace()]
    if not chars:
        return None
    sc = cfg.dpi / 72.0
    wx0, wy0 = min(c[0] for c in chars), min(c[1] for c in chars)
    wx1, wy1 = max(c[2] for c in chars), max(c[3] for c in chars)
    h = wy1 - wy0
    c0 = int(math.floor((wx0 - cfg.moved_margin_along * h) * sc))
    r0 = int(math.floor((wy0 - cfg.moved_margin_across * h) * sc))
    c1 = int(math.ceil((wx1 + cfg.moved_margin_along * h) * sc))
    r1 = int(math.ceil((wy1 + cfg.moved_margin_across * h) * sc))
    pix = page.get_pixmap(dpi=int(round(cfg.dpi)), clip=pymupdf.Rect(c0 / sc, r0 / sc, c1 / sc, r1 / sc),
                          colorspace=pymupdf.csGRAY, alpha=False)
    t = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).astype(np.float32)
    if t.std() < 1.0:
        return None
    return dict(t=t, c0=c0, r0=r0, c1=c0 + t.shape[1], r1=r0 + t.shape[0],
                box=(wx0 * sc, wy0 * sc, wx1 * sc, wy1 * sc), chars=[tuple(v * sc for v in c) for c in chars])


def char_check(tpl: dict, x: np.ndarray, s: float, tx: float, ty: float, cfg: EvalConfig) -> float:
    """Share of the rendered characters whose pixels correlate with the output at the transform
    (|NCC| >= cell_ncc with output contrast >= cell_min_visible_std)."""
    from .cells import bilinear

    t, c0, r0 = tpl["t"], tpl["c0"], tpl["r0"]
    ok = n = 0
    for a0, b0, a1, b1 in tpl["chars"]:
        cc, rr = np.meshgrid(np.arange(int(a0), int(math.ceil(a1))), np.arange(int(b0), int(math.ceil(b1))))
        cc, rr = cc.ravel(), rr.ravel()
        inside = (rr - r0 >= 0) & (rr - r0 < t.shape[0]) & (cc - c0 >= 0) & (cc - c0 < t.shape[1])
        cc, rr = cc[inside], rr[inside]
        if cc.size < 8:
            continue
        o = t[rr - r0, cc - c0].astype(np.float64)
        if o.std() < cfg.cell_min_orig_std:
            continue
        n += 1
        xs = bilinear(x, s * cc + tx, s * rr + ty)
        if xs.std() < cfg.cell_min_visible_std:
            continue
        ncc = float(np.mean((o - o.mean()) * (xs - xs.mean())) / (o.std() * xs.std()))
        ok += abs(ncc) >= cfg.cell_ncc
    return ok / n if n else 0.0


def own_image_score(o: np.ndarray, x: np.ndarray, w, cfg: EvalConfig, s: float, tx: float, ty: float) -> float:
    """Score of the word's original image (the moved search's template) at the same transform, within
    JITTER_PX: a restyled match must beat it by restyle_margin, so the original font is left to the
    moved search and its stricter checks."""
    tpl = template(o, w, cfg)
    if tpl is None:
        return -1.0
    t = tpl["t"].astype(np.float32)
    th, tw = max(3, int(round(t.shape[0] * s))), max(3, int(round(t.shape[1] * s)))
    t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    r0, c0 = int(round(s * tpl["r0"] + ty)) - JITTER_PX, int(round(s * tpl["c0"] + tx)) - JITTER_PX
    hh, ww = x.shape
    if r0 < 0 or c0 < 0 or r0 + th + 2 * JITTER_PX > hh or c0 + tw + 2 * JITTER_PX > ww:
        return -1.0
    win = x[r0:r0 + th + 2 * JITTER_PX, c0:c0 + tw + 2 * JITTER_PX].astype(np.float32)
    return float(np.nan_to_num(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED), nan=-1.0).max())


def find_restyled(gt: GTDocument, requirement: Mapping[str, str], pairs: Mapping[int, tuple[np.ndarray, np.ndarray]],
                  inplace: Mapping[str, tuple[str, float | None]], found: Mapping[str, Found],
                  anchors_by_page: Mapping[int, dict], unreliable: set[int], cfg: EvalConfig, src_doc, src_geoms,
                  out_doc) -> dict[str, Found]:
    """Words still DAMAGED or REMOVED after :func:`find_moved` found re-typeset in a substitute font.

    The GT text is typeset in each font of :func:`restyle_fonts` at the source's size and baseline and
    looked for like a moved word (same window, scales and anchors, two positions per scale). A position
    counts when it scores at least restyle_match, beats the original's own image at the same transform
    by restyle_margin, has at least restyle_chars of its characters correlating with the output, and,
    for a word that is not MUST, is a whole word. Horizontal words only. Assigned best score first,
    never over an anchor or a box already taken."""
    out: dict[str, Found] = {}
    font_cache: dict = {}
    for page in gt.pages:
        i = page.index
        if i in unreliable or i not in pairs:
            continue
        o, x = pairs[i]
        anchors = anchors_by_page.get(i, {})
        taken = list(anchors.values()) + [f.box for f in found.values() if f.page == i]
        words = [w for w in page.words if w.id not in found
                 and inplace.get(w.id, ("",))[0] in ("DAMAGED", "REMOVED")
                 and any(ch.isalnum() for ch in w.text) and not _vertical(w.angle)]
        if not words:
            continue
        fonts = restyle_fonts(out_doc, i, cfg, font_cache)
        glyphs = source_glyphs(src_doc[i], src_geoms[i]) if src_doc is not None and i < src_doc.page_count \
            else np.zeros((0, 4))
        lines = page_lines(page)
        paper = _paper(x)
        best: dict[str, dict] = {}
        for w in words:
            lo, hi = gap(w, lines, anchors, cfg)
            size, base = size_and_baseline(glyphs, w)
            gx0 = bbox(as_poly(w.poly))[0]
            hpt = word_frame(w)[0]
            req = requirement.get(w.id, "ORDINARY")
            for font in fonts:
                if not font.has(w.text):
                    continue
                tpl = render_text(w.text, font, size, gx0, base, page.width_pt, page.height_pt, cfg)
                if tpl is None:
                    continue
                tpl.update(hpt=hpt, ang=w.angle, vertical=False)
                for c in candidates(o, x, w, cfg, lo, hi, taken, tpl=tpl, n_keep=2):
                    if c["score"] < cfg.restyle_match:
                        continue
                    if c["score"] < own_image_score(o, x, w, cfg, c["s"], c["tx"], c["ty"]) + cfg.restyle_margin:
                        continue
                    v = char_check(tpl, x, c["s"], c["tx"], c["ty"], cfg)
                    if v < cfg.restyle_chars:
                        continue
                    if req != "MUST" and inky_beside(x, c["box"], False, cfg, paper):
                        continue
                    if w.id not in best or c["score"] > best[w.id]["score"]:
                        best[w.id] = dict(c, v=v, font=font.name)
        for wid, c in sorted(best.items(), key=lambda kv: -kv[1]["score"]):
            if any(overlap(c["box"], b) > cfg.moved_overlap or overlap(b, c["box"]) > cfg.moved_overlap for b in taken):
                continue
            taken.append(c["box"])
            out[wid] = Found(i, c["s"], c["tx"], c["ty"], tuple(c["box"]), round(c["dx_pt"], 2), round(c["dy_pt"], 2),
                             round(c["score"], 4), round(float(c["v"]), 4), requirement.get(wid, "ORDINARY"),
                             restyled=True, font=c["font"])
    return out


# -- shared structure: identifying slices and the state/postcode pair -----------------------------------------


def find_slices(gt: GTDocument, needle_cut: Mapping[str, tuple[int, int]], pcs: Mapping[int, PageCells],
                pairs: Mapping[int, tuple[np.ndarray, np.ndarray]], found: Mapping[str, Found],
                visible: Mapping[int, Mapping], informative: Mapping[int, set], unreliable: set[int],
                cfg: EvalConfig) -> list[tuple[int, tuple[str, int]]]:
    """Identifying slices of MUST words with shared characters (an email's local part, an identifier's
    or a phone number's other digits; word id -> [a, b)) found away from their place, as (page, unit)
    of their characters: a changed domain or prefix must not hide a kept local part or kept digits it
    pushed along. A slice is looked for when fewer than word_preserved_fraction of its characters are
    visible in place and its whole word was not found: its own original image, the moved search's
    window and scales, moved_match, and the band test at the transform finding at least
    word_preserved_fraction of its characters; matches at scale 1 within moved_zero_px of the GT place
    are left to the in-place test."""
    scale = cfg.dpi / 72.0
    words = gt.word_index()
    out: list[tuple[int, tuple[str, int]]] = []
    for wid, (a, b) in needle_cut.items():
        pg, w = words[wid]
        if pg in unreliable or pg not in pairs or wid in found or b - a < 2:
            continue
        us = [(wid, k) for k in range(a, b) if (wid, k) in informative.get(pg, set())]
        if not us or sum(bool(visible[pg].get(u, False)) for u in us) / len(us) >= cfg.word_preserved_fraction:
            continue
        pc = pcs[pg]
        o, x = pairs[pg]
        idxs = [i for i in pc.word_cells.get(wid, []) if a <= pc.cells[i].index < b]
        if not idxs:
            continue
        rr = np.concatenate([pc.rows[i] for i in idxs])
        cc = np.concatenate([pc.cols[i] for i in idxs])
        if rr.size == 0:
            continue
        hpt, _, ang = word_frame(w)
        vertical = _vertical(ang)
        x0, y0, x1, y1 = float(cc.min()), float(rr.min()), float(cc.max() + 1), float(rr.max() + 1)
        mh, mv = cfg.moved_margin_along * hpt * scale, cfg.moved_margin_across * hpt * scale
        if vertical:
            mh, mv = mv, mh
        c0, r0 = max(int(math.floor(x0 - mh)), 0), max(int(math.floor(y0 - mv)), 0)
        c1, r1 = min(int(math.ceil(x1 + mh)), o.shape[1]), min(int(math.ceil(y1 + mv)), o.shape[0])
        t = o[r0:r1, c0:c1]
        if t.size == 0 or t.std() < 1.0:
            continue
        tpl = dict(t=t, c0=c0, r0=r0, c1=c1, r1=r1, box=(x0, y0, x1, y1), hpt=hpt, ang=ang, vertical=vertical)
        best = None
        for c in candidates(o, x, w, cfg, None, None, [], tpl=tpl):
            if c["score"] < cfg.moved_match:
                continue
            if abs(c["s"] - 1.0) < 1e-6 and math.hypot(c["dx_pt"], c["dy_pt"]) * scale <= cfg.moved_zero_px:
                continue
            v = pc.word_fraction_at(wid, o, x, c["s"], c["tx"], c["ty"], cfg, chars=(a, b))
            if v is not None and v >= cfg.word_preserved_fraction and (best is None or c["score"] > best["score"]):
                best = c
        if best is not None:
            out.extend((pg, u) for u in us)
    return out


def refine_chars(pc: PageCells, wid: str, ks, o: np.ndarray, x: np.ndarray, s: float, tx: float, ty: float
                 ) -> tuple[float, float, float]:
    """The best whole-pixel offset (dx, dy, score) of the original glyphs of characters ``ks`` of a word,
    expected at output_px = s * original_px + (tx, ty), within half a character along the line and a
    quarter of their height across (TM_CCOEFF_NORMED); score -1 when it cannot be taken."""
    idxs = [i for i in pc.word_cells.get(wid, []) if pc.cells[i].index in ks]
    if not idxs:
        return 0.0, 0.0, -1.0
    rr = np.concatenate([pc.rows[i] for i in idxs])
    cc = np.concatenate([pc.cols[i] for i in idxs])
    if rr.size == 0:
        return 0.0, 0.0, -1.0
    r0, r1, c0, c1 = int(rr.min()) - 1, int(rr.max()) + 2, int(cc.min()) - 1, int(cc.max()) + 2
    t = o[max(r0, 0):r1, max(c0, 0):c1].astype(np.float32)
    if t.std() < 1.0:
        return 0.0, 0.0, -1.0
    th, tw = max(3, int(round(t.shape[0] * s))), max(3, int(round(t.shape[1] * s)))
    t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA)
    ea, ec = max(2, int(round(0.5 * (c1 - c0) / max(len(ks), 1)))), max(2, int(round(0.25 * (r1 - r0))))
    pr0, pc0 = s * max(r0, 0) + ty, s * max(c0, 0) + tx
    top, left = int(math.floor(pr0)) - ec, int(math.floor(pc0)) - ea
    hh, ww = x.shape
    if top < 0 or left < 0 or top + th + 2 * ec + 1 > hh or left + tw + 2 * ea + 1 > ww:
        return 0.0, 0.0, -1.0
    win = x[top:top + th + 2 * ec + 1, left:left + tw + 2 * ea + 1].astype(np.float32)
    res = np.nan_to_num(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED), nan=-1.0)
    k = np.unravel_index(int(np.argmax(res)), res.shape)
    return float(left + k[1] - pc0), float(top + k[0] - pr0), float(res.max())
