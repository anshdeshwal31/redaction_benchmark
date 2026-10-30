"""File-level content survival on one page (L2/L3, PLAN.md §5.3).

These checks ask whether the *original* content still exists in the output file,
whether or not it is visible: text-layer characters at the GT location, image layers
that still carry the original pixels (e.g. under an overlay box, or in one layer of an
MRC stack), and glyph paths that survive under a covering shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymupdf
import shapely
from rapidfuzz import fuzz

from ..geometry import Frame, PageGeometry, as_poly, dilate, to_shapely
from ..schemas import GTPage
from .cells import PageCells, correlate
from .config import EvalConfig


# -- text layer -----------------------------------------------------------------


def page_chars(page: pymupdf.Page, geom: PageGeometry) -> tuple[np.ndarray, list[str]]:
    """Centres (display_pt) and characters of every text object, visible or not."""
    pts, chars = [], []
    for block in page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_MEDIABOX_CLIP)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    if ch["c"].isspace():
                        continue
                    x0, y0, x1, y1 = ch["bbox"]
                    pts.append(((x0 + x1) / 2, (y0 + y1) / 2))
                    chars.append(ch["c"])
    if not pts:
        return np.zeros((0, 2)), []
    return geom.transform(np.array(pts), Frame.UNROTATED_PT, Frame.DISPLAY_PT), chars


def text_in_words(page_gt: GTPage, centres: np.ndarray, chars: list[str], word_ids: set[str]) -> dict[str, str]:
    """Extracted text (any render mode) whose character centres fall inside each GT word."""
    out: dict[str, str] = {}
    if not chars:
        return out
    for w in page_gt.words:
        if w.id not in word_ids:
            continue
        zone = to_shapely(dilate(as_poly(w.poly), 1.0))
        inside = shapely.contains_xy(zone, centres[:, 0], centres[:, 1])
        got = "".join(c for c, keep in zip(chars, inside) if keep)
        if got:
            out[w.id] = got
    return out


def text_leaks(needle: str, got: str | None, cfg: EvalConfig) -> bool:
    """Is the original token recoverable from the extracted text at its location?"""
    if not got or not needle.strip():
        return False
    a, b = needle.casefold(), got.casefold()
    return a in b or fuzz.ratio(a, b) >= cfg.text_ratio


# -- image layers ---------------------------------------------------------------


@dataclass
class Layer:
    xref: int
    width: int
    height: int
    is_mask: bool
    inverse: np.ndarray  # 3x3: unrotated page pt -> unit square
    gray: np.ndarray  # decoded image as grey (ink dark), raw data (soft masks ignored)


def _decode(doc: pymupdf.Document, xref: int, is_mask: bool) -> np.ndarray:
    pix = pymupdf.Pixmap(doc, xref)
    if is_mask:
        # Stencil masks decode as alpha-only pixmaps: 255 = painted with the fill colour.
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., -1]
        return (255 - a).astype(np.uint8)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()


def page_layers(doc: pymupdf.Document, page: pymupdf.Page, cache: dict[int, np.ndarray]) -> list[Layer]:
    layers = []
    for info in page.get_image_info(xrefs=True):
        xref = info.get("xref", 0)
        if not xref:
            continue  # inline images are rasterised in the page render; L1 covers them
        src = doc.xref_object(xref, compressed=True)
        is_mask = "/ImageMask true" in src
        if xref not in cache:
            try:
                cache[xref] = _decode(doc, xref, is_mask)
            except Exception:  # noqa: BLE001 - undecodable image: nothing recoverable here
                continue
        a, b, c, d, e, f = info["transform"]
        m = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        gray = cache[xref]
        layers.append(Layer(xref, gray.shape[1], gray.shape[0], is_mask, np.linalg.inv(m), gray))
    return layers


def layer_survival(
    cells: PageCells,
    indices: list[int],
    original: np.ndarray,
    layers: list[Layer],
    geom: PageGeometry,
    cfg: EvalConfig,
    strict_words: set[str] | frozenset[str] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Cells whose original pattern survives in some image layer.

    Returns ``(recoverable, latent)``: cell index -> layer xref. Recoverable means the
    layer alone shows the content with real contrast; latent means it correlates but
    is faint (MRC ghost text, near-transparent edits).

    Word characters then pass the same character identity gate as in place (eval-0.4): each layer
    that flags a word is resampled onto the render grid over the boxes of those words (2 px around),
    and a character whose identity, presence and clauses fail there loses its layer evidence
    (:meth:`PageCells.stats`; ``strict_words`` as there). Region slices are not gated.
    """
    recoverable: dict[int, int] = {}
    latent: dict[int, int] = {}
    if not layers or not indices:
        return recoverable, latent
    for i in indices:
        rr, cc = cells.rows[i], cells.cols[i]
        if rr.size < 4:
            continue
        o_all = original[rr, cc]
        if o_all.std() < cfg.cell_min_orig_std:
            continue
        unrot = geom.transform(cells.points(i), Frame.DISPLAY_PT, Frame.UNROTATED_PT)
        for layer in layers:
            uv = np.hstack([unrot, np.ones((len(unrot), 1))]) @ layer.inverse.T
            px = np.floor(uv[:, 0] * layer.width).astype(np.int64)
            py = np.floor(uv[:, 1] * layer.height).astype(np.int64)
            ok = (px >= 0) & (px < layer.width) & (py >= 0) & (py < layer.height)
            if ok.mean() < cfg.layer_min_inside:
                continue
            sample = layer.gray[py[ok], px[ok]]
            ncc, amp, _, sx = correlate(o_all[ok], sample)
            if abs(ncc) >= cfg.layer_ncc and amp >= cfg.layer_amp:
                recoverable[i] = layer.xref
                break
            if abs(ncc) >= cfg.layer_ncc and sx >= cfg.cell_latent_noise_std:
                latent.setdefault(i, layer.xref)
    if not recoverable and not latent:
        return recoverable, latent
    return _gate_layer_cells(cells, original, layers, geom, cfg, strict_words, recoverable, latent)


def _layer_on_render_grid(cells: PageCells, layer: Layer, geom: PageGeometry, shape: tuple[int, int],
                          words: set[str]) -> np.ndarray:
    """The layer as the output render would show it on its own, over the given words' boxes (2 px
    around): each render pixel centre taken from display pt to unrotated pt to the layer's unit square
    and sampled there (nearest pixel). Elsewhere, and outside the layer, the layer's paper level."""
    g = layer.gray
    h, w = shape
    s = cells.dpi / 72.0
    out = np.full((h, w), max(float(np.percentile(g, 95)), 64.0), np.float32)
    for wid in words:
        idx = cells.word_cells.get(wid, [])
        rr = np.concatenate([cells.rows[i] for i in idx]) if idx else np.zeros(0, np.int64)
        if rr.size == 0:
            continue
        cc = np.concatenate([cells.cols[i] for i in idx])
        r0, r1 = max(int(rr.min()) - 2, 0), min(int(rr.max()) + 3, h)
        c0, c1 = max(int(cc.min()) - 2, 0), min(int(cc.max()) + 3, w)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        pts = np.column_stack([(xx.ravel() + 0.5) / s, (yy.ravel() + 0.5) / s])
        unrot = geom.transform(pts, Frame.DISPLAY_PT, Frame.UNROTATED_PT)
        uv = np.hstack([unrot, np.ones((len(unrot), 1))]) @ layer.inverse.T
        px = np.floor(uv[:, 0] * layer.width).astype(np.int64)
        py = np.floor(uv[:, 1] * layer.height).astype(np.int64)
        inside = (px >= 0) & (px < layer.width) & (py >= 0) & (py < layer.height)
        vals = np.where(inside, g[np.clip(py, 0, layer.height - 1), np.clip(px, 0, layer.width - 1)],
                        out[yy.ravel(), xx.ravel()])
        out[yy.ravel(), xx.ravel()] = vals
    return out


def _gate_layer_cells(cells: PageCells, original: np.ndarray, layers: list[Layer], geom: PageGeometry,
                      cfg: EvalConfig, strict_words, recoverable: dict[int, int], latent: dict[int, int]
                      ) -> tuple[dict[int, int], dict[int, int]]:
    """Drop the layer evidence of word characters that fail the identity gate on the layer resampled
    onto the render grid."""
    informative = np.zeros(len(cells.cells), bool)
    for i in range(len(cells.cells)):
        rr, cc = cells.rows[i], cells.cols[i]
        informative[i] = rr.size >= 4 and float(original[rr, cc].std()) >= cfg.cell_min_orig_std
    by_xref = {la.xref: la for la in layers}
    passed: dict[int, dict] = {}
    for xref in {*recoverable.values(), *latent.values()}:
        flagged: dict[tuple[str, int], set[int]] = {}
        for i, x in [*recoverable.items(), *latent.items()]:
            c = cells.cells[i]
            if x == xref and c.word_id is not None:
                flagged.setdefault((c.word_id, c.index), set()).add(c.band)
        if not flagged:
            continue
        words = {u[0] for u in flagged}
        out = _layer_on_render_grid(cells, by_xref[xref], geom, original.shape, words)
        ci = cells.character_identity(original, out, informative, cfg, words=words)
        passed[xref] = cells.gate(ci, flagged, strict_words, cfg)

    def keep(i: int, xref: int) -> bool:
        c = cells.cells[i]
        return c.word_id is None or bool(passed.get(xref, {}).get((c.word_id, c.index)))

    return ({i: x for i, x in recoverable.items() if keep(i, x)}, {i: x for i, x in latent.items() if keep(i, x)})


# -- vector paths ---------------------------------------------------------------


def _path_key(d: dict) -> tuple:
    r = d["rect"]
    return (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1), d.get("fill") is not None)


def _straight(d: dict) -> bool:
    """Only axis-aligned lines and rectangles: a rule, box or field border, not pen strokes."""
    for item in d.get("items", []):
        if item[0] == "re":
            continue
        if item[0] != "l":
            return False  # curves and quads
        a, b = item[1], item[2]
        if abs(a.x - b.x) > 0.01 and abs(a.y - b.y) > 0.01:
            return False
    return True


def _points(d: dict) -> np.ndarray:
    """Every point a path's items pass through, in drawing order: line ends, curve control points,
    rectangle and quad corners."""
    pts: list[tuple[float, float]] = []
    for item in d.get("items", []):
        for p in item[1:]:
            if isinstance(p, pymupdf.Point):
                pts.append((p.x, p.y))
            elif isinstance(p, pymupdf.Rect):
                pts += [(p.x0, p.y0), (p.x1, p.y1)]
            elif isinstance(p, pymupdf.Quad):
                pts += [(q.x, q.y) for q in (p.ul, p.ur, p.ll, p.lr)]
    return np.array(pts, dtype=float).reshape(-1, 2)


def _same_path(a: dict, b: dict, tol: float = 0.1) -> bool:
    """One drawn path in both files: the same kinds of items through the same points (within ``tol``
    pt), filled or stroked alike. Colour and line width may differ: a stroke turned white or thinner
    is still the original stroke."""
    if a.get("type") != b.get("type") or [it[0] for it in a.get("items", [])] != [it[0] for it in b.get("items", [])]:
        return False
    pa, pb = _points(a), _points(b)
    return pa.shape == pb.shape and bool(np.all(np.abs(pa - pb) <= tol))


def _kept_glyph_paths(src: list[dict], src_geom: PageGeometry, page_gt: GTPage, word_ids: set[str]) -> set[int]:
    """Indices of the source paths that are glyphs of kept words drawn as outlines: a path's whole
    bounding box (display pt) lies inside the 1 pt dilated box of a kept word that the page draws as
    outlines, that is a word with at least max(1, ceil(half its letters and digits)) paths wholly
    inside that box. A word only in the text layer, such as a typed "______" signature line, has no
    such paths, so nothing drawn across it is taken for its glyphs."""
    kept = [w for w in page_gt.words if w.id not in word_ids]
    if not kept or not src:
        return set()
    corners = np.array([[(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                        for r in (d["rect"] for d in src)], float).reshape(-1, 2)
    shown = src_geom.transform(corners, Frame.UNROTATED_PT, Frame.DISPLAY_PT).reshape(-1, 4, 2)
    boxes = shapely.box(shown[:, :, 0].min(axis=1), shown[:, :, 1].min(axis=1),
                        shown[:, :, 0].max(axis=1), shown[:, :, 1].max(axis=1))
    words = [to_shapely(dilate(as_poly(w.poly), 1.0)) for w in kept]
    word_of, path_of = shapely.STRtree(boxes).query(words, predicate="covers")
    inside = np.bincount(word_of, minlength=len(kept))
    outlined = np.array([inside[k] >= max(1, math.ceil(0.5 * sum(c.isalnum() for c in w.text)))
                         for k, w in enumerate(kept)], bool)
    return {int(p) for k, p in zip(word_of, path_of) if outlined[k]}


def vector_survival(
    src_page: pymupdf.Page,
    out_page: pymupdf.Page,
    src_geom: PageGeometry,
    page_gt: GTPage,
    word_ids: set[str],
    region_ids: set[str] = frozenset(),
) -> dict[str, tuple[int, int]]:
    """word or region id -> (original glyph-like paths, surviving paths in the output).

    A word counts the glyph paths centred on it; one survives when the output has a path with the
    same bounding box and fill. A region (signature, mark) counts the pen strokes drawn in it; one
    survives only when the output still has that very path (the same items through the same points),
    so a new stand-in drawn to the original's size is not taken for it. Left out of a region: straight
    rules and boxes (the line a signature sits on belongs to the form) and glyphs of kept words drawn
    as outlines whose whole box lies inside the word (outlined text, such as a "Signed:" label the
    signature crosses). A kept word is drawn as outlines when the source page has at least half as
    many paths wholly inside its box (1 pt around) as it has letters and digits, and at least one; a
    typed "______" line in the text layer has none, so pen strokes across it count. Not covered: a
    mark drawn only with axis-aligned lines or rectangles, which counts as form, and other kept
    curved art inside a region's box, which counts as its strokes."""
    zones = {w.id: to_shapely(dilate(as_poly(w.poly), 1.0)) for w in page_gt.words if w.id in word_ids}
    zones.update({r.id: to_shapely(dilate(as_poly(r.poly), 1.0)) for r in page_gt.regions if r.id in region_ids})
    if not zones:
        return {}
    src = src_page.get_drawings()
    if not src:
        return {}
    out = out_page.get_drawings()
    out_keys = {_path_key(d) for d in out}
    out_rects = np.array([[d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1] for d in out], float).reshape(-1, 4)
    glyphs = _kept_glyph_paths(src, src_geom, page_gt, word_ids) if any(k in region_ids for k in zones) else set()

    def still_there(d: dict) -> bool:
        r = d["rect"]
        near = np.flatnonzero(np.all(np.abs(out_rects - [r.x0, r.y0, r.x1, r.y1]) <= 0.1, axis=1))
        return any(_same_path(d, out[k]) for k in near)

    areas = {k: z.area for k, z in zones.items()}
    result: dict[str, list[int]] = {k: [0, 0] for k in zones}
    for j, d in enumerate(src):
        r = d["rect"]
        area = r.width * r.height
        centre = shapely.Point(src_geom.transform([[(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2]], Frame.UNROTATED_PT,
                                                  Frame.DISPLAY_PT)[0])
        stroke = None  # in a region: a pen stroke, neither a form rule nor a kept word's glyph
        for key, zone in zones.items():
            if area > areas[key] or not zone.contains(centre):
                continue
            if key in region_ids:
                if stroke is None:
                    stroke = not _straight(d) and j not in glyphs
                if not stroke:
                    continue
                survived = still_there(d)
            else:
                survived = _path_key(d) in out_keys
            result[key][0] += 1
            result[key][1] += int(survived)
            break
    return {k: (v[0], v[1]) for k, v in result.items() if v[0] > 0}
