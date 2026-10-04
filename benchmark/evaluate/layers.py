"""File-level content survival on one page (L2/L3, PLAN.md §5.3).

These checks ask whether the *original* content still exists in the output file,
whether or not it is visible: text-layer characters at the GT location, image layers
that still carry the original pixels (e.g. under an overlay box, or in one layer of an
MRC stack), and glyph paths that survive under a covering shape.
"""

from __future__ import annotations

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
) -> tuple[dict[int, int], dict[int, int]]:
    """Cells whose original pattern survives in some image layer.

    Returns ``(recoverable, latent)``: cell index -> layer xref. Recoverable means the
    layer alone shows the content with real contrast; latent means it correlates but
    is faint (MRC ghost text, near-transparent edits).
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
    return recoverable, latent


# -- vector paths ---------------------------------------------------------------


def _path_key(d: dict) -> tuple:
    r = d["rect"]
    return (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1), d.get("fill") is not None)


def vector_survival(
    src_page: pymupdf.Page,
    out_page: pymupdf.Page,
    src_geom: PageGeometry,
    page_gt: GTPage,
    word_ids: set[str],
) -> dict[str, tuple[int, int]]:
    """word id -> (original glyph-like paths, surviving identical paths in the output)."""
    words = [w for w in page_gt.words if w.id in word_ids]
    if not words:
        return {}
    src = src_page.get_drawings()
    if not src:
        return {}
    out_keys = {_path_key(d) for d in out_page.get_drawings()}
    zones = {w.id: to_shapely(dilate(as_poly(w.poly), 1.0)) for w in words}
    areas = {w.id: zones[w.id].area for w in words}
    result: dict[str, list[int]] = {w.id: [0, 0] for w in words}
    for d in src:
        r = d["rect"]
        centre = src_geom.transform([[(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2]], Frame.UNROTATED_PT, Frame.DISPLAY_PT)[0]
        for wid, zone in zones.items():
            if r.width * r.height <= areas[wid] and zone.contains(shapely.Point(centre)):
                result[wid][0] += 1
                if _path_key(d) in out_keys:
                    result[wid][1] += 1
                break
    return {k: (v[0], v[1]) for k, v in result.items() if v[0] > 0}
