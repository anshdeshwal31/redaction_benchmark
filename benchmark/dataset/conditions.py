"""Automatic page-condition measurement (PLAN.md §1.2): encoding, DPI, skew, text axis."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf

from ..forensics.inspect import PageInventory
from ..schemas import PageConditions, PageEncoding

FULL_PAGE_COVERAGE = 0.85
VECTOR_OUTLINE_MIN_PATHS = 200


def classify_encoding(p: PageInventory) -> tuple[PageEncoding, float | None]:
    """Return the page encoding and the DPI of its full-page image (if any)."""
    full = [im for im in p.images if im.coverage >= FULL_PAGE_COVERAGE and not im.is_mask]
    bilevel = [im for im in p.images if im.is_mask or im.bpc == 1]
    dpi = None
    if full:
        big = max(full, key=lambda im: im.width * im.height)
        dpi = round(min(big.dpi_x or 0, big.dpi_y or 0), 1) or None
        if bilevel:
            return PageEncoding.RASTER_MRC, dpi
        if p.visible_chars == 0 and p.invisible_chars > 0:
            return PageEncoding.RASTER_HIDDEN_TEXT, dpi
        if p.visible_chars > 0:
            return PageEncoding.RASTER_VISIBLE_TEXT, dpi
        return PageEncoding.RASTER_SINGLE, dpi
    if p.visible_chars > 0:
        return PageEncoding.DIGITAL_TEXT, None
    if p.filled_paths >= VECTOR_OUTLINE_MIN_PATHS:
        return PageEncoding.VECTOR_OUTLINE, None
    if p.images or p.vector_paths:
        return PageEncoding.MIXED, None
    return PageEncoding.BLANK, None


def _profile_score(ys: np.ndarray, xs: np.ndarray, angle_deg: float) -> float:
    t = np.deg2rad(angle_deg)
    proj = np.round(ys * np.cos(t) - xs * np.sin(t)).astype(np.int64)
    hist = np.bincount(proj - proj.min())
    return float((hist.astype(np.float64) ** 2).sum())


def _best_angle(ys: np.ndarray, xs: np.ndarray, span: float = 5.0) -> tuple[float, float]:
    coarse = np.arange(-span, span + 1e-9, 0.25)
    scores = [_profile_score(ys, xs, a) for a in coarse]
    a0 = float(coarse[int(np.argmax(scores))])
    fine = np.arange(a0 - 0.25, a0 + 0.25 + 1e-9, 0.05)
    fscores = [_profile_score(ys, xs, a) for a in fine]
    i = int(np.argmax(fscores))
    return float(fine[i]), float(fscores[i])


def _remove_rules(ink: np.ndarray, dpi: float) -> np.ndarray:
    """Drop long straight lines (table rules, form boxes) that dominate projection profiles."""
    k = max(15, int(round(0.35 * dpi)))  # ~9 mm: much longer than any glyph stroke
    u8 = ink.astype(np.uint8)
    horiz = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    lines = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    return ink & ~lines.astype(bool)


def text_axis(ink: np.ndarray, dpi: float = 100, min_glyphs: int = 40) -> str:
    """Decide the reading axis from glyph neighbourhoods.

    Within a word the nearest glyph lies along the reading direction (inter-character
    gaps are smaller than line spacing), so the distribution of nearest-neighbour
    directions between glyph-sized components reveals the axis. Unlike projection
    profiles this is not fooled by aligned table columns.
    """
    from scipy.spatial import cKDTree

    n, _, stats, centroids = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    lo, hi = 0.02 * dpi, 0.6 * dpi  # glyph-sized: ~0.5 mm to ~15 mm
    keep = [
        i for i in range(1, n)
        if lo <= stats[i, cv2.CC_STAT_HEIGHT] <= hi and lo <= stats[i, cv2.CC_STAT_WIDTH] <= hi
    ]
    if len(keep) < min_glyphs:
        return "unknown"
    pts = centroids[keep]
    dist, idx = cKDTree(pts).query(pts, k=2)
    vec = pts[idx[:, 1]] - pts
    ang = np.degrees(np.arctan2(np.abs(vec[:, 1]), np.abs(vec[:, 0])))  # 0 = horizontal, 90 = vertical
    horizontal = float(np.mean(ang < 25))
    vertical = float(np.mean(ang > 65))
    if horizontal >= 1.5 * vertical and horizontal > 0.4:
        return "horizontal"
    if vertical >= 1.5 * horizontal and vertical > 0.4:
        return "vertical"
    return "unknown"


def measure_skew(gray: np.ndarray, dpi: float = 100, max_points: int = 200_000) -> tuple[float | None, str]:
    """Estimate (skew in degrees clockwise, text axis) from a greyscale rendering.

    The axis comes from glyph neighbourhoods (:func:`text_axis`); the skew from
    projection-profile sharpness along that axis (text lines give a spiky histogram at
    the right angle). Ruling lines are removed first so tables do not dominate either.
    """
    ink = _remove_rules(gray < 140, dpi)
    axis = text_axis(ink, dpi)
    ys, xs = np.nonzero(ink)
    if len(xs) < 500 or axis == "unknown":
        return None, axis
    if len(xs) > max_points:
        idx = np.random.default_rng(0).choice(len(xs), max_points, replace=False)
        ys, xs = ys[idx], xs[idx]
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)
    if axis == "horizontal":
        angle, _ = _best_angle(ys, xs)
    else:
        angle, _ = _best_angle(xs, -ys)  # same estimator on the 90°-rotated hypothesis
    return round(angle, 2), axis


def page_conditions(p: PageInventory, page: pymupdf.Page | None = None, skew_dpi: float = 100) -> PageConditions:
    encoding, dpi = classify_encoding(p)
    skew, axis = None, None
    if page is not None and encoding in (PageEncoding.RASTER_SINGLE, PageEncoding.RASTER_MRC,
                                         PageEncoding.RASTER_HIDDEN_TEXT, PageEncoding.RASTER_VISIBLE_TEXT):
        pix = page.get_pixmap(dpi=skew_dpi, colorspace=pymupdf.csGRAY, annots=False)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        skew, axis = measure_skew(gray, dpi=skew_dpi)
    mb, cb = p.mediabox, p.cropbox
    return PageConditions(
        index=p.index,
        display_w=round(p.display_size[0], 3),
        display_h=round(p.display_size[1], 3),
        rotation_attr=p.rotation,
        cropbox_offset=bool(abs(cb[0] - mb[0]) > 0.01 or abs(cb[1] - mb[1]) > 0.01),
        encoding=encoding,
        text_chars=p.visible_chars,
        invisible_text_chars=p.invisible_chars,
        image_count=len(p.images),
        bilevel_image_count=sum(1 for im in p.images if im.is_mask or im.bpc == 1),
        image_coverage=round(min(1.0, sum(im.coverage for im in p.images)), 4),
        full_page_image_dpi=dpi,
        vector_paths=p.vector_paths,
        skew_deg=skew,
        text_axis=axis,
    )
