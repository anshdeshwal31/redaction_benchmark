"""Controlled inputs (PLAN.md §4.3): clean twins and single-factor degradation ladders.

Both work on *any* document with GT, so they apply to real documents once annotated:

- clean_twin: a born-digital page with every GT word typeset at its GT polygon and
  orientation (non-text regions copied as image crops). Same content and layout as the
  source, perfect image quality: scan-vs-twin differences isolate scan/OCR difficulty for
  any tool, including final-PDF-only ones.
- degrade: renders a PDF through one controlled transform (DPI, skew, JPEG quality,
  noise, blur); GT polygons are transformed exactly.
"""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np
import pymupdf

from ..geometry import as_poly, bbox
from ..schemas import GTDocument, WordSource
from .fixtures import Fixture, transform_gt

_ASC, _DESC = 0.718, 0.207  # Helvetica ascender / descender (em)


def _frame(poly: np.ndarray, angle: float):
    t = math.radians(angle)
    u = np.array([math.cos(t), math.sin(t)])
    v = np.array([-math.sin(t), math.cos(t)])
    tu, sv = poly @ u, poly @ v
    return u, v, float(tu.min()), float(tu.max()), float(sv.min()), float(sv.max())


def clean_twin(source: bytes, gt: GTDocument, region_dpi: float = 300) -> Fixture:
    src = pymupdf.open("pdf", source)
    out = pymupdf.open()
    out.set_metadata({"producer": "redaction_benchmark twin", "creationDate": "D:20260101000000Z",
                      "modDate": "D:20260101000000Z"})
    for p in gt.pages:
        page = out.new_page(width=p.width_pt, height=p.height_pt)
        for w in p.words:
            if not w.text.strip():
                continue
            poly = as_poly(w.poly)
            u, v, t0, t1, s0, s1 = _frame(poly, w.angle)
            height = s1 - s0
            size = max(1.0, height / (_ASC + _DESC))
            length = pymupdf.get_text_length(w.text, fontname="helv", fontsize=size)
            sx = (t1 - t0) / length if length > 0 else 1.0
            origin = t0 * u + (s1 - _DESC * size) * v
            point = pymupdf.Point(*origin)
            # morph rotates counter-clockwise on the page; our angles are clockwise (y-down)
            m = pymupdf.Matrix(sx, 0, 0, 1, 0, 0) * pymupdf.Matrix(-w.angle)
            page.insert_text(point, w.text, fontname="helv", fontsize=size, morph=(point, m))
        for r in p.regions:
            x0, y0, x1, y1 = bbox(as_poly(r.poly))
            clip = pymupdf.Rect(x0, y0, x1, y1)
            s = region_dpi / 72.0
            pix = src[p.index].get_pixmap(matrix=pymupdf.Matrix(s, s), clip=clip, colorspace=pymupdf.csGRAY)
            page.insert_image(clip, pixmap=pix)
    pdf = out.tobytes(garbage=3, deflate=True, no_new_id=True)
    pages = [pg.model_copy(update={
        "rotation": 0,
        "conditions": ["clean_twin"],
        "words": [w.model_copy(update={"source": WordSource.TEXT_LAYER, "char_polys": None}) for w in pg.words],
    }) for pg in gt.pages]
    twin_gt = gt.model_copy(update={"pages": pages, "source_sha256": hashlib.sha256(pdf).hexdigest()})
    return Fixture(f"{gt.doc_id}-twin", pdf, twin_gt)


def degrade(source: bytes, gt: GTDocument, dpi: float = 200, skew_deg: float = 0.0, jpeg_quality: int = 90,
            noise_sigma: float = 0.0, blur_sigma: float = 0.0, seed: int = 0) -> Fixture:
    """Render every page through one controlled transform into an image-only PDF."""
    src = pymupdf.open("pdf", source)
    out = pymupdf.open()
    rng = np.random.default_rng(seed)
    s = dpi / 72.0
    mats = {}
    for p in gt.pages:
        pix = src[p.index].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY, annots=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).astype(np.float32)
        if skew_deg:
            m = cv2.getRotationMatrix2D((img.shape[1] / 2.0, img.shape[0] / 2.0), -skew_deg, 1.0)
            img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
            mats[p.index] = m
        if blur_sigma:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        if noise_sigma:
            img = img + rng.normal(0, noise_sigma, img.shape)
        ok, jpg = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        page = out.new_page(width=p.width_pt, height=p.height_pt)
        page.insert_image(page.rect, stream=jpg.tobytes())
    pdf = out.tobytes(garbage=3, deflate=True, no_new_id=True)

    def fn(page_index):
        m = mats.get(page_index)
        if m is None:
            return lambda pts: pts
        return lambda pts: (np.hstack([pts * s, np.ones((len(pts), 1))]) @ m.T) / s

    name = f"dpi{dpi:g}-skew{skew_deg:g}-q{jpeg_quality}-n{noise_sigma:g}-b{blur_sigma:g}"
    new_gt = transform_gt(gt, fn, lambda p: (p.width_pt, p.height_pt), pdf, WordSource.RASTER,
                          angle_delta=skew_deg, rotation=0, condition=f"degraded;{name}")
    return Fixture(f"{gt.doc_id}-{name}", pdf, new_gt)


LADDERS = {
    "dpi": [{"dpi": d} for d in (300, 200, 150, 100)],
    "skew": [{"skew_deg": a} for a in (0.0, 0.5, 1.5, 3.0)],
    "jpeg": [{"jpeg_quality": q} for q in (95, 60, 30, 10)],
    "noise": [{"noise_sigma": n} for n in (0.0, 10.0, 25.0, 45.0)],
    "blur": [{"blur_sigma": b} for b in (0.0, 0.8, 1.5, 2.5)],
}


def ladder(source: bytes, gt: GTDocument, factor: str) -> list[Fixture]:
    """One fixture per step of a single-factor degradation ladder."""
    return [degrade(source, gt, **params) for params in LADDERS[factor]]
