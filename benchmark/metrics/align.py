"""Projection of candidate artifacts into GT-word space (PLAN.md §3.2).

Every candidate claim (a detected mention, a planned region, an OCR token) is turned
into "which GT words, and how much of each word's ink, does it cover". Tools with
different OCR, tokenisation and reading order then become directly comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pymupdf

from ..geometry import Frame, PageGeometry, as_poly, bbox, load_page_geometries, oriented_cells, rasterize
from ..schemas import GTDocument

DEFAULT_THETA = 0.5  # a word is selected when at least half of its ink is covered


@dataclass
class _WordInk:
    word_id: str
    r0: int
    c0: int
    ink: np.ndarray  # bool, window-sized
    char_ink: list[np.ndarray]  # per character, window-sized

    @property
    def total(self) -> int:
        return int(self.ink.sum())


@dataclass
class InkIndex:
    """Ink masks of every GT word, from a rendering of the source document."""

    gt: GTDocument
    dpi: float
    geoms: list[PageGeometry]
    shapes: dict[int, tuple[int, int]] = field(default_factory=dict)
    words: dict[int, dict[str, _WordInk]] = field(default_factory=dict)

    @classmethod
    def build(cls, gt: GTDocument, source: bytes, dpi: float = 100) -> "InkIndex":
        import io
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            geoms = load_page_geometries(path)
        finally:
            import os

            os.unlink(path)
        idx = cls(gt=gt, dpi=dpi, geoms=geoms)
        doc = pymupdf.open("pdf", source)
        s = dpi / 72.0
        for page in gt.pages:
            pix = doc[page.index].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY, annots=False)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
            idx.shapes[page.index] = gray.shape
            idx.words[page.index] = {}
            for w in page.words:
                poly = as_poly(w.poly)
                x0, y0, x1, y1 = bbox(poly)
                c0, r0 = max(int(math.floor(x0 * s)) - 1, 0), max(int(math.floor(y0 * s)) - 1, 0)
                c1 = min(int(math.ceil(x1 * s)) + 1, gray.shape[1])
                r1 = min(int(math.ceil(y1 * s)) + 1, gray.shape[0])
                if c1 <= c0 or r1 <= r0:
                    continue
                origin = (c0 / s, r0 / s)
                pm = rasterize([poly], (r1 - r0, c1 - c0), s, origin=origin)
                win = gray[r0:r1, c0:c1].astype(np.int16)
                vals = win[pm]
                ink = np.zeros_like(pm)
                if vals.size:
                    bg = np.median(vals)
                    contrast = np.abs(win - bg)
                    thr = max(25.0, 0.35 * float(contrast[pm].max()))
                    ink = pm & (contrast > thr)
                if not ink.any():
                    ink = pm  # faint or tiny word: fall back to its footprint
                cells = ([as_poly(c) for c in w.char_polys] if w.char_polys
                         else oriented_cells(poly, max(1, len(w.text)), w.angle))
                char_ink = [ink & rasterize([c], pm.shape, s, origin=origin) for c in cells]
                idx.words[page.index][w.id] = _WordInk(w.id, r0, c0, ink, char_ink)
        return idx

    # -- frames ---------------------------------------------------------------------

    def to_display(self, page: int, poly, frame: Frame | str, dpi: float | None) -> np.ndarray:
        return self.geoms[page].transform(as_poly(poly), Frame(frame), Frame.DISPLAY_PT, dpi)

    def region_mask(self, page: int, polys: list[np.ndarray]) -> np.ndarray:
        return rasterize(polys, self.shapes[page], self.dpi / 72.0)

    # -- coverage ---------------------------------------------------------------------

    def coverage(self, page: int, mask: np.ndarray) -> dict[str, float]:
        """Fraction of each word's ink covered by ``mask`` (only words with any coverage)."""
        out = {}
        for wid, wi in self.words.get(page, {}).items():
            h, w = wi.ink.shape
            sub = mask[wi.r0:wi.r0 + h, wi.c0:wi.c0 + w]
            if sub.shape != wi.ink.shape:
                continue
            hit = int((sub & wi.ink).sum())
            if hit:
                out[wid] = hit / max(wi.total, 1)
        return out

    def char_coverage(self, page: int, mask: np.ndarray, word_id: str) -> list[float]:
        wi = self.words.get(page, {}).get(word_id)
        if wi is None:
            return []
        h, w = wi.ink.shape
        sub = mask[wi.r0:wi.r0 + h, wi.c0:wi.c0 + w]
        return [float((sub & ci).sum()) / max(int(ci.sum()), 1) if ci.any() else 1.0 for ci in wi.char_ink]

    def select(self, page: int, polys: list[np.ndarray], theta: float = DEFAULT_THETA) -> set[str]:
        if not polys:
            return set()
        cov = self.coverage(page, self.region_mask(page, polys))
        return {wid for wid, f in cov.items() if f >= theta}
