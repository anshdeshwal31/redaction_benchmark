"""Page rendering through independent engines, with caching.

The evaluator always compares original and output rendered by the *same* engine with
the *same* settings, so engine differences (anti-aliasing, one-pixel size rounding)
cancel out. Multiple engines and annotation states guard against output that is only
"redacted" in some viewers (PLAN.md §5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pymupdf
import pypdfium2 as pdfium


@dataclass(frozen=True)
class RenderConfig:
    engine: str  # "mupdf" | "pdfium"
    annots: bool = True

    @property
    def label(self) -> str:
        return f"{self.engine}{'' if self.annots else '-noannots'}"


def render_gray(pdf: bytes, page_index: int, dpi: float, cfg: RenderConfig) -> np.ndarray:
    if cfg.engine == "mupdf":
        doc = pymupdf.open("pdf", pdf)
        s = dpi / 72.0  # explicit matrix: get_pixmap(dpi=) only accepts integers
        pix = doc[page_index].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY, alpha=False,
                                         annots=cfg.annots)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()
    if cfg.engine == "pdfium":
        doc = pdfium.PdfDocument(pdf)
        try:
            page = doc[page_index]
            bitmap = page.render(scale=dpi / 72.0, grayscale=True, draw_annots=cfg.annots, may_draw_forms=cfg.annots)
            arr = bitmap.to_numpy()
            return (arr[..., 0] if arr.ndim == 3 else arr).copy()
        finally:
            doc.close()
    raise ValueError(f"unknown engine {cfg.engine}")


@dataclass
class RenderCache:
    dpi: float
    _store: dict = field(default_factory=dict)

    def get(self, key: str, pdf: bytes, page_index: int, cfg: RenderConfig) -> np.ndarray:
        k = (key, page_index, cfg)
        if k not in self._store:
            self._store[k] = render_gray(pdf, page_index, self.dpi, cfg)
        return self._store[k]


def match_shapes(reference: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, bool]:
    """Crop/pad ``other`` to ``reference``'s shape (renderers can differ by one pixel)."""
    if other.shape == reference.shape:
        return other, False
    h, w = reference.shape
    out = np.full((h, w), 255, dtype=other.dtype)
    hh, ww = min(h, other.shape[0]), min(w, other.shape[1])
    out[:hh, :ww] = other[:hh, :ww]
    big_change = abs(other.shape[0] - h) > 2 or abs(other.shape[1] - w) > 2
    return out, big_change
