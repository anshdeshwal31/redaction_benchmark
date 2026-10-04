"""Coordinate frames and the conversions between them.

Every geometric quantity in the benchmark is expressed in, or converted to, one
canonical frame:

``display_pt``
    PDF points of the page *as displayed*: the CropBox after applying ``/Rotate``,
    origin at the top-left, x to the right, y downwards. Ground truth, candidate
    artifacts and metrics all meet here.

Other frames that appear at the boundaries:

``unrotated_pt``
    CropBox-relative, origin top-left, y downwards, *before* ``/Rotate``. PyMuPDF
    reports extraction results (words, chars, drawings, image placements) in this
    frame and ``add_redact_annot`` accepts it. Verified for all four rotations with
    CropBox offsets in ``tests/test_geometry.py``.
``pdf_user``
    Raw PDF user space: origin bottom-left, y upwards, MediaBox coordinates. Content
    streams are written in this frame.
``image_px``
    Pixels of a rendering of the displayed page at a given DPI (``display_pt`` scaled
    by ``dpi / 72``).

Matrices use the column-vector convention: ``p' = M @ [x, y, 1]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np

Rect = tuple[float, float, float, float]


class Frame(str, Enum):
    DISPLAY_PT = "display_pt"
    UNROTATED_PT = "unrotated_pt"
    PDF_USER = "pdf_user"
    IMAGE_PX = "image_px"


def normalize_rect(rect: Iterable[float]) -> Rect:
    x0, y0, x1, y1 = (float(v) for v in rect)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def normalize_rotation(rotation: int) -> int:
    r = int(rotation) % 360
    if r % 90:
        raise ValueError(f"/Rotate must be a multiple of 90, got {rotation}")
    return r


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _scale(s: float) -> np.ndarray:
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class PageGeometry:
    """Page boxes and rotation, enough to convert between all frames.

    ``mediabox`` and ``cropbox`` are in PDF user space. The crop box is clipped to
    the media box, as required by ISO 32000-1 §14.11.2.
    """

    mediabox: Rect
    cropbox: Rect
    rotation: int = 0

    def __post_init__(self) -> None:
        mb = normalize_rect(self.mediabox)
        cb = normalize_rect(self.cropbox)
        cb = (max(cb[0], mb[0]), max(cb[1], mb[1]), min(cb[2], mb[2]), min(cb[3], mb[3]))
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            raise ValueError(f"crop box {self.cropbox} does not intersect media box {self.mediabox}")
        object.__setattr__(self, "mediabox", mb)
        object.__setattr__(self, "cropbox", cb)
        object.__setattr__(self, "rotation", normalize_rotation(self.rotation))

    @property
    def unrotated_size(self) -> tuple[float, float]:
        return (self.cropbox[2] - self.cropbox[0], self.cropbox[3] - self.cropbox[1])

    @property
    def display_size(self) -> tuple[float, float]:
        w, h = self.unrotated_size
        return (h, w) if self.rotation in (90, 270) else (w, h)

    def pixel_size(self, dpi: float) -> tuple[int, int]:
        """(width, height) in pixels of a MuPDF rendering at ``dpi``.

        MuPDF rounds the page rectangle outwards (``fz_round_rect``), so a partial
        last pixel counts. Other renderers may differ by one pixel; evaluators must
        use the actual array shape and only rely on the shared pixel grid.
        """
        w, h = self.display_size
        s = dpi / 72.0
        return (int(math.ceil(w * s - 0.001)), int(math.ceil(h * s - 0.001)))

    # -- building blocks (each paired with its exact inverse) ------------------

    def _user_to_unrotated(self) -> tuple[np.ndarray, np.ndarray]:
        cx0, _, _, cy1 = self.cropbox
        fwd = np.array([[1.0, 0.0, -cx0], [0.0, -1.0, cy1], [0.0, 0.0, 1.0]])
        inv = np.array([[1.0, 0.0, cx0], [0.0, -1.0, cy1], [0.0, 0.0, 1.0]])
        return fwd, inv

    def _unrotated_to_display(self) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.unrotated_size
        r = self.rotation
        if r == 0:
            return np.eye(3), np.eye(3)
        if r == 90:  # (x, y) -> (h - y, x)
            fwd = np.array([[0.0, -1.0, h], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            inv = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, h], [0.0, 0.0, 1.0]])
            return fwd, inv
        if r == 180:  # (x, y) -> (w - x, h - y)
            fwd = np.array([[-1.0, 0.0, w], [0.0, -1.0, h], [0.0, 0.0, 1.0]])
            return fwd, fwd.copy()
        # r == 270: (x, y) -> (y, w - x)
        fwd = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w], [0.0, 0.0, 1.0]])
        inv = np.array([[0.0, -1.0, w], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return fwd, inv

    def _to_display(self, frame: Frame, dpi: float | None) -> tuple[np.ndarray, np.ndarray]:
        frame = Frame(frame)
        if frame is Frame.DISPLAY_PT:
            return np.eye(3), np.eye(3)
        if frame is Frame.UNROTATED_PT:
            return self._unrotated_to_display()
        if frame is Frame.PDF_USER:
            u_fwd, u_inv = self._user_to_unrotated()
            r_fwd, r_inv = self._unrotated_to_display()
            return r_fwd @ u_fwd, u_inv @ r_inv
        if dpi is None or dpi <= 0:
            raise ValueError("a positive dpi is required for the image_px frame")
        s = dpi / 72.0
        return _scale(1.0 / s), _scale(s)

    def matrix(self, src: Frame | str, dst: Frame | str, dpi: float | None = None) -> np.ndarray:
        """3x3 matrix converting points from ``src`` to ``dst``."""
        src_fwd, _ = self._to_display(Frame(src), dpi)
        _, dst_inv = self._to_display(Frame(dst), dpi)
        return dst_inv @ src_fwd

    def transform(
        self, points: np.ndarray | Iterable, src: Frame | str, dst: Frame | str, dpi: float | None = None
    ) -> np.ndarray:
        """Convert an (N, 2) array of points from ``src`` to ``dst``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        m = self.matrix(src, dst, dpi)
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        return (homo @ m.T)[:, :2]


# -- reading page geometry from files ------------------------------------------


def _inherited(page_obj, key: str):
    """Resolve an inheritable page attribute by walking the /Parent chain."""
    node = page_obj
    seen = 0
    while node is not None and seen < 64:
        if key in node:
            return node[key]
        node = node.get("/Parent")
        seen += 1
    return None


def geometry_from_pikepdf_page(page) -> PageGeometry:
    """Build a :class:`PageGeometry` from a ``pikepdf.Page`` (inheritance resolved)."""
    obj = page.obj
    mediabox = _inherited(obj, "/MediaBox")
    if mediabox is None:
        mediabox = [0, 0, 612, 792]  # ISO 32000 has no default; US Letter is MuPDF's fallback
    cropbox = _inherited(obj, "/CropBox")
    rotation = _inherited(obj, "/Rotate")
    mb = normalize_rect([float(v) for v in mediabox])
    cb = normalize_rect([float(v) for v in cropbox]) if cropbox is not None else mb
    return PageGeometry(mediabox=mb, cropbox=cb, rotation=int(rotation) if rotation is not None else 0)


def load_page_geometries(path: str | Path) -> list[PageGeometry]:
    import pikepdf

    with pikepdf.open(str(path)) as pdf:
        return [geometry_from_pikepdf_page(p) for p in pdf.pages]


def check_against_pymupdf(geometries: list[PageGeometry], doc) -> list[str]:
    """Return mismatches between our geometry and what MuPDF will render.

    An empty list means the frames agree. A non-empty list means the PDF has
    malformed boxes that MuPDF repaired differently; callers should treat the page
    as unreliable rather than silently mis-mapping coordinates.
    """
    problems = []
    if len(geometries) != doc.page_count:
        return [f"page count differs: pikepdf={len(geometries)} mupdf={doc.page_count}"]
    for i, (g, page) in enumerate(zip(geometries, doc)):
        w, h = g.display_size
        if page.rotation != g.rotation:
            problems.append(f"page {i}: rotation {g.rotation} vs mupdf {page.rotation}")
        if abs(page.rect.width - w) > 0.01 or abs(page.rect.height - h) > 0.01:
            problems.append(f"page {i}: display size {(w, h)} vs mupdf {(page.rect.width, page.rect.height)}")
    return problems
