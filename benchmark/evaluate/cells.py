"""Character cells and pattern-survival statistics (PLAN.md §5.2).

A *cell* is one GT character (exact box for born-digital words, an oriented split of
the word polygon otherwise) or a slice of a non-text region (signature, avatar). For
each cell the original and output renderings are compared on the same pixels:

- ``ncc``  normalised cross-correlation of the two patterns (shape survives?)
- ``amp``  output/original contrast ratio (is what survives actually visible?)

A black box over black text gives a uniform output (ncc 0) and counts as obscured,
unlike a naive "is it still dark" test. A 90%-opaque box keeps ncc high but amp low,
which is *latent* content: invisible at a glance, recoverable by contrast stretching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geometry import as_poly, bbox, oriented_cells, rasterize
from ..schemas import GTPage
from .config import EvalConfig


BANDS = 4  # default sub-cells across the text line (EvalConfig.bands); calibrated in docs/metrics.md


@dataclass
class Cell:
    word_id: str | None
    region_id: str | None
    index: int  # character index within the word, or slice index within the region
    band: int  # 0..BANDS-1 across the line (a box shifted half a line leaves the top band legible)
    poly: np.ndarray


@dataclass
class CellStats:
    informative: np.ndarray
    visible: np.ndarray
    latent: np.ndarray
    new_ink: np.ndarray
    ncc: np.ndarray
    amp: np.ndarray


@dataclass
class PageCells:
    """All cells of one GT page, with their pixel footprints at a given DPI."""

    page: GTPage
    dpi: float
    shape: tuple[int, int]
    cells: list[Cell] = field(default_factory=list)
    word_cells: dict[str, list[int]] = field(default_factory=dict)
    region_cells: dict[str, list[int]] = field(default_factory=dict)
    rows: list[np.ndarray] = field(default_factory=list)
    cols: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def build(cls, page: GTPage, dpi: float, shape: tuple[int, int], bands: int = BANDS) -> "PageCells":
        pc = cls(page=page, dpi=dpi, shape=shape)
        for w in page.words:
            if w.char_polys:
                polys = [as_poly(p) for p in w.char_polys]
            else:
                polys = oriented_cells(as_poly(w.poly), max(1, len(w.text)), w.angle)
            idxs = []
            for k, p in enumerate(polys):
                for b, band in enumerate(oriented_cells(p, bands, w.angle + 90.0)):
                    idxs.append(pc._add(Cell(w.id, None, k, b, band)))
            pc.word_cells[w.id] = idxs
        for r in page.regions:
            poly = as_poly(r.poly)
            x0, y0, x1, y1 = bbox(poly)
            long_, short = max(x1 - x0, y1 - y0), max(1e-6, min(x1 - x0, y1 - y0))
            n = max(1, int(round(long_ / short)))
            angle = 0.0 if (x1 - x0) >= (y1 - y0) else 90.0
            idxs = []
            for k, p in enumerate(oriented_cells(poly, n, angle)):
                for b, band in enumerate(oriented_cells(p, bands, angle + 90.0)):
                    idxs.append(pc._add(Cell(None, r.id, k, b, band)))
            pc.region_cells[r.id] = idxs
        return pc

    def word_char_cells(self, word_id: str, start: int | None = None, end: int | None = None) -> list[int]:
        """Cell indices of a word, optionally restricted to characters [start, end)."""
        idxs = self.word_cells.get(word_id, [])
        if start is None:
            return idxs
        return [i for i in idxs if start <= self.cells[i].index < end]

    def _add(self, cell: Cell) -> int:
        s = self.dpi / 72.0
        x0, y0, x1, y1 = bbox(cell.poly)
        c0 = max(int(math.floor(x0 * s)) - 1, 0)
        r0 = max(int(math.floor(y0 * s)) - 1, 0)
        c1 = min(int(math.ceil(x1 * s)) + 1, self.shape[1])
        r1 = min(int(math.ceil(y1 * s)) + 1, self.shape[0])
        if c1 <= c0 or r1 <= r0:
            rr = cc = np.zeros(0, dtype=np.int64)
        else:
            mask = rasterize([cell.poly], (r1 - r0, c1 - c0), s, origin=(c0 / s, r0 / s))
            rr, cc = np.nonzero(mask)
            rr, cc = rr + r0, cc + c0
        self.cells.append(cell)
        self.rows.append(rr)
        self.cols.append(cc)
        return len(self.cells) - 1

    def points(self, i: int) -> np.ndarray:
        """Display-pt coordinates of the pixel centres of cell ``i``."""
        s = self.dpi / 72.0
        return np.column_stack([(self.cols[i] + 0.5) / s, (self.rows[i] + 0.5) / s])

    def stats(self, original: np.ndarray, output: np.ndarray, cfg: EvalConfig) -> CellStats:
        n = len(self.cells)
        informative = np.zeros(n, bool)
        visible = np.zeros(n, bool)
        latent = np.zeros(n, bool)
        new_ink = np.zeros(n, bool)
        nccs = np.zeros(n)
        amps = np.zeros(n)
        for i in range(n):
            rr, cc = self.rows[i], self.cols[i]
            if rr.size < 4:
                continue
            o = original[rr, cc].astype(np.float64)
            x = output[rr, cc].astype(np.float64)
            so, sx = o.std(), x.std()
            if so < cfg.cell_min_orig_std:
                continue
            informative[i] = True
            ncc = 0.0 if sx < 1e-6 else float(np.mean((o - o.mean()) * (x - x.mean())) / (so * sx))
            amp = sx / so
            nccs[i], amps[i] = ncc, amp
            correlated = abs(ncc) >= cfg.cell_ncc
            if correlated and amp >= cfg.cell_amp_visible and sx >= cfg.cell_min_visible_std:
                visible[i] = True
            elif correlated and sx >= cfg.cell_latent_noise_std:
                latent[i] = True
            elif sx >= cfg.cell_min_visible_std:
                new_ink[i] = True
        return CellStats(informative, visible, latent, new_ink, nccs, amps)


Unit = tuple[str, int]  # (word or region id, character / slice index)


def run_needed(word_length: int) -> int:
    """Consecutive visible characters needed for any of them to count: 1 for initials,
    2 for short words, 3 for words of 6+ characters (where two adjacent capitals of a
    different name can match the original's strokes by chance)."""
    if word_length <= 2:
        return 1
    return 2 if word_length < 6 else 3


def unit_flags(pc: PageCells, informative: np.ndarray, flags: np.ndarray) -> dict[Unit, bool]:
    """Aggregate band evidence to characters (units). Only informative units appear.

    A character counts when (a) at least two of its informative bands are flagged (all of
    them if it has only one, e.g. a dash), or one band is
    flagged together with the same band of a neighbouring character (a box shifted
    half a line leaves whole top bands legible), and (b) it belongs to a run of at
    least two such characters, unless its word has at most two characters (initials).
    Rule (b) suppresses coincidental matches, e.g. a replacement glyph that happens to
    equal the original glyph at the same position. Calibrated in tests/test_evaluator.py.
    """
    bands: dict[Unit, dict[int, bool]] = {}
    length: dict[str, int] = {}
    for i, c in enumerate(pc.cells):
        owner = c.word_id or c.region_id
        length[owner] = max(length.get(owner, 0), c.index + 1)
        if informative[i]:
            bands.setdefault((owner, c.index), {})[c.band] = bool(flags[i])
    char = {}
    for (owner, idx), b in bands.items():
        n = sum(b.values())
        neighbour = any(v and bands.get((owner, idx + d), {}).get(bi, False) for bi, v in b.items() for d in (-1, 1))
        # min(2, informative bands): a glyph with ink in a single band (a dash, a full stop)
        # counts when that band survives.
        char[(owner, idx)] = n >= min(2, len(b)) or (n >= 1 and neighbour)
    out: dict[Unit, bool] = {}
    for (owner, idx), v in char.items():
        need = run_needed(length[owner])
        if not v:
            out[(owner, idx)] = False
            continue
        lo = idx
        while char.get((owner, lo - 1), False):
            lo -= 1
        hi = idx
        while char.get((owner, hi + 1), False):
            hi += 1
        out[(owner, idx)] = hi - lo + 1 >= need
    return out


def correlate(o: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """(ncc, amp, std_o, std_x) of two equally-sized samples."""
    o = o.astype(np.float64)
    x = x.astype(np.float64)
    so, sx = o.std(), x.std()
    if so < 1e-6 or sx < 1e-6:
        return 0.0, (sx / so if so > 1e-6 else 0.0), so, sx
    return float(np.mean((o - o.mean()) * (x - x.mean())) / (so * sx)), sx / so, so, sx
