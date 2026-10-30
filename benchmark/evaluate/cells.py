"""Character cells and pattern-survival statistics (PLAN.md §5.2).

A *cell* is one GT character (exact box for born-digital words, an oriented split of
the word polygon otherwise) or a slice of a non-text region (signature, avatar). For
each cell the original and output renderings are compared on the same pixels:

- ``ncc``  normalised cross-correlation of the two patterns (shape survives?)
- ``amp``  output/original contrast ratio (is what survives actually visible?)

A black box over black text gives a uniform output (ncc 0) and counts as obscured,
unlike a naive "is it still dark" test. A 90%-opaque box keeps ncc high but amp low,
which is *latent* content: invisible at a glance, recoverable by contrast stretching.

Character identity (eval-0.4). The band test alone can be fooled by a different glyph that
lines up with the original's strokes in two bands (a fake typeset in the original's font, size
and place). A word character's band evidence therefore stands only when the character as a whole
is still the original glyph (see :meth:`PageCells.stats` and docs/metrics.md, "Character
identity"). Region slices (signatures, marks) are not gated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np

from ..geometry import as_poly, bbox, oriented_cells, rasterize
from ..schemas import GTPage
from .config import EvalConfig


BANDS = 4  # default sub-cells across the text line (EvalConfig.bands); calibrated in docs/metrics.md
SUBBANDS = 16  # default identity sub-bands across the line per character (EvalConfig.identity_subbands)

Unit = tuple[str, int]  # (word or region id, character / slice index)


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
    # eval-0.4: identity and presence of the characters of the words the call judged, and, for each
    # character the band test flagged, what let its evidence stand: "identity", "presence", "edge" or
    # "run", or "" when the gate dropped it (its bands then count as new ink).
    identity: dict[Unit, float] = field(default_factory=dict)
    presence: dict[Unit, float] = field(default_factory=dict)
    gate: dict[Unit, str] = field(default_factory=dict)


@dataclass
class CharacterIdentity:
    """Identity and presence of word characters between an original and an output render.

    ``num`` and ``den`` hold, per variant, the identity sums of each (character, band), so that
    identity can be pooled over any subset of a character's bands or over a run of characters."""

    rows: dict[Unit, int]  # character -> row of the arrays below
    informative: np.ndarray  # [characters]: has an informative band
    identity: np.ndarray  # [characters]: the largest |.| over the variants, sign kept
    presence: np.ndarray  # [characters]
    num: np.ndarray  # [variants, characters, bands]
    den: np.ndarray

    def of(self, u: Unit) -> float:
        r = self.rows.get(u)
        return float(self.identity[r]) if r is not None and self.informative[r] else 0.0

    def presence_of(self, u: Unit) -> float:
        r = self.rows.get(u)
        return float(self.presence[r]) if r is not None and self.informative[r] else 0.0

    def pooled(self, units: list[Unit], bands) -> float:
        """Identity over the given bands of the given characters together (largest |.| over the
        variants, sign kept)."""
        rows = [self.rows[u] for u in units if u in self.rows]
        bands = sorted(bands)
        if not rows or not bands:
            return 0.0
        num = self.num[:, rows][:, :, bands].sum(axis=(1, 2))
        den = self.den[:, rows][:, :, bands].sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(den > 1e-9, num / den, 0.0)
        return float(v[int(np.argmax(np.abs(v)))])


@dataclass
class _IdentityIndex:
    """The word cells of a page as flat pixel arrays, grouped by word (built once per page)."""

    rows: dict[Unit, int]
    unit_word: np.ndarray  # [characters] -> word number
    words: list[str]
    word_span: np.ndarray  # [words, 2]: first and last+1 pixel of each word's cells
    cell: np.ndarray  # [word cells]: index into PageCells.cells
    cell_unit: np.ndarray  # [word cells] -> character row
    cell_size: np.ndarray  # [word cells]: pixel count
    rr: np.ndarray  # [pixels]
    cc: np.ndarray
    pix_cell: np.ndarray  # [pixels] -> word cell number
    pix_unit: np.ndarray  # [pixels] -> character row
    pix_word: np.ndarray  # [pixels] -> word number
    key: np.ndarray  # [pixels]: character row * sub-bands + identity sub-band


# The originals' blurred renderings for character identity, most recent first (PageCells._originals).
_ORIGINALS: list[tuple] = []
_ORIGINALS_KEPT = 4


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
    bands: int = BANDS
    subbands: int = SUBBANDS
    # identity sub-band of each pixel of a word cell (empty for region cells)
    subs: list[np.ndarray] = field(default_factory=list)
    # Characters of shared structure the policy lets a fake keep (eval-0.4, set by the evaluator): they hold
    # nothing identifying, so the run clause never pools them with identifying characters (a kept postcode
    # prefix would otherwise carry a fake digit beside it). The run rule still counts them.
    apart: frozenset = field(default_factory=frozenset)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(cls, page: GTPage, dpi: float, shape: tuple[int, int], bands: int = BANDS,
              subbands: int = SUBBANDS) -> "PageCells":
        pc = cls(page=page, dpi=dpi, shape=shape, bands=bands, subbands=bands * max(1, subbands // bands))
        per_band = pc.subbands // bands
        for w in page.words:
            if w.char_polys:
                polys = [as_poly(p) for p in w.char_polys]
            else:
                polys = oriented_cells(as_poly(w.poly), max(1, len(w.text)), w.angle)
            idxs = []
            for k, p in enumerate(polys):
                for b, band in enumerate(oriented_cells(p, bands, w.angle + 90.0)):
                    i = pc._add(Cell(w.id, None, k, b, band))
                    pc.subs[i] = pc._split(i, w.angle, per_band)
                    idxs.append(i)
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
            rr = cc = np.zeros(0, dtype=np.int32)
        else:
            mask = rasterize([cell.poly], (r1 - r0, c1 - c0), s, origin=(c0 / s, r0 / s))
            rr, cc = np.nonzero(mask)
            # 32-bit pixel indices: a page keeps these for every cell until its document is done
            rr, cc = (rr + r0).astype(np.int32), (cc + c0).astype(np.int32)
        self.cells.append(cell)
        self.rows.append(rr)
        self.cols.append(cc)
        self.subs.append(np.zeros(0, dtype=np.int16))
        return len(self.cells) - 1

    def _split(self, i: int, angle: float, per_band: int) -> np.ndarray:
        """Identity sub-band of each pixel of word cell ``i``: its pixel centres binned across the
        line into ``per_band`` equal strips of the band polygon. Pixel for pixel the cells a build
        with ``bands * per_band`` bands would give, without rasterising them again."""
        c = self.cells[i]
        if self.rows[i].size == 0:
            return np.zeros(0, dtype=np.int16)
        t = math.radians(angle + 90.0)
        v = np.array([math.cos(t), math.sin(t)])
        proj = c.poly @ v
        lo, hi = float(proj.min()), float(proj.max())
        p = self.points(i) @ v
        j = np.clip(np.floor((p - lo) / max(hi - lo, 1e-9) * per_band).astype(np.int64), 0, per_band - 1)
        return (c.band * per_band + j).astype(np.int16)

    def points(self, i: int) -> np.ndarray:
        """Display-pt coordinates of the pixel centres of cell ``i``."""
        s = self.dpi / 72.0
        return np.column_stack([(self.cols[i] + 0.5) / s, (self.rows[i] + 0.5) / s])

    # -- the band test ------------------------------------------------------------------------

    def band_stats(self, original: np.ndarray, output: np.ndarray, cfg: EvalConfig) -> CellStats:
        """The band test on its own (eval-0.3): per cell, is the original pattern present?"""
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

    def stats(self, original: np.ndarray, output: np.ndarray, cfg: EvalConfig,
              strict_words: set[str] | frozenset[str] | None = None) -> CellStats:
        """The band test, then the character identity gate (eval-0.4).

        ``strict_words`` are the words held to the MUST/MAY thresholds (None: every word). For their
        characters a second band test first looks at the bands the first did not flag with the output's
        near-black ink left out (a fake drawn over the original). Then a word character's visible and
        latent bands stand only if (a) its identity reaches ``identity_min`` (strict words) or
        ``identity_min_kept`` (other words), plus ``identity_digit_extra`` for a digit; or (b) it lies
        in a run of ``overprint_run`` characters of its word (the whole word if shorter) whose presence
        reaches ``overprint_presence_min`` (an original under a fake drawn over it); or, for strict
        characters only, (c) the edge clause: at least two flagged bands, contiguous and touching the
        top or bottom band, whose pooled identity reaches ``identity_edge_min`` (a box shifted half a
        line over the original); or (d) the run clause: a flagged band shared by a run of at least
        max(2, run_needed(word length)) consecutive characters whose identity pooled over that band
        along the run reaches ``identity_run_min`` (both plus the digit extra). Dropped bands count as
        new ink. Region slices are not gated."""
        st = self.band_stats(original, output, cfg)
        masks = _OutputMasks.of(output, self.dpi, cfg)
        st = self._ink_band_test(st, original, output, cfg, strict_words, masks)
        flagged: dict[Unit, set[int]] = {}
        for i in np.flatnonzero(st.visible | st.latent):
            c = self.cells[i]
            if c.word_id is not None:
                flagged.setdefault((c.word_id, c.index), set()).add(c.band)
        if not flagged:
            return st
        ci = self.character_identity(original, output, st.informative, cfg, words={u[0] for u in flagged},
                                     masks=masks)
        gate = self.gate(ci, flagged, strict_words, cfg)
        drop = np.zeros(len(self.cells), bool)
        for u, why in gate.items():
            if not why:
                drop[[i for i in self.word_cells.get(u[0], []) if self.cells[i].index == u[1]]] = True
        identity = {u: float(ci.identity[r]) for u, r in ci.rows.items() if ci.informative[r]}
        presence = {u: float(ci.presence[r]) for u, r in ci.rows.items() if ci.informative[r]}
        return CellStats(st.informative, st.visible & ~drop, st.latent & ~drop,
                         st.new_ink | ((st.visible | st.latent) & drop), st.ncc, st.amp, identity, presence, gate)

    def _ink_band_test(self, st: CellStats, original: np.ndarray, output: np.ndarray, cfg: EvalConfig,
                       strict_words, masks: "_OutputMasks") -> CellStats:
        """The band test again on the informative bands of strict characters it did not flag, with the
        output's near-black ink (darker than ``identity_ink_level`` x paper, dilated) left out: an
        original under a translucent box with a fake typeset on top shows between the fake's strokes.
        At least half of the band's pixels must remain."""
        vis, lat, new_ink = st.visible.copy(), st.latent.copy(), st.new_ink.copy()
        for i in np.flatnonzero(st.informative & ~st.visible & ~st.latent):
            c = self.cells[i]
            if c.word_id is None or (strict_words is not None and c.word_id not in strict_words):
                continue
            rr, cc = self.rows[i], self.cols[i]
            keep = ~masks.ink[rr, cc]
            if keep.sum() < max(4, 0.5 * keep.size) or keep.all():
                continue
            o = original[rr, cc][keep].astype(np.float64)
            x = output[rr, cc][keep].astype(np.float64)
            so, sx = o.std(), x.std()
            if so < cfg.cell_min_orig_std or sx < 1e-6:
                continue
            ncc = float(np.mean((o - o.mean()) * (x - x.mean())) / (so * sx))
            if abs(ncc) >= cfg.cell_ncc and sx / so >= cfg.cell_amp_visible and sx >= cfg.cell_min_visible_std:
                vis[i], new_ink[i] = True, False
            elif abs(ncc) >= cfg.cell_ncc and sx >= cfg.cell_latent_noise_std:
                lat[i], new_ink[i] = True, False
        return CellStats(st.informative, vis, lat, new_ink, st.ncc, st.amp)

    # -- a word at another place (eval-0.4: moved words) --------------------------------------

    def word_fraction_at(self, word_id: str, original: np.ndarray, output: np.ndarray, s: float, tx: float,
                         ty: float, cfg: EvalConfig, gated: bool = False, strict_words=None,
                         chars: tuple[int, int] | None = None) -> float | None:
        """Share of the word's characters (those in [a, b) with ``chars``) the band test finds when the
        output is read at output_px = s * original_px + (tx, ty), sampled bilinearly (None: no
        informative character).

        With ``gated`` the character identity gate of :meth:`stats` judges the flagged characters too
        (``strict_words`` as there), on the output resampled to the original's grid around the word.
        At the identity transform the band-only share equals the in-place test's."""
        idxs = self.word_cells.get(word_id, [])
        if chars is not None:
            idxs = [i for i in idxs if chars[0] <= self.cells[i].index < chars[1]]
        if not idxs:
            return None
        inf, vis = self._band_test_at(idxs, original, output, s, tx, ty, cfg)
        if gated and vis.any():
            ci = self._identity_at(word_id, idxs, original, output, s, tx, ty, inf, cfg)
            flagged: dict[Unit, set[int]] = {}
            for j, i in enumerate(idxs):
                if vis[j]:
                    flagged.setdefault((word_id, self.cells[i].index), set()).add(self.cells[i].band)
            gate = self.gate(ci, flagged, strict_words, cfg)
            vis = vis & ~np.array([gate.get((word_id, self.cells[i].index), "x") == "" for i in idxs], bool)
        sub = SimpleNamespace(cells=[self.cells[i] for i in idxs])
        flags = unit_flags(sub, inf, vis)
        units = [u for u in flags if u[0] == word_id]
        return sum(flags[u] for u in units) / len(units) if units else None

    def char_flags_at(self, word_id: str, ks, original: np.ndarray, output: np.ndarray, s: float, tx: float,
                      ty: float, cfg: EvalConfig) -> dict[int, bool]:
        """For single known characters (a postcode's prefix digits): is each character ``k`` of ``ks``
        the original glyph at the transform? At least two of its informative bands visible (all when it
        has fewer) and identity at the MUST threshold (identity_min, plus identity_digit_extra for a
        digit); no run rule and no clause."""
        idxs = [i for i in self.word_cells.get(word_id, []) if self.cells[i].index in ks]
        if not idxs:
            return {}
        inf, vis = self._band_test_at(idxs, original, output, s, tx, ty, cfg)
        if not inf.any():
            return {}
        ci = self._identity_at(word_id, idxs, original, output, s, tx, ty, inf, cfg)
        text = self._word_texts().get(word_id, "")
        out: dict[int, bool] = {}
        for k in ks:
            cl = [j for j, i in enumerate(idxs) if self.cells[i].index == k and inf[j]]
            if not cl:
                continue
            need = cfg.identity_min + (cfg.identity_digit_extra if k < len(text) and text[k].isdigit() else 0.0)
            out[k] = int(vis[cl].sum()) >= min(2, len(cl)) and abs(ci.of((word_id, k))) >= need
        return out

    def _band_test_at(self, idxs: list[int], original: np.ndarray, output: np.ndarray, s: float, tx: float,
                      ty: float, cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray]:
        """(informative, visible) of the cells ``idxs`` with the output read at the transform."""
        inf = np.zeros(len(idxs), bool)
        vis = np.zeros(len(idxs), bool)
        for j, i in enumerate(idxs):
            rr, cc = self.rows[i], self.cols[i]
            if rr.size < 4:
                continue
            o = original[rr, cc].astype(np.float64)
            x = bilinear(output, s * cc + tx, s * rr + ty)
            so, sx = o.std(), x.std()
            if so < cfg.cell_min_orig_std:
                continue
            inf[j] = True
            ncc = 0.0 if sx < 1e-6 else float(np.mean((o - o.mean()) * (x - x.mean())) / (so * sx))
            vis[j] = abs(ncc) >= cfg.cell_ncc and sx / so >= cfg.cell_amp_visible and sx >= cfg.cell_min_visible_std
        return inf, vis

    def _identity_at(self, word_id: str, idxs: list[int], original: np.ndarray, output: np.ndarray, s: float,
                     tx: float, ty: float, inf: np.ndarray, cfg: EvalConfig) -> CharacterIdentity:
        """Character identity of the cells ``idxs`` of a word with the output read at the transform: the
        output resampled to the original's grid around them (paper elsewhere), then
        :meth:`character_identity` with its masks taken on that crop."""
        paper = self._cache.get(("paper", id(output)))
        if paper is None:
            paper = _paper(output)
            self._cache[("paper", id(output))] = paper
        kb = max(3, int(round(cfg.identity_box_pt * self.dpi / 72.0)))
        m = kb + int(cfg.identity_offset_px) + int(cfg.identity_ink_dilate_px) + 2
        rr = np.concatenate([self.rows[i] for i in idxs])
        cc = np.concatenate([self.cols[i] for i in idxs])
        h, w = self.shape
        r0, r1 = max(int(rr.min()) - m, 0), min(int(rr.max()) + m + 1, h)
        c0, c1 = max(int(cc.min()) - m, 0), min(int(cc.max()) + m + 1, w)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        crop = bilinear(output, s * xx + tx, s * yy + ty).astype(np.float32)
        virtual = np.full(self.shape, paper, np.float32)
        virtual[r0:r1, c0:c1] = crop
        local = _OutputMasks.of(crop, self.dpi, cfg, paper=paper)
        box = np.zeros(self.shape, bool)
        ink = np.zeros(self.shape, bool)
        box[r0:r1, c0:c1] = local.box
        ink[r0:r1, c0:c1] = local.ink
        informative = np.zeros(len(self.cells), bool)
        informative[idxs] = inf
        return self.character_identity(original, virtual, informative, cfg, words={word_id},
                                       masks=_OutputMasks(paper, box, ink))

    # -- character identity -------------------------------------------------------------------

    def _identity_index(self) -> _IdentityIndex:
        idx = self._cache.get("identity_index")
        if idx is not None:
            return idx
        rows: dict[Unit, int] = {}
        unit_word: list[int] = []
        words: list[str] = []
        spans: list[list[int]] = []
        cell, cell_unit, rr, cc, sub, pix_cell = [], [], [], [], [], []
        n_pix = 0
        for i, c in enumerate(self.cells):
            if c.word_id is None:
                continue
            if not words or words[-1] != c.word_id:  # a word's cells are added together
                words.append(c.word_id)
                spans.append([n_pix, n_pix])
            u = (c.word_id, c.index)
            if u not in rows:
                rows[u] = len(rows)
                unit_word.append(len(words) - 1)
            cell.append(i)
            cell_unit.append(rows[u])
            rr.append(self.rows[i])
            cc.append(self.cols[i])
            sub.append(self.subs[i])
            pix_cell.append(np.full(self.rows[i].size, len(cell) - 1, np.int32))
            n_pix += self.rows[i].size
            spans[-1][1] = n_pix

        # Per-pixel arrays in 32 bits: a page keeps its index until its document is done (eval-0.4 fix
        # round 1: 64-bit arrays made peak memory grow about 115 MB per dense page).
        def cat(a):
            return np.concatenate(a).astype(np.int32) if a else np.zeros(0, np.int32)

        cell_unit_a = np.array(cell_unit, np.int32)
        pix_cell_a = cat(pix_cell)
        pix_unit = cell_unit_a[pix_cell_a] if pix_cell_a.size else np.zeros(0, np.int32)
        unit_word_a = np.array(unit_word, np.int32)
        idx = _IdentityIndex(
            rows=rows, unit_word=unit_word_a, words=words, word_span=np.array(spans, np.int64).reshape(-1, 2),
            cell=np.array(cell, np.int64), cell_unit=cell_unit_a,
            cell_size=np.array([self.rows[i].size for i in cell], np.int64), rr=cat(rr), cc=cat(cc),
            pix_cell=pix_cell_a, pix_unit=pix_unit,
            pix_word=unit_word_a[pix_unit] if pix_unit.size else np.zeros(0, np.int32),
            key=(pix_unit * self.subbands + cat(sub)).astype(np.int32) if pix_unit.size else np.zeros(0, np.int32))
        self._cache["identity_index"] = idx
        return idx

    def _originals(self, original: np.ndarray, cfg: EvalConfig) -> tuple[float, list[np.ndarray]]:
        """The original's paper level and its renderings for identity: as rendered, then blurred
        (Gaussian, identity_blur_px). Kept for the last few originals seen across pages (one per render
        configuration), not per page: blurred copies of every page made peak memory grow with the
        document."""
        key = (id(original), tuple(cfg.identity_blur_px))
        for k, (got_key, got_orig, paper, blurred) in enumerate(_ORIGINALS):
            if got_key == key and got_orig is original:
                _ORIGINALS.insert(0, _ORIGINALS.pop(k))
                return paper, blurred
        blurred = [original] + [cv2.GaussianBlur(original, (0, 0), s) for s in cfg.identity_blur_px if s > 0]
        paper = _paper(original)
        _ORIGINALS.insert(0, (key, original, paper, blurred))
        del _ORIGINALS[_ORIGINALS_KEPT:]
        return paper, blurred

    def character_identity(self, original: np.ndarray, output: np.ndarray, informative: np.ndarray,
                           cfg: EvalConfig, words: set[str] | None = None,
                           masks: "_OutputMasks | None" = None) -> CharacterIdentity:
        """Identity and presence of the characters of ``words`` (None: every word) between two
        renders on the same pixel grid.

        identity: contrast-weighted mean NCC over the character's identity sub-bands,
        sum n*cov / sum n*so*sx over the sub-bands with original contrast (std >= cell_min_orig_std); a
        sub-band the output leaves uniform drops out. The largest |.| over 18 variants: 2 offsets (none,
        and the word's best whole-pixel offset within identity_offset_px by TM_CCOEFF_NORMED of its
        informative box) x 3 pixel sets (all; box pixels left out: output darker than half the paper
        level, opened with an identity_box_pt square; near-black ink left out: output darker than
        identity_ink_level x paper, dilated by identity_ink_dilate_px) x 3 originals (as rendered and
        blurred by identity_blur_px; the output is never blurred).
        presence: share of the original's ink present in the output, over the character's bands with
        output contrast (std >= cell_min_visible_std), the larger of the two offsets."""
        idx = self._identity_index()
        masks = masks or _OutputMasks.of(output, self.dpi, cfg)
        n_units, nb = len(idx.rows), self.bands
        per = self.subbands
        empty = CharacterIdentity(idx.rows, np.zeros(n_units, bool), np.zeros(n_units), np.zeros(n_units),
                                  np.zeros((0, n_units, nb)), np.zeros((0, n_units, nb)))
        if not n_units:
            return empty
        sel_words = (range(len(idx.words)) if words is None
                     else [k for k, w in enumerate(idx.words) if w in words])
        if not sel_words:
            return empty
        pix = np.concatenate([np.arange(*idx.word_span[k]) for k in sel_words])
        rr, cc = idx.rr[pix], idx.cc[pix]
        pix_cell, pix_unit, pix_word, key = idx.pix_cell[pix], idx.pix_unit[pix], idx.pix_word[pix], idx.key[pix]
        inf_cell = informative[idx.cell]
        unit_inf = np.zeros(n_units, bool)
        np.logical_or.at(unit_inf, idx.cell_unit, inf_cell)
        sel_units = np.zeros(n_units, bool)
        sel_units[np.unique(pix_unit)] = True
        unit_inf &= sel_units
        h, w = output.shape
        # the word's best whole-pixel offset (the output is already registered: offset 0 is the first)
        offs = np.zeros((len(idx.words), 2), np.int64)
        pix_inf = inf_cell[pix_cell]
        r_off = int(cfg.identity_offset_px)
        if r_off > 0:
            for k in sel_words:
                a, b = idx.word_span[k]
                m = pix_inf[np.searchsorted(pix, a):np.searchsorted(pix, b)]
                if m.any():
                    sl = slice(np.searchsorted(pix, a), np.searchsorted(pix, b))
                    offs[k] = _best_offset(original, output, rr[sl][m], cc[sl][m], r_off)
        paper_o, originals = self._originals(original, cfg)
        n_fine = n_units * per
        num_all, den_all = [], []
        best = np.zeros(n_units)
        best_pr = np.zeros(n_units)
        io = np.clip((paper_o - original[rr, cc].astype(np.float64)) / paper_o, 0, 1)
        n_cells = len(idx.cell)
        for shifted in (False, True):
            if shifted and not offs[sel_words].any():
                break
            if shifted:
                r2 = np.clip(rr + offs[pix_word, 1], 0, h - 1)
                c2 = np.clip(cc + offs[pix_word, 0], 0, w - 1)
            else:
                r2, c2 = rr, cc
            x = output[r2, c2].astype(np.float64)
            pixel_sets = [None, (~masks.box[r2, c2]).astype(np.float64), (~masks.ink[r2, c2]).astype(np.float64)]
            for wts in pixel_sets:
                def bc(v, wts=wts):
                    return np.bincount(key, weights=v if wts is None else v * wts, minlength=n_fine)
                n = bc(np.ones_like(x))
                sx, sxx = bc(x), bc(x * x)
                for o_img in originals:
                    o = o_img[rr, cc].astype(np.float64)
                    so, soo, sox = bc(o), bc(o * o), bc(o * x)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mo, mx = so / n, sx / n
                        vo = soo / n - mo * mo
                        vx = sxx / n - mx * mx
                        cov = sox / n - mo * mx
                    ok = (n >= 4) & (vo >= cfg.cell_min_orig_std ** 2)
                    num = np.where(ok, n * cov, 0.0).reshape(n_units, nb, per // nb).sum(axis=2)
                    den = np.where(ok, n * np.sqrt(np.clip(vo, 0, None)) * np.sqrt(np.clip(vx, 0, None)), 0.0)
                    den = den.reshape(n_units, nb, per // nb).sum(axis=2)
                    num_all.append(num)
                    den_all.append(den)
                    a, b = num.sum(axis=1), den.sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        v = np.where(b > 1e-9, a / b, 0.0)
                    best = np.where(np.abs(v) > np.abs(best), v, best)
            # presence over the bands with output contrast
            s1 = np.bincount(pix_cell, weights=x, minlength=n_cells)
            s2 = np.bincount(pix_cell, weights=x * x, minlength=n_cells)
            with np.errstate(invalid="ignore", divide="ignore"):
                m_ = s1 / idx.cell_size
                sd = np.sqrt(np.clip(s2 / idx.cell_size - m_ * m_, 0, None))
            live = ((idx.cell_size >= 4) & (sd >= cfg.cell_min_visible_std))[pix_cell]
            ix = np.clip((masks.paper - x) / masks.paper, 0, 1)
            tot = np.bincount(pix_unit, weights=np.where(live, io, 0.0), minlength=n_units)
            got = np.bincount(pix_unit, weights=np.where(live, np.minimum(ix, io), 0.0), minlength=n_units)
            with np.errstate(invalid="ignore", divide="ignore"):
                best_pr = np.maximum(best_pr, np.where(tot > 1e-9, got / tot, 0.0))
        return CharacterIdentity(idx.rows, unit_inf, np.where(unit_inf, best, 0.0), np.where(unit_inf, best_pr, 0.0),
                                 np.stack(num_all), np.stack(den_all))

    def _word_texts(self) -> dict[str, str]:
        t = self._cache.get("texts")
        if t is None:
            t = self._cache["texts"] = {w.id: w.text for w in self.page.words}
        return t

    def _word_chars(self) -> dict[str, list[int]]:
        ch = self._cache.get("chars")
        if ch is None:
            ch = {}
            for u in self._identity_index().rows:
                ch.setdefault(u[0], []).append(u[1])
            ch = self._cache["chars"] = {w: sorted(ks) for w, ks in ch.items()}
        return ch

    def gate(self, ci: CharacterIdentity, flagged: dict[Unit, set[int]], strict_words, cfg: EvalConfig
             ) -> dict[Unit, str]:
        """For each flagged character (unit -> its flagged bands), what lets its band evidence stand:
        "identity", "presence", "edge" or "run", or "" (see :meth:`stats`)."""
        texts, chars = self._word_texts(), self._word_chars()

        def strict(w: str) -> bool:
            return strict_words is None or w in strict_words

        def extra(u: Unit) -> float:
            t = texts.get(u[0], "")
            return cfg.identity_digit_extra if u[1] < len(t) and t[u[1]].isdigit() else 0.0

        overprinted: set[Unit] = set()
        for w in {u[0] for u in flagged}:
            ks = chars.get(w, [])
            need = min(cfg.overprint_run, max(len(ks), 1))
            run: list[int] = []
            for k in ks + [None]:
                present = k is not None and ci.presence_of((w, k)) >= cfg.overprint_presence_min
                if present and (not run or k == run[-1] + 1):
                    run.append(k)
                    continue
                if len(run) >= need:
                    overprinted.update((w, kk) for kk in run)
                run = [k] if present else []
        out: dict[Unit, str] = {}
        for u, bs in flagged.items():
            if not bs:
                continue
            x = extra(u)
            if abs(ci.of(u)) >= (cfg.identity_min if strict(u[0]) else cfg.identity_min_kept) + x:
                out[u] = "identity"
            elif u in overprinted:
                out[u] = "presence"
            elif not strict(u[0]):
                out[u] = ""
            elif len(bs) >= 2 and _edge(bs, self.bands) and abs(ci.pooled([u], bs)) >= cfg.identity_edge_min + x:
                out[u] = "edge"
            elif self._run_clause(ci, flagged, u, chars.get(u[0], []), cfg.identity_run_min + x, self.apart):
                out[u] = "run"
            else:
                out[u] = ""
        return out

    @staticmethod
    def _run_clause(ci: CharacterIdentity, flagged: dict[Unit, set[int]], u: Unit, ks: list[int],
                    threshold: float, apart: frozenset = frozenset()) -> bool:
        """Is a band of ``u`` flagged along a run of consecutive characters of its word long enough
        (max(2, run_needed(word length)), the whole word if shorter), with the identity pooled over that
        band along the run reaching ``threshold``? The longest run counts, then the higher identity. A run
        holds only shared characters (``apart``) or only identifying ones."""
        w, k = u
        side = u in apart
        best_v, best_len = 0.0, 0
        for b in sorted(flagged.get(u, ())):
            run = [k]
            j = k - 1
            while b in flagged.get((w, j), ()) and ((w, j) in apart) == side:
                run.insert(0, j)
                j -= 1
            j = k + 1
            while b in flagged.get((w, j), ()) and ((w, j) in apart) == side:
                run.append(j)
                j += 1
            v = ci.pooled([(w, kk) for kk in run], [b])
            if (len(run), abs(v)) > (best_len, abs(best_v)):
                best_v, best_len = v, len(run)
        n = len(ks) or 1
        return best_len >= min(max(2, run_needed(n)), n) and abs(best_v) >= threshold


def _edge(bands: set[int], n: int) -> bool:
    """Do the bands form one contiguous run touching the top or the bottom band?"""
    s = sorted(bands)
    if not s or s != list(range(s[0], s[-1] + 1)):
        return False
    return s[0] == 0 or s[-1] == n - 1


def _paper(img: np.ndarray) -> float:
    """Paper level of a render: its 95th-percentile grey (at least 64)."""
    return max(float(np.percentile(img, 95)), 64.0)


@dataclass
class _OutputMasks:
    """Pixel masks of an output render used by the identity gate."""

    paper: float
    box: np.ndarray  # dark pixels (below half the paper level) opened with an identity_box_pt square
    ink: np.ndarray  # near-black pixels (below identity_ink_level x paper), dilated

    @classmethod
    def of(cls, output: np.ndarray, dpi: float, cfg: EvalConfig, paper: float | None = None) -> "_OutputMasks":
        paper = _paper(output) if paper is None else paper
        kb = max(3, int(round(cfg.identity_box_pt * dpi / 72.0)))
        dark = (output < paper / 2).astype(np.uint8)
        box = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((kb, kb), np.uint8)).astype(bool)
        ink = (output < cfg.identity_ink_level * paper).astype(np.uint8)
        if cfg.identity_ink_dilate_px > 0:
            k = 2 * int(cfg.identity_ink_dilate_px) + 1
            ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
        return cls(paper, box, ink.astype(bool))


def _best_offset(original: np.ndarray, output: np.ndarray, rr: np.ndarray, cc: np.ndarray, r: int) -> tuple[int, int]:
    """(dx, dy): the whole-pixel shift within ``r`` px that best matches the original's patch (the
    bounding box of the given pixels) in the output, by |TM_CCOEFF_NORMED|."""
    if rr.size == 0:
        return 0, 0
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    h, w = output.shape
    tpl = original[r0:r1, c0:c1].astype(np.float32)
    if tpl.std() < 1e-3:
        return 0, 0
    sr0, sr1 = max(r0 - r, 0), min(r1 + r, h)
    sc0, sc1 = max(c0 - r, 0), min(c1 + r, w)
    win = output[sr0:sr1, sc0:sc1].astype(np.float32)
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return 0, 0
    res = np.nan_to_num(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED), nan=0.0)
    mn, mx, lmn, lmx = cv2.minMaxLoc(res)
    loc = lmx if abs(mx) >= abs(mn) else lmn
    return int(loc[0] + sc0 - c0), int(loc[1] + sr0 - r0)


def bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``img`` sampled at fractional pixel coordinates (x a column, y a row); paper white (255)
    outside."""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx, fy = xs - x0, ys - y0
    out = np.full(np.shape(xs), 255.0)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    a, b, fa, fb = x0[ok], y0[ok], fx[ok], fy[ok]
    out[ok] = (img[b, a] * (1 - fa) * (1 - fb) + img[b, a + 1] * fa * (1 - fb)
               + img[b + 1, a] * (1 - fa) * fb + img[b + 1, a + 1] * fa * fb)
    return out


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
