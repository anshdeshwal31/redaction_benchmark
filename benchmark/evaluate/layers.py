"""File-level content survival on one page (L2/L3, PLAN.md §5.3).

These checks ask whether the *original* content still exists in the output file,
whether or not it is visible: text-layer characters at the GT location, image layers
that still carry the original pixels (e.g. under an overlay box, or in one layer of an
MRC stack), and glyph paths that survive under a covering shape.

eval-0.4 adds the page's text in extraction order (:func:`page_text`) and the search for a
word at its displayed place (:func:`find_at_place`), used for searchable kept words and for
leaked text beside the 1 pt zone of a word's GT box.
"""

from __future__ import annotations

import math
import re
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


def slice_in(needle: str, text: str | None, shared: tuple[str, ...] = ()) -> bool:
    """Is the identifying slice of a word with shared characters in ``text``? The text's own copy of each
    shared part (an identifier's or a phone number's prefix, an email's domain) is cut out first, so the
    slice is never found inside the shared part a fake keeps ('04' of '0404' in a fake '0498', 'smith' in
    '@smithlaw.com.au'); a slice of digits counts only as a whole digit run of what is left."""
    n = needle.strip(STRIP).casefold()
    if not text or not n:
        return False
    t = text.casefold()
    for part in shared:
        i = t.find(part.casefold()) if part else -1
        if i >= 0:
            t = t[:i] + " " + t[i + len(part):]
    if n.isdigit():
        return any(m.group() == n for m in re.finditer(r"\d+", t))
    return n in t


def text_leaks(needle: str, got: str | None, cfg: EvalConfig) -> bool:
    """Is the original token recoverable from the extracted text at its location?"""
    if not got or not needle.strip():
        return False
    a, b = needle.casefold(), got.casefold()
    return a in b or fuzz.ratio(a, b) >= cfg.text_ratio


# -- text at its displayed place (eval-0.4) ------------------------------------------

STRIP = ".,;:()"


@dataclass(frozen=True)
class TextChar:
    c: str
    box: tuple[float, float, float, float]  # display pt (x0, y0, x1, y1)
    line: int  # MuPDF text line number: a change is a line break in the extracted text
    seq: int  # extraction order


@dataclass(frozen=True)
class TextRun:
    along: float  # centre of the occurrence minus the word's centre, along the line (pt)
    across: float  # the same across the line (pt)
    width_ratio: float  # length of the occurrence / length of the word, along the line
    first: int  # seq of its first character
    last: int  # seq of its last character


class PageText(list):
    """The characters of one page's text in extraction order (a list of :class:`TextChar`), with
    their box centres and whole tokens worked out once."""

    def __init__(self, chars=()):
        super().__init__(sorted(chars, key=lambda c: c.seq))
        boxes = np.array([c.box for c in self], float).reshape(-1, 4)
        self.mids = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        # whole tokens: maximal runs of characters with no whitespace between them on one text line
        self.tokens: list[list[int]] = []
        self.token_of = np.full(len(self), -1, np.int64)
        prev_line = None
        for k, ch in enumerate(self):
            if ch.c.isspace():
                prev_line = None
                continue
            if prev_line is None or ch.line != prev_line:
                self.tokens.append([])
            self.tokens[-1].append(k)
            self.token_of[k] = len(self.tokens) - 1
            prev_line = ch.line


def page_text(page: pymupdf.Page, geom: PageGeometry, scale: tuple[float, float] = (1.0, 1.0)) -> PageText:
    """Every character of the page's text (any render mode, as :func:`page_chars` reads it) in
    extraction order, boxes in display pt (multiplied by ``scale`` when the output page was resized to
    the source), including the spaces MuPDF's extraction adds for gaps."""
    out: list[TextChar] = []
    seq = line_no = 0
    raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_MEDIABOX_CLIP)
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    p = geom.transform(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), Frame.UNROTATED_PT,
                                       Frame.DISPLAY_PT) * np.asarray(scale)
                    out.append(TextChar(ch["c"], (float(p[:, 0].min()), float(p[:, 1].min()),
                                                  float(p[:, 0].max()), float(p[:, 1].max())), line_no, seq))
                    seq += 1
            line_no += 1
    return PageText(out)


def _frame(poly) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(centre, along unit vector, across unit vector, length, height) of a word polygon whose corners
    run TL, TR, BR, BL in reading order."""
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    u, v = p[1] - p[0], p[3] - p[0]
    length, height = float(np.hypot(*u)), float(np.hypot(*v))
    u = u / length if length > 0 else np.array([1.0, 0.0])
    v = v / height if height > 0 else np.array([0.0, 1.0])
    return p.mean(axis=0), u, v, length, height


def glued_parts(token: str) -> list[tuple[int, int]]:
    """The parts of a token's letter runs that are glued at a change of case: (start, end) of 'Example'
    and 'Street' in 'ExampleStreet,', of 'Faketown' and 'QLD' in 'FaketownQLD4000'. Only runs that split
    give parts, so a whole run ('Parkside', or 'example' in 'jane@example.com') gives none. MuPDF puts no
    space between words drawn less than about 1.5 pt apart, so their letters come out as one token."""
    out: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\W\d_]+", token):
        run, a = m.group(), 0
        cuts = [k for k in range(1, len(run))
                if (run[k - 1].islower() and run[k].isupper())
                or (run[k - 1].isupper() and run[k].isupper() and k + 1 < len(run) and run[k + 1].islower())]
        for b in [*cuts, len(run)] if cuts else ():
            out.append((m.start() + a, m.start() + b))
            a = b
    return out


def token_span(needle: str, token: str, ratio: float) -> tuple[int, int] | None:
    """The part of a whole output token that holds ``needle`` (both stripped of punctuation and case
    folded): a number only as a maximal digit run equal to it ('214' and '142750' do not hold '14');
    anything else as the whole token, equal or with a rapidfuzz ratio of at least ``ratio``, and a word
    of letters also as a maximal letter run equal to it (the state of a glued 'NSW2750', the suburb of
    a word the GT splits between the street line and the locality) or as a part of a letter run glued at
    a change of case (:func:`glued_parts`: 'Example' in 'ExampleStreet,', never 'Park' in 'Parkside')."""
    n = needle.strip(STRIP).casefold()
    t = token.casefold()
    if not n:
        return None
    if n.isdigit():
        for m in re.finditer(r"\d+", t):
            if m.group() == n:
                return m.start(), m.end()
        return None
    lead = len(t) - len(t.lstrip(STRIP))
    core = t.strip(STRIP)
    if core and (core == n or fuzz.ratio(n, core) >= ratio):
        return lead, lead + len(core)
    if n.isalpha():
        for m in re.finditer(r"[^\W\d_]+", t):
            if m.group() == n:
                return m.start(), m.end()
        for a, b in glued_parts(token):
            if t[a:b] == n:
                return a, b
    return None


def find_at_place(poly, needle: str, chars, shift=(0.0, 0.0), line_height: float | None = None,
                  band: float = 0.5, margin: float | None = 0.25, ratio: float = 90.0,
                  whole_token: bool = False, shared: tuple[str, ...] | None = None) -> TextRun | None:
    """Is ``needle`` in the text at the word's displayed place? The nearest accepted occurrence, or None.

    The place is the word's GT polygon moved by ``shift`` (pt): centre C, along-line direction u, length
    L, and H the line height (``line_height``, else the polygon's height). Candidate characters lie
    within ``band`` x H of C across the line and 1.5 L + H along it. Joined in extraction order and case
    folded, they must hold the needle (stripped of .,;:()) as a substring or, failing that, as a window
    of n-1..n+1 characters with a rapidfuzz ratio of at least ``ratio``; with ``whole_token`` an output
    token must hold it (:func:`token_span`); with ``shared`` (the shared parts of a word whose identifying
    slice the needle is) an output token must hold it by :func:`slice_in`. An occurrence is at the place
    when its centre lies within
    ``band`` x H across and L/2 + ``margin`` x H along (``margin`` None: anywhere along the line)."""
    pt = chars if isinstance(chars, PageText) else PageText(chars)
    n = needle.strip(STRIP).casefold()
    if not n or not len(pt):
        return None
    centre, u, v, length, height = _frame(poly)
    h = line_height or height
    centre = centre + np.asarray(shift, dtype=float)
    mid = pt.mids - centre
    al, ac = mid @ u, mid @ v
    keep = np.abs(ac) <= band * h
    if margin is not None:
        keep &= np.abs(al) <= 1.5 * length + h
    idx = [k for k in np.flatnonzero(keep) if not pt[k].c.isspace()]
    if not idx:
        return None
    spans: list[list[int]] = []
    if shared is not None:
        for t in sorted({int(pt.token_of[k]) for k in idx}):
            ids = pt.tokens[t]
            if slice_in(n, "".join(pt[k].c for k in ids), shared):
                spans.append(ids)
    elif whole_token:
        for t in sorted({int(pt.token_of[k]) for k in idx}):
            ids = pt.tokens[t]
            got = token_span(n, "".join(pt[k].c for k in ids), ratio)
            if got is not None:
                spans.append(ids[got[0]:got[1]])
    else:
        hay = "".join(pt[k].c for k in idx).casefold()
        s = hay.find(n)
        while s >= 0:
            spans.append(idx[s:s + len(n)])
            s = hay.find(n, s + 1)
        if not spans:
            m = len(n)
            for w in (m - 1, m, m + 1):
                for s in range(0, max(0, len(hay) - w) + 1) if w >= 1 else ():
                    if fuzz.ratio(n, hay[s:s + w]) >= ratio:
                        spans.append(idx[s:s + w])
    best = None
    for ids in spans:
        corners = np.array([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in (pt[k].box for k in ids)],
                           float).reshape(-1, 2) - centre
        a, c = corners @ u, corners @ v
        along, across = (a.min() + a.max()) / 2, (c.min() + c.max()) / 2
        if abs(across) > band * h or (margin is not None and abs(along) > length / 2 + margin * h):
            continue
        run = TextRun(float(along), float(across), float((a.max() - a.min()) / length) if length else 0.0,
                      pt[ids[0]].seq, pt[ids[-1]].seq)
        if best is None or abs(run.along) < abs(best.along):
            best = run
    return best


def tokens_in_zone(poly, chars, dilate_pt: float = 1.0) -> list[str]:
    """The whole output tokens with a character centre inside the word polygon grown by ``dilate_pt``:
    the token-wise form of :func:`text_in_words` (a token reaching beyond the box stays whole)."""
    pt = chars if isinstance(chars, PageText) else PageText(chars)
    if not len(pt):
        return []
    zone = to_shapely(dilate(as_poly(poly), dilate_pt))
    inside = shapely.contains_xy(zone, pt.mids[:, 0], pt.mids[:, 1])
    hit = sorted({int(pt.token_of[k]) for k in np.flatnonzero(inside) if pt.token_of[k] >= 0})
    return ["".join(pt[k].c for k in pt.tokens[t]) for t in hit]


def token_leaks(needle: str, tokens: list[str], cfg: EvalConfig) -> bool:
    """Is the original token one of the whole output tokens (:func:`token_span`)?"""
    return any(token_span(needle, t, cfg.text_ratio) is not None for t in tokens)


def separated(chars, first: TextRun, second: TextRun) -> bool | None:
    """Do two occurrences read as separate words in the extracted text: whitespace (in the file or added
    for a gap) or a line break between them? None when they are out of order."""
    if second.first <= first.last:
        return None
    by_seq = {c.seq: c for c in chars}
    between = [by_seq[s] for s in range(first.last + 1, second.first) if s in by_seq]
    if any(c.c.isspace() for c in between):
        return True
    a, b = by_seq.get(first.last), by_seq.get(second.first)
    return bool(a and b and a.line != b.line)


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
    and sampled there bilinearly (layer pixel centres at half-integers, edges held). Elsewhere, and
    outside the layer, the layer's paper level. Nearest-pixel sampling made a low-resolution layer
    blocky on the render grid, so the original's characters failed the identity gate and a
    recoverable layer was dropped (75-100 DPI backgrounds are common)."""
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
        fx, fy = uv[:, 0] * layer.width, uv[:, 1] * layer.height
        inside = (fx >= 0) & (fx < layer.width) & (fy >= 0) & (fy < layer.height)
        vals = np.where(inside, _sample_bilinear(g, fx - 0.5, fy - 0.5), out[yy.ravel(), xx.ravel()])
        out[yy.ravel(), xx.ravel()] = vals
    return out


def _sample_bilinear(g: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``g`` sampled bilinearly at fractional pixel coordinates (x a column, y a row, pixel centres
    at integers), coordinates held to the image's edge pixels."""
    h, w = g.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)
    x0 = np.minimum(np.floor(xs).astype(np.int64), max(w - 2, 0))
    y0 = np.minimum(np.floor(ys).astype(np.int64), max(h - 2, 0))
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xs - x0, ys - y0
    gf = g.astype(np.float32, copy=False)
    return (gf[y0, x0] * (1 - fx) * (1 - fy) + gf[y0, x1] * fx * (1 - fy)
            + gf[y1, x0] * (1 - fx) * fy + gf[y1, x1] * fx * fy)


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
    outlines, that is a word with letters or digits and at least ceil(half of them) paths wholly inside
    that box. A word with none, such as a typed "______" signature line, is never outlined: a pen stroke
    wholly inside its box (a flat signature or initials on the line) would otherwise be taken for its
    glyph, and a rule's own glyphs are straight, which vector_survival leaves out anyway."""
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
    alnum = [sum(c.isalnum() for c in w.text) for w in kept]
    outlined = np.array([n > 0 and inside[k] >= math.ceil(0.5 * n) for k, n in enumerate(alnum)], bool)
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
    many paths wholly inside its box (1 pt around) as it has letters and digits; a typed "______" line
    has none, so pen strokes across it or wholly inside its box count. Not covered: a
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
