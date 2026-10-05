"""Split words that PP-OCRv6 glued together ("ElianaGood", "Suite14", "inthe").

The recogniser sometimes drops the space between neighbouring words on tight scans, which
costs the benchmark: a glued given name and surname is one GT word to annotate and one
token to score. A split is only made where the *page image* shows a gap: the ink profile of
the word's own quadrilateral along its reading direction must have an interior run of blank
columns wider than a typical letter gap, and the character boundary nearest to that gap
(from a proportional-font width model, snapped to a case, digit or punctuation transition
when one is within a character) must agree with it. Anything less certain is left whole.

Pure image evidence on the same pixels the engine read: no dictionary and no second engine.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

# Helvetica/Arial advance widths per 1000 em. Only ratios matter (position of a character
# boundary along the word), so the match need not be exact for other sans or serif fonts.
_W = {
    **dict(zip("abcdefghijklmnopqrstuvwxyz",
               (556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278,
                556, 500, 722, 500, 500, 500))),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               (667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611,
                722, 667, 944, 667, 667, 611))),
    **{d: 556 for d in "0123456789"},
    **{".": 278, ",": 278, ":": 278, ";": 278, "/": 278, "|": 260, "-": 333, "(": 333, ")": 333, "[": 278, "]": 278,
       "@": 1015, "&": 667, "'": 191, '"': 355, "!": 278, "?": 556, "_": 556, "+": 584, "%": 889, "#": 556, "*": 389,
       "$": 556, "=": 584, "’": 222, "‘": 222, "“": 333, "”": 333, "–": 556, "—": 1000},
}

VERSION = 1  # bump when the rules or thresholds change: it is recorded with every OCR draft

MIN_GAP_FRAC = 0.22   # blank run, as a fraction of the word's height, that can count as a word space
REL_GAP = 2.6         # ... and it must be this many times the page's typical gap between letters
MAX_POS_ERR = 0.06    # boundary position error allowed, as a fraction of the word's width
MIN_PX_HEIGHT = 12    # below this the ink profile says nothing


def _adv(c: str) -> float:
    return float(_W.get(c, 556))


def boundary_candidates(text: str) -> list[int]:
    """Indices i (1..n-1) where a split between text[i-1] and text[i] would be natural:
    lower->Upper, letter<->digit, a closing mark before a word, a word before an opening mark."""
    out = []
    for i in range(1, len(text)):
        a, b = text[i - 1], text[i]
        if ((a.islower() and b.isupper()) or (a.isalpha() and b.isdigit()) or (a.isdigit() and b.isalpha())
                or (a in ".,:;)" and b.isalnum()) or (a.isalnum() and b in "(&")):
            out.append(i)
    return out


def _crop(gray: np.ndarray, quad_px: np.ndarray, angle: float) -> np.ndarray | None:
    """Grayscale pixels inside the word's quadrilateral, oriented so reading runs left to right
    (None for angles that are not a multiple of 90: skewed words are left alone)."""
    if angle % 90 != 0:
        return None
    x0, y0 = np.floor(quad_px.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad_px.max(axis=0)).astype(int)
    h, w = gray.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    crop = gray[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(quad_px - [x0, y0]).astype(np.int32)], 255)
    crop[mask == 0] = 255
    k = {0: 0, 90: 1, 180: 2, 270: 3}[int(angle) % 360]
    return np.rot90(crop, k) if k in (1, 3) else (crop[::-1, ::-1] if k == 2 else crop)


def interior_gaps(crop: np.ndarray) -> tuple[list[tuple[int, int]], int, int, list[tuple[int, int]]]:
    """Blank column runs strictly inside the ink extent, plus the inked runs between them:
    (gaps [(start, end)...], ink_start, ink_end, runs [(start, end)...])."""
    h, w = crop.shape
    thr, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if ink.sum() < 6:
        return [], 0, w, []
    # Rules and underlines span the gaps they run under: drop rows that are mostly ink.
    ink[ink.sum(axis=1) > 0.8 * w] = 0
    # Isolated specks (scan noise) do not close a gap: a column needs 2 ink pixels.
    cols = ink.sum(axis=0) >= 2
    idx = np.flatnonzero(cols)
    if idx.size == 0:
        return [], 0, w, []
    lo, hi = int(idx[0]), int(idx[-1])
    gaps, runs, start, rstart = [], [], None, lo
    for x in range(lo, hi + 1):
        if not cols[x]:
            if start is None:
                start = x
                runs.append((rstart, x))
        elif start is not None:
            gaps.append((start, x))
            start, rstart = None, x
    runs.append((rstart, hi + 1))
    return gaps, lo, hi + 1, runs


def decide(text: str, gaps: list[tuple[int, int]], lo: int, hi: int, runs: list[tuple[int, int]], height: int,
           min_gap_px: float, trace: list | None = None) -> list[int]:
    """Character indices at which to split ``text`` given its interior ink gaps."""
    if len(text) < 3 or height < MIN_PX_HEIGHT or hi - lo < 8 or not any(c.isalpha() for c in text):
        return []  # too little to go on; numbers and dates (no letters) are never split
    cum = np.concatenate([[0.0], np.cumsum([_adv(c) for c in text])])
    cum /= cum[-1]
    cands = boundary_candidates(text)
    cuts = []
    for g0, g1 in gaps:
        if (g1 - g0) < max(MIN_GAP_FRAC * height, min_gap_px):
            continue
        frac = ((g0 + g1) / 2 - lo) / (hi - lo)
        best = min(range(1, len(text)), key=lambda i: abs(cum[i] - frac))
        near = [i for i in cands if abs(i - best) <= 1 and abs(cum[i] - frac) <= MAX_POS_ERR * 2]
        pick = min(near, key=lambda i: abs(cum[i] - frac)) if near else best
        if not near:
            # No natural boundary: the width model can be a character off, so let the number of
            # separate ink runs left of the gap (letters mostly do not touch) arbitrate.
            left_runs = sum(1 for _, b in runs if b <= g0)
            around = [i for i in (best - 1, best, best + 1) if 1 <= i < len(text) and abs(cum[i] - frac) <= MAX_POS_ERR]
            exact = [i for i in around if i == left_runs]
            if len(exact) == 1:
                pick = exact[0]
        left, right = text[:pick], text[pick:]
        if abs(cum[pick] - frac) > MAX_POS_ERR:
            continue
        if not (any(c.isalnum() for c in left) and any(c.isalnum() for c in right)):
            continue  # punctuation stays attached to its word (annotation guidelines)
        if text[pick - 1].isdigit() and text[pick].isdigit() and (g1 - g0) < 0.30 * height:
            continue  # inside a number only a clearly wide gap is a space
        if any(len(part) == 1 and part.isalpha() and part not in "aAI" for part in (left, right)):
            continue  # a lone letter that is not a word ("QL|D", "N|ot|ice") is a false gap
        cuts.append(pick)
        if trace is not None:
            trace.append((pick, bool(near), (g1 - g0) / height))
    return sorted(set(cuts))


def split_glued_words(words: list, gray: np.ndarray, s: float) -> list:
    """Split OcrWords at ink gaps. ``gray`` is the page render (uint8 H x W) at ``s`` px per pt."""
    prepared = []
    for w in words:
        quad = np.asarray(w.poly, float) if w.poly else None
        crop = _crop(gray, quad * s, w.angle) if quad is not None else None
        prepared.append((w, quad, interior_gaps(crop) + (crop.shape[0],) if crop is not None else None))
    # The page's own letter spacing: median of every interior gap in the longer words.
    letter = [g1 - g0 for w, _, info in prepared if info and len(w.text) >= 5 for g0, g1 in info[0]]
    min_gap_px = REL_GAP * float(np.median(letter)) if len(letter) >= 20 else 0.0
    out = []
    for w, quad, info in prepared:
        cuts = decide(w.text, *info, min_gap_px) if info else []
        if not cuts:
            out.append(w)
            continue
        bounds = [0, *cuts, len(w.text)]
        cum = np.concatenate([[0.0], np.cumsum([_adv(c) for c in w.text])])
        cum /= cum[-1]
        for a, b in zip(bounds, bounds[1:]):
            t0, t1 = float(cum[a]), float(cum[b])
            q = np.array([quad[0] + (quad[1] - quad[0]) * t0, quad[0] + (quad[1] - quad[0]) * t1,
                          quad[3] + (quad[2] - quad[3]) * t1, quad[3] + (quad[2] - quad[3]) * t0])
            xs, ys = q[:, 0], q[:, 1]
            out.append(replace(w, text=w.text[a:b], bbox=(float(xs.min()), float(ys.min()), float(xs.max()),
                                                          float(ys.max())), poly=[tuple(map(float, p)) for p in q]))
    return out

