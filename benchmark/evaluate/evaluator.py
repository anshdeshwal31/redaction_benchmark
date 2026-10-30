"""Evaluate one candidate output PDF against GT (PLAN.md §5): the actual document,
never the candidate's claims.

Evidence is gathered per *band* (a third of a character cell, across the line) and
aggregated to *units* (characters, or slices of non-text regions) by
:func:`benchmark.evaluate.cells.unit_flags`, identically for every channel:

- L1 visible: original pattern present with real contrast in any render configuration
- L3 latent: original pattern present but faint
- L2 image layer: original pattern recoverable from an image layer on its own
- L2 earlier revision: visible in an earlier incremental revision
- L2 text layer / vector paths: original characters or glyph paths survive in the file
"""

from __future__ import annotations

import hashlib
import io
from collections import defaultdict

import cv2
import numpy as np
import pikepdf
import pymupdf
import shapely

from .. import ocr
from ..forensics.inspect import revision_prefixes
from ..geometry import as_poly, dilate, geometry_from_pikepdf_page, rasterize, to_shapely
from ..gt.address_split import STATE_CODES, postcode_in_state, read_end
from ..gt.policy import Effective, resolve
from ..gt.shared_roles import STRIP as SHARED_STRIP
from ..gt.shared_roles import SharedUnits, document_shared_units
from ..schemas import Action, GTDocument, PiiType, PolicyProfile, Requirement
from .addresses import (
    AddressUnit,
    Token,
    at_place,
    is_new_text,
    locality_roles,
    locality_shape,
    ocr_tokens,
    page_lines,
    page_tokens,
    token_survivals,
    unit_readback,
    unit_regions,
    units_of,
    widened,
)
from .cells import CellStats, PageCells, Unit, unit_flags
from .config import EvalConfig
from .document import document_findings
from .moved import Found, find_moved, find_restyled, find_slices, refine_chars
from .layers import (
    PageText,
    find_at_place,
    layer_survival,
    page_chars,
    page_layers,
    page_text,
    separated,
    slice_in,
    text_in_words,
    text_leaks,
    token_leaks,
    tokens_in_zone,
    vector_survival,
)
from .outcomes import (
    HIDDEN,
    L1_L2_LEAKS,
    PRESENT,
    SAFE,
    DocumentEvaluation,
    Integrity,
    MentionOutcome,
    MentionStatus,
    PageOutcome,
    WordOutcome,
    WordStatus,
)
from .render import RenderCache, RenderConfig, match_shapes, render_gray


# -- integrity --------------------------------------------------------------------


def _geometries(data: bytes):
    with pikepdf.open(io.BytesIO(data)) as pdf:
        return [geometry_from_pikepdf_page(p) for p in pdf.pages], [str(w) for w in pdf.check_pdf_syntax()]


def check_integrity(source: bytes, output: bytes) -> tuple[Integrity, list, list]:
    src_geoms, _ = _geometries(source)
    try:
        out_geoms, warnings = _geometries(output)
        mu = pymupdf.open("pdf", output)
    except Exception as exc:  # noqa: BLE001 - any failure to open is an integrity failure
        return Integrity(False, False, [f"output does not open: {exc}"], []), src_geoms, []
    problems = []
    if mu.is_repaired:
        problems.append("output is damaged: MuPDF had to repair it")
    for engine in readable_engines(output, ("pdfium",), want_missing=True):
        problems.append(f"output cannot be opened by {engine}")
    if len(out_geoms) != len(src_geoms):
        problems.append(f"page count {len(out_geoms)} != source {len(src_geoms)}")
    for i, (a, b) in enumerate(zip(src_geoms, out_geoms)):
        aw, ah = a.display_size
        bw, bh = b.display_size
        if abs(aw - bw) > 0.5 or abs(ah - bh) > 0.5:
            # The client accepts a changed page size (Q11); the evaluator rescales the page, and a
            # changed aspect ratio marks the page's pixel evidence unreliable.
            warnings.append(f"page {i}: display size {bw:.1f}x{bh:.1f} != source {aw:.1f}x{ah:.1f}")
        elif a.rotation != b.rotation:
            # Rotation flattened into the content: what a reader sees is unchanged.
            warnings.append(f"page {i}: /Rotate {b.rotation} != source {a.rotation} (displayed geometry unchanged)")
    return Integrity(True, len(out_geoms) == len(src_geoms), problems, warnings), src_geoms, out_geoms


def readable_engines(data: bytes, engines, want_missing: bool = False) -> list[str]:
    """Engines that can open ``data`` (or, with ``want_missing``, those that cannot)."""
    import pypdfium2 as pdfium

    ok, bad = [], []
    for e in engines:
        try:
            if e == "pdfium":
                pdfium.PdfDocument(data).close()
            else:
                pymupdf.open("pdf", data)
            ok.append(e)
        except Exception:  # noqa: BLE001 - unreadable by this engine
            bad.append(e)
    return bad if want_missing else ok


# -- registration -----------------------------------------------------------------


def estimate_shift(original: np.ndarray, output: np.ndarray) -> tuple[float, float, float]:
    """Translation (dx, dy) in pixels such that ``output`` ≈ ``original`` shifted by it."""
    f = min(1.0, 800.0 / original.shape[1])
    a = cv2.resize(original, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(output, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(255 - a, 255 - b, win)
    return dx / f, dy / f, float(response)


def _global_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    sa, sb = a.std(), b.std()
    return 0.0 if sa < 1e-6 or sb < 1e-6 else float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)


# -- helpers -----------------------------------------------------------------------


def _mention_units(m, words) -> list[tuple[int, Unit]]:
    out = []
    for seg in m.segments:
        for wid in seg.words:
            n = len(words[wid][1].text)
            lo, hi = (seg.char_start, seg.char_end) if seg.char_start is not None else (0, n)
            out.extend((seg.page, (wid, k)) for k in range(lo, hi))
        for rid in seg.regions:
            out.append((seg.page, (rid, -1)))  # expanded against the page's region units below
    return out


def _required_text(m, words) -> dict[tuple[int, str], str]:
    out = {}
    for seg in m.segments:
        for wid in seg.words:
            text = words[wid][1].text
            if seg.char_start is not None:
                text = text[seg.char_start:seg.char_end]
            out[(seg.page, wid)] = text.strip(".,;:()")
    return out


def _fraction(flags: list[bool]) -> float:
    return float(sum(flags)) / len(flags) if flags else 0.0


def _alnum(s: str) -> str:
    return "".join(ch for ch in s.casefold() if ch.isalnum())


def _is_new_text(word_text: str, needle: str, got: str | None, cfg: EvalConfig) -> bool:
    """Extracted text at a mention word that is neither the original token nor the part of
    the word outside the required span (e.g. a trailing full stop left in place)."""
    if not got or text_leaks(needle, got, cfg):
        return False
    new = _alnum(got)
    rest = _alnum(word_text.replace(needle, "", 1))
    return bool(new) and new not in rest


def _only_slid(got: str | None, slid: set[str]) -> bool:
    """Is the text at a word (letters and digits) made only of words its line kept, whole, with at most the
    end of one before them and the start of one after them? A line re-typeset after a deletion slides
    those words (the mention's shared words, the line's other words) into an erased word's place; they
    are never its replacement. Kept words of one character are left out, so a fake is never cut up."""
    left = _alnum(got or "")
    whole = {t for t in slid if len(t) >= 2}
    if not left or not whole:
        return False
    n = len(left)
    # ok[i]: left[:i] is an end fragment, then whole kept words
    ok = [False] * (n + 1)
    ok[0] = True
    for i in range(1, n + 1):
        ok[i] = any(t.endswith(left[:i]) for t in whole) or any(ok[i - len(t)] and left[i - len(t):i] == t
                                                               for t in whole if len(t) <= i)
    return any(ok[i] and (i == n or any(t.startswith(left[i:]) for t in whole)) for i in range(n + 1)
               if i > 0 or any(t.startswith(left) for t in whole))


def _text_like_ink(gray: np.ndarray, polys: list[np.ndarray], scale: float, min_transitions: int = 4) -> bool:
    """Does the region carry text-like ink (many stroke transitions), rather than a solid
    fill or the edge of a box? Used to tell replacement text from plain removal. A word of one or
    two characters has fewer strokes across it: ``min_transitions`` 2 for those."""
    if not polys:
        return False
    mask = rasterize(polys, gray.shape, scale)
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    win = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ink = (win < 128) & m
    dark = ink.sum() / max(m.sum(), 1)
    if dark < 0.03 or dark > 0.6:
        return False
    i8 = ink.astype(np.int8)
    rows = np.abs(np.diff(i8, axis=1)).sum(axis=1)[ink.any(axis=1)]
    cols = np.abs(np.diff(i8, axis=0)).sum(axis=0)[ink.any(axis=0)]
    med = max(float(np.median(rows)) if rows.size else 0.0, float(np.median(cols)) if cols.size else 0.0)
    return med >= min_transitions


def _region_stand_in(gray: np.ndarray, polys: list[np.ndarray], scale: float, min_std: float) -> bool:
    """Is a hidden region filled with new picture content (a synthetic signature, avatar or
    silhouette) rather than a uniform box or blank paper? The mask is eroded so the edge of a
    box drawn exactly over the region does not count as variation."""
    if not polys:
        return False
    mask = rasterize(polys, gray.shape, scale)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return bool(mask.sum() >= 25 and float(gray[mask].std()) >= min_std)


def _has_alnum(s: str, n: int = 1) -> bool:
    return sum(ch.isalnum() for ch in s) >= n


def _line_heights(gt: GTDocument) -> dict[str, float]:
    """Word id -> H, the median height (across the line) of the GT words on its line."""
    by_line: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    for p in gt.pages:
        for w in p.words:
            q = np.asarray(w.poly, float).reshape(-1, 2)
            by_line[(p.index, w.line) if w.line else (p.index, w.id)].append((w.id, float(np.hypot(*(q[3] - q[0])))))
    out: dict[str, float] = {}
    for ws in by_line.values():
        h = float(np.median([x for _, x in ws]))
        out.update((wid, h) for wid, _ in ws)
    return out


def _searchable(gt: GTDocument, word_outcomes: list[WordOutcome], page_texts: dict[int, PageText],
                shift_pt: dict[int, tuple[float, float]], line_h: dict[str, float], cfg: EvalConfig,
                found: dict[str, Found] | None = None) -> tuple[dict[str, bool], dict]:
    """Kept words found in the text layer at their displayed place (client Q11), and the text layer's
    alignment and word breaks, which are reported and never fail a document.

    Counted: PRESERVED and MOVED kept words (ORDINARY, MUST_NOT) with a letter or digit, looked for at
    their displayed place plus the page's registration shift with :func:`find_at_place`
    (searchable_band across, L/2 + searchable_margin x H along): the GT polygon, or for a MOVED word
    the polygon at its found transform (owner, 2026-10-29). A word not found is misplaced when it is on
    its own line further along, else missing. Word breaks: of every two consecutive GT words of a line
    both found, whether the extracted text separates them (a space or a line break), runs them together
    (glued) or puts the second first (out of order); examples name word ids only."""
    found = found or {}
    scale = cfg.dpi / 72.0
    status = {wo.word_id: wo for wo in word_outcomes}
    searchable: dict[str, bool] = {}
    moved: dict[str, bool] = {}
    runs: dict[str, tuple[int, object]] = {}
    misplaced = missing = 0
    empty = PageText()
    for page in gt.pages:
        chars = page_texts.get(page.index, empty)
        shift = shift_pt.get(page.index, (0.0, 0.0))
        for w in page.words:
            wo = status.get(w.id)
            if not (wo and wo.status in PRESENT and _has_alnum(w.text)
                    and wo.requirement in (Effective.ORDINARY.value, Effective.MUST_NOT.value)):
                continue
            poly, h = w.poly, line_h.get(w.id)
            if wo.status is WordStatus.MOVED:
                if w.id not in found:
                    continue
                poly, h = found[w.id].poly(w.poly, scale), (h * found[w.id].s if h else None)
            run = find_at_place(poly, w.text, chars, shift, h, cfg.searchable_band, cfg.searchable_margin,
                                cfg.text_ratio)
            searchable[w.id] = run is not None
            if wo.status is WordStatus.MOVED:
                moved[w.id] = run is not None
            if run is not None:
                runs[w.id] = (page.index, run)
                wo.text_offset_pt = round(float(np.hypot(run.along, run.across)), 2)
            elif find_at_place(poly, w.text, chars, shift, h, cfg.searchable_band, None,
                               cfg.text_ratio) is not None:
                misplaced += 1
            else:
                missing += 1
    preserved = {wid: v for wid, v in searchable.items() if wid not in moved}
    pairs = glued = out_of_order = 0
    examples: list[str] = []
    for page in gt.pages:
        lines: dict = defaultdict(list)
        for w in page.words:
            lines[w.line].append(w)
        for ws in lines.values():
            ws.sort(key=lambda w: w.order)
            for a, b in zip(ws, ws[1:]):
                if a.id in runs and b.id in runs:
                    pairs += 1
                    sep = separated(page_texts.get(page.index, empty), runs[a.id][1], runs[b.id][1])
                    if sep is None:
                        out_of_order += 1
                    elif not sep:
                        glued += 1
                        examples.append(f"{a.id}|{b.id}")
    off = np.array([np.hypot(r.along, r.across) for _, r in runs.values()])
    ratio = np.array([r.width_ratio for _, r in runs.values()])

    def q(a, p):
        return round(float(np.percentile(a, p)), 2) if a.size else None

    return searchable, {
        "searchable_preserved_rate": round(sum(preserved.values()) / len(preserved), 4) if preserved else None,
        "searchable_moved_words": len(moved),
        "searchable_moved_rate": round(sum(moved.values()) / len(moved), 4) if moved else None,
        "searchable_misplaced_words": misplaced,
        "searchable_missing_words": missing,
        "text_offset_median_pt": q(off, 50),
        "text_offset_p95_pt": q(off, 95),
        "text_offset_max_pt": round(float(off.max()), 2) if off.size else None,
        "text_width_ratio_median": q(ratio, 50),
        "text_word_pairs": pairs,
        "text_glued_pairs": glued,
        "text_out_of_order_pairs": out_of_order,
        "text_glued_rate": round(glued / pairs, 4) if pairs else None,
        "text_glued_examples": examples[:10],
    }


def _flags_from(indices: dict[int, int], n: int) -> np.ndarray:
    arr = np.zeros(n, bool)
    if indices:
        arr[list(indices)] = True
    return arr


# -- main ----------------------------------------------------------------------------


def evaluate_output(
    source: bytes,
    output: bytes,
    gt: GTDocument,
    policy: PolicyProfile,
    config: EvalConfig | None = None,
) -> DocumentEvaluation:
    cfg = config or EvalConfig()
    res = resolve(gt, policy)
    integrity, src_geoms, out_geoms = check_integrity(source, output)
    entities = {e.id: e for e in gt.entities}
    words = gt.word_index()
    regions_idx = gt.region_index()
    scale = cfg.dpi / 72.0

    engines = readable_engines(output, cfg.renderers) if integrity.opens else list(cfg.renderers)
    if not engines:
        engines = list(cfg.renderers)
    configs = [RenderConfig(e, True) for e in engines]
    out_doc = pymupdf.open("pdf", output) if integrity.opens else None
    if out_doc is not None and any(p.first_annot for p in out_doc):
        configs += [RenderConfig(e, False) for e in engines]
    labels = [rc.label for rc in configs]
    src_doc = pymupdf.open("pdf", source)
    cache = RenderCache(cfg.dpi)

    pcs: dict[int, PageCells] = {}
    region_units: dict[int, dict[str, list[Unit]]] = {}
    informative: dict[int, set[Unit]] = {}
    vis: dict[int, dict[Unit, bool]] = {}
    vis_by_cfg: dict[int, dict[str, dict[Unit, bool]]] = {}
    lat: dict[int, dict[Unit, bool]] = {}
    img: dict[int, dict[Unit, bool]] = {}
    img_lat: dict[int, dict[Unit, bool]] = {}
    rev: dict[int, dict[Unit, bool]] = {}
    texts: dict[int, dict[str, str]] = {}
    rev_texts: dict[int, dict[str, str]] = defaultdict(dict)
    # The output's text per page, in extraction order and in display pt of the source page, for the checks
    # at the displayed place; the whole tokens in the 1 pt zone of address words (eval-0.4).
    page_texts: dict[int, PageText] = {}
    zone_tokens: dict[int, dict[str, list[str]]] = {}
    rev_tokens: dict[int, dict[str, list[str]]] = defaultdict(dict)
    vector: dict[int, dict[str, tuple[int, int]]] = {}
    primary_out: dict[int, np.ndarray] = {}
    # (original, registered output) of the first render configuration per page, for the searches
    # that run after the page loop (moved words, displaced originals)
    prim_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    prim_stats: dict[int, CellStats] = {}  # the band test of the first render configuration, for E1's new ink
    text_scale: dict[int, tuple[float, float]] = {}  # source / output page size, for the OCR's words
    page_outcomes: list[PageOutcome] = []
    unreliable: set[int] = set()

    must_or_may_words = {wid for wid, eff in res.words.items() if eff in (Effective.MUST, Effective.MAY)}
    # -- shared structure (eval-0.4): what a realistic fake may keep, for the roles the policy enables ---
    shared: dict[str, SharedUnits] = document_shared_units(gt, policy)
    # Its characters never pool with identifying ones in the identity gate's run clause (PageCells.apart).
    apart = frozenset({(wid, k) for su in shared.values() for wid in su.full for k in range(len(words[wid][1].text))}
                      | {u for su in shared.values() for u in su.chars})
    # Held to the stricter identity thresholds and clauses of the character gate (cells.PageCells.stats).
    strict_words = frozenset(must_or_may_words)
    mention_words = {wid for m in gt.mentions for seg in m.segments for wid in seg.words}
    mention_regions = {rid for m in gt.mentions for seg in m.segments for rid in seg.regions}
    # Words of addresses and localities match only as whole tokens of the output's text, numbers as
    # maximal digit runs: '14' is not found in '214' or '142750' (E4).
    whole_token_words = {wid for m in gt.mentions
                         if m.type is PiiType.ADDRESS or (m.type is PiiType.LOCATION and m.subtype == "locality")
                         for seg in m.segments for wid in seg.words}
    layer_cache: dict[int, np.ndarray] = {}

    for page in gt.pages:
        i = page.index
        notes: list[str] = []
        base = cache.get("src", source, i, configs[0])
        pc = PageCells.build(page, cfg.dpi, base.shape, cfg.bands, cfg.identity_subbands)
        pc.apart = apart
        pcs[i] = pc
        region_units[i] = defaultdict(list)
        for c in pc.cells:
            if c.region_id is not None and (c.region_id, c.index) not in region_units[i][c.region_id]:
                region_units[i][c.region_id].append((c.region_id, c.index))
        exists = out_doc is not None and i < out_doc.page_count
        out_geom = out_geoms[i] if i < len(out_geoms) else src_geoms[i]
        if exists and i >= len(out_geoms):
            notes.append("output page structure unreadable by pikepdf; source geometry assumed")
        geometry_changed, shift = False, None
        if exists and i < len(out_geoms):
            a, b = src_geoms[i], out_geoms[i]
            if abs(a.display_size[0] / a.display_size[1] - b.display_size[0] / b.display_size[1]) > 0.02:
                unreliable.add(i)
                notes.append("displayed aspect ratio differs from source; pixel evidence unreliable")

        band_stats = {}
        for k, rc in enumerate(configs):
            o = cache.get("src", source, i, rc)
            if exists:
                x = cache.get("out", output, i, rc)
                if x.shape != o.shape:
                    if abs(x.shape[0] - o.shape[0]) > 2 or abs(x.shape[1] - o.shape[1]) > 2:
                        x = cv2.resize(x, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
                        geometry_changed = True
                    else:
                        x, _ = match_shapes(o, x)
                if k == 0:
                    dx, dy, resp = estimate_shift(o, x)
                    if np.hypot(dx, dy) >= cfg.registration_min_shift_px and resp > 0.05:
                        moved = _translate(x, -dx, -dy)
                        if _global_ncc(o, moved) > _global_ncc(o, x) + 0.02:
                            shift = (round(dx, 2), round(dy, 2))
                            notes.append(f"output content shifted by ({dx:.1f}, {dy:.1f}) px; registered")
                if shift is not None:
                    x = _translate(x, -shift[0], -shift[1])
            else:
                x = np.full_like(o, 255)
                if k == 0:
                    notes.append("page missing from output")
            band_stats[rc.label] = pc.stats(o, x, cfg, strict_words)
            if k == 0:
                primary_o, primary_x = o, x
                prim_pairs[i] = (o, x)
                # For the "replaced?" test, blank out bands where the original survives, so a
                # leftover original glyph is not mistaken for replacement text.
                masked = x.copy()
                for ci in np.flatnonzero(band_stats[rc.label].visible):
                    masked[pc.rows[ci], pc.cols[ci]] = 255
                primary_out[i] = masked

        prim = band_stats[labels[0]]
        prim_stats[i] = prim
        vis_by_cfg[i] = {lb: unit_flags(pc, band_stats[lb].informative, band_stats[lb].visible) for lb in labels}
        informative[i] = set(vis_by_cfg[i][labels[0]])
        vis[i] = {u: any(vis_by_cfg[i][lb].get(u, False) for lb in labels) for u in informative[i]}
        any_visible_band = np.vstack([band_stats[lb].visible for lb in labels]).any(axis=0)
        latent_band = np.vstack([band_stats[lb].latent for lb in labels]).any(axis=0) & ~any_visible_band
        lat[i] = {u: v and not vis[i].get(u, False)
                  for u, v in unit_flags(pc, prim.informative, latent_band).items()}

        # -- L2: file-level survival on this page ---------------------------------
        if exists:
            out_page = out_doc[i]
            centres, chars = page_chars(out_page, out_geom)
            texts[i] = text_in_words(page, centres, chars, mention_words)
            (sw, sh), (ow, oh) = src_geoms[i].display_size, out_geom.display_size
            text_scale[i] = (sw / ow, sh / oh)
            page_texts[i] = page_text(out_page, out_geom, text_scale[i])
            zone_tokens[i] = {w.id: tokens_in_zone(w.poly, page_texts[i]) for w in page.words
                              if w.id in whole_token_words}
            mention_cells = [ci for ci, c in enumerate(pc.cells)
                             if (c.word_id in mention_words) or c.region_id is not None]
            layers = page_layers(out_doc, out_page, layer_cache)
            rec, ltn = layer_survival(pc, mention_cells, primary_o, layers, out_geom, cfg, strict_words)
            n = len(pc.cells)
            img[i] = unit_flags(pc, prim.informative, _flags_from(rec, n))
            img_lat[i] = {u: v and not img[i].get(u, False)
                          for u, v in unit_flags(pc, prim.informative, _flags_from(ltn, n)).items()}
            # Drawn signatures and marks too: their strokes can survive under a box like glyph paths.
            vector[i] = vector_survival(src_doc[i], out_page, src_geoms[i], page, mention_words, mention_regions)
        else:
            texts[i], img[i], img_lat[i], vector[i] = {}, {}, {}, {}

        # -- page-level change and spill ----------------------------------------------
        changed = np.abs(cv2.blur(primary_o, (3, 3)).astype(np.int16)
                         - cv2.blur(primary_x, (3, 3)).astype(np.int16)) > cfg.change_threshold
        zone_polys = [dilate(as_poly(w.poly), cfg.tolerance_pt) for w in page.words if w.id in must_or_may_words]
        zone_polys += [dilate(as_poly(r.poly), cfg.tolerance_pt) for r in page.regions
                       if res.regions.get(r.id) in (Effective.MUST, Effective.MAY)]
        zone = rasterize(zone_polys, changed.shape, scale) if zone_polys else np.zeros_like(changed)
        spill = changed & ~zone
        orphans = []
        n_cc, _, st, _ = cv2.connectedComponentsWithStats(spill.astype(np.uint8), connectivity=8)
        for k in range(1, n_cc):
            if st[k, cv2.CC_STAT_AREA] >= (2 * scale) ** 2:
                x0, y0 = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP]
                w_, h_ = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
                orphans.append((round(x0 / scale, 1), round(y0 / scale, 1), round((x0 + w_) / scale, 1),
                                round((y0 + h_) / scale, 1)))
        px_area = (1.0 / scale) ** 2
        page_outcomes.append(PageOutcome(
            index=i, evaluated=i not in unreliable, geometry_changed=geometry_changed,
            registration_shift_px=shift, changed_area_pt2=round(float(changed.sum()) * px_area, 2),
            spill_area_pt2=round(float(spill.sum()) * px_area, 2),
            spill_fraction=round(float(spill.sum()) / spill.size, 6), orphan_regions=orphans[:200],
            blank_output=bool(exists and primary_x.std() < 1.0 and primary_o.std() > 5.0), notes=notes))
        # The page's renders are done with, but the primary source render (the earlier-revision check
        # reads it again).
        cache.evict(i, keep=(("src", configs[0]),))

    # -- L2: earlier revisions ------------------------------------------------------------
    prefixes = revision_prefixes(output) if (cfg.check_revisions and integrity.opens) else []
    for prefix in prefixes:
        rdoc = pymupdf.open("pdf", prefix)
        rgeoms, _ = _geometries(prefix)
        for page in gt.pages:
            i = page.index
            if i >= rdoc.page_count:
                continue
            o = cache.get("src", source, i, configs[0])
            x = render_gray(prefix, i, cfg.dpi, configs[0])
            if x.shape != o.shape:
                x = cv2.resize(x, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
            st_ = pcs[i].stats(o, x, cfg, strict_words)
            flags = unit_flags(pcs[i], st_.informative, st_.visible)
            rev[i] = {u: rev.get(i, {}).get(u, False) or v for u, v in flags.items()}
            centres, chars = page_chars(rdoc[i], rgeoms[i])
            for wid, got in text_in_words(page, centres, chars, mention_words).items():
                rev_texts[i][wid] = rev_texts[i].get(wid, "") + got
            if any(w.id in whole_token_words for w in page.words):
                rpt = page_text(rdoc[i], rgeoms[i])
                for w in page.words:
                    if w.id in whole_token_words:
                        rev_tokens[i].setdefault(w.id, []).extend(tokens_in_zone(w.poly, rpt))

    # -- document level --------------------------------------------------------------------
    findings, surfaces = document_findings(output, gt, res, policy) if integrity.opens else ([], [])

    shift_pt = {p.index: ((p.registration_shift_px[0] / scale, p.registration_shift_px[1] / scale)
                          if p.registration_shift_px else (0.0, 0.0)) for p in page_outcomes}
    line_h = _line_heights(gt)

    # -- moved words and displaced originals (eval-0.4) --------------------------------------------
    # The in-place test on the primary render configuration, which the word statuses report.
    inplace: dict[str, tuple[str, float | None]] = {}
    for page in gt.pages:
        prim_units = vis_by_cfg[page.index][labels[0]]
        for w in page.words:
            units = [u for u in ((w.id, k) for k in range(len(w.text))) if u in prim_units]
            if not units or page.index in unreliable:
                inplace[w.id] = (WordStatus.NOT_EVALUATED.value, None)
                continue
            frac = _fraction([prim_units[u] for u in units])
            ws = (WordStatus.PRESERVED if frac >= cfg.word_preserved_fraction
                  else WordStatus.REMOVED if frac <= cfg.word_removed_fraction else WordStatus.DAMAGED)
            inplace[w.id] = (ws.value, frac)
    requirement = {w.id: res.words.get(w.id, Effective.ORDINARY).value for p in gt.pages for w in p.words}
    # A word with shared characters is found away from its place only where its glyphs pass the identity gate:
    # a fake keeping those characters (a mobile's '04', a postcode's prefix) is not the original moved.
    gated = frozenset(wid for su in shared.values() for wid, _ in su.chars)
    found, anchors = find_moved(gt, requirement, pcs, prim_pairs, inplace, unreliable, cfg, strict_words, gated)
    # Words still damaged or removed, looked for re-typeset in a substitute font.
    found.update(find_restyled(gt, requirement, prim_pairs, inplace, found, anchors, unreliable, cfg, src_doc,
                               src_geoms, out_doc))
    # A MUST or MAY word found away from its place is a displaced original: its characters are visible,
    # and its ink is never taken for replacement ink.
    displaced: set[tuple[int, Unit]] = set()
    for wid, f in found.items():
        if f.requirement not in (Effective.MUST.value, Effective.MAY.value):
            continue
        for k in range(len(words[wid][1].text)):
            u = (wid, k)
            if u in informative.get(f.page, set()):
                if not vis[f.page].get(u, False):
                    displaced.add((f.page, u))
                vis[f.page][u] = True
                vis_by_cfg[f.page][labels[0]][u] = True
        if f.page in primary_out:
            x0, y0, x1, y1 = (int(round(v * scale)) for v in f.box)
            primary_out[f.page][max(y0 - 1, 0):max(y1 + 2, 0), max(x0 - 1, 0):max(x1 + 2, 0)] = 255

    # The identifying slice of a MUST word with shared characters (an email's local part, an identifier's
    # digits) is looked for on its own: a changed domain or prefix cannot hide a kept slice it pushed along.
    must_cut = {wid: cut for mid, su in shared.items() if res.mentions[mid].requirement is Requirement.MUST
                for wid, cut in su.needle_cut.items()}
    for pg, u in find_slices(gt, must_cut, pcs, prim_pairs, found, vis, informative, unreliable, cfg):
        if not vis[pg].get(u, False):
            displaced.add((pg, u))
        vis[pg][u] = True
        vis_by_cfg[pg][labels[0]][u] = True

    def text_pair(pg: int, s_wid: str, p_wid: str, ks, seen: bool = False) -> bool:
        """The pair read from the text layer: the state's text at its displayed place followed by a 4-digit
        token starting with the original's prefix digits or, when the original postcode starts with no
        digit of its state (NSW 1xxx, VIC 8xxx, QLD 9xxx, ACT 02xx), a postcode of that state. No state text
        there: for a state not seen in pixels (``seen`` False) nothing is kept and the pair stands; for one
        seen, the pair cannot be read and fails."""
        pt = page_texts.get(pg, PageText())
        state = words[s_wid][1].text.strip(SHARED_STRIP)
        run = find_at_place(words[s_wid][1].poly, state, pt, shift_pt.get(pg, (0.0, 0.0)), line_h.get(s_wid),
                            cfg.searchable_band, cfg.searchable_margin, cfg.text_ratio, whole_token=True)
        if run is None:
            return not seen
        # The postcode: the text right after the state in extraction order, or a 4-digit token at the
        # postcode's displayed place (a fake written into the page after the kept state comes later in the
        # extraction order).
        got = ["".join(c.c for c in pt if c.seq > run.last).lstrip()[:4]]
        pw = words[p_wid][1]
        h = line_h.get(p_wid) or float(np.hypot(*(as_poly(pw.poly)[3] - as_poly(pw.poly)[0])))
        got += [t.text.strip(SHARED_STRIP) for t in tokens_by_page.get(pg) or []
                if at_place(t, pw.poly, h, shift_pt.get(pg, (0.0, 0.0)), cfg)]
        # The original postcode (kept in the text under a box, say) is no same-state fake.
        got = [g for g in got if len(g) == 4 and g.isdigit() and g != pw.text.strip(SHARED_STRIP)]
        if not ks:
            code = STATE_CODES.get(_alnum(state))
            return code is not None and any(postcode_in_state(code, g) for g in got)
        prefix = "".join(pw.text[k] for k in ks)
        return any(g.startswith(prefix) for g in got)

    def pair_ok(pg: int, s_wid: str, state_units: list[Unit], p_wid: str, ks) -> bool:
        """Are the postcode's prefix digits the original glyphs where the state word is seen (in place, or
        at its found transform, refined within half a character)? A same-state fake has the state's digit
        there; another state's fake does not. Where they are not, or the original postcode has no prefix
        digit to read, the pair is read from the text layer (:func:`text_pair`): a same-state fake that
        redraws the postcode at another size, or replaces a postcode such as NSW 1xxx, still pairs where
        the output's text says so; with no text it fails."""
        if pg in unreliable or pg not in prim_pairs:
            return False
        if not ks:
            return text_pair(pg, s_wid, p_wid, ks, seen=True)
        f = found.get(s_wid)
        if f is not None:
            t = (f.s, f.tx, f.ty)
        elif any(vis[pg].get(u, False) for u in state_units):
            t = (1.0, 0.0, 0.0)
        else:
            return text_pair(pg, s_wid, p_wid, ks)
        o, x = prim_pairs[pg]
        dx, dy, score = refine_chars(pcs[pg], p_wid, ks, o, x, *t)
        if score >= cfg.moved_match:
            t = (t[0], t[1] + dx, t[2] + dy)
        flags = pcs[pg].char_flags_at(p_wid, ks, o, x, *t, cfg)
        return all(flags.get(k, False) for k in ks) or text_pair(pg, s_wid, p_wid, ks, seen=True)

    def text_leaked(pg: int, wid: str, needle: str, shared: tuple[str, ...] | None = None) -> bool:
        """The original token in the output's text: in the 1 pt zone of its word or, with text_place_leak,
        at its displayed place (the GT polygon plus the page's registration shift, the searchable
        tolerance), and for a displaced original also at its found place. With ``shared`` the needle is the
        identifying slice of a word with shared characters, matched by :func:`slice_in` (never inside the
        shared part a fake keeps)."""
        whole = wid in whole_token_words
        if shared is not None:
            if slice_in(needle, texts.get(pg, {}).get(wid), shared):
                return True
        elif whole and token_leaks(needle, zone_tokens.get(pg, {}).get(wid, []), cfg):
            return True
        elif not whole and text_leaks(needle, texts.get(pg, {}).get(wid), cfg):
            return True
        if not cfg.text_place_leak:
            return False
        places = [(words[wid][1].poly, line_h.get(wid))]
        f = found.get(wid)
        if f is not None and f.requirement in (Effective.MUST.value, Effective.MAY.value):
            places.append((f.poly(words[wid][1].poly, scale), line_h[wid] * f.s if line_h.get(wid) else None))
        return any(find_at_place(poly, needle, page_texts.get(pg, PageText()), shift_pt.get(pg, (0.0, 0.0)), h,
                                 cfg.searchable_band, cfg.searchable_margin, cfg.text_ratio,
                                 whole_token=whole, shared=shared) is not None for poly, h in places)

    def rev_text_leaked(pg: int, wid: str, needle: str, shared: tuple[str, ...] | None = None) -> bool:
        if shared is not None:
            return slice_in(needle, rev_texts.get(pg, {}).get(wid), shared)
        if wid in whole_token_words:
            return token_leaks(needle, rev_tokens.get(pg, {}).get(wid, []), cfg)
        return text_leaks(needle, rev_texts.get(pg, {}).get(wid), cfg)

    # -- address units (eval-0.4: E1, E3, E4; docs/metrics.md, "Address units") -----------------------------
    addr_units = units_of(gt)
    unit_of: dict[str, AddressUnit] = {m.id: u for u in addr_units for m in u.mentions}
    lines = page_lines(gt, line_h)
    tokens_by_page = {pg: page_tokens(pt, pg) for pg, pt in page_texts.items()}
    # Words the output still shows (PRESERVED in place, or MOVED), and the kept ones at their displayed place:
    # the anchors that bound a unit's read-back.
    present: set[str] = set()
    kept_place: dict[str, np.ndarray] = {}
    for page in gt.pages:
        for w in page.words:
            if w.id in found or inplace[w.id][0] == WordStatus.PRESERVED.value:
                present.add(w.id)
                if requirement[w.id] in (Effective.ORDINARY.value, Effective.MUST_NOT.value):
                    kept_place[w.id] = found[w.id].poly(w.poly, scale) if w.id in found else as_poly(w.poly)
    readback = {u.key: unit_readback(u, gt, tokens_by_page, kept_place, shift_pt, line_h, cfg, lines)
                for u in addr_units}
    regions = {u.key: unit_regions(u, lines, cfg) for u in addr_units}
    # E3 leaves out the tokens at the displayed place of words a kept (MUST_NOT) or optional (MAY) rule
    # decides, and such words never raise the allowance either.
    keep_zones: dict[int, list] = defaultdict(list)
    for page in gt.pages:
        sh = np.asarray(shift_pt.get(page.index, (0.0, 0.0)))
        for w in page.words:
            if requirement[w.id] in (Effective.MUST_NOT.value, Effective.MAY.value):
                poly = found[w.id].poly(w.poly, scale) if w.id in found else as_poly(w.poly)
                keep_zones[page.index].append(to_shapely(dilate(poly + sh, 1.0)))
    ocr_cache: dict[int, list[Token]] = {}

    def ocr_page_tokens(pg: int) -> list[Token]:
        """Words the measurement OCR reads on an output page with no text layer (review flags only)."""
        if pg not in ocr_cache:
            got = None
            if integrity.opens and out_doc is not None and pg < out_doc.page_count and ocr.available():
                try:
                    got = ocr.ocr_page(output, pg)
                except RuntimeError:
                    got = None
            ocr_cache[pg] = ocr_tokens(got, pg, text_scale.get(pg, (1.0, 1.0)))
        return ocr_cache[pg]

    new_ink_cache: dict[int, np.ndarray] = {}

    def new_ink(pg: int) -> np.ndarray:
        """New ink per band of the primary configuration, read on the output with the visible original
        bands and the boxes of displaced originals blanked: their ink is never the replacement."""
        if pg not in new_ink_cache:
            st, pc, x = prim_stats[pg], pcs[pg], primary_out[pg]
            arr = np.zeros(len(pc.cells), bool)
            for ci in np.flatnonzero(st.new_ink):
                rr, cc = pc.rows[ci], pc.cols[ci]
                arr[ci] = rr.size >= 4 and float(x[rr, cc].std()) >= cfg.cell_min_visible_std
            new_ink_cache[pg] = arr
        return new_ink_cache[pg]

    def blanked(pg: int, toks: list[Token]) -> np.ndarray:
        """The primary output of a page with the boxes of the given output-text tokens blanked (1 px
        around): ink that is never the replacement of the word under it."""
        pt = page_texts.get(pg)
        boxes = [pt[k].box for t in toks if t.page == pg and t.index >= 0 and pt is not None
                 for k in pt.tokens[t.index]]
        if not boxes:
            return primary_out[pg]
        x = primary_out[pg].copy()
        dx, dy = shift_pt.get(pg, (0.0, 0.0))
        for x0, y0, x1, y1 in boxes:
            r0, r1 = int((y0 - dy) * scale) - 1, int(np.ceil((y1 - dy) * scale)) + 2
            c0, c1 = int((x0 - dx) * scale) - 1, int(np.ceil((x1 - dx) * scale)) + 2
            x[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = 255
        return x

    def slid_tokens(pg: int, polys: list[np.ndarray], slid: set[str]) -> list[Token]:
        """Output tokens over the given word polygons (1 pt around) that are words their lines keep."""
        sh = np.asarray(shift_pt.get(pg, (0.0, 0.0)), float)
        out = []
        for t in tokens_by_page.get(pg) or []:
            if _alnum(t.text) not in slid:
                continue
            c = np.asarray(t.centre, float) - sh
            if any(p[:, 0].min() - 1 <= c[0] <= p[:, 0].max() + 1 and p[:, 1].min() - 1 <= c[1] <= p[:, 1].max() + 1
                   for p in polys):
                out.append(t)
        return out

    def e3(unit: AddressUnit, needles: dict[str, str]) -> tuple[set[str], bool]:
        """E3: the identifying words (word id -> whole-token needle) that survive in the text layer of the
        unit's region, and whether the measurement OCR suspects one on a page with no text layer."""
        survived: set[str] = set()
        suspected = False
        others = {(t.page, t.index) for k, rb in readback.items() if k != unit.key for t in rb}
        for reg in regions[unit.key]:
            pg = reg.page
            toks, by_ocr = tokens_by_page.get(pg), False
            if not toks:
                if pg in unreliable or pg not in primary_out:
                    continue
                toks, by_ocr = ocr_page_tokens(pg), True
            if not toks:
                continue
            shift = shift_pt.get(pg, (0.0, 0.0))
            zones = keep_zones.get(pg, [])
            out_toks = [t for t in toks if reg.contains(t.centre, shift) and (t.page, t.index) not in others
                        and not any(z.contains(shapely.Point(t.centre)) for z in zones)]
            allow = [w.text for w in gt.page(pg).words
                     if requirement[w.id] == Effective.ORDINARY.value and w.id in present
                     and reg.contains(as_poly(w.poly).mean(axis=0))]
            hit = token_survivals(needles, out_toks, allow)
            if by_ocr:
                suspected |= bool(hit)
            else:
                survived |= hit
        return survived, suspected

    def e1(m, unit: AddressUnit, ident: dict[tuple[int, str], list[int]], kept: dict[str, str],
           slid: set[str] = frozenset()) -> tuple[bool, str | None, list[str]]:
        """E1: is this part of an address replaced? (replaced, locality shape, review flags). ``ident`` maps
        each identifying word to its identifying character indices; ``slid`` holds the words its lines keep
        (letters and digits only), never the replacement where a re-typeset line slides them along."""
        rb = readback[unit.key]
        flags: list[str] = []
        outs = primary_out

        def text_like(pg: int, wid: str) -> bool:
            # New ink on a word's bands counts only when it looks like text there: the edges of a plain box
            # (and their JPEG ringing) make the word's edge bands new ink, and a box is never a fake.
            w = words[wid][1]
            short = sum(ch.isalnum() for ch in w.text) <= cfg.address_short_chars
            return _text_like_ink(outs[pg], [as_poly(w.poly)], scale, 2 if short else 4)

        # A state kept as shared structure has passed its pair check: the text layer need not hold it.
        state_kept = any(unit.roles.get(w) in ("state", "state+postcode") for w in kept)
        if m.type is PiiType.LOCATION:
            if rb:
                holds, shape = locality_shape([t.text for t in rb], *locality_roles(unit, gt), state_kept=state_kept)
                return holds, shape, flags
            ok = bool(ident)
            for (pg, wid), ks in ident.items():
                if pg not in primary_out:
                    ok = False
                    break
                chars = unit_flags(pcs[pg], prim_stats[pg].informative, new_ink(pg))
                if not any(chars.get((wid, k), False) for k in ks) or not text_like(pg, wid):
                    ok = False
                    break
            if not ok:
                polys: dict[int, list[np.ndarray]] = defaultdict(list)
                for (pg, wid) in ident:
                    w = words[wid][1]
                    polys[pg].append(widened(w.poly, w.angle, 0.5 * (line_h.get(wid) or 0.0)))
                ok = any(_text_like_ink(primary_out[pg], ps, scale) for pg, ps in polys.items() if pg in primary_out)
            pages_ = {pg for pg, _ in ident}
            if pages_ and ocr.available() and not any(tokens_by_page.get(pg) for pg in pages_):
                # The measurement OCR may only ask for a review: it never changes the verdict.
                read = unit_readback(unit, gt, {pg: ocr_page_tokens(pg) for pg in pages_}, kept_place, shift_pt,
                                     line_h, cfg, lines)
                if not locality_shape([t.text for t in read], *locality_roles(unit, gt), state_kept=state_kept)[0]:
                    flags.append("LOCALITY_SHAPE_UNCONFIRMED")
            return ok, "no_text_at_unit", flags
        # The street line's own new text: read-back tokens before the locality's, never the words its lines
        # keep. A line re-typeset after a deletion slides those, and the locality's fake, into an erased street
        # word's place; their ink is blanked for the pixel tests too.
        street_rb, other_rb = list(rb), []
        if unit.locality is not None and rb:
            end = read_end([t.text for t in rb], overseas=True)
            if end is not None and end.anchor is not None and end.anchor >= 1:
                street_rb, other_rb = rb[:end.anchor - 1], rb[end.anchor - 1:]
        other_rb += [t for t in street_rb if _alnum(t.text) in slid]
        street_rb = [t for t in street_rb if _alnum(t.text) not in slid]
        polys = defaultdict(list)
        for (pg, wid) in ident:
            polys[pg].append(as_poly(words[wid][1].poly))
        outs = {pg: blanked(pg, other_rb + slid_tokens(pg, polys[pg], slid)) for pg in polys if pg in primary_out}
        for (pg, wid), ks in ident.items():
            w = words[wid][1]
            h = line_h.get(wid) or float(np.hypot(*(as_poly(w.poly)[3] - as_poly(w.poly)[0])))
            sh = shift_pt.get(pg, (0.0, 0.0))
            if any(t.page == pg and at_place(t, w.poly, h, sh, cfg) and is_new_text(w.text, t.text) for t in street_rb):
                return True, None, flags
            if pg in outs:
                st, pc, x = prim_stats[pg], pcs[pg], outs[pg]
                cells = [ci for ci in pc.word_cells.get(wid, []) if pc.cells[ci].index in ks and st.informative[ci]]
                ink = [bool(new_ink(pg)[ci]) and float(x[pc.rows[ci], pc.cols[ci]].std()) >= cfg.cell_min_visible_std
                       for ci in cells]
                if cells and np.mean(ink) >= cfg.address_new_ink_fraction and text_like(pg, wid):
                    return True, None, flags
        return any(_text_like_ink(outs[pg], ps, scale) for pg, ps in polys.items() if pg in outs), None, flags

    # -- mentions ----------------------------------------------------------------------------
    line_words: dict[tuple[int, str], list] = defaultdict(list)
    for page in gt.pages:
        for w in page.words:
            line_words[(page.index, w.line)].append(w)
    outcomes: list[MentionOutcome] = []
    excused_words: set[str] = set()  # whole words excused as shared structure (pairs that held)
    for m in gt.mentions:
        r = res.mentions[m.id]
        ent = entities.get(m.entity) if m.entity else None
        units: list[tuple[int, Unit]] = []
        for pg, u in _mention_units(m, words):
            if u[1] == -1:
                units.extend((pg, ru) for ru in region_units.get(pg, {}).get(u[0], []))
            else:
                units.append((pg, u))
        units = [(pg, u) for pg, u in units if u in informative.get(pg, set())]
        pages = sorted({seg.page for seg in m.segments})
        # Shared structure leaves every residual; the state and its postcode prefix only together, when the
        # prefix reads as the original glyph where the state is seen.
        su = shared.get(m.id, SharedUnits())
        page_of = {wid: seg.page for seg in m.segments for wid in seg.words}
        pair_log: list[list] = []
        for s_wid, p_wid, ks in list(su.pairs):
            state_units = [(s_wid, k) for k in range(len(words[s_wid][1].text))
                           if (s_wid in su.full or (s_wid, k) in su.chars) and k not in ks]
            ok = pair_ok(page_of[s_wid], s_wid, state_units, p_wid, ks)
            pair_log.append([s_wid, ok])
            if not ok:
                su.without_pair(s_wid, p_wid, ks)
        shared_units = [(pg, u) for pg, u in units if su.shared(u)]
        units = [(pg, u) for pg, u in units if not su.shared(u)]
        excused_words.update(su.full)
        req_all = _required_text(m, words)
        unit = unit_of.get(m.id)
        # The identifying words of an address part (shared words left out) and their identifying characters.
        ident: dict[tuple[int, str], list[int]] = {}
        if unit is not None:
            for seg in m.segments:
                for wid in seg.words:
                    if wid in su.full:
                        continue
                    lo, hi = (seg.char_start, seg.char_end) if seg.char_start is not None \
                        else (0, len(words[wid][1].text))
                    ks = [k for k in range(lo, hi) if (wid, k) not in su.chars]
                    if ks:
                        ident[(seg.page, wid)] = ks
        # E4: in an address part a word of at most address_short_chars letters or digits counts in a channel
        # only when all its characters do there ('14' with only its '1' seen counts nothing).
        short = {wid for (pg, wid), t in req_all.items() if (pg, wid) in ident
                 and sum(ch.isalnum() for ch in t) <= cfg.address_short_chars}
        short_at: dict[str, list[int]] = defaultdict(list)
        for k_, (pg, u) in enumerate(units):
            if u[0] in short:
                short_at[u[0]].append(k_)

        def flags_of(get) -> list[bool]:
            vals = [bool(get(pg, u)) for pg, u in units]
            for ks_ in short_at.values():
                if not all(vals[k_] for k_ in ks_):
                    for k_ in ks_:
                        vals[k_] = False
            return vals

        vis_flags = flags_of(lambda pg, u: vis[pg].get(u, False))
        f_vis = _fraction(vis_flags)
        f_disp = _fraction(flags_of(lambda pg, u: (pg, u) in displaced))
        f_lat = _fraction(flags_of(lambda pg, u: lat[pg].get(u, False)))
        f_img = _fraction(flags_of(lambda pg, u: img.get(pg, {}).get(u, False)))
        f_img_lat = _fraction(flags_of(lambda pg, u: img_lat.get(pg, {}).get(u, False)))
        f_rev_px = _fraction(flags_of(lambda pg, u: rev.get(pg, {}).get(u, False)))
        # ...and a fully visible identifying word makes the part at least PARTIAL, whatever its residual.
        seen_word: dict[str, list[bool]] = defaultdict(list)
        for (pg, u), v in zip(units, vis_flags):
            if (pg, u[0]) in ident:
                seen_word[u[0]].append(v)
        word_seen = any(vs and all(vs) for vs in seen_word.values())
        # In the text layer a shared word is left out and a word with shared characters is looked for by
        # its identifying slice (an email's local part, an identifier's digits).
        req_text = {}
        cut_parts: dict[str, tuple[str, ...]] = {}  # word id -> its shared parts, for a slice's search
        for (pg, wid), t in req_all.items():
            if wid in su.full:
                continue
            if wid in su.needle_cut:
                a, b = su.needle_cut[wid]
                wt = words[wid][1].text
                t = wt[a:b].strip(SHARED_STRIP)
                cut_parts[wid] = tuple(p for p in (wt[:a].strip(SHARED_STRIP), wt[b:].strip(SHARED_STRIP)) if p)
            if t:
                req_text[(pg, wid)] = t
        total_chars = sum(len(t) for t in req_text.values())
        leaked_words = {k for k, t in req_text.items() if text_leaked(k[0], k[1], t, cut_parts.get(k[1]))}
        review_flags: list[str] = []
        if unit is not None:
            # E3: an identifying word of the unit found more often in the unit's region than the input kept it
            # there is a surviving original wherever it was drawn.
            needles = {wid: req_all[(pg, wid)] for (pg, wid) in ident if req_all.get((pg, wid))}
            survived, suspected = e3(unit, needles)
            leaked_words |= {k for k in req_text if k[1] in survived}
            if suspected:
                review_flags.append("LEAK_SUSPECTED")
        hit = sum(len(t) for k, t in req_text.items() if k in leaked_words)
        rev_hit = sum(len(t) for (pg, wid), t in req_text.items() if rev_text_leaked(pg, wid, t, cut_parts.get(wid)))
        f_text = hit / total_chars if total_chars else 0.0
        f_rev = max(f_rev_px, rev_hit / total_chars if total_chars else 0.0)
        vec_orig = vec_surv = 0
        for (pg, key) in [*req_text, *((seg.page, rid) for seg in m.segments for rid in seg.regions)]:
            o_s = vector.get(pg, {}).get(key)
            if o_s:
                vec_orig, vec_surv = vec_orig + o_s[0], vec_surv + o_s[1]
        f_vec = vec_surv / vec_orig if vec_orig else 0.0
        per_cfg = {lb: _fraction(flags_of(lambda pg, u, lb=lb: vis_by_cfg[pg][lb].get(u, False))) for lb in labels}
        worst_cfg = max(per_cfg, key=per_cfg.get) if units and max(per_cfg.values()) > 0 else None

        polys_by_page: dict[int, list[np.ndarray]] = defaultdict(list)
        for seg in m.segments:
            polys_by_page[seg.page].extend(as_poly(words[wid][1].poly) for wid in seg.words)
        # The words the mention's lines keep (its shared words, the lines' other words): a line re-typeset
        # after a deletion slides them into an erased word's place, and they are never its replacement.
        ident_ids = {wid for (_, wid) in ident} if unit is not None else \
            {wid for (_, wid) in req_all if wid not in su.full}
        slid = {_alnum(x.text) for pg, wid in req_all for x in line_words.get((pg, words[wid][1].line), ())
                if x.id not in ident_ids and _alnum(x.text)}

        def got_new(pg: int, wid: str, t: str) -> bool:
            # A word's shared characters (an email's domain, an identifier's prefix), and any fragment of
            # the original left beside them, are never the replacement.
            got = texts.get(pg, {}).get(wid)
            if _only_slid(got, slid):
                return False
            if got and wid in su.needle_cut:
                a, b = su.needle_cut[wid]
                wt = words[wid][1].text
                for part in (wt[:a], wt[b:]):
                    part = part.strip(SHARED_STRIP)
                    i_ = got.casefold().find(part.casefold()) if part else -1
                    if i_ >= 0:
                        got = got[:i_] + got[i_ + len(part):]
                left = _alnum(got)
                if left and left in _alnum(wt):
                    return False
            return _is_new_text(words[wid][1].text, t, got, cfg)

        shape = None
        if unit is not None and ident:
            # E1: an address part is judged by its identifying words and the unit's read-back.
            replaced, shape, more = e1(m, unit, ident, su.full, slid)
            review_flags += more
        else:
            new_text = any(got_new(pg, wid, t) for (pg, wid), t in req_all.items())
            replaced = new_text or any(_text_like_ink(blanked(pg, slid_tokens(pg, polys, slid)), polys, scale)
                                       for pg, polys in polys_by_page.items() if pg in primary_out)
        region_polys: dict[int, list[np.ndarray]] = defaultdict(list)
        for seg in m.segments:
            region_polys[seg.page].extend(as_poly(regions_idx[rid][1].poly) for rid in seg.regions)
        replaced = replaced or any(_region_stand_in(primary_out[pg], polys, scale, cfg.region_replaced_min_std)
                                   for pg, polys in region_polys.items() if pg in primary_out)

        l2 = []
        # A whole identifying word of an address in the text layer is a leak whatever its share (E4).
        if f_text > cfg.safe_max_residual or (unit is not None and leaked_words):
            l2.append("text_layer")
        if f_img > cfg.safe_max_residual:
            l2.append("image_layer")
        if f_vec > cfg.vector_survival:
            l2.append("vector_paths")
        if f_rev > cfg.safe_max_residual:
            l2.append("earlier_revision")
        if any(p in unreliable for p in pages) or (not units and not total_chars):
            status = MentionStatus.NOT_EVALUATED
        elif f_vis >= cfg.leak_min_residual:
            status = MentionStatus.LEAK_VISIBLE
        elif f_vis > cfg.safe_max_residual or word_seen:
            status = MentionStatus.PARTIAL
        elif l2:
            status = MentionStatus.LEAK_RECOVERABLE
        elif max(f_lat, f_img_lat) > cfg.safe_max_residual:
            status = MentionStatus.LEAK_LATENT
        elif replaced:
            status = MentionStatus.SAFE_REPLACED
        elif r.action is Action.REPLACE:
            status = MentionStatus.NOT_REPLACED
        else:
            status = MentionStatus.SAFE_REMOVED
        outcomes.append(MentionOutcome(
            mention_id=m.id, type=m.type.value, subtype=m.subtype, entity=m.entity,
            role=ent.role.value if ent else None, requirement=r.requirement.value, pending=r.pending,
            pages=pages, conditions=list(m.conditions), cells=len(units), visible_residual=round(f_vis, 4),
            worst_config=worst_cfg, latent_residual=round(f_lat, 4), text_residual=round(f_text, 4),
            image_residual=round(f_img, 4), image_latent_residual=round(f_img_lat, 4),
            vector_residual=round(f_vec, 4), revision_residual=round(f_rev, 4), replaced=replaced,
            l2_sources=l2, status=status, displaced_residual=round(f_disp, 4), shared_words=sorted(su.full),
            shared_chars=sorted([w, k] for w, k in su.chars),
            shared_visible=round(_fraction([vis[pg].get(u, False) for pg, u in shared_units]), 4),
            shared_pairs=pair_log, address_unit=unit.key if unit is not None else None, locality_shape=shape,
            review_flags=review_flags))

    # -- words: the in-place test, or MOVED where the word was found away from its place ------------
    word_outcomes: list[WordOutcome] = []
    for page in gt.pages:
        for w in page.words:
            eff = res.words.get(w.id, Effective.ORDINARY)
            st, frac = inplace[w.id]
            if st == WordStatus.NOT_EVALUATED.value:
                word_outcomes.append(WordOutcome(w.id, page.index, eff.value, None, WordStatus.NOT_EVALUATED))
            elif w.id in found:
                f = found[w.id]
                word_outcomes.append(WordOutcome(
                    w.id, page.index, eff.value, f.v, WordStatus.MOVED, in_place_fraction=round(frac, 4),
                    dx_pt=f.dx_pt, dy_pt=f.dy_pt, scale=f.s, match_score=f.score, restyled=f.restyled))
            else:
                word_outcomes.append(WordOutcome(w.id, page.index, eff.value, round(frac, 4), WordStatus(st)))

    # -- searchable text (client Q11): kept words found in the text layer at their displayed place ---
    word_text = {w.id: w.text for p in gt.pages for w in p.words}
    searchable, text_info = _searchable(gt, word_outcomes, page_texts, shift_pt, line_h, cfg, found)

    summary = _summarise(outcomes, word_outcomes, findings, surfaces, integrity, page_outcomes,
                         word_text=word_text, searchable=searchable, cfg=cfg, excused=excused_words)
    summary.update(text_info)
    return DocumentEvaluation(
        doc_id=gt.doc_id, eval_version=cfg.version, eval_digest=cfg.digest(), policy=res.policy,
        source_sha256=hashlib.sha256(source).hexdigest(), output_sha256=hashlib.sha256(output).hexdigest(),
        render_configs=labels, integrity=integrity, mentions=outcomes, words=word_outcomes,
        pages=page_outcomes, findings=findings, surfaces=surfaces, revisions_checked=len(prefixes), summary=summary)


def _summarise(mentions, words, findings, surfaces, integrity, pages, word_text: dict[str, str] | None = None,
               searchable: dict[str, bool] | None = None, cfg: EvalConfig | None = None,
               excused: set[str] | None = None) -> dict:
    cfg = cfg or EvalConfig()
    word_text = word_text or {}
    searchable = searchable or {}
    must = [m for m in mentions if m.requirement == Requirement.MUST.value]
    evaluated = [m for m in must if m.status is not MentionStatus.NOT_EVALUATED]
    by_status = {s.value: sum(1 for m in must if m.status is s) for s in MentionStatus}
    not_replaced = [m for m in evaluated if m.status is MentionStatus.NOT_REPLACED]
    leaked_entities = {m.entity for m in evaluated if m.entity and m.status not in HIDDEN}
    leaked_entities |= {e for f in findings for e in f.entity_ids}
    must_entities = {m.entity for m in must if m.entity}
    done = [w for w in words if w.status is not WordStatus.NOT_EVALUATED]
    ordinary = [w for w in done if w.requirement == Effective.ORDINARY.value]
    protected = [w for w in done if w.requirement == Effective.MUST_NOT.value]
    # A MOVED word's content is intact (displaced or re-typeset): neither collateral nor damage.
    collateral = [w for w in ordinary if w.status not in PRESENT]
    protected_damage = [w for w in protected if w.status not in PRESENT]
    # Only real words count towards the collateral pass mark; stray punctuation is reported.
    real = [w for w in ordinary if _has_alnum(word_text.get(w.word_id, "x" * cfg.collateral_min_chars),
                                              cfg.collateral_min_chars)]
    real_collateral = [w for w in real if w.status not in PRESENT]
    collateral_ok = not real or len(real_collateral) / len(real) <= cfg.collateral_fail_rate
    found = sum(searchable.values())
    searchable_rate = found / len(searchable) if searchable else None
    searchable_ok = searchable_rate is None or searchable_rate >= cfg.searchable_min_rate
    # Shared structure excused in its mention is neutral: neither a redacted token nor a missed one.
    must_words = [w for w in done if w.requirement == Effective.MUST.value and w.word_id not in (excused or ())]

    def redacted(w):  # a MOVED word is present, wherever it was drawn
        return w.status is not WordStatus.MOVED and w.visible_fraction is not None and w.visible_fraction < 0.5

    tp = sum(1 for w in must_words if redacted(w))
    fn = len(must_words) - tp
    fp = sum(1 for w in ordinary + protected if redacted(w))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    must_surface_leaks = [s.surface_id for s in surfaces if s.leaked and s.requirement is Requirement.MUST]
    moved = [w for w in done if w.status is WordStatus.MOVED]
    l1_l2 = [m for m in evaluated if m.status in L1_L2_LEAKS]
    may = [m for m in mentions if m.requirement == Requirement.MAY.value]

    def rate(n, d):
        return round(n / d, 4) if d else None

    return {
        "must_mentions": len(must),
        "must_evaluated": len(evaluated),
        "must_by_status": by_status,
        "must_leak_rate": rate(sum(1 for m in evaluated if m.status not in HIDDEN), len(evaluated)),
        "must_l1_l2_leak_rate": rate(len(l1_l2), len(evaluated)),
        "must_not_replaced": len(not_replaced),
        "must_not_replaced_rate": rate(len(not_replaced), len(evaluated)),
        "must_fail_rate": rate(sum(1 for m in evaluated if m.status not in SAFE), len(evaluated)),
        "entities_with_must_mentions": len(must_entities),
        "entities_leaked": sorted(e for e in leaked_entities if e in must_entities),
        "entity_any_leak_rate": rate(len(leaked_entities & must_entities), len(must_entities)),
        "document_findings": len(findings),
        "must_surface_leaks": must_surface_leaks,
        "ordinary_words": len(ordinary),
        "collateral_words": len(collateral),
        "collateral_rate": rate(len(collateral), len(ordinary)),
        "collateral_real_words": len(real_collateral),
        "collateral_real_rate": rate(len(real_collateral), len(real)),
        "collateral_ok": collateral_ok,
        "searchable_words": len(searchable),
        "searchable_rate": round(searchable_rate, 4) if searchable_rate is not None else None,
        "searchable_ok": searchable_ok,
        "protected_words": len(protected),
        "protected_damaged": [w.word_id for w in protected_damage],
        # eval-0.4: words found away from their place (content intact), and MUST words found so
        # (displaced originals, which count as visible for their mention)
        "moved_words": sum(1 for w in moved if w.requirement == Effective.ORDINARY.value),
        "moved_real_words": sum(1 for w in moved if w.requirement == Effective.ORDINARY.value
                                and _has_alnum(word_text.get(w.word_id, "x" * cfg.collateral_min_chars),
                                               cfg.collateral_min_chars)),
        "protected_moved": sum(1 for w in moved if w.requirement == Effective.MUST_NOT.value),
        "must_words_displaced": sorted(w.word_id for w in moved if w.requirement == Effective.MUST.value),
        "effective_token_precision": round(precision, 4) if precision is not None else None,
        "effective_token_recall": round(recall, 4) if recall is not None else None,
        "effective_token_f1": round(f1, 4) if f1 is not None else None,
        "may_by_status": {s.value: sum(1 for m in may if m.status is s) for s in MentionStatus
                          if any(m.status is s for m in may)},
        "pending_mentions": sorted({m.pending for m in may if m.pending}),
        "integrity_ok": integrity.ok,
        "pages_not_evaluated": [p.index for p in pages if not p.evaluated],
        "clean_document": bool(integrity.ok and not l1_l2 and not findings and not must_surface_leaks
                               and not protected_damage and len(evaluated) == len(must)
                               and not not_replaced and collateral_ok and searchable_ok),
    }
