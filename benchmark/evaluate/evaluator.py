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

from ..forensics.inspect import revision_prefixes
from ..geometry import as_poly, dilate, geometry_from_pikepdf_page, rasterize
from ..gt.policy import Effective, resolve
from ..schemas import GTDocument, PolicyProfile, Requirement
from .cells import PageCells, Unit, unit_flags
from .config import EvalConfig
from .document import document_findings
from .layers import layer_survival, page_chars, page_layers, text_in_words, text_leaks, vector_survival
from .outcomes import (
    L1_L2_LEAKS,
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
            problems.append(f"page {i}: display size {bw:.1f}x{bh:.1f} != source {aw:.1f}x{ah:.1f}")
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


def _text_like_ink(gray: np.ndarray, polys: list[np.ndarray], scale: float) -> bool:
    """Does the region carry text-like ink (many stroke transitions), rather than a solid
    fill or the edge of a box? Used to tell replacement text from plain removal."""
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
    return med >= 4


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
    vector: dict[int, dict[str, tuple[int, int]]] = {}
    primary_out: dict[int, np.ndarray] = {}
    page_outcomes: list[PageOutcome] = []
    unreliable: set[int] = set()

    must_or_may_words = {wid for wid, eff in res.words.items() if eff in (Effective.MUST, Effective.MAY)}
    mention_words = {wid for m in gt.mentions for seg in m.segments for wid in seg.words}
    layer_cache: dict[int, np.ndarray] = {}

    for page in gt.pages:
        i = page.index
        notes: list[str] = []
        base = cache.get("src", source, i, configs[0])
        pc = PageCells.build(page, cfg.dpi, base.shape)
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
            band_stats[rc.label] = pc.stats(o, x, cfg)
            if k == 0:
                primary_o, primary_x = o, x
                # For the "replaced?" test, blank out bands where the original survives, so a
                # leftover original glyph is not mistaken for replacement text.
                masked = x.copy()
                for ci in np.flatnonzero(band_stats[rc.label].visible):
                    masked[pc.rows[ci], pc.cols[ci]] = 255
                primary_out[i] = masked

        prim = band_stats[labels[0]]
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
            mention_cells = [ci for ci, c in enumerate(pc.cells)
                             if (c.word_id in mention_words) or c.region_id is not None]
            layers = page_layers(out_doc, out_page, layer_cache)
            rec, ltn = layer_survival(pc, mention_cells, primary_o, layers, out_geom, cfg)
            n = len(pc.cells)
            img[i] = unit_flags(pc, prim.informative, _flags_from(rec, n))
            img_lat[i] = {u: v and not img[i].get(u, False)
                          for u, v in unit_flags(pc, prim.informative, _flags_from(ltn, n)).items()}
            vector[i] = vector_survival(src_doc[i], out_page, src_geoms[i], page, mention_words)
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
            st_ = pcs[i].stats(o, x, cfg)
            flags = unit_flags(pcs[i], st_.informative, st_.visible)
            rev[i] = {u: rev.get(i, {}).get(u, False) or v for u, v in flags.items()}
            centres, chars = page_chars(rdoc[i], rgeoms[i])
            for wid, got in text_in_words(page, centres, chars, mention_words).items():
                rev_texts[i][wid] = rev_texts[i].get(wid, "") + got

    # -- document level --------------------------------------------------------------------
    findings, surfaces = document_findings(output, gt, res, policy) if integrity.opens else ([], [])

    # -- mentions ----------------------------------------------------------------------------
    outcomes: list[MentionOutcome] = []
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
        f_vis = _fraction([vis[pg].get(u, False) for pg, u in units])
        f_lat = _fraction([lat[pg].get(u, False) for pg, u in units])
        f_img = _fraction([img.get(pg, {}).get(u, False) for pg, u in units])
        f_img_lat = _fraction([img_lat.get(pg, {}).get(u, False) for pg, u in units])
        f_rev_px = _fraction([rev.get(pg, {}).get(u, False) for pg, u in units])
        req_text = _required_text(m, words)
        total_chars = sum(len(t) for t in req_text.values())
        hit = sum(len(t) for (pg, wid), t in req_text.items() if text_leaks(t, texts.get(pg, {}).get(wid), cfg))
        rev_hit = sum(len(t) for (pg, wid), t in req_text.items()
                      if text_leaks(t, rev_texts.get(pg, {}).get(wid), cfg))
        f_text = hit / total_chars if total_chars else 0.0
        f_rev = max(f_rev_px, rev_hit / total_chars if total_chars else 0.0)
        vec_orig = vec_surv = 0
        for (pg, wid) in req_text:
            o_s = vector.get(pg, {}).get(wid)
            if o_s:
                vec_orig, vec_surv = vec_orig + o_s[0], vec_surv + o_s[1]
        f_vec = vec_surv / vec_orig if vec_orig else 0.0
        per_cfg = {lb: _fraction([vis_by_cfg[pg][lb].get(u, False) for pg, u in units]) for lb in labels}
        worst_cfg = max(per_cfg, key=per_cfg.get) if units and max(per_cfg.values()) > 0 else None

        polys_by_page: dict[int, list[np.ndarray]] = defaultdict(list)
        for seg in m.segments:
            polys_by_page[seg.page].extend(as_poly(words[wid][1].poly) for wid in seg.words)
        new_text = any(_is_new_text(words[wid][1].text, t, texts.get(pg, {}).get(wid), cfg)
                       for (pg, wid), t in req_text.items())
        replaced = new_text or any(_text_like_ink(primary_out[pg], polys, scale)
                                   for pg, polys in polys_by_page.items() if pg in primary_out)

        l2 = []
        if f_text > cfg.safe_max_residual:
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
        elif f_vis > cfg.safe_max_residual:
            status = MentionStatus.PARTIAL
        elif l2:
            status = MentionStatus.LEAK_RECOVERABLE
        elif max(f_lat, f_img_lat) > cfg.safe_max_residual:
            status = MentionStatus.LEAK_LATENT
        elif replaced:
            status = MentionStatus.SAFE_REPLACED
        else:
            status = MentionStatus.SAFE_REMOVED
        outcomes.append(MentionOutcome(
            mention_id=m.id, type=m.type.value, subtype=m.subtype, entity=m.entity,
            role=ent.role.value if ent else None, requirement=r.requirement.value, pending=r.pending,
            pages=pages, conditions=list(m.conditions), cells=len(units), visible_residual=round(f_vis, 4),
            worst_config=worst_cfg, latent_residual=round(f_lat, 4), text_residual=round(f_text, 4),
            image_residual=round(f_img, 4), image_latent_residual=round(f_img_lat, 4),
            vector_residual=round(f_vec, 4), revision_residual=round(f_rev, 4), replaced=replaced,
            l2_sources=l2, status=status))

    # -- words ---------------------------------------------------------------------------------
    word_outcomes: list[WordOutcome] = []
    for page in gt.pages:
        prim_units = vis_by_cfg[page.index][labels[0]]
        for w in page.words:
            eff = res.words.get(w.id, Effective.ORDINARY)
            units = [u for u in ((w.id, k) for k in range(len(w.text))) if u in prim_units]
            if not units or page.index in unreliable:
                word_outcomes.append(WordOutcome(w.id, page.index, eff.value, None, WordStatus.NOT_EVALUATED))
                continue
            frac = _fraction([prim_units[u] for u in units])
            if frac >= cfg.word_preserved_fraction:
                ws = WordStatus.PRESERVED
            elif frac <= cfg.word_removed_fraction:
                ws = WordStatus.REMOVED
            else:
                ws = WordStatus.DAMAGED
            word_outcomes.append(WordOutcome(w.id, page.index, eff.value, round(frac, 4), ws))

    summary = _summarise(outcomes, word_outcomes, findings, surfaces, integrity, page_outcomes)
    return DocumentEvaluation(
        doc_id=gt.doc_id, eval_version=cfg.version, eval_digest=cfg.digest(), policy=res.policy,
        source_sha256=hashlib.sha256(source).hexdigest(), output_sha256=hashlib.sha256(output).hexdigest(),
        render_configs=labels, integrity=integrity, mentions=outcomes, words=word_outcomes,
        pages=page_outcomes, findings=findings, surfaces=surfaces, revisions_checked=len(prefixes), summary=summary)


def _summarise(mentions, words, findings, surfaces, integrity, pages) -> dict:
    must = [m for m in mentions if m.requirement == Requirement.MUST.value]
    evaluated = [m for m in must if m.status is not MentionStatus.NOT_EVALUATED]
    by_status = {s.value: sum(1 for m in must if m.status is s) for s in MentionStatus}
    leaked_entities = {m.entity for m in evaluated if m.entity and m.status not in SAFE}
    leaked_entities |= {e for f in findings for e in f.entity_ids}
    must_entities = {m.entity for m in must if m.entity}
    done = [w for w in words if w.status is not WordStatus.NOT_EVALUATED]
    ordinary = [w for w in done if w.requirement == Effective.ORDINARY.value]
    protected = [w for w in done if w.requirement == Effective.MUST_NOT.value]
    collateral = [w for w in ordinary if w.status is not WordStatus.PRESERVED]
    protected_damage = [w for w in protected if w.status is not WordStatus.PRESERVED]
    must_words = [w for w in done if w.requirement == Effective.MUST.value]

    def redacted(w):
        return w.visible_fraction is not None and w.visible_fraction < 0.5

    tp = sum(1 for w in must_words if redacted(w))
    fn = len(must_words) - tp
    fp = sum(1 for w in ordinary + protected if redacted(w))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    must_surface_leaks = [s.surface_id for s in surfaces if s.leaked and s.requirement is Requirement.MUST]
    l1_l2 = [m for m in evaluated if m.status in L1_L2_LEAKS]
    may = [m for m in mentions if m.requirement == Requirement.MAY.value]

    def rate(n, d):
        return round(n / d, 4) if d else None

    return {
        "must_mentions": len(must),
        "must_evaluated": len(evaluated),
        "must_by_status": by_status,
        "must_leak_rate": rate(sum(1 for m in evaluated if m.status not in SAFE), len(evaluated)),
        "must_l1_l2_leak_rate": rate(len(l1_l2), len(evaluated)),
        "entities_with_must_mentions": len(must_entities),
        "entities_leaked": sorted(e for e in leaked_entities if e in must_entities),
        "entity_any_leak_rate": rate(len(leaked_entities & must_entities), len(must_entities)),
        "document_findings": len(findings),
        "must_surface_leaks": must_surface_leaks,
        "ordinary_words": len(ordinary),
        "collateral_words": len(collateral),
        "collateral_rate": rate(len(collateral), len(ordinary)),
        "protected_words": len(protected),
        "protected_damaged": [w.word_id for w in protected_damage],
        "effective_token_precision": round(precision, 4) if precision is not None else None,
        "effective_token_recall": round(recall, 4) if recall is not None else None,
        "effective_token_f1": round(f1, 4) if f1 is not None else None,
        "may_by_status": {s.value: sum(1 for m in may if m.status is s) for s in MentionStatus
                          if any(m.status is s for m in may)},
        "pending_mentions": sorted({m.pending for m in may if m.pending}),
        "integrity_ok": integrity.ok,
        "pages_not_evaluated": [p.index for p in pages if not p.evaluated],
        "clean_document": bool(integrity.ok and not l1_l2 and not findings and not must_surface_leaks
                               and not protected_damage and len(evaluated) == len(must)),
    }
