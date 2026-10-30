"""The character identity gate (eval-0.4, benchmark/evaluate/cells.py): band evidence of a word
character stands only when the character as a whole is still the original glyph.

A fake typeset in the original's font, size and place can line up with the original's strokes in two
bands; the gate drops that. The original itself, shifted by a pixel, half under a box, blurred, inverted
or under a fake drawn over it, keeps its evidence. Fictional fixture content only.
"""

import io
import time
from functools import lru_cache

import cv2
import numpy as np
import pikepdf
import pymupdf
import pytest

from benchmark.evaluate import SAFE, EvalConfig, evaluate_output
from benchmark.evaluate.cells import CharacterIdentity, PageCells, unit_flags
from benchmark.evaluate.evaluator import _translate
from benchmark.evaluate.layers import layer_survival, page_layers
from benchmark.evaluate.render import RenderCache, RenderConfig
from benchmark.geometry import as_poly, bbox, geometry_from_pikepdf_page
from benchmark.mocks import MockSpec, run_mock
from benchmark.synth import ALL_VARIANTS

from tests import calib


@lru_cache(maxsize=None)
def strict() -> frozenset:
    return frozenset(w for w, r in calib.requirements().items() if r in ("MUST", "MAY"))


def _cells(o, dpi, page=0):
    cfg = EvalConfig(dpi=dpi)
    return cfg, PageCells.build(calib.base().gt.pages[page], dpi, o.shape, cfg.bands, cfg.identity_subbands)


def _counted(pc, st, words):
    flags = unit_flags(pc, st.informative, st.visible | st.latent)
    return {u for u, v in flags.items() if v and u[0] in words}


def test_a_fake_state_over_the_erased_original_counts_no_character():
    # The true-replace mock writes 'SA' where 'QLD' was, in the same font and size. At 150 DPI the band
    # test sees the tops of Q and L (eval-0.3.1 judged the locality PARTIAL); identity rejects them.
    fx = ALL_VARIANTS["digital"]()
    out = run_mock(fx.pdf, fx.gt, calib.V1, MockSpec("true-replace", replace_all=True, stand_ins=True)).pdf
    o, x = calib.render_pair(fx.pdf, out, 0, EvalConfig(dpi=150))
    cfg, pc = _cells(o, 150)
    qld = {"p0w22"}
    assert calib.words()["p0w22"].text == "QLD"
    assert _counted(pc, pc.band_stats(o, x, cfg), qld) == {("p0w22", 0), ("p0w22", 1)}
    st = pc.stats(o, x, cfg, strict())
    assert _counted(pc, st, qld) == set()
    assert {st.gate[u] for u in st.gate if u[0] == "p0w22"} == {""}


def _must_on_page0():
    return {w for w in calib.must_words() if w.startswith("p0")}


def _boxes(dpi):
    s = dpi / 72.0
    ws = calib.words()
    out = []
    for w in sorted(_must_on_page0()):
        x0, y0, x1, y1 = bbox(as_poly(ws[w].poly))
        out.append((int(x0 * s) - 1, int(y0 * s), int(np.ceil(x1 * s)) + 1, int(np.ceil(y1 * s)), (y0 + y1) / 2 * s))
    return out


def _one_px_off(o, dpi):
    return _translate(o, 1, 0)


def _half_under_a_black_box(o, dpi):
    x = o.copy()
    for c0, r0, c1, r1, mid in _boxes(dpi):
        x[int(mid):r1 + 1, c0:c1] = 0
    return x


def _blurred(o, dpi):
    return cv2.GaussianBlur(o, (0, 0), 1.2)


def _inverted(o, dpi):
    x = o.copy()
    for c0, r0, c1, r1, _ in _boxes(dpi):
        x[r0:r1, c0 + 1:c1 - 1] = 255 - x[r0:r1, c0 + 1:c1 - 1]
    return x


@pytest.mark.parametrize("dpi", [150, 200])
@pytest.mark.parametrize("change", [_one_px_off, _half_under_a_black_box, _blurred, _inverted],
                         ids=["1px-off", "half-boxed", "blurred-1.2px", "white-on-black"])
def test_the_original_degraded_keeps_its_evidence(change, dpi):
    b = calib.base()
    o, _ = calib.render_pair(b.pdf, b.pdf, 0, EvalConfig(dpi=dpi))
    cfg, pc = _cells(o, dpi)
    x = change(o, dpi)
    words = _must_on_page0()
    stock = _counted(pc, pc.band_stats(o, x, cfg), words)
    st = pc.stats(o, x, cfg, strict())
    assert len(stock) >= 100  # the change leaves the original legible to the band test...
    assert _counted(pc, st, words) == stock  # ...and the gate drops none of it
    assert all(st.gate[u] for u in st.gate if u[0] in words)


@pytest.mark.parametrize("dpi", [150, 200])
def test_a_fake_drawn_over_the_original_keeps_it_counted_through_presence(dpi):
    fk = calib.fakes()
    out = calib.build(word_ops={w: ("overprint", fk[w]) for w in calib.must_words()})
    b = calib.base()
    o, x = calib.render_pair(b.pdf, out, 0, EvalConfig(dpi=dpi))
    cfg, pc = _cells(o, dpi)
    words = _must_on_page0()
    stock = _counted(pc, pc.band_stats(o, x, cfg), words)
    st = pc.stats(o, x, cfg, strict())
    assert len(stock) >= 140 and _counted(pc, st, words) == stock
    assert sum(1 for u, why in st.gate.items() if u[0] in words and why == "presence") >= 50


def _identity_of(pc, values: dict) -> CharacterIdentity:
    rows = pc._identity_index().rows
    n = len(rows)
    ident = np.zeros(n)
    for u, v in values.items():
        ident[rows[u]] = v
    zeros = np.zeros((1, n, pc.bands))
    return CharacterIdentity(rows, np.ones(n, bool), ident, np.zeros(n), zeros, zeros)


def test_identity_thresholds_for_mention_and_kept_characters():
    # 0.80 for a letter of a MUST or MAY word, 0.85 for its digits; 0.70 and 0.75 for kept words. The
    # sign of the correlation does not matter (white text on black is the original too).
    b = calib.base()
    o, _ = calib.render_pair(b.pdf, b.pdf, 0, EvalConfig(dpi=150))
    cfg, pc = _cells(o, 150)
    ws, req = calib.words(), calib.requirements()
    claim = next(w for w in ws if req[w] == "MUST" and ws[w].text.startswith("CLM-"))  # letters and digits
    kept_digit = next(w for w in ws if req[w] in ("ORDINARY", "MUST_NOT") and ws[w].text[:2].isdigit())
    kept_letter = next(w for w in ws if req[w] == "ORDINARY" and ws[w].text[:2].isalpha())
    cases = [((claim, 0), 0.801, 0.799), ((claim, 4), 0.851, 0.849), ((kept_letter, 0), 0.701, 0.699),
             ((kept_digit, 0), 0.751, 0.749)]
    for u, passes, fails in cases:
        for value, why in ((passes, "identity"), (-passes, "identity"), (fails, "")):
            gate = pc.gate(_identity_of(pc, {u: value}), {u: {1, 2}}, strict(), cfg)
            assert gate[u] == why, (u, value)
    # Without the strict words every word is held to the MUST thresholds.
    assert pc.gate(_identity_of(pc, {(kept_letter, 0): 0.75}), {(kept_letter, 0): {1, 2}}, None, cfg) == {
        (kept_letter, 0): ""}


# -- image layers (S10): the same gate on the layer resampled to the render grid ----------------------


@pytest.mark.parametrize("raster_dpi", [150, 300])
def test_image_layer_gate_drops_same_font_fakes(raster_dpi):
    # Every MUST word replaced in place by a same-font fake, the page rasterised (image only). The layer
    # holds the fakes, not the originals, so no fake character keeps layer evidence that counts; the kept
    # 'CLM-' of the claim numbers is original and keeps it (shared from S14). In the 150-DPI layer two
    # fake letters where org1's legal form 'Pty' was pass the run clause (the documented limit of random
    # same-font matches); the legal form is shared from S14.
    b = calib.base()
    out = calib.same_font_fakes(raster_dpi)
    cfg = EvalConfig(dpi=200)
    strict_ = strict()
    doc = pymupdf.open("pdf", out)
    with pikepdf.open(io.BytesIO(out)) as pdf:
        geoms = [geometry_from_pikepdf_page(p) for p in pdf.pages]
    ws = calib.words()
    fake = calib.fake_characters(calib.fakes())
    claims = {(w, k) for w in calib.mention_words("clm0") + calib.mention_words("clm1") for k in range(4)}
    expected = {("p1w11", 0), ("p1w11", 1)} if raster_dpi == 150 else set()
    kept = set()  # characters whose layer evidence counts (two bands and the run rule, as for a mention)
    for page in b.gt.pages:
        o = RenderCache(200).get("src", b.pdf, page.index, RenderConfig("mupdf", True))
        pc = PageCells.build(page, 200, o.shape, cfg.bands, cfg.identity_subbands)
        cells = [i for i, c in enumerate(pc.cells) if c.word_id in strict_]
        layers = page_layers(doc, doc[page.index], {})
        rec, lat = layer_survival(pc, cells, o, layers, geoms[page.index], cfg, strict_)
        flags = np.zeros(len(pc.cells), bool)
        flags[[*rec, *lat]] = True
        informative = np.array([pc.rows[i].size >= 4 and o[pc.rows[i], pc.cols[i]].std() >= cfg.cell_min_orig_std
                                for i in range(len(pc.cells))])
        kept |= {u for u, v in unit_flags(pc, informative, flags).items() if v}
    assert kept & fake == expected, sorted((u, ws[u[0]].text[u[1]]) for u in (kept & fake) ^ expected)
    assert claims <= kept  # the original 'CLM-' keeps its layer evidence
    t0 = time.time()
    ev = evaluate_output(b.pdf, out, b.gt, calib.V1, cfg)
    print(f"P1 {raster_dpi}-DPI raster evaluated at 200 DPI with the layer gate in {time.time() - t0:.1f} s")
    text = set(calib.must_text_mentions())
    with_layer = {m.mention_id for m in ev.mentions if m.mention_id in text
                  and ("image_layer" in m.l2_sources or m.image_latent_residual > cfg.safe_max_residual)}
    assert with_layer == {"clm0", "clm1"} | ({"org1"} if raster_dpi == 150 else set()), with_layer
    assert all(m.status in SAFE for m in ev.mentions if m.mention_id in text - with_layer)


@pytest.mark.parametrize("variant", ["digital", "rotated_content"])
def test_identity_subbands_are_the_band_cells_split(variant):
    fx = ALL_VARIANTS[variant]()
    for page in fx.gt.pages:
        shape = (int(round(page.height_pt * 200 / 72)) + 1, int(round(page.width_pt * 200 / 72)) + 1)
        if page.rotation in (90, 270):
            shape = shape[::-1]
        pc = PageCells.build(page, 200, shape, 4, 16)
        fine = PageCells.build(page, 200, shape, 16)
        by = {(c.word_id, c.index, c.band): i for i, c in enumerate(fine.cells) if c.word_id is not None}
        n = 0
        for i, c in enumerate(pc.cells):
            if c.word_id is None:
                continue
            for j in range(4):
                sel = pc.subs[i] == c.band * 4 + j
                k = by[(c.word_id, c.index, c.band * 4 + j)]
                got = set(zip(pc.rows[i][sel].tolist(), pc.cols[i][sel].tolist()))
                assert got == set(zip(fine.rows[k].tolist(), fine.cols[k].tolist())), (c.word_id, c.index, c.band, j)
                n += len(got)
        assert n > 10000
