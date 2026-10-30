"""eval-0.4 calibration on fictional adversarial outputs (tests/calib.py), the measurements behind the
thresholds in docs/metrics.md. Marked 'calibration': a quick loop can deselect them with
``-m 'not calibration'``; the full suite runs them.

Real leaks stay caught (a white box shifted half a line or a translucent box, with the fake typeset in
or over it), correct same-font fakes in place are not taken for the original, and the known limits are
pinned so that a change to them shows.
"""

import time
from collections import Counter

import pytest

from benchmark.evaluate import SAFE, EvalConfig, evaluate_output
from benchmark.evaluate.cells import PageCells, unit_flags
from benchmark.evaluate.render import RenderCache, RenderConfig, match_shapes
from benchmark.mocks import run_mock
from benchmark.mocks.named import NAMED
from tests import calib

pytestmark = pytest.mark.calibration
CFG = EvalConfig(dpi=200)  # the prototype's measurements: 200 DPI, MuPDF and PDFium


def _evaluate(out, cfg=CFG):
    b = calib.base()
    return evaluate_output(b.pdf, out, b.gt, calib.V1, cfg)


def _not_safe(ev) -> dict[str, str]:
    """MUST text mentions (the 20 the cases replace; the signature is left as it is) not judged safe."""
    text = set(calib.must_text_mentions())
    return {m.mention_id: m.status.value for m in ev.mentions if m.mention_id in text and m.status not in SAFE}


# -- real leaks the identity gate must not accept (holes 1 and 2) -------------------------------------


@pytest.mark.parametrize("f, fit, raster_dpi", [(-0.5, True, 300), (0.5, True, 300), (0.5, False, None),
                                                (0.5, False, 300)],
                         ids=["X1-0.5h-r300", "X1+0.5h-r300", "V1+0.5h-vector", "V1+0.5h-r300"])
def test_shift_white_with_fake_inside_is_a_leak(f, fit, raster_dpi):
    # A white box over each MUST mention shifted half a line, the fake typeset inside it: the uncovered
    # half of the original is a leak (measured 20, 18, 20, 20 of 20 with every eval-0.4 part on).
    ev = _evaluate(calib.shift_white(f, fit, raster_dpi))
    assert len(_not_safe(ev)) >= 18, Counter(_not_safe(ev).values())


@pytest.mark.parametrize("f", [0.3, 0.4])
def test_shift_white_by_less_than_half_a_line_is_pinned_at_the_stock_level(f):
    # Known limit: a box shifted 0.3-0.4 of a line with the fake inside leaves too little of the original
    # for the band test; eval-0.4 stays at eval-0.3.1's level (at most 14 of 20 judged safe).
    ev = _evaluate(calib.shift_white(f, True, 300))
    assert 20 - len(_not_safe(ev)) <= 14, _not_safe(ev)


@pytest.mark.parametrize("opacity", [0.3, 0.6, 0.9])
def test_translucent_box_with_fake_on_top_is_a_leak(opacity):
    # The original under a translucent white box with the fake drawn on top shows between the fake's
    # strokes (measured 18, 18, 17 of 20).
    ev = _evaluate(calib.veil(opacity))
    assert len(_not_safe(ev)) >= 17, Counter(_not_safe(ev).values())


# -- correct fakes in place ---------------------------------------------------------------------------


# Same-font fake characters the run clause admits at 200 DPI (measured; none of them makes its word
# count). The prototype's offline estimate of the run clause found none; the exact pooled identity finds
# this one in the PDFium render of the 150-DPI JPEG.
KNOWN_CLAUSE_MATCHES = {"P1-r150": [("pdfium", ("p0w32", 1))]}


@pytest.mark.parametrize("case", ["P1-vector", "P1-r300", "P1-r150", "K1-vector", "K1-r300"])
def test_partial_identity_clauses_admit_no_same_font_fake(case):
    # The edge and run clauses find partly covered originals; a correct fake in the original's font and
    # size must not pass them.
    kind, form = case.split("-")
    raster = {"vector": None, "r300": 300, "r150": 150}[form]
    out = calib.same_font_fakes(raster) if kind == "P1" else calib.shared_redraw(raster)
    fake = calib.fake_characters(calib.fakes() if kind == "P1" else calib.shared_fakes())
    b = calib.base()
    strict = frozenset(w for w, r in calib.requirements().items() if r in ("MUST", "MAY"))
    rejected, admitted, counted = 0, [], []
    for engine in ("mupdf", "pdfium"):
        cache = RenderCache(CFG.dpi)
        for page in b.gt.pages:
            o = cache.get("src", b.pdf, page.index, RenderConfig(engine, True))
            x, _ = match_shapes(o, cache.get("out", out, page.index, RenderConfig(engine, True)))
            pc = PageCells.build(page, CFG.dpi, o.shape, CFG.bands, CFG.identity_subbands)
            st = pc.stats(o, x, CFG, strict)
            flags = unit_flags(pc, st.informative, st.visible | st.latent)
            rejected += sum(1 for u, why in st.gate.items() if u in fake and why == "")
            admitted += [(engine, u) for u, why in st.gate.items() if u in fake and why in ("edge", "run")]
            counted += [(engine, u) for u in fake if flags.get(u)]
    assert rejected >= 20 and admitted == KNOWN_CLAUSE_MATCHES.get(case, []), admitted
    assert not set(admitted) & set(counted), counted  # an admitted fake character never counts


@pytest.mark.parametrize("form, dpi", [("vector", 150), ("vector", 200), ("r300", 200)])
def test_same_font_fakes_in_place_are_safe(form, dpi):
    # Every MUST word replaced at its origin and size by a fake in the same font (P1), as a PDF and as a
    # 300-DPI JPEG. Until shared roles (S14) the claim numbers keep their original 'CLM-', so clm0 and
    # clm1 are PARTIAL, and in the JPEG their image layer holds it. At 150 DPI two fake letters where
    # org1's legal form 'Pty' was pass the run clause (the documented limit of random same-font matches
    # in name words); the legal form is shared from S14.
    ev = _evaluate(calib.same_font_fakes(300 if form == "r300" else None), EvalConfig(dpi=dpi))
    expected = {"clm0", "clm1"} | ({"org1"} if dpi == 150 else set())
    assert set(_not_safe(ev)) == expected, _not_safe(ev)
    layer = {m.mention_id for m in ev.mentions if m.mention_id in calib.must_text_mentions()
             and ("image_layer" in m.l2_sources or m.image_latent_residual > CFG.safe_max_residual)}
    assert layer == (set() if form == "vector" else {"clm0", "clm1"}), layer


def test_random_other_state_localities_in_place_are_safe():
    # E5: the address line truly redacted and a random realistic whole-address fake in another state
    # typeset in its place, 15 seeded fakes. eval-0.3.1 flagged 7 of 15. The identity gate leaves 5:
    # four fakes whose house number starts with '1', the very glyph '14' starts with at the same place
    # (E4, S15: a 1-2 character address word counts only when all its characters are visible), and one
    # whose postcode digits pass the run clause in the PDFium render ('7486' over '4000').
    flagged = {}
    for k, fake in enumerate(calib.address_fakes()):
        ev = _evaluate(calib.address_in_place(fake))
        bad = {m.mention_id: m.status.value for m in ev.mentions
               if m.mention_id in ("addr1", "addr1-loc") and m.status not in SAFE}
        if bad:
            flagged[k] = bad
    assert flagged == {2: {"addr1": "PARTIAL"}, 4: {"addr1-loc": "PARTIAL"}, 6: {"addr1": "PARTIAL"},
                       12: {"addr1": "PARTIAL"}, 14: {"addr1": "PARTIAL"}}, flagged


@pytest.mark.parametrize("font", ["cour", "tiro", "hebo"])
def test_original_retypeset_in_another_font_is_a_leak(font):
    # Hole 3, vector part: each MUST original re-typeset in place in Courier, Times or Helvetica-Bold is
    # in the text layer at its displayed place, a text-layer leak on its own whatever the pixels show.
    ev = _evaluate(calib.retypeset_in_font(font))
    text = set(calib.must_text_mentions())
    missed = {m.mention_id for m in ev.mentions if m.mention_id in text
              and (m.status in SAFE or "text_layer" not in m.l2_sources)}
    assert missed == set(), missed


def test_lookalike_digit_in_protected_word_is_a_known_limit():
    # Hole 6: a protected date or dose altered by one look-alike digit in place (3/8, 5/6, 6/8) keeps a
    # high identity and reads PRESERVED; only a change the identity sees is damage.
    ev = _evaluate(calib.lookalikes())
    words = {w.word_id: w for w in ev.words}
    for wid in ("p0w16", "p0w45", "p0w48", "p0w69"):  # 12/08/1970, 60mg, 04/06/2021., (01.02.2028)
        assert words[wid].status.value == "PRESERVED", wid
    assert (words["p1w39"].status.value, words["p1w39"].visible_fraction) == ("DAMAGED", 0.75)  # (07.02.2026)


# -- moved words and displaced originals (holes 4 and 5) ---------------------------------------------


def test_kept_word_inside_a_fake_is_not_moved():
    # Hole 4 (M2): a name and the kept word after it ('on', 'is', 'in', 'and') erased by an over-extended
    # span and one fake typeset over it that happens to contain the kept word ('Ann Wilson'). The kept word
    # is not found inside the fake (it is not a whole word there): REMOVED, the same collateral as the
    # control whose fakes do not contain it (M2c).
    b = calib.ext_base()
    out, kept = calib.overextended_span()
    ev = evaluate_output(b.pdf, out, b.gt, calib.V1, CFG)
    words = {w.word_id: w.status.value for w in ev.words}
    assert [words[k] for k in kept] == ["REMOVED"] * 4
    ctrl, _ = calib.overextended_span(control=True)
    ev_c = evaluate_output(b.pdf, ctrl, b.gt, calib.V1, CFG)
    assert ev.summary["collateral_real_words"] == ev_c.summary["collateral_real_words"] == 4


def test_altered_word_in_place_is_not_moved():
    # Hole 5: a match at the GT place must pass the identity gate there. M1's 'far' typeset over 'for' stays
    # damage (REMOVED: its 'f' and 'r' are not a run), not MOVED; W1's same-state fake postcode '4006' is
    # not a displaced '4000'; B11's moves of 0.5 pt are real and stay MOVED (every moved word PRESERVED
    # or MOVED, none damaged).
    ev = _evaluate(calib.lookalikes())
    w = {x.word_id: x for x in ev.words}
    assert calib.words()["p1w13"].text == "for" and w["p1w13"].status.value in ("DAMAGED", "REMOVED")
    ev = _evaluate(calib.same_city_postcode())
    postcode = calib.mention_words("addr1-loc")[2]
    assert {x.word_id: x.status.value for x in ev.words}[postcode] != "MOVED"
    assert ev.summary["must_words_displaced"] == []
    truth: dict = {}
    ev = _evaluate(calib.moved_case("B11", truth=truth))
    st = {x.word_id: x.status.value for x in ev.words}
    assert {st[k] for k in truth} <= {"PRESERVED", "MOVED"} and "MOVED" in {st[k] for k in truth}, st


def test_displaced_originals_are_leaks():
    # K0c: every line with a MUST word re-typeset with its originals (1.04x, 0.3 pt down, 1 pt right),
    # rasterised at 300 DPI: the originals are found away from their place and every MUST text mention is
    # a visible leak.
    ev = _evaluate(calib.originals_redrawn())
    assert Counter(m.status.value for m in ev.mentions if m.mention_id in calib.must_text_mentions()) == {
        "LEAK_VISIBLE": 20}
    assert all(m.displaced_residual > 0 for m in ev.mentions if m.mention_id in calib.must_text_mentions())


def test_kept_locality_moved_by_a_longer_fake_street_is_a_leak():
    # K2: the redrawn address line keeps the original locality, which the longer fake street pushes along
    # the line: a displaced original, LEAK_VISIBLE in full.
    ev = _evaluate(calib.locality_kept_redrawn())
    m = ev.mention("addr1-loc")
    assert (m.status.value, m.visible_residual) == ("LEAK_VISIBLE", 1.0)


# -- substitute fonts: the restyled search (holes 3 and 9) ---------------------------------------------


@pytest.mark.parametrize("font, minimum", [("tiro", 20), ("hebo", 20), ("cour", 16)])
def test_original_retypeset_in_another_font_in_a_raster_is_a_leak(font, minimum):
    # Hole 3, raster part: each MUST original re-typeset in place in Times, Helvetica-Bold or Courier and
    # rasterised at 300 DPI (no text layer): the restyled search finds it, a displaced original. Courier
    # is the documented limit (measured 16 of 20).
    ev = _evaluate(calib.retypeset_in_font(font, raster_dpi=300))
    assert len(_not_safe(ev)) >= minimum, _not_safe(ev)


def test_lines_retypeset_in_a_substitute_font_are_moved():
    # Hole 9 (W4): the lines with fakes re-typeset in Times: their kept words are MOVED (restyled), not
    # collateral (the 22 Oct plan: 35 collateral and 8 protected words damaged).
    t0 = time.time()
    ev = _evaluate(calib.lines_in_font("tiro"))
    print(f"W4 evaluated at 200 DPI in {time.time() - t0:.1f} s")
    s = ev.summary
    assert s["collateral_real_words"] == 0 and s["protected_damaged"] == [], (s["collateral_real_words"],
                                                                              s["protected_damaged"])
    assert any(w.restyled for w in ev.words if w.status.value == "MOVED")


def _restyled(ev) -> list[str]:
    return [w.word_id for w in ev.words if w.restyled]


@pytest.mark.parametrize("case", ["P1-vector", "P1-r300", "P1-r150", "K1", "C1", "C3", "D1", "D2", "D3", "D4", "D5",
                                  "E1", "E2", "E3", "E4", "E5", "E6", "E7", "M1", "reference-replace"])
def test_restyled_search_finds_no_fake(case):
    # Fakes in the original's font, different words, erased, boxed and partly boxed words and the traps
    # with repeated words: the restyled search finds nothing. The margin over the original's own image
    # rejects M1's altered protected date '(07.02.2026)' and C1's one-letter garbles.
    if case.startswith("P1"):
        out = calib.same_font_fakes({"vector": None, "r300": 300, "r150": 150}[case.split("-")[1]])
    elif case == "K1":
        out = calib.shared_redraw()
    elif case == "M1":
        out = calib.lookalikes()
    elif case == "reference-replace":
        b = calib.base()
        out = run_mock(b.pdf, b.gt, calib.V1, NAMED["reference-replace"]).pdf
    elif case in ("C3", "D3", "D5", "E2", "E5", "E7"):
        out = calib.moved_case(case)
    else:
        out = calib.damage_case(case)
    assert _restyled(_evaluate(out)) == []
