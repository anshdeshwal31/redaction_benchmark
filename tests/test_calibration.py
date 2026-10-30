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
    # 300-DPI JPEG, under the shared roles of policy 1.2.0: every MUST text mention SAFE_REPLACED and none
    # with image-layer evidence. The claim numbers keep their 'CLM-' prefix (shared, owner Q3); at 150 DPI
    # two fake letters where org1's 'Pty' was pass the run clause, inside the shared legal form (owner Q2).
    b = calib.base()
    out = calib.same_font_fakes(300 if form == "r300" else None)
    ev = evaluate_output(b.pdf, out, b.gt, calib.V1_SHARED, EvalConfig(dpi=dpi))
    assert _not_safe(ev) == {}
    layer = {m.mention_id for m in ev.mentions if m.mention_id in calib.must_text_mentions()
             and ("image_layer" in m.l2_sources or m.image_latent_residual > CFG.safe_max_residual)}
    assert layer == set(), layer


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


# -- shared structure (holes 10 and 11, owner decisions of 2026-10-22 and 2026-10-29) ---------------------
# Policy: the client's with the shared roles of 1.2.0 (calib.V1_SHARED).


def _shared(out, cfg=CFG):
    b = calib.base()
    return evaluate_output(b.pdf, out, b.gt, calib.V1_SHARED, cfg)


def _st(ev, mid) -> tuple[str, float]:
    m = ev.mention(mid)
    return m.status.value, m.visible_residual


@pytest.mark.parametrize("case", ["S1", "K1-vector", "K1-r300"])
def test_shared_words_are_safe(case):
    # Fakes that keep the shared words ('40 Bridge Street,', 'Bundall QLD 4217', 'Jabiru Pty Ltd'), in place
    # (S1) or on redrawn lines (K1): every MUST text mention SAFE_REPLACED, the shared words reported.
    out = {"S1": calib.shared_in_place(), "K1-vector": calib.shared_redraw(), "K1-r300": calib.shared_redraw(300)}[case]
    ev = _shared(out)
    assert _not_safe(ev) == {}
    ws = calib.words()
    assert [ws[w].text for w in ev.mention("addr1").shared_words] == ["Street,"]
    assert [ws[w].text for w in ev.mention("addr1-loc").shared_words] == ["QLD"]
    assert sorted(ws[w].text for w in ev.mention("org1").shared_words) == ["Ltd", "Pty"]
    assert [ok for _, ok in ev.mention("addr1-loc").shared_pairs] == [True]


def test_shared_words_never_excuse_a_surviving_original():
    # The state kept and the rest boxed (S2): no same-state postcode after the kept 'QLD', so it counts.
    ev = _shared(calib.shared_case("S2"))
    assert _st(ev, "addr1-loc") == ("PARTIAL", 0.2) and _st(ev, "addr1")[0] == "LEAK_RECOVERABLE"
    # The original postcode kept with a fake suburb (S3): the state and the prefix pass; '000' counts.
    assert _st(_shared(calib.shared_case("S3")), "addr1-loc") == ("PARTIAL", 0.2727)
    # The original locality kept on a redrawn line (K2), the street name kept (K3).
    assert _st(_shared(calib.locality_kept_redrawn()), "addr1-loc") == ("LEAK_VISIBLE", 1.0)
    assert _st(_shared(calib.shared_case("K3")), "addr1") == ("PARTIAL", 0.7778)


def test_shared_words_alone_are_not_a_replacement():
    # Only the shared words left, the rest erased (S4): the street type and the legal form are no
    # replacement; the kept 'QLD' has no same-state postcode after it and counts.
    ev = _shared(calib.shared_case("S4"))
    assert _st(ev, "addr1")[0] == "NOT_REPLACED" and _st(ev, "org1")[0] == "NOT_REPLACED"
    assert _st(ev, "addr1-loc") == ("PARTIAL", 0.2)


@pytest.mark.parametrize("raster_dpi", [None, 300], ids=["S5-vector", "S5r-r300"])
def test_kept_state_next_to_another_states_fake_is_a_leak(raster_dpi):
    # 'Parramatta QLD 2150' typed over 'Faketown QLD 4000': the kept 'QLD' beside another state's postcode
    # is not shared (the pair fails), so it counts. Same-state fakes (W0, K1) stay safe.
    assert _st(_shared(calib.shared_case("S5", raster_dpi)), "addr1-loc") == ("PARTIAL", 0.2)
    if raster_dpi is None:
        assert _not_safe(_shared(calib.shared_case("W0"))) == {}


def test_legal_form_is_shared():
    # Owner Q2: 'Jabiru Pty Ltd' for 'Acme Widgets Pty Ltd' is a replacement (S1, above); the company name
    # kept with the legal form (K4) is a visible leak.
    assert _st(_shared(calib.shared_in_place()), "org1")[0] == "SAFE_REPLACED"
    assert _st(_shared(calib.shared_case("K4")), "org1")[0] == "LEAK_VISIBLE"


def _q3(**changes) -> dict[str, str]:
    fk = calib.q3_fakes()
    for mid, text in changes.items():
        fk[calib.mention_words(mid)[0]] = text if text is not None else calib.words()[calib.mention_words(mid)[0]].text
    return fk


@pytest.mark.parametrize("case, expected", [
    ("Q3a-vector", {"em1": "SAFE_REPLACED", "clm0": "SAFE_REPLACED", "clm1": "SAFE_REPLACED"}),
    ("Q3a-r300", {"em1": "SAFE_REPLACED", "clm0": "SAFE_REPLACED", "clm1": "SAFE_REPLACED"}),
    ("Q3b", {"em1": "LEAK_VISIBLE", "clm0": "LEAK_VISIBLE", "clm1": "LEAK_VISIBLE"}),
    ("Q3c", {"em1": "LEAK_VISIBLE"}),
    ("Q3d-vector", {"clm0": "LEAK_VISIBLE", "clm1": "LEAK_VISIBLE"}),
    ("Q3d-r300", {"clm0": "LEAK_VISIBLE", "clm1": "LEAK_VISIBLE"}),
    ("Q3e", {"em1": "SAFE_REPLACED", "clm0": "SAFE_REPLACED", "clm1": "SAFE_REPLACED"}),
    ("Q3f", {"em1": "LEAK_VISIBLE", "clm0": "LEAK_VISIBLE", "clm1": "LEAK_VISIBLE"}),
])
def test_owner_q3_shared_structure(case, expected):
    # Owner Q3: a fake may keep the email domain and the identifier prefix, nothing else. Q3a keeps both
    # and replaces the rest; Q3b keeps the email and the claim numbers whole; Q3c keeps the local part with
    # another domain; Q3d keeps the digits with another prefix (the slice search finds them pushed along,
    # with or without a text layer); Q3e/Q3f as Q3a/Q3b on redrawn lines.
    raster = 300 if case.endswith("r300") else None
    kind = case.split("-")[0]
    if kind == "Q3a":
        out = calib.shared_in_place(_q3(), raster)
    elif kind == "Q3b":
        out = calib.shared_in_place(_q3(em1=None, clm0=None, clm1=None))
    elif kind == "Q3c":
        out = calib.shared_in_place(_q3(em1="jane.smithers@mail.test"))
    elif kind == "Q3d":
        out = calib.shared_in_place(_q3(clm0="BSA-48213", clm1="BSA-48213"), raster)
    elif kind == "Q3e":
        out = calib.build(line_ops=calib.redraw(_q3()))
    else:
        out = calib.build(line_ops=calib.redraw(_q3(em1=None, clm0=None, clm1=None)))
    ev = _shared(out)
    assert {mid: ev.mention(mid).status.value for mid in expected} == expected


def test_identifying_slice_is_searched():
    # Q3d as a 300-DPI raster (no text layer): the kept digits '48213' after a changed prefix 'BSA-' are
    # found pushed along the line, displaced originals.
    ev = _shared(calib.shared_in_place(_q3(clm0="BSA-48213", clm1="BSA-48213"), 300))
    for mid in ("clm0", "clm1"):
        assert ev.mention(mid).status.value == "LEAK_VISIBLE" and ev.mention(mid).displaced_residual > 0


@pytest.mark.parametrize("dpi", [150, 200])
def test_shared_structure_alone_is_not_a_replacement(dpi):
    # The partial-box mock (the left half of every mention word boxed) leaves '@example.com' in sight: a
    # domain alone is no replacement, so em1 is NOT_REPLACED, never SAFE_REPLACED. Under policy v0, which
    # shares nothing, every required mention stays PARTIAL.
    from benchmark.mocks import MockSpec

    b = calib.base()
    cfg = EvalConfig(dpi=dpi)
    out = run_mock(b.pdf, b.gt, calib.V1_SHARED, MockSpec("partial", partial=0.5)).pdf
    ev = evaluate_output(b.pdf, out, b.gt, calib.V1_SHARED, cfg)
    assert ev.mention("em1").status.value == "NOT_REPLACED"
    out0 = run_mock(b.pdf, b.gt, calib.V0, MockSpec("partial", partial=0.5)).pdf
    ev0 = evaluate_output(b.pdf, out0, b.gt, calib.V0, cfg)
    assert {m.status.value for m in ev0.mentions if m.requirement == "MUST"} == {"PARTIAL"}


def test_mobile_prefix_is_shared():
    # Owner answer (a), 2026-10-29: a fake mobile may keep its '04'. A realistic fake typed in place
    # ('0412 987 654') is a replacement under the 1.2.0 roles (without them it is PARTIAL); the number kept
    # whole, or its other digits kept behind a changed first word, still count.
    ph = calib.mention_words("ph1")
    ws = calib.words()
    assert ws[ph[0]].text.startswith("04")
    fk = calib.realistic_fakes()
    fk.update(zip(ph, ["0412", "987", "654"]))
    out = calib.build(word_ops=calib.in_place(fk))
    assert _st(_shared(out), "ph1")[0] == "SAFE_REPLACED"
    assert _st(_evaluate(out), "ph1")[0] == "PARTIAL"
    kept = calib.build(word_ops=calib.in_place(calib.keep(fk, ph)))
    assert _st(_shared(kept), "ph1")[0] == "LEAK_VISIBLE"
    rest = calib.build(word_ops=calib.in_place(calib.keep({**fk, ph[0]: "0491"}, ph[1:])))
    assert _st(_shared(rest), "ph1")[0] in ("PARTIAL", "LEAK_VISIBLE")
