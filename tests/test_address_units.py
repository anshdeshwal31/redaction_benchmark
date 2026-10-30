"""Addresses judged as units (eval-0.4 S15: E1 replacement, E3 token survival, E4 short words).

Fictional cases from docs/address_cases.json, or written in its format here, rendered by
tests/address_fixtures.py and changed by simulated tools. Policy: the client's policy with the shared roles of
policy 1.2.0 (calib.V1_SHARED), so a fake keeps the state with a same-state postcode prefix, a street type
and the box words.
"""

import random
from collections import defaultdict

import pytest

from benchmark import ocr
from benchmark.evaluate import SAFE, EvalConfig, MentionStatus, evaluate_output
from benchmark.mocks import run_mock
from benchmark.mocks.named import NAMED
from benchmark.ocr import OcrWord
from tests import address_fixtures as af
from tests import calib

P = calib.V1_SHARED
CFG = EvalConfig(dpi=200, renderers=("mupdf",))

# 'Address: 12 Smith Street, Parramatta NSW 2150' on row 2, nothing on the next two rows.
ONE_LINE = af.one_case("t-one-line", [("Report prepared for the claimant.", 0),
                                      ("Address: 12 Smith Street, Parramatta NSW 2150", 2),
                                      ("The claimant attended on time.", 5)],
                       [("A1", [[1, 1], [1, 2], [1, 3]], [[1, 4], [1, 5], [1, 6]])])


def _ev(fx, out, cfg=CFG, policy=P):
    return evaluate_output(fx.pdf, out, fx.gt, policy, cfg)


def _st(ev) -> dict[str, str]:
    return {m.mention_id: m.status.value for m in ev.mentions if m.address_unit}


# -- E1: replacement per part ------------------------------------------------------------------------------


def test_an_erased_locality_is_not_replaced():
    fx = af.render(ONE_LINE)
    ev = _ev(fx, af.draw(fx, af.erases_locality(fx, P, "A1")))
    assert _st(ev) == {"A1": "SAFE_REPLACED", "A1-loc": "NOT_REPLACED"}
    # The street line's fake is no locality: the read-back has text but no locality shape.
    assert ev.mention("A1-loc").locality_shape == "none"


def test_a_flowing_fake_shorter_than_the_original_is_a_replacement():
    # One fake for the whole address from its first word: a longer street, then 'Ryde NSW 2112', ending before
    # where 'Parramatta NSW 2150' ended. The same-state 'NSW' and the postcode's '2' are drawn elsewhere along
    # the line, and the pair still holds.
    fx = af.render(ONE_LINE)
    ev = _ev(fx, af.draw(fx, af.flowing(fx, "A1", ["98 Coolabah Esplanade, Ryde NSW 2112"])))
    assert _st(ev) == {"A1": "SAFE_REPLACED", "A1-loc": "SAFE_REPLACED"}
    assert ev.mention("A1-loc").locality_shape == "suburb+state+postcode"


@pytest.mark.parametrize("image_only", [False, True])
def test_a_single_replaced_digit_is_a_replacement(image_only):
    # 'PO Box 9' -> 'PO Box 4' and 'Level 4' -> 'Level 9': the box and level words are shared structure, so the
    # one changed digit is the whole replacement, read from the text layer or, with no text layer, from the new
    # ink on its bands.
    case = af.one_case("t-digit", [("PO Box 9", 1), ("Penrith NSW 2750", 2), ("Level 4, 12 Smith Street", 5),
                                   ("Ryde NSW 2112", 6)],
                       [("A1", [[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]),
                        ("A2", [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]], [[3, 0], [3, 1], [3, 2]])])
    fx = af.render(case)
    out = af.draw(fx, af.correct(fx, P, fakes={"w0_2": "4", "w2_1": "9,"}))
    if image_only:
        out = af.raster(out)
    st = _st(_ev(fx, out))
    assert st["A1"] == st["A2"] == "SAFE_REPLACED", st


def test_a_locality_with_no_text_at_the_unit_is_not_replaced():
    # syn-rotate-attr-like: the page has a text layer, the street line's fake is in it, and the locality's fake
    # is only pixels. The text at the unit holds no locality, so the locality is not replaced.
    fx = af.render(ONE_LINE)
    street = af.correct(fx, P, keep=set(af.mention_words(fx, "A1-loc")))
    ops = street + [("erase", w) for w in af.mention_words(fx, "A1-loc")]
    p, x, y = fx.origin["w1_4"]
    ops.append(("image", p, x, y, "Ryde NSW 2112"))
    st = _st(_ev(fx, af.draw(fx, ops)))
    assert st == {"A1": "SAFE_REPLACED", "A1-loc": "NOT_REPLACED"}


# -- E3: token survival in the text layer --------------------------------------------------------------------


def test_a_kept_locality_pushed_to_the_next_line_is_a_text_leak():
    fx = af.render(ONE_LINE)
    ev = _ev(fx, af.draw(fx, af.next_line_locality(fx, P, "A1")))
    loc = ev.mention("A1-loc")
    assert loc.status in (MentionStatus.LEAK_RECOVERABLE, MentionStatus.PARTIAL, MentionStatus.LEAK_VISIBLE)
    assert "text_layer" in loc.l2_sources and loc.text_residual > 0
    assert ev.mention("A1").status is MentionStatus.SAFE_REPLACED


@pytest.mark.parametrize("case_id", ["document-full-letter", "document-verification-neighbours",
                                     "form-then-other-rows"])
def test_kept_text_sharing_a_units_numbers_is_no_survival(case_id):
    # '3 May 2024' and '12 June 2023' (unit 3, house number 12), 'Pain level 4' and '14/03/2021' (level 4,
    # postcode 2021), '12/03/1980' (house number 12): kept text, never a surviving original.
    fx = af.rendered(case_id)
    ev = _ev(fx, af.draw(fx, af.correct(fx, P)))
    assert set(_st(ev).values()) == {"SAFE_REPLACED"}, _st(ev)
    assert not [m.mention_id for m in ev.mentions if m.address_unit and m.l2_sources]


def test_a_fake_repeating_an_original_word_in_another_role_is_a_survival():
    # Reuse across roles: 'Wattle Grove' as the fake suburb of '42 Wattle Grove Road'.
    case = af.one_case("t-reuse", [("Address: 42 Wattle Grove Road, Penrith NSW 2750", 2)],
                       [("A1", [[0, 1], [0, 2], [0, 3], [0, 4]], [[0, 5], [0, 6], [0, 7]])])
    fx = af.render(case)
    ev = _ev(fx, af.draw(fx, af.flowing(fx, "A1", ["87 Banksia Road, Wattle Grove NSW 2173"])))
    street = ev.mention("A1")
    assert "text_layer" in street.l2_sources and street.status not in SAFE
    # The same fake with another suburb is a replacement.
    ok = _ev(fx, af.draw(fx, af.flowing(fx, "A1", ["87 Banksia Road, Lalor Park NSW 2147"])))
    assert set(_st(ok).values()) == {"SAFE_REPLACED"}, _st(ok)


# -- E4: short words --------------------------------------------------------------------------------------------


def test_a_short_number_counts_only_when_all_of_it_is_seen():
    # '17' typed over '14' at its origin shows the original '1': E4 counts '14' only when both digits are seen.
    fx = af.render(ONE_LINE)
    out = af.draw(fx, af.correct(fx, P, fakes={"w1_1": "17"}))
    ev = _ev(fx, out)
    assert ev.mention("A1").status is MentionStatus.SAFE_REPLACED and ev.mention("A1").visible_residual == 0
    # Without the rule the '1' counts.
    raw = _ev(fx, out, EvalConfig(dpi=200, renderers=("mupdf",), address_short_chars=0))
    assert raw.mention("A1").status is MentionStatus.PARTIAL


def test_a_fully_visible_short_word_makes_its_part_partial():
    # A 60-character street line with only its house number '7' kept: 1 of 55 identifying characters (under the
    # 0.02 safe limit), yet a fully visible identifying word makes the part at least PARTIAL.
    case = af.one_case("t-long", [("7 Coolangatta Bottlebrush Wollondilly Murrumbidgee Kurrajong Esplanade,", 1),
                                  ("Penrith NSW 2750", 2)],
                       [("A1", [[0, k] for k in range(7)], [[1, 0], [1, 1], [1, 2]])])
    fx = af.render(case)
    ev = _ev(fx, af.draw(fx, af.correct(fx, P, keep={"w0_0"})))
    street = ev.mention("A1")
    assert street.status is MentionStatus.PARTIAL and 0 < street.visible_residual <= CFG.safe_max_residual
    assert ev.mention("A1-loc").status is MentionStatus.SAFE_REPLACED


def test_a_number_holding_the_original_is_no_survival():
    # '214' over '14': a maximal digit run that only contains the original number is not it.
    fx = af.render(ONE_LINE)
    ev = _ev(fx, af.draw(fx, af.flowing(fx, "A1", ["214 Banksia Avenue, Ryde NSW 2112"])))
    assert _st(ev) == {"A1": "SAFE_REPLACED", "A1-loc": "SAFE_REPLACED"}


# -- the measurement OCR only asks for a review ------------------------------------------------------------


def test_the_ocr_adds_review_flags_and_never_changes_a_status(monkeypatch):
    fx = af.render(ONE_LINE)
    # Image only: the kept locality pushed to the next line, and a locality erased with no fake.
    pushed = af.raster(af.draw(fx, af.next_line_locality(fx, P, "A1")))
    erased = af.raster(af.draw(fx, af.erases_locality(fx, P, "A1")))
    before = {name: _st(_ev(fx, out)) for name, out in (("pushed", pushed), ("erased", erased))}
    p, x, y = fx.origin["w1_4"]

    def fake_ocr(pdf, page, dpi=300, lang="eng"):  # what an OCR would read on the pushed output
        return [OcrWord(t, (x + 40 * k, y + af.PITCH - 8, x + 40 * k + 30, y + af.PITCH + 2), 0.9, "l")
                for k, t in enumerate(["Parramatta", "NSW", "2150"])]

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_page", fake_ocr)
    ev = _ev(fx, pushed)
    assert _st(ev) == before["pushed"]
    assert "LEAK_SUSPECTED" in ev.mention("A1-loc").review_flags
    monkeypatch.setattr(ocr, "ocr_page", lambda *a, **k: [])
    ev = _ev(fx, erased)
    assert _st(ev) == before["erased"] and before["erased"]["A1-loc"] == "NOT_REPLACED"
    assert "LOCALITY_SHAPE_UNCONFIRMED" in ev.mention("A1-loc").review_flags


# -- regression: correct tools stay safe ----------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["reference-replace", "K1-redrawn"])
def test_correct_tools_on_the_fixture_stay_safe(tool):
    b = calib.base()
    if tool == "reference-replace":
        out = run_mock(b.pdf, b.gt, P, NAMED["reference-replace"]).pdf
    else:
        out = calib.build(line_ops=calib.redraw(calib.shared_fakes()))
    ev = evaluate_output(b.pdf, out, b.gt, P, EvalConfig(dpi=200))
    assert {m.mention_id: m.status.value for m in ev.mentions if m.address_unit} == {
        "addr1": "SAFE_REPLACED", "addr1-loc": "SAFE_REPLACED"}


def _family_cases() -> list[str]:
    """One single-page core case with address units per family of the case table (seeded). Left out: units whose
    state has no postcode prefix to pair with (NSW 1xxx, VIC 8xxx, QLD 9xxx, ACT 02xx: a kept state is counted
    there, a documented limit) and the two OCR-shaped units the role reader leaves to review."""
    skip_ids = {"ocr-letter-o-in-postcode", "ocr-state-n5w"}
    by_family = defaultdict(list)
    for c in af.cases().values():
        if c["tier"] != "core" or not c["expect"]["addresses"] or c["id"] in skip_ids:
            continue
        if len({line["page"] for line in c["lines"]}) != 1:
            continue
        if any(_no_prefix(u) for u in c["expect"]["addresses"]):
            continue
        by_family[c["family"]].append(c["id"])
    rng = random.Random(20261029)
    return [rng.choice(sorted(ids)) for _, ids in sorted(by_family.items())]


def _no_prefix(unit) -> bool:
    words = {w[3]: w[2] for w in unit["words"]}
    pc = next((w[2] for w in unit["words"] if w[3] == "postcode"), None)
    state = words.get("state")
    if not pc or not state or not pc.strip(".,").isdigit():
        return False
    first = {"nsw": "2", "act": "2", "vic": "3", "qld": "4", "sa": "5", "wa": "6", "tas": "7", "nt": "08"}
    code = state.strip(".,").replace(".", "").casefold()
    return code in first and not pc.strip(".,").startswith(first[code])


@pytest.mark.calibration
@pytest.mark.parametrize("case_id", _family_cases())
def test_the_correct_tool_is_safe_on_every_family(case_id):
    fx = af.rendered(case_id)
    ev = _ev(fx, af.draw(fx, af.correct(fx, P)), EvalConfig(dpi=200))
    assert set(_st(ev).values()) == {"SAFE_REPLACED"}, _st(ev)
