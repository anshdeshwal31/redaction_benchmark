"""Moved words and displaced originals (eval-0.4): kept words a tool redrew away from their place are
MOVED, not damage; a MUST word found away from its place is a displaced original, a leak. Fictional
outputs built from the digital fixture (tests/calib.py, the moved study's cases), scored under the
client's policy at 200 DPI.
"""

import math

import pytest

from benchmark.evaluate import SAFE, EvalConfig, evaluate_output
from benchmark.geometry import as_poly, bbox
from benchmark.gt import load_policy
from benchmark.mocks import MockSpec, run_mock
from benchmark.selfcheck import CHECKS, half_line_shift
from benchmark.synth import digital, mrc, rotated_content
from tests import calib

CFG = EvalConfig(dpi=200)
V0 = load_policy(calib.ROOT / "policies" / "au-medicolegal-v0.yaml")


def _evaluate(out, cfg=CFG):
    b = calib.base()
    return evaluate_output(b.pdf, out, b.gt, calib.V1, cfg)


def _statuses(ev) -> dict:
    return {w.word_id: w for w in ev.words}


def _centre_shift(wid: str, move: tuple[float, float, float]) -> tuple[float, float]:
    """Where the centre of a word's GT box goes when the word is drawn at its origin + (dx, dy), size x s."""
    dx, dy, s = move
    _, _, x0, baseline, _, _ = calib.base().layout[wid]
    bx0, by0, bx1, by1 = bbox(as_poly(calib.words()[wid].poly))
    return dx + (s - 1) * ((bx0 + bx1) / 2 - x0), dy + (s - 1) * ((by0 + by1) / 2 - baseline)


@pytest.mark.parametrize("raster_dpi", [None, 300, 150], ids=["vector", "r300", "r150"])
@pytest.mark.parametrize("case", ["A1", "A2", "A5"])
def test_redrawn_lines_are_moved_not_removed(case, raster_dpi):
    # Every line holding a MUST word re-typeset with fakes at 1.04x or 1.12x (A2 also 0.5 pt down and
    # 1 pt right), the way tool 2 redraws a line: its kept words are intact, so they are PRESERVED or
    # MOVED, never collateral, and each MOVED word is found where it was drawn.
    truth: dict = {}
    ev = _evaluate(calib.moved_case(case, raster_dpi, truth))
    words = _statuses(ev)
    assert truth and all(calib.requirements()[w] != "MUST" for w in truth)
    bad = {w: words[w].status.value for w in truth if words[w].status.value not in ("PRESERVED", "MOVED")}
    assert bad == {}, bad
    moved = [w for w in truth if words[w].status.value == "MOVED"]
    errors = {w: round(math.hypot(words[w].dx_pt - _centre_shift(w, truth[w])[0],
                                  words[w].dy_pt - _centre_shift(w, truth[w])[1]), 2) for w in moved}
    assert max(errors.values()) < 1.0, errors
    if case != "A1":  # A1 starts each line in place: its first words stay PRESERVED
        assert len(moved) == len(truth) == 65, (len(moved), len(truth))
    s = ev.summary
    assert s["collateral_real_words"] == 0 and s["protected_damaged"] == [] and s["must_words_displaced"] == []
    text = set(calib.must_text_mentions())
    # The locality's fake (two pool names and a 5-digit number) is no locality: with a text layer, E1 finds no
    # locality shape at the unit and the locality is NOT_REPLACED (S15).
    expected = {"addr1-loc": "NOT_REPLACED"} if raster_dpi is None else {}
    assert {m.mention_id: m.status.value for m in ev.mentions if m.mention_id in text and m.status not in SAFE} == \
        expected


@pytest.mark.parametrize("case", ["B4", "B12"])
def test_isolated_moves_are_found(case):
    # The last kept word of each line moved on its own: 58 pt along the line (B4), or 1 pt along and 0.3 pt
    # down at 1.02x (B12). All ten are found.
    truth: dict = {}
    ev = _evaluate(calib.moved_case(case, truth=truth))
    words = _statuses(ev)
    assert len(truth) == 10
    assert sorted(words[w].status.value for w in truth) == ["MOVED"] * 10
    assert ev.summary["collateral_real_words"] == 0


def test_a_drop_of_8_pt_is_a_known_limit():
    # The last kept word of each line moved 8 pt along and 8 pt down, more than half a line: the search
    # looks half a line across, so most are not found (documented limit; at most 4 of 10).
    truth: dict = {}
    ev = _evaluate(calib.moved_case("B9", truth=truth))
    words = _statuses(ev)
    assert sum(words[w].status.value == "MOVED" for w in truth) <= 4


def test_a_different_word_is_not_moved():
    # Every kept word replaced in place by a different word of about its length (C3): nothing is MOVED.
    ev = _evaluate(calib.moved_case("C3"))
    assert [w.word_id for w in ev.words if w.status.value == "MOVED"] == []


@pytest.mark.parametrize("case, collateral", [("D3", 24), ("D5", 3)])
def test_boxed_words_stay_damage(case, collateral):
    # A black box over the first half of every other kept word (D3), or shifted 7 pt below it (D5): those
    # words are damaged where they are, never MOVED (measured collateral 24 and 3).
    ev = _evaluate(calib.moved_case(case))
    assert [w.word_id for w in ev.words if w.status.value == "MOVED"] == []
    assert ev.summary["collateral_real_words"] == collateral


def test_repeated_words_traps():
    # E5: the protected 'strong' erased while the surname 'Strong' stays in place 73 pt to its left:
    # nothing is MOVED (the surname is an anchor; 'strong' is not found in it).
    ev = _evaluate(calib.moved_case("E5"))
    assert [w.word_id for w in ev.words if w.status.value == "MOVED"] == []
    # E7: no anchors left on the line and the surname redrawn 40 pt right: found, a displaced original,
    # so m3 is a visible leak.
    ev = _evaluate(calib.moved_case("E7"))
    assert ev.summary["must_words_displaced"] == ["p0w32"]
    assert ev.mention("m3").status.value == "LEAK_VISIBLE"
    assert ev.mention("m3").displaced_residual > 0
    # E2 (documented limit): 'is' erased, 'further' replaced by 'is' and its anchor 'treatment' erased:
    # the original 'is' is taken as moved onto the new 'is' (one false MOVED).
    ev = _evaluate(calib.moved_case("E2"))
    assert len([w for w in ev.words if w.status.value == "MOVED"]) == 1


FIXTURES = {"digital": digital, "mrc": mrc, "rotated_content": rotated_content}


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_no_mock_moves_a_word(fixture):
    # The self-check's mocks redact in place: any word they leave MOVED is within 1 pt of its place.
    fx = FIXTURES[fixture]()
    cfg = EvalConfig(dpi=150, renderers=("mupdf",))
    far = {}
    for c in CHECKS:
        spec = c.spec
        if c.name == "shift-half-line":
            spec = MockSpec("shift", method="rasterize", shift_pt=half_line_shift(fx.gt))
        mock = run_mock(fx.pdf, fx.gt, V0, spec)
        ev = evaluate_output(fx.pdf, mock.pdf, fx.gt, V0, cfg)
        far.update({(c.name, w.word_id): (w.dx_pt, w.dy_pt) for w in ev.words
                    if w.status.value == "MOVED" and math.hypot(w.dx_pt, w.dy_pt) > 1.0})
    assert far == {}, far


def test_words_retypeset_in_a_substitute_font_are_moved():
    # Every other kept word re-typeset at its place in Times (a substitute font): the original's own image
    # is not there, the GT text typeset in Times is. The words are MOVED (restyled), not collateral.
    ws = calib.words()
    some = calib.kept_real_words()[::2]
    ev = _evaluate(calib.build(word_ops={w: ("font", "tiro", None) for w in some}))
    words = _statuses(ev)
    assert {w: words[w].status.value for w in some if words[w].status.value not in ("PRESERVED", "MOVED")} == {}
    restyled = [w for w in some if words[w].restyled]
    assert len(restyled) >= 0.8 * len(some), (len(restyled), len(some))
    assert ev.summary["collateral_real_words"] == 0
    assert all(ws[w].text for w in restyled)
