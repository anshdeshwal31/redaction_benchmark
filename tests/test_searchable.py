"""Searchable text found at the word's displayed place (eval-0.4, client Q11).

A kept word is searchable when the output's text layer holds it on its own line, within the word plus
a quarter line height along it (the page's registration shift and a resized page followed). The cases
put a hidden Helvetica text layer over the synthetic scan (image only, so every kept word is preserved)
in the ways tools write theirs. Text-layer alignment and word breaks are reported, never a failure.
Fictional fixture content only.
"""

import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pymupdf
import pytest

from benchmark.evaluate import EvalConfig, evaluate_output
from benchmark.evaluate.layers import (
    PageText,
    TextChar,
    find_at_place,
    separated,
    token_leaks,
    token_span,
    tokens_in_zone,
)
from benchmark.gt.policy import Effective, resolve
from benchmark.synth import ALL_VARIANTS
from tests.calib import V1

CFG = EvalConfig(dpi=150)
HELV = pymupdf.Font("helv")
HELV_LINE, HELV_MID = 1.075 + 0.299, (1.075 - 0.299) / 2  # Helvetica ascender + descender, and centring


@lru_cache(maxsize=None)
def scan():
    return ALL_VARIANTS["raster"]()


@lru_cache(maxsize=None)
def kept() -> frozenset:
    res = resolve(scan().gt, V1)
    return frozenset(w for w, e in res.words.items() if e in (Effective.ORDINARY, Effective.MUST_NOT))


def _place(page, text, x0, x1, yc, h, dx=0.0, dy=0.0, size_factor=1.0, fit=True, start_off=0.0, k=1.0):
    size = h / HELV_LINE * size_factor * k
    natural = HELV.text_length(text, fontsize=size)
    sx = (x1 - x0) * k / natural if (fit and natural > 0) else 1.0
    start = pymupdf.Point(x0 * k + start_off + dx, yc * k + HELV_MID * size + dy)
    page.insert_text(start, text, fontsize=size, fontname="helv", render_mode=3,
                     morph=(start, pymupdf.Matrix(sx, 1)) if abs(sx - 1) > 1e-3 else None)


@lru_cache(maxsize=None)
def layer(case: str, seed: int = 7) -> bytes:
    """The scan with a hidden text layer for its kept words, written the way ``case`` says."""
    rng = random.Random(seed)
    src = pymupdf.open("pdf", scan().pdf)
    k, offset = 1.0, (0.0, 0.0)
    if case in ("resized", "shifted"):  # the page itself changed: draw the scan into a new page
        k = 1.25 if case == "resized" else 1.0
        offset = (6.0, 4.0) if case == "shifted" else (0.0, 0.0)
        doc = pymupdf.open()
        for p in src:
            new = doc.new_page(width=p.rect.width * k, height=p.rect.height * k)
            r = pymupdf.Rect(0, 0, p.rect.width * k, p.rect.height * k) + (*offset, *offset)
            new.show_pdf_page(r, src, p.number)
    else:
        doc = src
    for p in scan().gt.pages:
        page = doc[p.index]
        ws = sorted((w for w in p.words if w.id in kept()), key=lambda w: (w.line or "", w.order))
        if case == "reversed":  # each line's words written last to first
            by_line = defaultdict(list)
            for w in ws:
                by_line[w.line].append(w)
            ws = [w for line in by_line.values() for w in reversed(line)]
        for w in ws:
            q = np.array(w.poly)
            x0, x1, yc, h = q[:, 0].min(), q[:, 0].max(), q[:, 1].mean(), q[:, 1].max() - q[:, 1].min()
            if case == "drop10" and rng.random() < 0.10:
                continue
            if case.startswith("along"):
                _place(page, w.text, x0, x1, yc, h, dx=float(case[5:]))
            elif case == "next_line":
                _place(page, w.text, x0, x1, yc, h, dy=20.0)
            elif case == "tool2_like":  # Helvetica, size from the box height, no horizontal scaling
                _place(page, w.text, x0, x1, yc, h, size_factor=rng.uniform(0.85, 1.35), fit=False,
                       start_off=rng.uniform(-1.5, 0.5))
            elif case == "misscaled":  # coordinates taken as 300-DPI pixels read as points
                _place(page, w.text, x0 * 72 / 300, x1 * 72 / 300, yc * 72 / 300, h)
            elif case in ("resized", "shifted"):
                _place(page, w.text, x0, x1, yc, h, dx=offset[0], dy=offset[1], k=k)
            else:  # exact, drop10, reversed
                _place(page, w.text, x0, x1, yc, h)
    return doc.tobytes()


@lru_cache(maxsize=None)
def summary(case: str):
    fx = scan()
    return evaluate_output(fx.pdf, layer(case), fx.gt, V1, CFG).summary


@pytest.mark.parametrize("case", ["exact", "along3", "along4", "tool2_like"])
def test_a_few_points_of_drift_and_tool_style_retypesetting_stay_searchable(case):
    s = summary(case)
    assert s["searchable_words"] >= 60 and s["searchable_rate"] >= 0.95 and s["searchable_ok"], s["searchable_rate"]


@pytest.mark.parametrize("case, below", [("next_line", 0.01), ("along10", 0.95), ("along25", 0.30),
                                         ("misscaled", 0.01)])
def test_text_on_another_line_or_far_along_it_is_not_searchable(case, below):
    s = summary(case)
    assert s["searchable_rate"] < below and not s["searchable_ok"], s["searchable_rate"]


def test_dropped_words_are_never_found():
    s = summary("drop10")
    rng, dropped = random.Random(7), 0
    for p in scan().gt.pages:  # the builder's draws, in its order; counted words have a letter or digit
        for w in sorted((w for w in p.words if w.id in kept()), key=lambda w: (w.line or "", w.order)):
            dropped += rng.random() < 0.10 and any(c.isalnum() for c in w.text)
    assert dropped >= 3
    assert s["searchable_rate"] == round((s["searchable_words"] - dropped) / s["searchable_words"], 4)
    assert s["searchable_missing_words"] + s["searchable_misplaced_words"] == dropped and not s["searchable_ok"]


@pytest.mark.parametrize("case", ["shifted", "resized"])
def test_the_registration_shift_and_a_resized_page_are_followed(case):
    fx = scan()
    ev = evaluate_output(fx.pdf, layer(case), fx.gt, V1, CFG)
    if case == "shifted":
        assert all(p.registration_shift_px for p in ev.pages)
    else:
        assert all(p.geometry_changed for p in ev.pages)
    s = ev.summary
    assert s["searchable_words"] >= 60 and s["searchable_rate"] >= 0.95 and s["searchable_ok"], s["searchable_rate"]


def test_text_alignment_is_reported():
    s = summary("along3")
    assert abs(s["text_offset_median_pt"] - 3.0) < 0.25 and abs(s["text_width_ratio_median"] - 1.0) < 0.05
    assert s["text_offset_p95_pt"] >= s["text_offset_median_pt"] and s["text_offset_max_pt"] >= s["text_offset_p95_pt"]
    exact = summary("exact")
    assert exact["text_offset_median_pt"] < 0.25 and exact["searchable_misplaced_words"] == 0


def test_word_breaks_are_informational():
    exact, tool2, rev = summary("exact"), summary("tool2_like"), summary("reversed")
    # Separate word objects with gaps and no space characters read as separate words.
    assert exact["text_word_pairs"] >= 30 and exact["text_glued_pairs"] == 0 and exact["text_out_of_order_pairs"] == 0
    # Words typeset like tool 2 (no horizontal scaling) run into each other...
    assert tool2["text_glued_rate"] > 0.3 and all("|" in e for e in tool2["text_glued_examples"])
    assert all(part.startswith("p") for e in tool2["text_glued_examples"] for part in e.split("|"))  # word ids only
    # ...and a line written last word first reads out of order; neither fails the document.
    assert rev["text_out_of_order_pairs"] >= 20
    for s in (tool2, rev):
        assert s["searchable_ok"]
        assert {k: s[k] for k in ("integrity_ok", "collateral_ok", "protected_damaged", "must_by_status")} == {
            k: exact[k] for k in ("integrity_ok", "collateral_ok", "protected_damaged", "must_by_status")}


# -- find_at_place on hand-made text ---------------------------------------------------------------


def _chars(text, x0, y0, w=5.0, h=10.0, line=0, seq0=0, step=None):
    """Characters of ``text`` left to right from (x0, y0), one ``w`` wide, extraction order seq0..."""
    step = w if step is None else step
    return [TextChar(c, (x0 + k * step, y0, x0 + k * step + w, y0 + h), line, seq0 + k) for k, c in enumerate(text)]


def _poly(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_find_at_place_cases():
    word = _poly(100, 100, 135, 110)  # 'Medical', 7 characters of 5 pt
    assert abs(find_at_place(word, "Medical", _chars("Medical", 100, 100)).along) < 1e-9
    assert find_at_place(word, "Medical.", _chars("Medical", 102, 100)) is not None  # stripped, 2 pt drift
    assert find_at_place(word, "Medical", _chars("Medical", 100, 125)) is None  # next line
    assert find_at_place(word, "Medical", _chars("Medical", 100, 104.9)) is not None  # 0.49 H across
    assert find_at_place(word, "Medical", _chars("Medical", 100, 105.1)) is None  # 0.51 H across
    # glued tokens: 'of' after 'Date' with no space is still at its place
    glued = _chars("Dateof", 80, 100)
    assert find_at_place(_poly(100, 100, 110, 110), "of", glued) is not None
    # one OCR error in a long word
    long_word = _poly(100, 100, 165, 110)
    assert find_at_place(long_word, "physiotherapy", _chars("physiotberapy", 100, 100)) is not None
    # the same word twice: the nearest is taken, a far one alone is rejected
    near_far = _chars("and", 100, 100) + _chars(" ", 115, 100, seq0=3) + _chars("and", 140, 100, seq0=4)
    assert abs(find_at_place(_poly(100, 100, 115, 110), "and", near_far).along) < 1e-9
    assert find_at_place(_poly(100, 100, 115, 110), "and", _chars("and", 140, 100)) is None
    # a missing word between neighbours that are present
    line = _chars("The", 80, 100) + _chars(" ", 95, 100, seq0=3) + _chars("was", 140, 100, seq0=4)
    assert find_at_place(_poly(100, 100, 135, 110), "report", line) is None


@pytest.mark.parametrize("angle", [90, 270])
def test_find_at_place_on_rotated_words(angle):
    if angle == 90:  # reads top to bottom, letter tops towards +x
        word = [(210, 100), (210, 130), (200, 130), (200, 100)]
        chars = [TextChar(c, (200, 100 + 5 * k, 210, 105 + 5 * k), 0, k) for k, c in enumerate("Report")]
    else:  # reads bottom to top
        word = [(200, 130), (200, 100), (210, 100), (210, 130)]
        chars = [TextChar(c, (200, 125 - 5 * k, 210, 130 - 5 * k), 0, k) for k, c in enumerate("Report")]
    run = find_at_place(word, "Report", chars)
    assert run is not None and abs(run.along) < 1e-9 and abs(run.width_ratio - 1.0) < 1e-9
    moved = [TextChar(c.c, (c.box[0] + 12, c.box[1], c.box[2] + 12, c.box[3]), 0, c.seq) for c in chars]
    assert find_at_place(word, "Report", moved) is None  # a line away across


def test_whole_tokens_match_numbers_as_maximal_digit_runs():
    word = _poly(100, 100, 110, 110)  # '14'
    glued = _chars("142750", 100, 100)
    assert find_at_place(word, "14", glued) is not None  # a substring...
    assert find_at_place(word, "14", glued, whole_token=True) is None  # ...but not a whole token
    assert find_at_place(word, "14", _chars("214", 95, 100), whole_token=True) is None
    own = _chars("14", 100, 100) + _chars(" ", 110, 100, seq0=2) + _chars("2750", 115, 100, seq0=3)
    assert find_at_place(word, "14", own, whole_token=True) is not None
    assert token_span("4000", "QLD4000", 90) == (3, 7) and token_span("4000", "40001", 90) is None
    assert token_span("Faketown,", "faketown", 90) is not None and token_span("St", "Sts", 90) is None
    # letters as a maximal letter run: a glued state, a suburb the GT splits off a word
    assert token_span("NSW", "NSW2750", 90) == (0, 3) and token_span("Penrith", "St,Penrith", 90) == (3, 10)
    assert token_span("Ann", "Annabel", 90) is None
    assert tokens_in_zone(word, PageText(glued)) == ["142750"] and not token_leaks("14", ["142750"], CFG)
    assert tokens_in_zone(word, PageText(own)) == ["14"] and token_leaks("14", ["14"], CFG)


def test_separated_reads_spaces_line_breaks_and_order():
    a = _chars("Date", 80, 100)
    run_a = find_at_place(_poly(80, 100, 100, 110), "Date", a + _chars("of", 100, 100, seq0=4))
    run_b = find_at_place(_poly(100, 100, 110, 110), "of", a + _chars("of", 100, 100, seq0=4))
    assert separated(a + _chars("of", 100, 100, seq0=4), run_a, run_b) is False  # glued
    spaced = a + _chars(" ", 100, 100, seq0=4) + _chars("of", 103, 100, seq0=5)
    sa = find_at_place(_poly(80, 100, 100, 110), "Date", spaced)
    sb = find_at_place(_poly(103, 100, 113, 110), "of", spaced)
    assert separated(spaced, sa, sb) is True
    broken = a + _chars("of", 100, 100, line=1, seq0=4)
    assert separated(broken, find_at_place(_poly(80, 100, 100, 110), "Date", broken),
                     find_at_place(_poly(100, 100, 110, 110), "of", broken)) is True
    assert separated(spaced, sb, sa) is None  # out of order
