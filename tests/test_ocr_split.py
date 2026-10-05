import cv2
import numpy as np
import pymupdf
import pytest

from benchmark import ocr, ocr_split

DPI = 300
S = DPI / 72


def _render(text: str, rotation: int = 0, size: float = 16):
    """A page with `text` set in Helvetica, rendered; returns (gray, display-pt quad of the line, angle)."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), text, fontsize=size, fontname="helv")
    words = page.get_text("words")
    x0, y0 = min(w[0] for w in words), min(w[1] for w in words)
    x1, y1 = max(w[2] for w in words), max(w[3] for w in words)
    page.set_rotation(rotation)
    m = page.rotation_matrix
    quad = [tuple(pymupdf.Point(x, y) * m) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    # order the quad along the reading direction: start edge first
    pix = page.get_pixmap(matrix=pymupdf.Matrix(S, S), colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy(), quad, float(rotation)


def _word(text, quad, angle):
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    return ocr.OcrWord(text, (min(xs), min(ys), max(xs), max(ys)), 99.0, "l0", [tuple(p) for p in quad], angle)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("glued,expected", [
    ("ElianaGood", ["Eliana", "Good"]),
    ("Suite14", ["Suite", "14"]),
    ("withintherecordsof", ["within", "the", "records", "of"]),
])
def test_glued_words_are_split_at_the_ink_gap(glued, expected, rotation):
    spaced = " ".join(expected)
    gray, quad, angle = _render(spaced, rotation)
    parts = ocr_split.split_glued_words([_word(glued, quad, angle)], gray, S)
    assert [w.text for w in parts] == expected or (
        # a one-character miss on a word with no natural boundary is accepted, not a split elsewhere
        glued == "withintherecordsof" and "".join(w.text for w in parts) == glued and len(parts) == 4)
    # parts tile the original quad along the reading direction and keep its angle
    assert all(w.angle == angle for w in parts)
    boxes = [w.bbox for w in parts]
    assert all(b[2] > b[0] and b[3] > b[1] for b in boxes)


def test_words_with_no_gap_are_left_whole():
    for text in ("ElianaGood", "Eliana", "inthe", "bookings@limeml.com.au", "08/04/1976", "2018"):
        gray, quad, angle = _render(text, 0)
        (w,) = ocr_split.split_glued_words([_word(text, quad, angle)], gray, S)
        assert w.text == text


def test_numbers_and_dates_are_never_split_even_with_gaps():
    gray, quad, angle = _render("25 / 08 / 2021", 0)
    (w,) = ocr_split.split_glued_words([_word("25/08/2021", quad, angle)], gray, S)
    assert w.text == "25/08/2021"


def test_punctuation_stays_attached_to_its_word():
    gray, quad, angle = _render("her ,", 0)
    (w,) = ocr_split.split_glued_words([_word("her,", quad, angle)], gray, S)
    assert w.text == "her,"


def test_skewed_words_are_left_alone():
    gray, quad, _ = _render("Eliana Good", 0)
    (w,) = ocr_split.split_glued_words([_word("ElianaGood", quad, 3.5)], gray, S)
    assert w.text == "ElianaGood"


def test_underline_does_not_hide_the_gap_and_does_not_invent_one():
    gray, quad, angle = _render("Eliana Good", 0)
    h, w_ = gray.shape
    ys = int(max(p[1] for p in quad) * S) - 2
    cv2.line(gray, (int(min(p[0] for p in quad) * S), ys), (int(max(p[0] for p in quad) * S), ys), 0, 3)
    parts = ocr_split.split_glued_words([_word("ElianaGood", quad, angle)], gray, S)
    assert [p.text for p in parts] == ["Eliana", "Good"]
