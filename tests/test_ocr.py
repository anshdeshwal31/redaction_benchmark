import math

import pytest

from benchmark import ocr


def _line(text, poly, tokens, boxes, score=0.99):
    return {"text": text, "score": score, "poly": poly, "tokens": tokens, "token_boxes": boxes}


def test_tesseract_tsv_quote_does_not_swallow_following_rows():
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [f"5\t1\t1\t1\t1\t{n}\t{10 * n}\t10\t8\t10\t95.0\t{t}" for n, t in enumerate(['See', 'Annexure', '"D"'])]
    rows += [f"5\t1\t1\t1\t2\t{n}\t{10 * n}\t30\t8\t10\t95.0\t{t}" for n, t in enumerate(["Claimant", "Smith"])]
    words = ocr.parse_tesseract_tsv("\n".join([header, *rows]), 1.0)
    assert [w.text for w in words] == ["See", "Annexure", '"D"', "Claimant", "Smith"]


def test_unknown_engine_is_an_error(monkeypatch):
    monkeypatch.setenv("RB_OCR_ENGINE", "docTR")
    with pytest.raises(ValueError):
        ocr.engine()


def test_engine_not_installed_means_not_evaluable():
    # tests/conftest.py points RB_PPOCR_PYTHON at a missing interpreter
    assert not ocr.available() and ocr.version() is None and ocr.ocr_page(b"", 0) is None


def test_ppocr_words_split_tokens_with_embedded_spaces():
    s = 300 / 72
    toks = ["Ask", " & ", "clarify", " (", "NSSI", ") ", "quiz"]
    boxes = [[100 + 50 * k, 100, 140 + 50 * k, 140] for k in range(len(toks))]
    res = {"angle": 0, "lines": [_line("Ask & clarify (NSSI) quiz", [[100, 100], [440, 100], [440, 140], [100, 140]],
                                       toks, boxes)]}
    words = ocr.ppocr_words(res, 2480, 3508, s, 595, 842)
    assert [w.text for w in words] == ["Ask", "&", "clarify", "(NSSI)", "quiz"]
    nssi = words[3]
    assert nssi.bbox[0] == pytest.approx(250 / s) and nssi.bbox[2] == pytest.approx(390 / s)
    assert all(w.angle == 0 for w in words) and all(w.conf == 99.0 for w in words)


def test_ppocr_words_follow_a_skewed_line():
    # A line rising 10 px over 1000 px: word quads follow the slope; the angle snaps to 0.
    poly = [[0, 110], [1000, 100], [1000, 140], [0, 150]]
    res = {"angle": 0, "lines": [_line("a b", poly, ["a", " ", "b"], [[0, 100, 400, 150], [400, 100, 500, 150],
                                                                     [500, 100, 1000, 150]])]}
    a, b = ocr.ppocr_words(res, 2000, 2000, 1.0, 2000, 2000)
    assert a.poly[0] == pytest.approx((0, 110)) and a.poly[1] == pytest.approx((400, 106))
    assert b.poly[2] == pytest.approx((1000, 140))
    assert a.angle == 0


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_ppocr_words_map_orientation_corrected_boxes_back(angle):
    # Forward-rotate a known word box like PaddleX does, then check the mapping returns it.
    import cv2
    import numpy as np

    w, h = 2480, 3508
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s_ = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    fwd = lambda x, y: tuple(m @ np.array([x, y, 1.0]))  # noqa: E731
    # In the corrected image the word sits on a horizontal line at (500..900, 1000..1040).
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
    orig = [tuple(inv @ np.array([x, y, 1.0])) for x, y in [(500, 1000), (900, 1000), (900, 1040), (500, 1040)]]
    assert [fwd(*p) for p in orig][0] == pytest.approx((500, 1000))
    res = {"angle": angle, "lines": [_line("word", [[500, 1000], [900, 1000], [900, 1040], [500, 1040]],
                                           ["word"], [[500, 1000, 900, 1040]])]}
    (word,) = ocr.ppocr_words(res, w, h, 1.0, w, h)
    assert np.allclose(np.array(word.poly), np.array(orig), atol=1e-6)
    # Text that reads left-to-right once the image is turned `angle`° counter-clockwise reads at
    # `angle`° clockwise in the original (e.g. 270: bottom-to-top, as on the rotated NOC annexures).
    assert word.angle == angle


def test_ppocr_words_vertical_line_reads_top_to_bottom():
    poly = [[100, 100], [140, 100], [140, 500], [100, 500]]
    res = {"angle": 0, "lines": [_line("WHSQ 12286", poly, ["WHSQ", " ", "12286"],
                                       [[100, 100, 140, 260], [100, 260, 140, 300], [100, 300, 140, 500]])]}
    a, b = ocr.ppocr_words(res, 1000, 1000, 1.0, 1000, 1000)
    assert (a.text, b.text) == ("WHSQ", "12286")
    assert a.angle == 90 and a.bbox[1] == pytest.approx(100) and a.bbox[3] == pytest.approx(260)
    assert b.bbox[1] == pytest.approx(300) and math.isclose(b.bbox[3], 500)


def test_ppocr_words_fall_back_to_proportional_split():
    poly = [[0, 0], [100, 0], [100, 10], [0, 10]]
    res = {"angle": 0, "lines": [_line("ab cd", poly, ["ab cd"], [])]}
    a, b = ocr.ppocr_words(res, 200, 200, 1.0, 200, 200)
    assert (a.text, b.text) == ("ab", "cd") and a.bbox[2] == pytest.approx(40) and b.bbox[0] == pytest.approx(60)
