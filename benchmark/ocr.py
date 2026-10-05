"""Measurement OCR (PLAN.md §5.9): a pinned, local instrument used for GT pre-annotation
and for reading replacement text from image-only outputs.

Two engines, selected with ``$RB_OCR_ENGINE``:

- ``ppocrv6`` (default): PaddleOCR PP-OCRv6 (medium detection and recognition, page
  orientation classifier) on ONNX Runtime, CPU. PaddleOCR pins numpy, PyYAML and OpenCV
  to versions that conflict with this package's pins, so it runs in a separate venv
  (``.venv-ocr``, locked by ``requirements-ocr.txt``; ``$RB_PPOCR_PYTHON`` points at
  another interpreter) through a persistent worker process (``_ppocr_worker.py``). Pages
  are rendered here with the pinned renderer and all geometry is computed here. Words the
  engine glued together are split where the page image shows a gap (``ocr_split``;
  ``RB_OCR_SPLIT=0`` turns that off).
- ``tesseract``: the Tesseract CLI (``$RB_TESSERACT`` if it is not on PATH).

The engine is never switched silently: when the selected engine is not installed, callers
get ``None`` and report the dependent evaluations as not evaluable. ``version()`` names the
engine, package versions and model-weight hashes so results can record what measured them.
"""

from __future__ import annotations

import atexit
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from . import ocr_split

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("_ppocr_worker.py")
ENGINES = ("ppocrv6", "tesseract")


@dataclass
class OcrWord:
    text: str
    bbox: tuple[float, float, float, float]  # display_pt
    conf: float
    line: str
    poly: list[tuple[float, float]] | None = None  # display_pt quadrilateral, when the engine gives one
    angle: float = 0.0  # baseline direction, degrees clockwise from +x (GTWord.angle convention)


def split_enabled() -> bool:
    return os.environ.get("RB_OCR_SPLIT", "1") != "0"


def engine() -> str:
    name = (os.environ.get("RB_OCR_ENGINE") or "ppocrv6").strip().lower()
    if name not in ENGINES:
        raise ValueError(f"RB_OCR_ENGINE={name!r}: expected one of {', '.join(ENGINES)}")
    return name


def tesseract_path() -> str | None:
    return os.environ.get("RB_TESSERACT") or shutil.which("tesseract")


def ppocr_python() -> str | None:
    env = os.environ.get("RB_PPOCR_PYTHON")
    exe = Path(env) if env else ROOT / ".venv-ocr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(exe) if exe.is_file() else None


def available() -> bool:
    return (tesseract_path() if engine() == "tesseract" else ppocr_python()) is not None


def version() -> str | None:
    if not available():
        return None
    if engine() == "tesseract":
        out = subprocess.run([tesseract_path(), "--version"], capture_output=True, encoding="utf-8",
                             errors="replace", timeout=30)
        first = (out.stdout or out.stderr).splitlines()
        return first[0] if first else None
    v = _worker().call({"op": "version"})["version"]
    pkgs = ", ".join(f"{k} {v['packages'][k]}" for k in ("paddleocr", "paddlex", "onnxruntime"))
    models = ", ".join(f"{name} {h or 'unhashed'}" for name, h in v["models"].items())
    split = f"; glued-word splitting v{ocr_split.VERSION}" if split_enabled() else ""
    return f"PP-OCRv6 ({pkgs}; {models}{split})"


def ocr_page(pdf: bytes, page_index: int, dpi: float = 300, lang: str = "eng") -> list[OcrWord] | None:
    """Words on the displayed page, or None when the selected engine is not installed.

    ``lang`` applies to Tesseract only; the PP-OCRv6 recogniser covers English and the
    other Latin-script languages with one model.
    """
    if not available():
        return None
    s = dpi / 72.0
    page = pymupdf.open("pdf", pdf)[page_index]
    if engine() == "tesseract":
        return _tesseract_page(page, s, lang)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csRGB)
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "page.png")
        pix.save(img)
        res = _worker().call({"op": "ocr", "image": img})
    words = ppocr_words(res, pix.width, pix.height, s, page.rect.width, page.rect.height)
    if split_enabled():
        gray = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2GRAY)
        words = ocr_split.split_glued_words(words, gray, s)
    return words


# ---------------------------------------------------------------- Tesseract

def _tesseract_page(page: pymupdf.Page, s: float, lang: str) -> list[OcrWord]:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "page.png")
        pix.save(img)
        # Tesseract writes UTF-8 whatever the locale (cp1252 decoding loses whole pages on Windows).
        out = subprocess.run([tesseract_path(), img, "stdout", "-l", lang, "--psm", "3", "tsv"], capture_output=True,
                             encoding="utf-8", errors="replace", timeout=600)
    return parse_tesseract_tsv(out.stdout, s)


def parse_tesseract_tsv(tsv: str, s: float) -> list[OcrWord]:
    words = []
    # TSV is unquoted: a token starting with '"' must not open a quoted field (QUOTE_NONE).
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        x, y, w, h = (float(row[k]) / s for k in ("left", "top", "width", "height"))
        words.append(OcrWord(text, (x, y, x + w, y + h), float(row.get("conf") or -1),
                             f"b{row['block_num']}p{row['par_num']}l{row['line_num']}"))
    return words


# ---------------------------------------------------------------- PP-OCRv6

class _Worker:
    """Persistent PP-OCRv6 process (model start-up takes seconds; pages are OCRed in turn)."""

    def __init__(self, python: str):
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen([python, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._err, encoding="utf-8")

    def call(self, request: dict) -> dict:
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except OSError:
            line = ""
        if not line:
            raise RuntimeError(f"PP-OCRv6 worker exited unexpectedly:\n{self._stderr_tail()}")
        out = json.loads(line)
        if not out.get("ok"):
            raise RuntimeError(f"PP-OCRv6 worker: {out.get('error')}")
        return out

    def _stderr_tail(self, n: int = 2000) -> str:
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace")[-n:]

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._err.close()


_worker_instance: _Worker | None = None


def _worker() -> _Worker:
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = _Worker(ppocr_python())
        atexit.register(_worker_instance.close)
    return _worker_instance


def _unrotate_affine(angle: int, w: int, h: int) -> np.ndarray:
    """2x3 affine from PaddleX's orientation-corrected image back to the input image.

    Mirrors ``paddlex ... warp_image.rotate_image``: ``cv2.getRotationMatrix2D`` about the
    centre (counter-clockwise for positive angles) on a canvas enlarged to fit; inverted.
    """
    if not angle:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t = math.radians(angle)
    a, b = math.cos(t), math.sin(t)
    cx, cy = w / 2, h / 2
    m = np.array([[a, b, (1 - a) * cx - b * cy], [-b, a, b * cx + (1 - a) * cy], [0.0, 0.0, 1.0]])
    new_w, new_h = int(h * abs(b) + w * abs(a)), int(h * abs(a) + w * abs(b))
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return np.linalg.inv(m)[:2]


def _split_words(tokens: list[str], boxes: list[list[float]], vertical: bool) -> list[tuple[str, float, float]] | None:
    """Whitespace words (the GT word definition) as (text, start, end) along the line axis.

    PaddleOCR tokens can carry spaces (' & ', ' (', ') '): each whitespace run is a word
    boundary and a punctuation piece takes its token's extent, joining the adjacent word.
    Returns None when tokens and boxes do not pair up.
    """
    if len(tokens) != len(boxes):
        return None
    words, text, lo, hi = [], "", math.inf, -math.inf

    def flush():
        nonlocal text, lo, hi
        if text:
            words.append((text, lo, hi))
        text, lo, hi = "", math.inf, -math.inf

    for tok, box in zip(tokens, boxes):
        a, b = (box[1], box[3]) if vertical else (box[0], box[2])
        for part in re.split(r"(\s+)", tok):
            if not part:
                continue
            if part.isspace():
                flush()
            else:
                text += part
                lo, hi = min(lo, a), max(hi, b)
    flush()
    return words


def _proportional_words(text: str, start: float, end: float) -> list[tuple[str, float, float]]:
    out, n, pos = [], max(len(text), 1), 0
    for part in text.split(" "):
        if part:
            out.append((part, start + (end - start) * pos / n, start + (end - start) * (pos + len(part)) / n))
        pos += len(part) + 1
    return out


def _snap(angle: float, tol: float = 10.0) -> float:
    """Scan skew is not reading direction: snap to the nearest multiple of 90 within ``tol``."""
    a = angle % 360.0
    k = round(a / 90.0) * 90.0
    return k % 360.0 if abs(a - k) <= tol else round(a, 1)


def ppocr_words(res: dict, width: int, height: int, s: float, page_w: float, page_h: float) -> list[OcrWord]:
    """Worker output -> words in display_pt with tight quadrilaterals following each line.

    Lines come as quadrilaterals (tl, tr, br, bl) in the orientation-corrected image, with
    word extents along the line axis (x for horizontal lines; y for vertical lines, which
    PaddleX reads top-to-bottom). Each word's quad is the line quad cut at its extent, then
    mapped back to the input image and scaled to points.
    """
    inv = _unrotate_affine(int(res.get("angle") or 0), width, height)

    def to_pt(p):
        x, y = inv @ np.array([p[0], p[1], 1.0])
        return (float(min(max(x / s, 0.0), page_w)), float(min(max(y / s, 0.0), page_h)))

    words: list[OcrWord] = []
    for k, line in enumerate(res.get("lines", [])):
        q = np.asarray(line["poly"], dtype=float)
        if q.shape != (4, 2):
            x0, y0 = q.min(axis=0)
            x1, y1 = q.max(axis=0)
            q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        tl, tr, br, bl = q
        ext = q.max(axis=0) - q.min(axis=0)
        vertical = ext[1] > 1.5 * max(ext[0], 1e-6)  # PaddleX's own vertical-text rule
        # Edges running along the reading direction, and the coordinate that parameterises them.
        (e1a, e1b), (e2a, e2b), axis = ((tl, bl), (tr, br), 1) if vertical else ((tl, tr), (bl, br), 0)
        parts = _split_words(line.get("tokens", []), line.get("token_boxes", []), vertical)
        if parts is None:
            parts = _proportional_words(line.get("text", ""), e1a[axis], e1b[axis])

        def at(a, b, v):
            d = b[axis] - a[axis]
            return a + (b - a) * ((v - a[axis]) / d if abs(d) > 1e-6 else 0.0)

        dvec = inv[:, :2] @ (e1b - e1a)
        angle = _snap(math.degrees(math.atan2(dvec[1], dvec[0])))
        for text, lo, hi in parts:
            if hi - lo < 1.0:
                hi = lo + 1.0
            quad = [to_pt(at(e1a, e1b, lo)), to_pt(at(e1a, e1b, hi)), to_pt(at(e2a, e2b, hi)), to_pt(at(e2a, e2b, lo))]
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            words.append(OcrWord(text, (min(xs), min(ys), max(xs), max(ys)), round(float(line["score"]) * 100, 1),
                                 f"l{k}", quad, angle))
    return words
