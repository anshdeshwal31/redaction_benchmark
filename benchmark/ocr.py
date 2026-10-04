"""Measurement OCR (PLAN.md §5.9): a pinned, local instrument used for GT pre-annotation
and for reading replacement text from image-only outputs.

Only Tesseract (CLI) is supported for now. It is optional: if the binary is missing,
callers get ``None`` and report the dependent evaluations as not evaluable. Install it
(e.g. the UB Mannheim Windows build) and set $RB_TESSERACT if it is not on PATH.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import pymupdf


@dataclass
class OcrWord:
    text: str
    bbox: tuple[float, float, float, float]  # display_pt
    conf: float
    line: str


def tesseract_path() -> str | None:
    return os.environ.get("RB_TESSERACT") or shutil.which("tesseract")


def version() -> str | None:
    exe = tesseract_path()
    if not exe:
        return None
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    first = (out.stdout or out.stderr).splitlines()
    return first[0] if first else None


def ocr_page(pdf: bytes, page_index: int, dpi: float = 300, lang: str = "eng") -> list[OcrWord] | None:
    """Words on the displayed page, or None when no OCR engine is installed."""
    exe = tesseract_path()
    if not exe:
        return None
    s = dpi / 72.0
    pix = pymupdf.open("pdf", pdf)[page_index].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY)
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "page.png")
        pix.save(img)
        out = subprocess.run([exe, img, "stdout", "-l", lang, "--psm", "3", "tsv"], capture_output=True, text=True,
                             timeout=600)
    words = []
    for row in csv.DictReader(io.StringIO(out.stdout), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        x, y, w, h = (float(row[k]) / s for k in ("left", "top", "width", "height"))
        words.append(OcrWord(text, (x, y, x + w, y + h), float(row.get("conf") or -1),
                             f"b{row['block_num']}p{row['par_num']}l{row['line_num']}"))
    return words
