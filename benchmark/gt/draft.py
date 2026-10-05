"""GT pre-annotation and review sheets (PLAN.md §7.4).

Drafts come only from benchmark-owned tools (text layer, measurement OCR, regex), never
from candidate output. Every pre-labelled mention carries the condition ``prelabel`` and
no entity: a human must confirm boundaries, types, entities, roles and attributes, add
the missed mentions, classified spans and regions, and remove the false ones. Drafts of
real documents contain PII and are written under the git-ignored data directory.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .. import ocr
from ..geometry import Frame, as_poly, geometry_from_pikepdf_page, rect_poly
from ..schemas import GTDocument, GTMention, GTPage, GTWord, PiiType, Segment, WordSource

# Pre-label patterns over whitespace tokens (token-level, so multi-token matches are windows).
TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "a/prof", "mx"}
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
PATTERNS: list[tuple[PiiType, str | None, re.Pattern, int]] = [
    (PiiType.EMAIL, None, re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+[.,;:)]?$"), 1),
    (PiiType.PHONE, "mobile", re.compile(r"^04\d{2}\s?\d{3}\s?\d{3}$"), 3),
    (PiiType.PHONE, "landline", re.compile(r"^\(0\d\)\s?\d{4}\s?\d{4}$"), 3),
    (PiiType.PHONE, "landline", re.compile(r"^(1300|1800)\s?\d{3}\s?\d{3}$"), 3),
    (PiiType.DATE, None, re.compile(r"^\(?\d{1,2}\s?[/.\-]\s?\d{1,2}\s?[/.\-]\s?\d{2,4}\)?[.,;:]?$"), 5),
    (PiiType.DATE, None, re.compile(rf"^\d{{1,2}}(st|nd|rd|th)?\s(({_MONTHS})[a-z]*)\s\d{{4}}[.,;:]?$", re.I), 3),
    (PiiType.ID_BUSINESS, "abn", re.compile(r"^ABN\s\d{2}\s\d{3}\s\d{3}\s\d{3}$"), 5),
    (PiiType.ID_OTHER, None, re.compile(r"^[A-Z]{1,3}\d{2}[A-Z]{2}\d{5,7}[.,]?$"), 1),
    (PiiType.ADDRESS, "state_postcode", re.compile(r"^(QLD|NSW|VIC|TAS|SA|WA|NT|ACT|Qld)\s\d{4}[.,]?$"), 2),
]


def _words_from_text_layer(page: pymupdf.Page, geom) -> list[tuple[str, np.ndarray, str]]:
    out = []
    for x0, y0, x1, y1, text, b, ln, _ in page.get_text("words", sort=True):
        poly = geom.transform(rect_poly(x0, y0, x1, y1), Frame.UNROTATED_PT, Frame.DISPLAY_PT)
        out.append((text, poly, f"b{b}l{ln}"))
    return out


def draft_gt(pdf: bytes, doc_id: str, matter: str | None = None) -> tuple[GTDocument, list[str]]:
    import io

    import pikepdf

    notes: list[str] = []
    doc = pymupdf.open("pdf", pdf)
    with pikepdf.open(io.BytesIO(pdf)) as pk:
        geoms = [geometry_from_pikepdf_page(p) for p in pk.pages]
    pages: list[GTPage] = []
    mentions: list[GTMention] = []
    ocr_used = False
    for i, page in enumerate(doc):
        g = geoms[i]
        w, h = g.display_size
        words = [(t, poly, ln, 0.0) for t, poly, ln in _words_from_text_layer(page, g)]
        source = WordSource.TEXT_LAYER
        if not words:
            got = ocr.ocr_page(pdf, i)
            if got is None:
                notes.append(f"page {i}: no text layer and no measurement OCR installed; words must be annotated")
            else:
                source, ocr_used = WordSource.RASTER, True
                words = [(o.text, as_poly(o.poly) if o.poly else rect_poly(*o.bbox), o.line, o.angle) for o in got]
        gt_words = [GTWord(id=f"p{i}w{n}", text=t, poly=[tuple(pt) for pt in poly.tolist()], order=n, line=f"p{i}{ln}",
                           angle=a, source=source, legibility="clear") for n, (t, poly, ln, a) in enumerate(words)]
        pages.append(GTPage(index=i, width_pt=round(w, 3), height_pt=round(h, 3), rotation=g.rotation,
                            words=gt_words, conditions=["draft"]))
        mentions.extend(_prelabel(gt_words, i, len(mentions)))
    note = "Machine pre-annotation. Not ground truth until reviewed and adjudicated."
    if ocr_used:
        note += f" Words on pages without a text layer: measurement OCR {ocr.version()}."
    gt = GTDocument(doc_id=doc_id, source_sha256=hashlib.sha256(pdf).hexdigest(), gt_version="draft-0",
                    matter=matter, pages=pages, mentions=mentions, annotators=["prelabel"], notes=note)
    return gt, notes


def _prelabel(words: list[GTWord], page: int, start: int) -> list[GTMention]:
    out: list[GTMention] = []
    used: set[int] = set()
    n = len(words)
    for pii, subtype, pat, max_len in PATTERNS:
        for i in range(n):
            for k in range(max_len, 0, -1):
                idx = list(range(i, min(i + k, n)))
                if len(idx) < k or used & set(idx):
                    continue
                text = " ".join(words[j].text for j in idx)
                if pat.match(text):
                    out.append(GTMention(id=f"pre{start + len(out)}", type=pii, subtype=subtype, text=text,
                                         segments=[Segment(page=page, words=[words[j].id for j in idx])],
                                         conditions=["prelabel"]))
                    used.update(idx)
                    break
    # Title + capitalised words -> candidate person names
    for i, wd in enumerate(words[:-1]):
        if wd.text.strip(".").casefold() not in TITLES or i in used:
            continue
        idx = []
        for j in range(i + 1, min(i + 4, n)):
            t = words[j].text.strip(".,;:()'’s")
            if t[:1].isupper() and j not in used and t.casefold() not in TITLES:
                idx.append(j)
            else:
                break
        if idx:
            out.append(GTMention(id=f"pre{start + len(out)}", type=PiiType.PERSON_NAME,
                                 text=" ".join(words[j].text for j in idx),
                                 segments=[Segment(page=page, words=[words[j].id for j in idx])],
                                 optional_words=[wd.id], conditions=["prelabel"]))
            used.update(idx)
    return out


COLORS = {"PERSON_NAME": (220, 40, 40), "ADDRESS": (230, 120, 0), "PHONE": (180, 0, 180), "EMAIL": (180, 0, 180),
          "DATE": (0, 140, 0), "SIGNATURE": (120, 60, 200)}


def review_sheets(gt: GTDocument, pdf: bytes, out_dir: Path, dpi: float = 110) -> list[Path]:
    """Overlay PNGs: words grey, mentions coloured by type (with ids), classified spans blue,
    regions purple. For visual QC of drafts and adjudicated GT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open("pdf", pdf)
    s = dpi / 72.0
    words = gt.word_index()
    regions = gt.region_index()
    paths = []
    for p in gt.pages:
        pix = doc[p.index].get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csRGB)
        img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3), cv2.COLOR_RGB2BGR).copy()

        def draw(poly, color, thick=1):
            pts = np.round(as_poly(poly) * s).astype(np.int32)
            cv2.polylines(img, [pts], True, color, thick)
            return pts

        for w in p.words:
            draw(w.poly, (190, 190, 190))
        for c in gt.classified:
            for seg in c.segments:
                if seg.page == p.index:
                    for wid in seg.words:
                        draw(words[wid][1].poly, (200, 120, 0), 2)
        for r in p.regions:
            draw(r.poly, (200, 60, 120), 2)
        for m in gt.mentions:
            color = COLORS.get(m.type.value, (0, 90, 220))[::-1]
            for seg in m.segments:
                if seg.page != p.index:
                    continue
                pts = None
                for wid in seg.words:
                    pts = draw(words[wid][1].poly, color, 2)
                for rid in seg.regions:
                    pts = draw(regions[rid][1].poly, color, 2)
                if pts is not None:
                    cv2.putText(img, m.id, tuple(int(v) for v in pts.min(axis=0) + [0, -2]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, color, 1)
        path = out_dir / f"p{p.index + 1:02d}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths
