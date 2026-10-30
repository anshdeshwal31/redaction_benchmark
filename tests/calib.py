"""Fictional calibration outputs for the eval-0.4 tests, built from the digital fixture's own layout.

Ported from the eval-0.4 prototype's builders (the challenge cases X1, X2, V1, M1, the moved study's
synthetic cases and the merge study's P1 and K1, the address study's random locality fakes E5). Every
word is drawn on its own at the fixture's (x0, baseline, size), so untouched words land exactly where
the fixture put them. All content is the fixture's fictional text.

Operations per word id (``word_ops``):
  ("text", new)                  another word at the same origin and size
  ("text_at", new, dx, dy, s)    another word, displaced and scaled
  ("erase",)                     not drawn
  ("move", dx, dy, s)            the word itself, displaced and scaled
  ("box",)                       drawn, then a black box over its GT polygon
  ("partial", f)                 drawn, then a black box over the first fraction f of its width
  ("shiftbox", dy)               drawn, then a black box over its polygon shifted down by dy pt
  ("shiftwhite_fake", fake, f)   drawn, a white box over its polygon shifted f x its height across the
                                 line, then the fake typeset inside that box
  ("veil_fake", fake, opacity)   drawn, a white box of that opacity over it, the fake on top
  ("overprint", fake)            drawn, then the fake typeset on top of it at the same origin
  ("font", fontname, text|None)  the word (or another text) at the same origin and size in another font
  ("tz", text|None, sx)          the word (or another text) condensed or widened by sx at the same origin
Per line (``line_ops``): ("redraw", fakes, s, dy, dx0[, font[, sx]]): the whole line re-typeset from its
first word's origin (+dx0, +dy) at size*s with MUST words swapped for fakes (word id -> fake); the kept
words move with it.
``post``: drawings made after every word of the page: boxes of any opacity and fakes inside them.
``raster_dpi``: the result rasterised to a grey JPEG (quality ``jpeg_quality``), image only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from benchmark.evaluate import EvalConfig
from benchmark.evaluate.evaluator import _global_ncc, _translate, estimate_shift
from benchmark.evaluate.render import RenderCache, RenderConfig, match_shapes
from benchmark.geometry import as_poly, bbox
from benchmark.gt import load_policy, resolve
from benchmark.gt.address_split import POSTCODES
from benchmark.gt.policy import Effective
from benchmark.synth.fixtures import FONT, _digital_pages, build_digital_doc, digital
from benchmark.synth.spec import medico_legal_spec

ROOT = Path(__file__).resolve().parents[1]
V1 = load_policy(ROOT / "policies" / "au-medicolegal-v1.yaml")


@dataclass(frozen=True)
class Base:
    pdf: bytes
    gt: object  # GTDocument
    layout: dict  # word id -> (page, text, x0, baseline, size, line)
    spec: object  # DocSpec


_BASES: dict = {}


def base(spec=None, key: str = "digital") -> Base:
    """The digital fixture of ``spec`` (default: the medico-legal spec) and its word layout."""
    if key in _BASES:
        return _BASES[key]
    spec = spec or medico_legal_spec()
    fx = digital(spec)
    pages, _, layout = _digital_pages(build_digital_doc(spec), spec)
    lay, k = {}, 0
    for p in pages:
        for w in p.words:
            wl = layout[k]
            assert wl.text == w.text
            lay[w.id] = (p.index, w.text, wl.x0, wl.baseline, wl.size, w.line)
            k += 1
    _BASES[key] = Base(fx.pdf, fx.gt, lay, spec)
    return _BASES[key]


def _text(page, x, y, t, size, font=FONT, sx=1.0, color=(0, 0, 0)):
    if not t:
        return
    if abs(sx - 1.0) > 1e-9:
        page.insert_text((x, y), t, fontname=font, fontsize=size, color=color,
                         morph=(pymupdf.Point(x, y), pymupdf.Matrix(sx, 1.0)))
    else:
        page.insert_text((x, y), t, fontname=font, fontsize=size, color=color)


def width(t: str, size: float, font: str = FONT) -> float:
    return pymupdf.get_text_length(t, fontname=font, fontsize=size)


def build(word_ops=None, line_ops=None, raster_dpi=None, jpeg_quality=80, post=None, relocate=None, extra=None,
          spec=None, key="digital", truth: dict | None = None) -> bytes:
    """The output PDF. When ``truth`` is a dict, it receives word id -> (dx, dy, s) of every kept word
    a move or a redrawn line displaced."""
    word_ops, line_ops, relocate = word_ops or {}, line_ops or {}, relocate or {}
    b = base(spec, key)
    doc = pymupdf.open()
    for p in b.gt.pages:
        page = doc.new_page(width=b.spec.width, height=b.spec.height)
        by_line: dict = {}
        for w in p.words:
            by_line.setdefault(w.line, []).append(w)
        boxes = []
        for line_id, ws in by_line.items():
            ws.sort(key=lambda w: w.order)
            if line_id in line_ops:
                op = line_ops[line_id]
                _, fakes, s, dy, dx0 = op[:5]
                font = op[5] if len(op) > 5 else FONT
                sx = op[6] if len(op) > 6 else 1.0
                _, _, x_start, baseline, size, _ = b.layout[ws[0].id]
                xcur, fs = x_start + dx0, size * s
                for w in ws:
                    t = fakes.get(w.id, w.text)
                    if t:
                        _text(page, xcur, baseline + dy, t, fs, font, sx)
                    if truth is not None and w.id not in fakes:
                        truth[w.id] = (xcur - b.layout[w.id][2], dy, s)
                    xcur += width(t + " ", fs, font) * sx if t else 0.0
                continue
            for w in ws:
                _, text, x0, baseline, size, _ = b.layout[w.id]
                if w.id in relocate:
                    pg, nx, nb, ns, t = relocate[w.id]
                    if pg == p.index:
                        _text(page, nx, nb, t, ns)
                    continue
                op = word_ops.get(w.id)
                if op is None:
                    _text(page, x0, baseline, text, size)
                    continue
                kind = op[0]
                bx0, by0, bx1, by1 = bbox(as_poly(w.poly))
                h = by1 - by0
                if kind == "text":
                    _text(page, x0, baseline, op[1], size)
                elif kind == "text_at":
                    _, new, dx, dy, s = op
                    _text(page, x0 + dx, baseline + dy, new, size * s)
                elif kind == "erase":
                    pass
                elif kind == "move":
                    _, dx, dy, s = op
                    _text(page, x0 + dx, baseline + dy, text, size * s)
                    if truth is not None:
                        truth[w.id] = (dx, dy, s)
                elif kind in ("box", "partial", "shiftbox"):
                    _text(page, x0, baseline, text, size)
                    if kind == "box":
                        boxes.append((bx0, by0, bx1, by1))
                    elif kind == "partial":
                        boxes.append((bx0, by0, bx0 + op[1] * (bx1 - bx0), by1))
                    else:
                        boxes.append((bx0, by0 + op[1], bx1, by1 + op[1]))
                elif kind == "shiftwhite_fake":
                    _, fake, f = op
                    _text(page, x0, baseline, text, size)
                    dy = f * h
                    page.draw_rect(pymupdf.Rect(bx0 - 0.5, by0 + dy, max(bx1, x0 + width(fake, size)) + 0.5, by1 + dy),
                                   color=None, fill=(1, 1, 1))
                    _text(page, x0, baseline + dy, fake, size)
                elif kind == "veil_fake":
                    _, fake, opacity = op
                    _text(page, x0, baseline, text, size)
                    page.draw_rect(pymupdf.Rect(bx0 - 0.5, by0 - 0.5, bx1 + 0.5, by1 + 0.5), color=None,
                                   fill=(1, 1, 1), fill_opacity=opacity)
                    _text(page, x0, baseline, fake, size)
                elif kind == "overprint":
                    _text(page, x0, baseline, text, size)
                    _text(page, x0, baseline, op[1], size)
                elif kind == "font":
                    _, fontname, t = op
                    _text(page, x0, baseline, t if t is not None else text, size, font=fontname)
                elif kind == "tz":
                    _, t, sx = op
                    _text(page, x0, baseline, t if t is not None else text, size, sx=sx)
                else:
                    raise ValueError(kind)
        for (pg, x, y, t, size) in (extra or []):
            if pg == p.index:
                _text(page, x, y, t, size)
        for d in (post or []):
            if d["page"] != p.index:
                continue
            if d.get("rect") is not None:
                page.draw_rect(pymupdf.Rect(*d["rect"]), color=None, fill=d.get("fill", (1, 1, 1)),
                               fill_opacity=d.get("opacity", 1.0))
            if d.get("fake"):
                sx = 1.0
                if d.get("fit_w"):  # fit to the box: the fake condensed or widened to fill its width
                    sx = d["fit_w"] / max(width(d["fake"], d["size"], d.get("font", FONT)), 1e-6)
                _text(page, d["x"], d["y"], d["fake"], d["size"], font=d.get("font", FONT), sx=sx,
                      color=d.get("color", (0, 0, 0)))
        for sq in b.spec.squiggles.get(p.index, []):
            page.draw_polyline(list(sq.points), color=(0, 0, 0), width=1.6)
        for r in boxes:
            page.draw_rect(pymupdf.Rect(*r), color=None, fill=(0, 0, 0))
    if raster_dpi:
        out = pymupdf.open()
        for pg in doc:
            pix = pg.get_pixmap(dpi=int(raster_dpi), colorspace=pymupdf.csGRAY, alpha=False)
            g = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
            _, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            new = out.new_page(width=pg.rect.width, height=pg.rect.height)
            new.insert_image(new.rect, stream=buf.tobytes())
        doc = out
    return doc.tobytes(garbage=3, deflate=True)


# -- the fixture's words and mentions under the client's policy ----------------------------------------


@lru_cache(maxsize=None)
def requirements() -> dict[str, str]:
    """Word id -> requirement under the client's policy."""
    b = base()
    res = resolve(b.gt, V1)
    return {w.id: res.words.get(w.id, Effective.ORDINARY).value for p in b.gt.pages for w in p.words}


def words() -> dict:
    return {w.id: w for p in base().gt.pages for w in p.words}


def lines() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in base().gt.pages:
        for w in p.words:
            out.setdefault(w.line, []).append(w.id)
    return out


def must_words() -> list[str]:
    req = requirements()
    return [w for w in words() if req[w] == "MUST"]


def must_lines() -> set[str]:
    req = requirements()
    return {ln for ln, ids in lines().items() if any(req[i] == "MUST" for i in ids)}


def mention_words(mid: str) -> list[str]:
    m = next(m for m in base().gt.mentions if m.id == mid)
    return [w for seg in m.segments for w in seg.words]


def must_text_mentions() -> list[str]:
    """Mentions made only of MUST words (the signature, a region, is not among them)."""
    req = requirements()
    return [m.id for m in base().gt.mentions
            if mention_words(m.id) and all(req[w] == "MUST" for w in mention_words(m.id))]


POOL = ["Ashdown", "Merrow", "Tolland", "Quill", "Pryce", "Holloway", "Baxter", "Fenwick", "Ivy", "Nora", "Mabel",
        "Hugh", "Felix", "Ralph", "Silas", "Owen", "Grace", "Ruth", "Clare", "Anna"]


def fakes_for(ids: list[str]) -> dict[str, str]:
    """A fictional fake per word in the original's case and trailing punctuation; digits mapped one by one
    (other characters kept: an identifier's 'CLM-', a date's '/')."""
    ws = words()
    out = {}
    for k, wid in enumerate(ids):
        t = ws[wid].text
        core = t.strip(".,;:()|")
        trail = t[len(t.rstrip(".,;:()")):]
        if not core:
            out[wid] = t
            continue
        f = POOL[(k * 7 + len(core)) % len(POOL)]
        if core.isupper() and len(core) > 1:
            f = f.upper()
        if any(ch.isdigit() for ch in core):
            f = "".join(str((int(ch) * 3 + 7) % 10) if ch.isdigit() else ch for ch in core) + ("9" if k % 2 else "")
        out[wid] = f + trail
    return out


@lru_cache(maxsize=None)
def _fakes() -> tuple:
    return tuple(fakes_for(must_words()).items())


def fakes() -> dict[str, str]:
    """The fake of every MUST word (the moved study's FAKES)."""
    return dict(_fakes())


def fake_characters(fk: dict[str, str]) -> set[tuple[str, int]]:
    """Characters of the faked words where the output shows fake text (or nothing), not the original's:
    a fake that happens to have the original's character at the same index shows the original glyph."""
    ws = words()
    out = set()
    for w, fake in fk.items():
        t = ws[w].text
        out.update((w, k) for k in range(len(t)) if k >= len(fake) or fake[k] != t[k])
    return out


def in_place(fk: dict[str, str]) -> dict:
    """Every MUST word with a fake replaced at its origin and size; an empty fake erases the word."""
    must = set(must_words())
    return {w: (("text", f) if f != "" else ("erase",)) for w, f in fk.items() if w in must}


def redraw(fk: dict[str, str], s=1.02, dy=0.2, dx0=0.5) -> dict:
    """Every line holding a MUST word re-typeset with the fakes (the kept words move with it)."""
    ls = lines()
    return {ln: ("redraw", {w: fk[w] for w in ls[ln] if w in fk}, s, dy, dx0) for ln in must_lines()}


def shared_fakes() -> dict[str, str]:
    """The fakes with a realistic address and organisation that keep the shared words (street type,
    state and its postcode digit, legal form): the merge study's K1."""
    fk = fakes()
    fk.update(zip(mention_words("addr1"), ["40", "Bridge", "Street,"]))
    fk.update(zip(mention_words("addr1-loc"), ["Bundall", "QLD", "4217"]))
    fk.update(zip(mention_words("org1"), ["Jabiru", "", "Pty", "Ltd"]))
    return fk


def span_post(mid: str, fake_words: list[str], f: float = 0.0, opacity=None, pad: float = 0.5, fakes_on=True,
              color=(0, 0, 0), fit=False) -> dict:
    """A white box over a mention's words (shifted across the line by f x its height) and the fake
    typeset inside it."""
    ids = mention_words(mid)
    b = base()
    ws = words()
    pg = b.layout[ids[0]][0]
    boxes = [bbox(as_poly(ws[w].poly)) for w in ids]
    x0, y0 = min(r[0] for r in boxes), min(r[1] for r in boxes)
    x1, y1 = max(r[2] for r in boxes), max(r[3] for r in boxes)
    h = y1 - y0
    _, _, fx, base_y, size, _ = b.layout[ids[0]]
    fake = " ".join(t for t in fake_words if t)
    if not fit:
        x1 = max(x1, fx + width(fake, size))
    d = dict(page=pg, rect=(x0 - pad, y0 + f * h, x1 + pad, y1 + f * h), x=fx, y=base_y + f * h, size=size,
             fake=fake if fakes_on else None, color=color)
    if fit:
        d["fit_w"] = x1 - fx
    if opacity is not None:
        d["opacity"] = opacity
    return d


def all_spans(**kw) -> list[dict]:
    fk = fakes()
    return [span_post(m, [fk.get(w, "") for w in mention_words(m)], **kw) for m in must_text_mentions()]


# -- the calibration cases ---------------------------------------------------------------------------


def shift_white(f: float, fit: bool, raster_dpi=None) -> bytes:
    """X1 (fit=True) and V1: a white box over each MUST mention, shifted f x its height across the line,
    with the fake typeset inside it; the original is still there."""
    return build(post=all_spans(f=f, fit=fit), raster_dpi=raster_dpi)


def veil(opacity: float, raster_dpi=300) -> bytes:
    """X2: a white box of the given opacity over each MUST mention, the fake fitted on top."""
    return build(post=all_spans(opacity=opacity, fit=True), raster_dpi=raster_dpi)


def same_font_fakes(raster_dpi=None) -> bytes:
    """P1: every MUST word replaced in place by its fake, in the original's font and size."""
    return build(word_ops=in_place(fakes()), raster_dpi=raster_dpi)


def shared_redraw(raster_dpi=None) -> bytes:
    """K1: every line with a MUST word re-typeset (1.02x, 0.2 pt down, 0.5 pt right) with fakes that keep
    the shared words."""
    return build(line_ops=redraw(shared_fakes()), raster_dpi=raster_dpi)


# M1: protected dates, a dose and kept words altered in place by one look-alike character.
LOOKALIKE = {"p0w16": "12/08/1970", "p0w45": "60mg", "p0w48": "04/06/2021.", "p0w69": "(01.02.2028)",
             "p1w39": "(07.02.2026)", "p1w14": "sir", "p1w13": "far", "p0w14": "at"}


def lookalikes() -> bytes:
    return build(word_ops={w: ("text", t) for w, t in LOOKALIKE.items()})


# E5: random realistic whole-address fakes typeset in place of the synthetic address line after a true
# redaction (one string in the original's font and size), the address study's experiment.
STREETS = ["Banksia", "Coolabah", "Grevillea", "Jacaranda", "Kurrajong", "Waratah", "Melaleuca", "Bottlebrush",
           "Ironbark", "Wattle", "Bluegum", "Myrtle"]
STREET_TYPES = ["Road", "Street", "Avenue", "Parade", "Crescent", "Lane", "Close", "Terrace", "Drive", "Court"]
SUBURBS = ["Oakvale", "Brushmere", "Tallowby", "Karribrook", "Wattlecombe", "Bimbadeen", "Fernlea", "Ironmere",
           "Kelvin Grove", "Redcliffe", "Ashgrove", "Paddington"]
ADDRESS_WORDS = ("14", "Example", "Street,", "Faketown", "QLD", "4000")


def address_fakes(n: int = 15, other_state: bool = True, seed: int = 20261022) -> list[str]:
    """``n`` seeded whole-address fakes, in another state than the original's QLD (or in QLD)."""
    rng = random.Random(seed)

    def postcode(state):
        while True:
            lo, hi = rng.choice(POSTCODES[state])
            pc = f"{rng.randint(max(lo, 200), hi):04d}"
            if pc != "4000":
                return pc

    out = []
    for _ in range(n):
        state = rng.choice(["NSW", "VIC", "SA", "WA", "TAS"]) if other_state else "QLD"
        out.append(f"{rng.randint(1, 299)} {rng.choice(STREETS)} {rng.choice(STREET_TYPES)}, "
                   f"{rng.choice(SUBURBS)} {state} {postcode(state)}")
    return out


def address_in_place(fake: str) -> bytes:
    """The fixture with its address line truly redacted and ``fake`` typeset at its start."""
    doc = pymupdf.open("pdf", base().pdf)
    page = doc[0]
    found = [w for w in page.get_text("words") if w[4] in ADDRESS_WORDS]
    for w in found:
        page.add_redact_annot(pymupdf.Rect(w[:4]), fill=(1, 1, 1))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
    page.insert_text((found[0][0], 160.0), fake, fontname="helv", fontsize=11)
    return doc.tobytes(garbage=4, deflate=True)


# -- render pairs as the evaluator sees them ---------------------------------------------------------


def render_pair(source: bytes, output: bytes, page: int, cfg: EvalConfig) -> tuple[np.ndarray, np.ndarray]:
    """(original, output) renders of the first render configuration, the output registered as the
    evaluator registers it."""
    rc = RenderConfig("mupdf", True)
    cache = RenderCache(cfg.dpi)
    o = cache.get("src", source, page, rc)
    x = cache.get("out", output, page, rc)
    if x.shape != o.shape:
        if abs(x.shape[0] - o.shape[0]) > 2 or abs(x.shape[1] - o.shape[1]) > 2:
            x = cv2.resize(x, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            x, _ = match_shapes(o, x)
    dx, dy, resp = estimate_shift(o, x)
    if np.hypot(dx, dy) >= cfg.registration_min_shift_px and resp > 0.05:
        moved = _translate(x, -dx, -dy)
        if _global_ncc(o, moved) > _global_ncc(o, x) + 0.02:
            x = _translate(x, -round(dx, 2), -round(dy, 2))
    return o, x
