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
from benchmark.schemas import PiiType
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


def retypeset_in_font(fontname: str, raster_dpi=None) -> bytes:
    """V3/X3: every MUST word's original re-typeset at its origin and size in another font ('tiro'
    Times, 'cour' Courier, 'hebo' Helvetica-Bold): the original text, in the text layer too."""
    return build(word_ops={w: ("font", fontname, None) for w in must_words()}, raster_dpi=raster_dpi)


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


# -- the moved study's cases (eval-0.4 S12): planted moves, different words, erasures, traps --------------


def kept_words() -> list[str]:
    req = requirements()
    return [w for w in words() if req[w] in ("ORDINARY", "MUST_NOT", "MAY")]


def kept_real_words() -> list[str]:
    ws = words()
    return [w for w in kept_words() if sum(ch.isalnum() for ch in ws[w].text) >= 2]


def last_kept_of_lines() -> list[str]:
    """The last word of each line when it is a kept word (the isolated moves move these)."""
    req = requirements()
    out = []
    for ids in lines().values():
        ks = [i for i in ids if req[i] in ("ORDINARY", "MUST_NOT", "MAY")]
        if ks and ids[-1] == ks[-1]:
            out.append(ks[-1])
    return out


OTHER = {2: ["um", "ox", "ew", "yo"], 3: ["owl", "mug", "zip", "elk"], 4: ["lamp", "wove", "mint", "buck"],
         5: ["plumb", "vixen", "quirk", "mango"], 6: ["walrus", "hiccup", "zenith", "bounty"],
         7: ["quantum", "gazelle", "trolley", "biscuit"], 8: ["mahogany", "quixotic", "envelope", "seashell"],
         9: ["porcupine", "vineyards", "quarterly", "blueberry"],
         10: ["lighthouse", "watermelon", "kingfisher", "background"]}


def other_word(t: str, k: int) -> str:
    """A different word of about the same length, in the original's case and trailing punctuation."""
    core = t.strip(".,;:()|")
    trail = t[len(t.rstrip(".,;:()")):]
    n = len(core)
    w = OTHER.get(min(max(n, 2), 10), OTHER[10])[k % 4]
    if n > 10:
        w = (w * 2)[:n]
    if core[:1].isupper():
        w = w.capitalize()
    return w + trail


# Redrawn lines (like tool 2): (scale, dy, dx0) of the moved study's A1, A2 and A5.
REDRAW = {"A1": (1.04, 0.0, 0.0), "A2": (1.04, 0.5, 1.0), "A5": (1.12, 0.0, 0.5)}
# Isolated moves of the last kept word of each line: (dx, dy, s) of B4, B9, B11 and B12.
ISOLATED = {"B4": (58, 0, 1), "B9": (8, 8, 1), "B11": (0.5, 0, 1), "B12": (1.0, 0.3, 1.02)}


def moved_case(name: str, raster_dpi=None, truth: dict | None = None) -> bytes:
    """The moved study's fictional outputs: A* redrawn lines with fakes, B* isolated moves, C3 a different
    word at every kept word's place, D3 and D5 a box over half of / shifted 7 pt below every other kept
    word, E2/E5/E7 traps with repeated words."""
    if name in REDRAW:
        s, dy, dx0 = REDRAW[name]
        return build(line_ops=redraw(fakes(), s, dy, dx0), raster_dpi=raster_dpi, truth=truth)
    if name in ISOLATED:
        dx, dy, s = ISOLATED[name]
        return build(word_ops={w: ("move", dx, dy, s) for w in last_kept_of_lines()}, raster_dpi=raster_dpi,
                     truth=truth)
    ws = words()
    every_other = kept_words()[::2]
    p1 = {w.text: w.id for w in base().gt.pages[1].words if w.line == "p1l4"}
    ops = {
        "C3": lambda: {w: ("text", other_word(ws[w].text, k)) for k, w in enumerate(kept_real_words())},
        "D3": lambda: {w: ("partial", 0.5) for w in every_other},
        "D5": lambda: {w: ("shiftbox", 7.0) for w in every_other},
        # 'is' erased, 'further' replaced by 'is', and the anchor 'treatment' erased too
        "E2": lambda: {p1["further"]: ("text", "is"), p1["is"]: ("erase",), p1["treatment"]: ("erase",)},
        # the protected word 'strong' erased while the surname 'Strong' stays 73 pt to its left
        "E5": lambda: {"p0w35": ("erase",)},
        # no anchors left and the surname 'Strong' redrawn 40 pt right, 33 pt from where 'strong' was
        "E7": lambda: {"p0w35": ("erase",), "p0w33": ("erase",), "p0w34": ("erase",), "p0w32": ("move", 40.0, 0.0, 1.0)},
    }[name]()
    return build(word_ops=ops, raster_dpi=raster_dpi, truth=truth)


def keep(fk: dict[str, str], ids: list[str]) -> dict[str, str]:
    """The fakes with the originals of ``ids`` kept."""
    ws = words()
    return {**fk, **{w: ws[w].text for w in ids}}


def originals_redrawn(raster_dpi=300) -> bytes:
    """K0c: every line with a MUST word re-typeset with its originals (1.04x, 0.3 pt down, 1 pt right)."""
    return build(line_ops={ln: ("redraw", {}, 1.04, 0.3, 1.0) for ln in must_lines()}, raster_dpi=raster_dpi)


def locality_kept_redrawn(raster_dpi=None) -> bytes:
    """K2: K1 with the original locality kept; the longer fake street moves it along the line."""
    return build(line_ops=redraw(keep(shared_fakes(), mention_words("addr1-loc"))), raster_dpi=raster_dpi)


def realistic_fakes() -> dict[str, str]:
    """The challenge's REAL fakes: K1's with another street number and name."""
    fk = fakes()
    fk.update(zip(mention_words("addr1"), ["83", "Harbour", "Street,"]))
    fk.update(zip(mention_words("addr1-loc"), ["Bundall", "QLD", "4217"]))
    fk.update(zip(mention_words("org1"), ["Jabiru", "", "Pty", "Ltd"]))
    return fk


def same_city_postcode() -> bytes:
    """W1: realistic fakes in place, the locality faked as 'Newstead QLD 4006' for 'Faketown QLD 4000'."""
    fk = realistic_fakes()
    fk.update(zip(mention_words("addr1-loc"), ["Newstead", "QLD", "4006"]))
    return build(word_ops=in_place(fk))


def ext_spec():
    """The medico-legal spec with four lines of names whose fakes an over-extended span (the NER span
    took the kept word after the name) replaces (the challenge's M2)."""
    from benchmark.schemas import PiiType
    from benchmark.synth.spec import L, MentionSpec, T

    spec = medico_legal_spec()
    spec.pages[1] = list(spec.pages[1]) + [
        L(72, 400, 11, T("Review"), T("by"), T("Dr", "opt:m10"), T("Lee", "m:m10:given"), T("Hart", "m:m10:surname"),
          T("on"), T("3"), T("May"), T("was"), T("normal.")),
        L(72, 420, 11, T("The"), T("claimant"), T("Ms", "opt:m11"), T("Kay", "m:m11:given"),
          T("Ross", "m:m11:surname"), T("is"), T("well"), T("today.")),
        L(72, 440, 11, T("She"), T("saw"), T("Dr", "opt:m12"), T("Ian", "m:m12:given"), T("Lam", "m:m12:surname"),
          T("in"), T("clinic"), T("today.")),
        L(72, 460, 11, T("Seen"), T("with"), T("Ms", "opt:m13"), T("Sue", "m:m13:given"), T("Tan", "m:m13:surname"),
          T("and"), T("her"), T("son.")),
    ]
    for m in ("m10", "m11", "m12", "m13"):
        spec.mentions[m] = MentionSpec(PiiType.PERSON_NAME, None, form="Given Surname")
    return spec


_EXT: list = []


def ext_base() -> Base:
    if not _EXT:
        _EXT.append(base(ext_spec(), "ext"))
    return _EXT[0]


OVER = [  # (baseline, MUST words, the kept word the span over-extends to, fake, control fake without it)
    (400, ["Lee", "Hart"], "on", "Ann Wilson", "Ann Wilby"),
    (420, ["Kay", "Ross"], "is", "Ruth Harris", "Ruth Hardy"),
    (440, ["Ian", "Lam"], "in", "Ken Martin", "Ken Marsh"),
    (460, ["Sue", "Tan"], "and", "Sandra Lee", "Sophie Lee"),
]


def _ext_id(page: int, baseline: float, text: str) -> str:
    for wid, (pg, t, _, b, _, _) in ext_base().layout.items():
        if pg == page and abs(b - baseline) < 0.5 and t == text:
            return wid
    raise KeyError(text)


def overextended_span(control: bool = False) -> tuple[bytes, list[str]]:
    """M2 (M2c with ``control``): the names and the kept word after them erased, one fake typeset over the
    span; the fake contains the kept word ('on' in 'Wilson'), the control's does not. Returns the output
    and the kept words' ids."""
    b = ext_base()
    ops, post, kept = {}, [], []
    for y, names, word, fake, ctrl in OVER:
        ids = [_ext_id(1, y, t) for t in names]
        k = _ext_id(1, y, word)
        for w in ids + [k]:
            ops[w] = ("erase",)
        _, _, fx, by, size, _ = b.layout[ids[0]]
        post.append(dict(page=1, rect=None, x=fx, y=by, size=size, fake=ctrl if control else fake))
        kept.append(k)
    return build(word_ops=ops, post=post, spec=b.spec, key="ext"), kept


# -- substitute fonts (eval-0.4 S13) ------------------------------------------------------------------

SIM = {'a': 'o', 'o': 'a', 'e': 'c', 'c': 'e', 'n': 'u', 'u': 'n', 'r': 'n', 'i': 'l', 'l': 'i', 't': 'f', 'f': 't',
       'h': 'b', 'b': 'h', 'm': 'n', 's': 'e', 'd': 'b', 'p': 'q', 'g': 'q', 'y': 'v', 'v': 'y', 'k': 'h', 'w': 'v',
       'x': 'k', 'j': 'i', 'q': 'g', 'z': 's', '0': '8', '8': '0', '1': '7', '7': '1', '2': '3', '3': '2', '4': '9',
       '9': '4', '5': '6', '6': '5'}


def one_letter(t: str) -> str:
    """The word with its middle letter or digit swapped for a look-alike (the moved study's C1)."""
    idx = [i for i, ch in enumerate(t) if ch.isalnum()]
    if not idx:
        return t
    i = idx[len(idx) // 2]
    ch = t[i]
    rep = SIM.get(ch.lower(), "x")
    return t[:i] + (rep.upper() if ch.isupper() else rep) + t[i + 1:]


def lines_in_font(fontname: str = "tiro", raster_dpi=None) -> bytes:
    """W4: every line with a MUST word re-typeset with the realistic fakes in a substitute font."""
    fk = realistic_fakes()
    ls = lines()
    return build(line_ops={ln: ("redraw", {w: fk[w] for w in ls[ln] if w in fk}, 1.0, 0.0, 0.0, fontname, 1.0)
                           for ln in must_lines()}, raster_dpi=raster_dpi)


def damage_case(name: str) -> bytes:
    """The moved study's other negatives: C1 a one-letter look-alike at every kept word's place, D1 every
    other kept word erased, D2 boxed, D4 boxed over 70% of its width, E1/E3/E4/E6 the remaining traps."""
    ws = words()
    every_other = kept_words()[::2]
    p0 = {w.text: w.id for w in base().gt.pages[0].words if w.line == "p0l6"}
    p1 = {w.text: w.id for w in base().gt.pages[1].words if w.line == "p1l4"}
    ops = {
        "C1": lambda: {w: ("text", one_letter(ws[w].text)) for w in kept_real_words()},
        "D1": lambda: {w: ("erase",) for w in every_other},
        "D2": lambda: {w: ("box",) for w in every_other},
        "D4": lambda: {w: ("partial", 0.7) for w in every_other},
        "E1": lambda: {p1["further"]: ("text", "is"), p1["is"]: ("erase",)},
        "E3": lambda: {p0["a"]: ("erase",)},
        "E4": lambda: {p0["a"]: ("erase",), p0["and"]: ("text", "or")},
        "E6": lambda: {"p0w35": ("erase",), "p0w33": ("erase",), "p0w34": ("erase",)},
    }[name]()
    return build(word_ops=ops)


# -- shared structure (eval-0.4 S14) ---------------------------------------------------------------------

V0 = load_policy(ROOT / "policies" / "au-medicolegal-v0.yaml")
# The client's policy (V1, 1.2.0) carries the shared roles the owner decided on 2026-10-22 and 2026-10-29.
# Two copies of it for the tests that need less: one without the phone prefix (a fake mobile keeping '04'
# then shows the original's glyphs), and one sharing nothing (the shared structure of 1.1.1).


def _with_shared_roles(keep, suffix):
    return V1.model_copy(update={"version": f"{V1.version}+{suffix}",
                                 "shared_roles": [r for r in V1.shared_roles if keep(r)]})


V1_PHONE_OFF = _with_shared_roles(lambda r: PiiType.PHONE not in r.types, "no-phone-prefix")
V1_NO_SHARED = _with_shared_roles(lambda r: False, "no-shared-roles")


def shared_in_place(fk: dict[str, str] | None = None, raster_dpi=None) -> bytes:
    """S1 (and the Q3 cases): every MUST word replaced in place by K1's fakes, which keep the shared words
    (an empty fake erases its word)."""
    return build(word_ops=in_place(fk or shared_fakes()), raster_dpi=raster_dpi)


def q3_fakes() -> dict[str, str]:
    """K1's fakes with an email and claim numbers that keep the domain and the 'CLM-' prefix (owner Q3)."""
    fk = shared_fakes()
    fk.update({mention_words("em1")[0]: "liam.hughes@example.com", mention_words("clm0")[0]: "CLM-93571",
               mention_words("clm1")[0]: "CLM-93571"})
    return fk


def shared_case(name: str, raster_dpi=None) -> bytes:
    """The merge study's shared-word cases under K1's fakes: S2 the state kept and the rest of the address
    boxed, S3 the postcode kept, S4 only the shared words left (the rest erased; S4r with the lines
    re-typeset, so they slide into the erased words' places), K3 the street name kept,
    K4 the company name kept, S5 another state's fake beside the kept state."""
    ws = words()
    addr, loc, org = mention_words("addr1"), mention_words("addr1-loc"), mention_words("org1")
    if name == "S2":
        return build(word_ops={w: ("box",) for w in addr + loc if ws[w].text != "QLD"}, raster_dpi=raster_dpi)
    if name == "S3":
        return build(word_ops={**in_place(shared_fakes()), loc[2]: ("text", ws[loc[2]].text)}, raster_dpi=raster_dpi)
    if name == "S4":
        return build(word_ops={w: ("erase",) for w in addr + loc + org
                               if ws[w].text.strip(".,") not in ("Street", "QLD", "Pty", "Ltd")}, raster_dpi=raster_dpi)
    if name == "S4r":  # S4 on re-typeset lines: the kept words (and the locality's fake) slide left
        fk = dict(shared_fakes())
        fk.update({w: (ws[w].text if ws[w].text.strip(".,") in ("Street", "Pty", "Ltd") else "") for w in addr + org})
        return build(line_ops={ln: ("redraw", {w: fk[w] for w in lines()[ln] if w in fk}, 1.0, 0.0, 0.0)
                               for ln in must_lines()}, raster_dpi=raster_dpi)
    if name == "K3":
        return build(line_ops=redraw(keep(shared_fakes(), [addr[1]])), raster_dpi=raster_dpi)
    if name == "K4":
        return build(line_ops=redraw(keep(shared_fakes(), org[:2])), raster_dpi=raster_dpi)
    if name == "S5":
        fk = realistic_fakes()
        fk.update(zip(loc, ["Parramatta", "QLD", "2150"]))
        return build(word_ops=in_place(fk), raster_dpi=raster_dpi)
    if name == "W0":
        return build(word_ops=in_place(realistic_fakes()), raster_dpi=raster_dpi)
    raise ValueError(name)


# -- fix round 1 cases ---------------------------------------------------------------------------------


def layer_under_boxes(raster_dpi: int) -> bytes:
    """The unredacted page as one grey JPEG image layer at ``raster_dpi``, with a black vector box over
    every MUST word: the originals are recoverable from the layer under the boxes."""
    b, ws = base(), words()
    doc = pymupdf.open("pdf", build(raster_dpi=raster_dpi))
    for w in must_words():
        x0, y0, x1, y1 = bbox(as_poly(ws[w].poly))
        doc[b.layout[w][0]].draw_rect(pymupdf.Rect(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3), color=None,
                                      fill=(0, 0, 0))
    return doc.tobytes(garbage=3, deflate=True)


def image_only_png(pdf: bytes, dpi: int) -> bytes:
    """Each page of ``pdf`` rendered to a grey PNG at ``dpi`` and placed as the page's only content."""
    src, out = pymupdf.open("pdf", pdf), pymupdf.open()
    for p in src:
        pix = p.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        q = out.new_page(width=p.rect.width, height=p.rect.height)
        q.insert_image(q.rect, stream=pix.tobytes("png"))
    return out.tobytes()


def hidden_address(fake: str, hidden: list[str], glued: bool) -> bytes:
    """The address line truly redacted with ``fake`` typeset in place (as :func:`address_in_place`), plus
    an invisible text layer (render mode 3) on the address line holding ``hidden``: spaced as the original
    words were, or glued (each word ending 1 pt before the next, which MuPDF extracts as one token)."""
    doc = pymupdf.open("pdf", address_in_place(fake))
    page = doc[0]
    src = pymupdf.open("pdf", base().pdf)[0]
    x = [w[0] for w in src.get_text("words") if w[4] in ADDRESS_WORDS][0]
    for t in hidden:
        page.insert_text((x, 160.0), t, fontname="helv", fontsize=11, render_mode=3)
        x += pymupdf.get_text_length(t + ("" if glued else " "), fontname="helv", fontsize=11) + (1.0 if glued else 0.0)
    return doc.tobytes(garbage=4, deflate=True)
