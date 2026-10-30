"""Address cases rendered to PDF, and simulated tools, for the address-unit tests (eval-0.4 S15).

Every case is fictional: a case of docs/address_cases.json, or one written in its format in a test. Each case
line is typeset with PyMuPDF's Helvetica at 10 pt from (x0, 60 + 14 x row) on its page. GT words and their
character boxes come from the rendered text layer; the GT mentions are the table's split
(``tests/address_cases.build_split_gt``): a street line (ADDRESS, ``address_street_line``) and its ``<id>-loc``
locality, a locality-only LOCATION/locality, or an unsplit ADDRESS. Lines carry no block, so one page is one
block for E3's adjacent lines.

A simulated tool is a list of operations applied with true redaction, then drawing in Helvetica at the
original size (:func:`draw`):

  ("replace", word id, text)          the word removed and ``text`` typeset at its origin
  ("erase", word id)                  the word removed, nothing drawn
  ("text", page, x, baseline, text)   new text drawn (a text layer and pixels)
  ("image", page, x, baseline, text)  the same drawn as a picture: pixels with no text layer
  ("hidden", page, x, baseline, text) invisible text: a text layer with no pixels

:func:`correct` replaces every identifying word of every unit (the words outside the shared structure the
policy enables) with a fake that follows the table's fake constraints: digits mapped one by one with every
digit changed and no look-alike pair (3/8, 5/6, 6/8), the postcode's shared prefix kept, letters shifted so
that no letter stays at its index, no wider than the original, and no fake equal to, beginning with or
holding an identifying word of 3 or more letters of any unit.
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from functools import lru_cache

import pymupdf

from benchmark.gt.shared_roles import document_shared_units
from benchmark.schemas import GTDocument, GTPage, GTWord, WordSource
from tests.address_cases import build_split_gt, table

FONT, SIZE, PITCH, TOP = "helv", 10.0, 14.0, 60.0
STRIP = ".,;:()"
# Digit maps with every digit changed and no tabular look-alike pair (3/8, 5/6, 6/8), never to a leading 0.
DIGITS = ({"0": "7", "1": "4", "2": "9", "3": "1", "4": "2", "5": "7", "6": "1", "7": "2", "8": "4", "9": "3"},
          {"0": "2", "1": "7", "2": "4", "3": "9", "4": "7", "5": "2", "6": "4", "7": "9", "8": "2", "9": "1"})


@dataclass(frozen=True)
class Fixture:
    pdf: bytes
    gt: GTDocument
    origin: dict  # word id -> (page, x, baseline)
    case: dict


def cases() -> dict[str, dict]:
    return {c["id"]: c for c in table()["cases"]}


def baseline(row: int) -> float:
    return TOP + PITCH * row


def render(case: dict) -> Fixture:
    """The case typeset to PDF, with GT words and character boxes read back from its text layer."""
    pages = max(line["page"] for line in case["lines"]) + 1
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=595, height=842)
    for line in case["lines"]:
        doc[line["page"]].insert_text((float(line.get("x0", 72)), baseline(line["row"])), line["text"],
                                      fontname=FONT, fontsize=SIZE)
    pdf = doc.tobytes(garbage=4, deflate=True)
    doc = pymupdf.open("pdf", pdf)
    chars: dict[int, list] = {}
    for p in range(pages):
        raw = doc[p].get_text("rawdict")
        chars[p] = [ch for b in raw["blocks"] for ln in b.get("lines", []) for sp in ln["spans"] for ch in sp["chars"]]
    words: dict[int, list[GTWord]] = {p: [] for p in range(pages)}
    origin = {}
    for li, line in enumerate(case["lines"]):
        p, y, x0 = line["page"], baseline(line["row"]), float(line.get("x0", 72))
        x1 = x0 + pymupdf.get_text_length(line["text"], fontname=FONT, fontsize=SIZE) + 0.5
        mine = sorted((ch for ch in chars[p] if abs(ch["origin"][1] - y) < 0.5 and x0 - 0.5 <= ch["origin"][0] <= x1),
                      key=lambda ch: ch["origin"][0])
        groups, cur = [], []
        for ch in mine:
            if ch["c"].isspace():
                if cur:
                    groups.append(cur)
                cur = []
            else:
                cur.append(ch)
        if cur:
            groups.append(cur)
        texts = line["text"].split()
        assert len(groups) == len(texts), (case["id"], li)
        for wi, (text, g) in enumerate(zip(texts, groups)):
            assert "".join(ch["c"] for ch in g) == text, (case["id"], li, wi)
            boxes = [ch["bbox"] for ch in g]
            polys = [[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])] for b in boxes]
            bx0, by0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            bx1, by1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            wid = f"w{li}_{wi}"
            words[p].append(GTWord(id=wid, text=text, poly=[(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)],
                                   order=len(words[p]), line=f"l{li}", source=WordSource.TEXT_LAYER, char_polys=polys))
            origin[wid] = (p, g[0]["origin"][0], y)
    split = build_split_gt(case)
    gt = GTDocument(doc_id=case["id"], source_sha256=hashlib.sha256(pdf).hexdigest(), gt_version="0.1",
                    pages=[GTPage(index=p, width_pt=595.0, height_pt=842.0, words=sorted(ws, key=lambda w: w.order))
                           for p, ws in words.items()], mentions=split.mentions)
    return Fixture(pdf, gt, origin, case)


@lru_cache(maxsize=None)
def rendered(case_id: str) -> Fixture:
    return render(cases()[case_id])


def _rect(fx: Fixture, wid: str) -> pymupdf.Rect:
    w = fx.gt.word_index()[wid][1]
    xs = [x for x, _ in w.poly]
    ys = [y for _, y in w.poly]
    return pymupdf.Rect(min(xs) - 0.2, min(ys), max(xs) + 0.2, max(ys))


def _picture(text: str) -> tuple[bytes, float, float]:
    """``text`` as a 600-DPI greyscale picture (PNG), with its width and height in pt."""
    w = pymupdf.get_text_length(text, fontname=FONT, fontsize=SIZE) + 2
    h = SIZE * 1.4
    d = pymupdf.open()
    pg = d.new_page(width=w, height=h)
    pg.insert_text((1, SIZE * 1.075), text, fontname=FONT, fontsize=SIZE)
    pix = pg.get_pixmap(dpi=600, colorspace=pymupdf.csGRAY)
    return pix.tobytes("png"), w, h


def draw(fx: Fixture, ops: list[tuple]) -> bytes:
    """The output of a simulated tool (operations in the module docstring)."""
    doc = pymupdf.open("pdf", fx.pdf)
    gone = [op for op in ops if op[0] in ("replace", "erase")]
    for op in gone:
        p = fx.origin[op[1]][0]
        doc[p].add_redact_annot(_rect(fx, op[1]), fill=(1, 1, 1))
    for page in doc:
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
    for op in ops:
        if op[0] == "replace" and op[2]:
            p, x, y = fx.origin[op[1]]
            doc[p].insert_text((x, y), op[2], fontname=FONT, fontsize=SIZE)
        elif op[0] in ("text", "hidden"):
            doc[op[1]].insert_text((op[2], op[3]), op[4], fontname=FONT, fontsize=SIZE,
                                   render_mode=3 if op[0] == "hidden" else 0)
        elif op[0] == "image":
            png, w, h = _picture(op[4])
            doc[op[1]].insert_image(pymupdf.Rect(op[2] - 1, op[3] - SIZE * 1.075, op[2] - 1 + w, op[3] - SIZE * 1.075 + h),
                                    stream=png)
    return doc.tobytes(garbage=4, deflate=True)


# -- fakes ------------------------------------------------------------------------------------------------


def _alnum(t: str) -> str:
    return "".join(ch for ch in t.casefold() if ch.isalnum())


def identifying(fx: Fixture, policy) -> dict[str, set[int]]:
    """Word id -> indices of its characters a fake must change: every character of every unit word except the
    shared structure the policy enables (whole words and characters)."""
    shared = document_shared_units(fx.gt, policy)
    words = fx.gt.word_index()
    out: dict[str, set[int]] = {}
    for m in fx.gt.mentions:
        su = shared.get(m.id)
        for seg in m.segments:
            for wid in seg.words:
                if su is not None and wid in su.full:
                    continue
                text = words[wid][1].text
                lo, hi = (seg.char_start, seg.char_end) if seg.char_start is not None else (0, len(text))
                ks = {k for k in range(lo, hi) if text[k].isalnum() and not (su is not None and (wid, k) in su.chars)}
                if ks:
                    out.setdefault(wid, set()).update(ks)
    return out


def _shift(c: str, k: int) -> str:
    letters = string.ascii_uppercase if c.isupper() else string.ascii_lowercase
    return letters[(letters.index(c) + k) % 26] if c in letters else c


# Letters a fake must not put over one another: the same letter, or a look-alike in the same font
# (narrow stems, round letters, arches), which can pass for the original glyph at the same place.
_ALIKE = ["iljI1|!tfr", "aceo", "OUQDCG0", "nhurm", "vwyVWY", "bdpqg"]


def _xs(text: str) -> list[tuple[float, float]]:
    out, x = [], 0.0
    for c in text:
        w = pymupdf.get_text_length(c, fontname=FONT, fontsize=SIZE)
        out.append((x, x + w))
        x += w
    return out


def _over_alike(text: str, fake: str, ks: set[int]) -> bool:
    """Does a changed letter of the fake sit over the same or a look-alike letter of the original?"""
    a, b = _xs(text), _xs(fake)
    for i, c in enumerate(fake):
        if i not in ks or not c.isalpha():
            continue
        for j, o in enumerate(text):
            lo, hi = max(a[j][0], b[i][0]), min(a[j][1], b[i][1])
            if hi - lo <= 0.3 * min(a[j][1] - a[j][0], b[i][1] - b[i][0]):
                continue
            if c.casefold() == o.casefold() or any(c in g and o in g for g in _ALIKE):
                return True
    return False


def fake_word(text: str, ks: set[int], banned: set[str], numbers: set[str]) -> str:
    """A fake of one word changing the characters ``ks``: digits through a look-alike-free map, letters shifted
    by one amount so that none stays at its index or sits over the same or a look-alike letter, no wider than
    the original, equal to no identifying number (``numbers``), and neither equal to nor beginning with nor
    holding a banned word."""
    width = pymupdf.get_text_length(text, fontname=FONT, fontsize=SIZE)
    best = None
    for dmap in DIGITS:
        for k in list(range(1, 26)):
            fake = "".join((dmap.get(c, c) if c.isdigit() else _shift(c, k)) if i in ks else c
                           for i, c in enumerate(text))
            kk = set(ks)
            # A fake wider than the original loses changed letters from its end until it fits (shorter fakes
            # are realistic; a wider one would run into the next word, in pixels and in the text layer).
            while pymupdf.get_text_length(fake, fontname=FONT, fontsize=SIZE) > width + 0.3:
                cut = max((i for i in kk if i < len(fake) and fake[i].isalpha()), default=None)
                if cut is None or sum(ch.isalnum() for ch in fake) <= 2:
                    break
                fake = fake[:cut] + fake[cut + 1:]
                kk = {i if i < cut else i - 1 for i in kk if i != cut}
            if _over_alike(text, fake, kk):
                continue
            core = _alnum(fake)
            if any(n and len(n) >= 3 and (n in core or core.startswith(n)) for n in banned) or core in banned:
                continue
            runs = {"".join(ch for ch in part if ch.isdigit()) for part in fake.split()}
            if runs & numbers:
                continue
            w = pymupdf.get_text_length(fake, fontname=FONT, fontsize=SIZE)
            if w <= width + 0.3:
                return fake
            if best is None or w < best[0]:
                best = (w, fake)
    return best[1] if best else text


def correct_fakes(fx: Fixture, policy, keep: set[str] = frozenset()) -> dict[str, str]:
    """A fake for every identifying word of the case (``keep``: word ids left as they are)."""
    ident = identifying(fx, policy)
    words = fx.gt.word_index()
    banned = {_alnum(words[w][1].text) for w in ident if sum(ch.isalpha() for ch in words[w][1].text) >= 3}
    numbers = {_alnum(words[w][1].text) for w in ident if _alnum(words[w][1].text).isdigit()}
    return {wid: fake_word(words[wid][1].text, ks, banned, numbers) for wid, ks in ident.items() if wid not in keep}


def correct(fx: Fixture, policy, fakes: dict[str, str] | None = None, keep: set[str] = frozenset()) -> list[tuple]:
    """A correct tool: every identifying word replaced in place (``fakes`` override the generated ones)."""
    fk = {**correct_fakes(fx, policy, keep), **(fakes or {})}
    return [("replace", wid, t) for wid, t in fk.items()]


def mention_words(fx: Fixture, mid: str) -> list[str]:
    m = next(x for x in fx.gt.mentions if x.id == mid)
    return [wid for seg in m.segments for wid in seg.words]


def keeps_locality(fx: Fixture, policy, unit: str) -> list[tuple]:
    """The street line faked, the original locality kept as it is."""
    return correct(fx, policy, keep=set(mention_words(fx, f"{unit}-loc")))


def erases_locality(fx: Fixture, policy, unit: str) -> list[tuple]:
    """The street line faked and the locality erased with no fake."""
    loc = mention_words(fx, f"{unit}-loc")
    return correct(fx, policy, keep=set(loc)) + [("erase", wid) for wid in loc]


def flowing(fx: Fixture, unit: str, texts: list[str], as_image: bool = False) -> list[tuple]:
    """One flowing fake per line of the unit: every word of the unit erased, then ``texts`` (one per line, in
    order) typeset from the first unit word of each line."""
    ids = [w for mid in (unit, f"{unit}-loc") if any(m.id == mid for m in fx.gt.mentions)
           for w in mention_words(fx, mid)]
    ops: list[tuple] = [("erase", wid) for wid in ids]
    starts: dict[tuple[int, float], tuple[int, float, float]] = {}
    for wid in ids:
        p, x, y = fx.origin[wid]
        starts.setdefault((p, y), (p, x, y))
    for (p, x, y), text in zip(starts.values(), texts):
        ops.append(("image" if as_image else "text", p, x, y, text))
    return ops


def next_line_locality(fx: Fixture, policy, unit: str) -> list[tuple]:
    """The street line faked and the original locality pushed to the next line, with its text layer."""
    loc = mention_words(fx, f"{unit}-loc")
    words = fx.gt.word_index()
    p, x, y = fx.origin[loc[0]]
    return erases_locality(fx, policy, unit) + [("text", p, x, y + PITCH, " ".join(words[w][1].text for w in loc))]


def raster(pdf: bytes, dpi: int = 300) -> bytes:
    """``pdf`` as image-only pages (a scan with no text layer)."""
    src = pymupdf.open("pdf", pdf)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
    return out.tobytes(garbage=4, deflate=True)


def one_case(case_id: str, lines: list[tuple[str, int]], units: list[tuple[str, list, list]]) -> dict:
    """A fictional case in the table's format: ``lines`` as (text, row) on page 0 at x0 72, ``units`` as (id,
    street-line positions, locality positions)."""
    return {"id": case_id, "lines": [{"text": t, "page": 0, "row": r, "x0": 72} for t, r in lines],
            "expect": {"addresses": [{"id": u, "street_line": s, "locality": loc} for u, s, loc in units]}}
