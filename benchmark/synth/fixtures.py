"""Synthetic PDFs with exact ground truth, in the encodings observed in the seed set.

The digital base is laid out with PyMuPDF; word and character boxes are read back
from the text layer, so GT geometry is exact. Every other variant is derived from the
base by a known geometric transform that is applied to the GT as well.

Encodings (mirroring PLAN.md §1.2):
    digital            born-digital text layer (like PSY/TRN)
    raster             single full-page grey JPEG, optional skew (like IME)
    raster_hidden_text raster + invisible OCR text layer ("sandwich" PDF)
    mrc                150-DPI background (graphics + faint ghost text) + 300-DPI
                       1-bit stencil masks for the text (like NOC)
    vector             glyphs as filled vector paths, no text layer (like DPI)
    rotated_content    landscape content rotated 90° inside a portrait image (NOC annexure)
    rotate_attr        digital page with /Rotate
    cropbox            digital page whose CropBox is offset inside a larger MediaBox
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pikepdf
import pymupdf

from ..geometry import Frame, PageGeometry, bbox, rect_poly
from ..schemas import (
    ClassifiedClass,
    GTClassified,
    GTDocument,
    GTMention,
    GTPage,
    GTRegion,
    GTWord,
    NameComponent,
    RegionKind,
    Segment,
    WordSource,
)
from .spec import DocSpec, Line, medico_legal_spec, narrative_spec

FONT = "helv"
GT_VERSION = "synthetic-0.1"


@dataclass
class Fixture:
    name: str
    pdf: bytes
    gt: GTDocument

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.pdf).hexdigest()


@dataclass
class _WordLayout:
    page: int
    text: str
    x0: float
    baseline: float
    size: float


# ----------------------------------------------------------------------------
# digital base
# ----------------------------------------------------------------------------


def _new_doc() -> pymupdf.Document:
    doc = pymupdf.open()
    doc.set_metadata({"producer": "redaction_benchmark synth", "creationDate": "D:20260101000000Z",
                      "modDate": "D:20260101000000Z"})
    return doc


def _save(doc: pymupdf.Document) -> bytes:
    return doc.tobytes(garbage=3, deflate=True, no_new_id=True)


def _draw_page_content(page: pymupdf.Page, spec: DocSpec, page_index: int, text: bool, graphics: bool) -> None:
    if text:
        for line in spec.pages[page_index]:
            page.insert_text((line.x, line.y), " ".join(t.text for t in line.toks), fontname=FONT,
                             fontsize=line.size, color=(0, 0, 0))
    if graphics:
        for sq in spec.squiggles.get(page_index, []):
            page.draw_polyline(list(sq.points), color=(0, 0, 0), width=1.6)


def build_digital_doc(spec: DocSpec, text: bool = True, graphics: bool = True) -> pymupdf.Document:
    doc = _new_doc()
    for i in range(len(spec.pages)):
        page = doc.new_page(width=spec.width, height=spec.height)
        _draw_page_content(page, spec, i, text, graphics)
    return doc


def _extract_words(page: pymupdf.Page, lines: list[Line], page_index: int) -> tuple[list[GTWord], list[_WordLayout], list]:
    """Read back word and character boxes and zip them with the spec tokens."""
    words = page.get_text("words", sort=False)
    toks = [(line, t) for line in lines for t in line.toks]
    if len(words) != len(toks):
        raise RuntimeError(f"page {page_index}: {len(words)} extracted words for {len(toks)} spec tokens")
    chars = []
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        for ln in block.get("lines", []):
            for span in ln["spans"]:
                for ch in span["chars"]:
                    if not ch["c"].isspace():
                        chars.append(ch)
    out_words: list[GTWord] = []
    layout: list[_WordLayout] = []
    pos = 0
    for n, (w, (line, tok)) in enumerate(zip(words, toks)):
        if w[4] != tok.text:
            raise RuntimeError(f"page {page_index}: extracted {w[4]!r} but spec has {tok.text!r}")
        cps = []
        for _ in tok.text:
            c = chars[pos]
            pos += 1
            cps.append(rect_poly(*c["bbox"]).tolist())
        out_words.append(GTWord(
            id=f"p{page_index}w{n}",
            text=tok.text,
            poly=rect_poly(w[0], w[1], w[2], w[3]).tolist(),
            order=n,
            line=f"p{page_index}l{lines.index(line)}",
            source=WordSource.TEXT_LAYER,
            char_polys=cps,
        ))
        layout.append(_WordLayout(page_index, tok.text, w[0], line.y, line.size))
    return out_words, layout, toks


def _assemble_gt(spec: DocSpec, pages: list[GTPage], toks_per_page: list[list], pdf: bytes) -> GTDocument:
    mention_parts: dict[str, list[tuple[int, str, tuple[int, int] | None]]] = {}
    components: dict[str, dict[NameComponent, list[str]]] = {}
    optional: dict[str, list[str]] = {}
    classified_parts: dict[str, list[tuple[int, str]]] = {}
    for page, toks in zip(pages, toks_per_page):
        for word, (_, tok) in zip(page.words, toks):
            for tag in tok.tags:
                kind, _, rest = tag.partition(":")
                ident, _, comp = rest.partition(":")
                if kind == "m":
                    mention_parts.setdefault(ident, []).append((page.index, word.id, tok.span))
                    if comp:
                        components.setdefault(ident, {}).setdefault(NameComponent(comp), []).append(word.id)
                elif kind == "opt":
                    optional.setdefault(ident, []).append(word.id)
                elif kind == "c":
                    classified_parts.setdefault(ident, []).append((page.index, word.id))
    words = {w.id: w for p in pages for w in p.words}

    def segments_for(parts):
        segs: list[Segment] = []
        for page_index, wid, span in parts:
            if span is not None:
                segs.append(Segment(page=page_index, words=[wid], char_start=span[0], char_end=span[1]))
            elif segs and segs[-1].page == page_index and segs[-1].char_start is None:
                segs[-1].words.append(wid)
            else:
                segs.append(Segment(page=page_index, words=[wid]))
        return segs

    mentions: list[GTMention] = []
    for mid, ms in spec.mentions.items():
        parts = mention_parts.get(mid, [])
        if parts:
            segs = segments_for(parts)
            text = " ".join(words[w].text[s[0]:s[1]] if s else words[w].text for _, w, s in parts)
        else:  # region-only mention (signature)
            region_pages = [(pi, sq) for pi, sqs in spec.squiggles.items() for sq in sqs if sq.mention == mid]
            if not region_pages:
                continue
            segs = [Segment(page=pi, regions=[f"p{pi}{sq.region_id}"]) for pi, sq in region_pages]
            text = ""
        mentions.append(GTMention(
            id=mid, type=ms.type, subtype=ms.subtype, entity=ms.entity, text=text, segments=segs,
            optional_words=optional.get(mid, []), components=components.get(mid, {}), form=ms.form,
            conditions=list(ms.conditions),
        ))
    classified = [
        GTClassified(id=cid, cls=spec.classified[cid],
                     segments=segments_for([(p, w, None) for p, w in parts]),
                     text=" ".join(words[w].text for _, w in parts))
        for cid, parts in classified_parts.items()
    ]
    return GTDocument(
        doc_id=spec.name,
        source_sha256=hashlib.sha256(pdf).hexdigest(),
        gt_version=GT_VERSION,
        matter=spec.matter,
        pages=pages,
        mentions=mentions,
        entities=spec.entities,
        relations=spec.relations,
        classified=classified,
        annotators=["synthetic"],
    )


def _digital_pages(doc: pymupdf.Document, spec: DocSpec) -> tuple[list[GTPage], list[list], list[_WordLayout]]:
    pages, toks_all, layout_all = [], [], []
    for i, page in enumerate(doc):
        words, layout, toks = _extract_words(page, spec.pages[i], i)
        regions = []
        for sq in spec.squiggles.get(i, []):
            x0, y0, x1, y1 = bbox(np.array(sq.points))
            regions.append(GTRegion(id=f"p{i}{sq.region_id}", kind=RegionKind.SIGNATURE,
                                    poly=rect_poly(x0 - 2, y0 - 2, x1 + 2, y1 + 2).tolist()))
        pages.append(GTPage(index=i, width_pt=spec.width, height_pt=spec.height, rotation=0, words=words,
                            regions=regions, conditions=["digital_text"]))
        toks_all.append(toks)
        layout_all.extend(layout)
    return pages, toks_all, layout_all


def digital(spec: DocSpec | None = None) -> Fixture:
    spec = spec or medico_legal_spec()
    doc = build_digital_doc(spec)
    pages, toks, _ = _digital_pages(doc, spec)
    pdf = _save(doc)
    return Fixture(f"{spec.name}-digital", pdf, _assemble_gt(spec, pages, toks, pdf))


# ----------------------------------------------------------------------------
# GT transforms for derived variants
# ----------------------------------------------------------------------------

PointMap = Callable[[np.ndarray], np.ndarray]


def _map_poly(poly, fn: PointMap) -> list:
    return [tuple(pt) for pt in fn(np.asarray(poly, dtype=np.float64)).tolist()]


def transform_gt(
    gt: GTDocument,
    fn: Callable[[int], PointMap],
    page_size: Callable[[GTPage], tuple[float, float]],
    pdf: bytes,
    source: WordSource | None = None,
    angle_delta: float = 0.0,
    rotation: int | None = None,
    condition: str | None = None,
) -> GTDocument:
    pages = []
    for p in gt.pages:
        f = fn(p.index)
        w, h = page_size(p)
        words = [
            w_.model_copy(update={
                "poly": _map_poly(w_.poly, f),
                "char_polys": [_map_poly(c, f) for c in w_.char_polys] if w_.char_polys else None,
                "angle": (w_.angle + angle_delta) % 360,
                "source": source or w_.source,
            })
            for w_ in p.words
        ]
        regions = [r.model_copy(update={"poly": _map_poly(r.poly, f)}) for r in p.regions]
        conditions = [condition] if condition else p.conditions
        pages.append(p.model_copy(update={
            "width_pt": w, "height_pt": h, "words": words, "regions": regions,
            "rotation": p.rotation if rotation is None else rotation, "conditions": conditions,
        }))
    return gt.model_copy(update={"pages": pages, "source_sha256": hashlib.sha256(pdf).hexdigest()})


# ----------------------------------------------------------------------------
# raster variants
# ----------------------------------------------------------------------------


def _render_gray(page: pymupdf.Page, dpi: float) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()


def _rotation_px(shape: tuple[int, int], angle_cw: float) -> np.ndarray:
    h, w = shape
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_cw, 1.0)  # cv2: positive = counter-clockwise


def _jpeg(gray: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def raster(
    spec: DocSpec | None = None,
    dpi: float = 200,
    skew_deg: float = 0.0,
    jpeg_quality: int = 80,
    hidden_text: bool = False,
) -> Fixture:
    spec = spec or medico_legal_spec()
    base = build_digital_doc(spec)
    pages, toks, layout = _digital_pages(base, spec)
    base_gt = _assemble_gt(spec, pages, toks, b"")
    out = _new_doc()
    mats = {}
    for i, bpage in enumerate(base):
        gray = _render_gray(bpage, dpi)
        if skew_deg:
            m = _rotation_px(gray.shape, skew_deg)
            gray = cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR,
                                  borderValue=255)
            mats[i] = m
        page = out.new_page(width=spec.width, height=spec.height)
        page.insert_image(page.rect, stream=_jpeg(gray, jpeg_quality))
        if hidden_text and not skew_deg:
            for wl in layout:
                if wl.page == i:
                    page.insert_text((wl.x0, wl.baseline), wl.text, fontname=FONT, fontsize=wl.size, render_mode=3)
    pdf = _save(out)
    s = dpi / 72.0

    def fn(page_index: int) -> PointMap:
        m = mats.get(page_index)
        if m is None:
            return lambda p: p
        return lambda p: ((np.hstack([p * s, np.ones((len(p), 1))]) @ m.T) / s)

    name = "raster_hidden_text" if hidden_text else "raster"
    cond = "raster_hidden_text" if hidden_text else "raster_single"
    gt = transform_gt(base_gt, fn, lambda p: (p.width_pt, p.height_pt), pdf, WordSource.RASTER,
                      angle_delta=skew_deg, condition=cond)
    suffix = f"-skew{skew_deg:g}" if skew_deg else ""
    return Fixture(f"{spec.name}-{name}{suffix}", pdf, gt)


def raster_hidden_text(spec: DocSpec | None = None, dpi: float = 200) -> Fixture:
    return raster(spec, dpi=dpi, hidden_text=True)


def rotated_content(spec: DocSpec | None = None, dpi: float = 200, jpeg_quality: int = 80) -> Fixture:
    """Landscape content rotated 90° counter-clockwise into a portrait page image.

    Text then reads bottom-to-top (angle 270), like the narrative annexure of the
    seed Notice of Claim. Page /Rotate stays 0.
    """
    spec = spec or narrative_spec()
    base = build_digital_doc(spec)
    pages, toks, _ = _digital_pages(base, spec)
    base_gt = _assemble_gt(spec, pages, toks, b"")
    out = _new_doc()
    for bpage in base:
        gray = np.ascontiguousarray(np.rot90(_render_gray(bpage, dpi), k=1))
        page = out.new_page(width=spec.height, height=spec.width)
        page.insert_image(page.rect, stream=_jpeg(gray, jpeg_quality))
    pdf = _save(out)
    w_land = spec.width
    gt = transform_gt(base_gt, lambda _: (lambda p: np.column_stack([p[:, 1], w_land - p[:, 0]])),
                      lambda p: (p.height_pt, p.width_pt), pdf, WordSource.RASTER, angle_delta=270,
                      condition="raster_single;content_rotated_90")
    return Fixture(f"{spec.name}-rotated_content", pdf, gt)


# ----------------------------------------------------------------------------
# MRC (layered) variant
# ----------------------------------------------------------------------------


def _ink_bands(ink: np.ndarray, min_gap: int = 6) -> list[tuple[int, int, int, int]]:
    """Group ink rows into bands and return their tight (x0, y0, x1, y1) boxes."""
    rows = np.flatnonzero(ink.any(axis=1))
    if rows.size == 0:
        return []
    bands, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > min_gap:
            bands.append((start, prev + 1))
            start = r
        prev = r
    bands.append((start, prev + 1))
    boxes = []
    for y0, y1 in bands:
        cols = np.flatnonzero(ink[y0:y1].any(axis=0))
        boxes.append((int(cols[0]), int(y0), int(cols[-1]) + 1, int(y1)))
    return boxes


def mrc(spec: DocSpec | None = None, bg_dpi: float = 150, fg_dpi: float = 300, ghost: float = 0.06) -> Fixture:
    spec = spec or medico_legal_spec()
    base = build_digital_doc(spec)
    pages, toks, _ = _digital_pages(base, spec)
    base_gt = _assemble_gt(spec, pages, toks, b"")
    text_only = build_digital_doc(spec, text=True, graphics=False)
    graphics_only = build_digital_doc(spec, text=False, graphics=True)

    pdf = pikepdf.new()
    W, H = spec.width, spec.height
    for i in range(len(spec.pages)):
        fg = _render_gray(text_only[i], fg_dpi)
        ink = fg < 160
        bg_graphics = _render_gray(graphics_only[i], bg_dpi).astype(np.float64)
        ghost_layer = 255.0 - _render_gray(text_only[i], bg_dpi).astype(np.float64)
        bg = np.clip(bg_graphics - ghost * ghost_layer, 0, 255).astype(np.uint8)
        xobjects = pikepdf.Dictionary()
        xobjects["/Bg"] = pdf.make_stream(
            bg.tobytes(), Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=bg.shape[1],
            Height=bg.shape[0], ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8)
        ops = [f"q {W:.4f} 0 0 {H:.4f} 0 0 cm /Bg Do Q".encode()]
        s = 72.0 / fg_dpi
        for k, (x0, y0, x1, y1) in enumerate(_ink_bands(ink)):
            band = ink[y0:y1, x0:x1]
            bits = np.packbits(~band, axis=1)  # ImageMask: 0 paints with the fill colour
            xobjects[f"/M{k}"] = pdf.make_stream(
                bits.tobytes(), Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=band.shape[1],
                Height=band.shape[0], ImageMask=True, BitsPerComponent=1)
            ops.append(
                f"q 0 g {band.shape[1] * s:.4f} 0 0 {band.shape[0] * s:.4f} {x0 * s:.4f} {H - y1 * s:.4f} cm /M{k} Do Q"
                .encode())
        page = pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, W, H],
                                  Resources=pikepdf.Dictionary(XObject=xobjects),
                                  Contents=pdf.make_stream(b"\n".join(ops)))
        pdf.pages.append(pikepdf.Page(page))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True, compress_streams=True)
    data = buf.getvalue()
    gt = transform_gt(base_gt, lambda _: (lambda p: p), lambda p: (p.width_pt, p.height_pt), data,
                      WordSource.RASTER, condition="raster_mrc")
    return Fixture(f"{spec.name}-mrc", data, gt)


# ----------------------------------------------------------------------------
# vector outline variant
# ----------------------------------------------------------------------------


def vector(spec: DocSpec | None = None, dpi: float = 600) -> Fixture:
    """Glyphs as filled paths (traced from a high-resolution render); no text layer."""
    spec = spec or medico_legal_spec()
    base = build_digital_doc(spec)
    pages, toks, _ = _digital_pages(base, spec)
    base_gt = _assemble_gt(spec, pages, toks, b"")
    text_only = build_digital_doc(spec, text=True, graphics=False)
    pdf = pikepdf.new()
    W, H = spec.width, spec.height
    s = 72.0 / dpi
    for i in range(len(spec.pages)):
        ink = (_render_gray(text_only[i], dpi) < 128).astype(np.uint8)
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ops = ["0 g"]
        if hierarchy is not None:
            hier = hierarchy[0]
            for ci, cnt in enumerate(contours):
                if hier[ci][3] != -1:
                    continue  # holes are emitted with their outer contour
                group = [cnt]
                child = hier[ci][2]
                while child != -1:
                    group.append(contours[child])
                    child = hier[child][0]
                for c in group:
                    pts = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2)
                    if len(pts) < 3:
                        continue
                    coords = [((x + 0.5) * s, H - (y + 0.5) * s) for x, y in pts]
                    ops.append(f"{coords[0][0]:.3f} {coords[0][1]:.3f} m")
                    ops.extend(f"{x:.3f} {y:.3f} l" for x, y in coords[1:])
                    ops.append("h")
                ops.append("f*")
        for sq in spec.squiggles.get(i, []):
            pts = [(x, H - y) for x, y in sq.points]
            ops.append("1.6 w 0 G " + f"{pts[0][0]:.3f} {pts[0][1]:.3f} m " +
                       " ".join(f"{x:.3f} {y:.3f} l" for x, y in pts[1:]) + " S")
        page = pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, W, H],
                                  Contents=pdf.make_stream("\n".join(ops).encode()))
        pdf.pages.append(pikepdf.Page(page))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True, compress_streams=True)
    data = buf.getvalue()
    gt = transform_gt(base_gt, lambda _: (lambda p: p), lambda p: (p.width_pt, p.height_pt), data,
                      WordSource.VECTOR, condition="vector_outline")
    return Fixture(f"{spec.name}-vector", data, gt)


# ----------------------------------------------------------------------------
# page-box variants
# ----------------------------------------------------------------------------


def rotate_attr(spec: DocSpec | None = None, rotate: int = 90) -> Fixture:
    spec = spec or medico_legal_spec()
    base = digital(spec)
    with pikepdf.open(io.BytesIO(base.pdf)) as pdf:
        for p in pdf.pages:
            p.obj.Rotate = rotate
        buf = io.BytesIO()
        pdf.save(buf, deterministic_id=True)
    data = buf.getvalue()
    geom = PageGeometry((0, 0, spec.width, spec.height), (0, 0, spec.width, spec.height), rotate)
    gt = transform_gt(base.gt, lambda _: (lambda p: geom.transform(p, Frame.UNROTATED_PT, Frame.DISPLAY_PT)),
                      lambda p: geom.display_size, data, angle_delta=rotate, rotation=rotate,
                      condition=f"digital_text;rotate_attr_{rotate}")
    return Fixture(f"{spec.name}-rotate{rotate}", data, gt)


def cropbox(spec: DocSpec | None = None, offset: tuple[float, float] = (50, 60)) -> Fixture:
    """Same displayed page, but content is translated and the CropBox offset to match."""
    spec = spec or medico_legal_spec()
    base = digital(spec)
    dx, dy = offset
    with pikepdf.open(io.BytesIO(base.pdf)) as pdf:
        for p in pdf.pages:
            p.contents_coalesce()
            body = p.obj.Contents.read_bytes()
            p.obj.Contents = pdf.make_stream(f"q 1 0 0 1 {dx} {dy} cm\n".encode() + body + b"\nQ")
            p.obj.MediaBox = [0, 0, spec.width + 2 * dx, spec.height + 2 * dy]
            p.obj.CropBox = [dx, dy, spec.width + dx, spec.height + dy]
        buf = io.BytesIO()
        pdf.save(buf, deterministic_id=True)
    data = buf.getvalue()
    gt = transform_gt(base.gt, lambda _: (lambda p: p), lambda p: (p.width_pt, p.height_pt), data,
                      condition="digital_text;cropbox_offset")
    return Fixture(f"{spec.name}-cropbox", data, gt)


ALL_VARIANTS: dict[str, Callable[[], Fixture]] = {
    "digital": digital,
    "raster": raster,
    "raster_skew": lambda: raster(skew_deg=1.2),
    "raster_hidden_text": raster_hidden_text,
    "mrc": mrc,
    "vector": vector,
    "rotated_content": rotated_content,
    "rotate_attr": rotate_attr,
    "cropbox": cropbox,
}
