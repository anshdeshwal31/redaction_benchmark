"""Candidate-independent PDF inventory (PLAN.md §5.3, §14.2 `forensics`).

Lists every place content can live: visible and invisible text, each image layer
(including stencil masks and soft masks), vector paths, annotations, metadata, XMP,
outlines, form fields, attachments, thumbnails, structure-tree alternate text, optional
content, JavaScript, incremental revisions and unreferenced objects.

Values that can be personal data (metadata values, outline titles, field values,
alternate text) are only included when ``include_values=True``; reports written to
disk default to keys and counts.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pikepdf
import pymupdf

from ..geometry import Frame, PageGeometry, geometry_from_pikepdf_page


@dataclass
class ImageLayer:
    xref: int
    width: int
    height: int
    bpc: int
    is_mask: bool
    has_smask: bool
    filters: list[str]
    bbox: tuple[float, float, float, float]  # display_pt
    dpi_x: float | None
    dpi_y: float | None
    coverage: float  # fraction of page area


@dataclass
class AnnotationInfo:
    type: str
    bbox: tuple[float, float, float, float]  # display_pt
    has_contents: bool
    contents: str | None = None


@dataclass
class PageInventory:
    index: int
    mediabox: tuple[float, float, float, float]
    cropbox: tuple[float, float, float, float]
    rotation: int
    display_size: tuple[float, float]
    visible_chars: int
    invisible_chars: int
    fonts: list[str]
    images: list[ImageLayer]
    vector_paths: int
    filled_paths: int
    annotations: list[AnnotationInfo]
    links: int
    has_thumbnail: bool
    widgets: int


@dataclass
class DocumentInventory:
    sha256: str
    size_bytes: int
    pdf_version: str | None
    page_count: int
    encrypted: bool
    producer: str | None
    metadata: dict[str, str | None]  # key -> value (value None unless include_values)
    has_xmp: bool
    xmp: str | None
    outline_entries: int
    outline_titles: list[str] | None
    embedded_files: list[str]
    form_fields: dict[str, str | None]
    has_javascript: bool
    optional_content_groups: list[str]
    struct_tree: bool
    alt_texts: list[str] | None
    alt_text_count: int
    eof_markers: int
    mupdf_versions: int
    linearized: bool
    unreferenced_objects: int
    unreferenced_streams: int
    pages: list[PageInventory] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _display_bbox(geom: PageGeometry, rect) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    pts = geom.transform([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], Frame.UNROTATED_PT, Frame.DISPLAY_PT)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def _filters(obj_src: str) -> list[str]:
    m = re.search(r"/Filter\s*(\[[^\]]*\]|/\w+)", obj_src)
    return re.findall(r"/(\w+)", m.group(1)) if m else []


def _page_inventory(doc: pymupdf.Document, page: pymupdf.Page, geom: PageGeometry, pk_page,
                    include_values: bool) -> PageInventory:
    visible = invisible = 0
    for span in page.get_texttrace():
        n = len(span["chars"])
        if span["type"] == 3 or span.get("opacity", 1.0) == 0:
            invisible += n
        else:
            visible += n
    fonts = sorted({f[3] for f in page.get_fonts(full=True)})
    w, h = geom.display_size
    page_area = w * h
    images: list[ImageLayer] = []
    for info in page.get_image_info(xrefs=True):
        xref = info.get("xref", 0)
        src = doc.xref_object(xref, compressed=True) if xref else ""
        bb = _display_bbox(geom, info["bbox"])
        a, b, c, d = info["transform"][:4]
        sx = (a * a + b * b) ** 0.5
        sy = (c * c + d * d) ** 0.5
        images.append(ImageLayer(
            xref=xref,
            width=int(info["width"]),
            height=int(info["height"]),
            bpc=int(info["bpc"]),
            is_mask="/ImageMask true" in src,
            has_smask="/SMask" in src,
            filters=_filters(src),
            bbox=bb,
            dpi_x=round(info["width"] / (sx / 72.0), 2) if sx > 0 else None,
            dpi_y=round(info["height"] / (sy / 72.0), 2) if sy > 0 else None,
            coverage=round(max(0.0, (bb[2] - bb[0]) * (bb[3] - bb[1])) / page_area, 4),
        ))
    drawings = page.get_drawings()
    annots = []
    for an in page.annots() or []:
        contents = an.info.get("content") or ""
        annots.append(AnnotationInfo(type=an.type[1], bbox=_display_bbox(geom, an.rect),
                                     has_contents=bool(contents), contents=contents if include_values else None))
    return PageInventory(
        index=page.number,
        mediabox=geom.mediabox,
        cropbox=geom.cropbox,
        rotation=geom.rotation,
        display_size=geom.display_size,
        visible_chars=visible,
        invisible_chars=invisible,
        fonts=fonts,
        images=images,
        vector_paths=len(drawings),
        filled_paths=sum(1 for d in drawings if d.get("fill") is not None),
        annotations=annots,
        links=len(page.get_links()),
        has_thumbnail="/Thumb" in pk_page.obj,
        widgets=len(list(page.widgets() or [])),
    )


def _reachable_objgens(pdf: pikepdf.Pdf) -> set[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    stack = [pdf.trailer]
    while stack:
        obj = stack.pop()
        if isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream)):
            if obj.is_indirect:
                og = obj.objgen
                if og in seen:
                    continue
                seen.add(og)
            items = obj.items() if isinstance(obj, pikepdf.Dictionary) else obj.stream_dict.items()
            stack.extend(v for _, v in items)
        elif isinstance(obj, pikepdf.Array):
            if obj.is_indirect:
                og = obj.objgen
                if og in seen:
                    continue
                seen.add(og)
            stack.extend(obj)
    return seen


def _struct_alt_texts(pdf: pikepdf.Pdf, limit: int = 100_000) -> list[str]:
    root = pdf.Root.get("/StructTreeRoot")
    if root is None:
        return []
    out, stack, seen, steps = [], [root], set(), 0
    while stack and steps < limit:
        node = stack.pop()
        steps += 1
        if isinstance(node, pikepdf.Dictionary):
            if node.is_indirect:
                if node.objgen in seen:
                    continue
                seen.add(node.objgen)
            for key in ("/Alt", "/ActualText", "/E", "/T"):
                if key in node and isinstance(node[key], pikepdf.String):
                    out.append(str(node[key]))
            kids = node.get("/K")
            if kids is not None:
                stack.extend(kids if isinstance(kids, pikepdf.Array) else [kids])
        elif isinstance(node, pikepdf.Array):
            stack.extend(node)
    return out


def _has_javascript(pdf: pikepdf.Pdf) -> bool:
    names = pdf.Root.get("/Names")
    if names is not None and "/JavaScript" in names:
        return True
    oa = pdf.Root.get("/OpenAction")
    if isinstance(oa, pikepdf.Dictionary) and oa.get("/S") == pikepdf.Name.JavaScript:
        return True
    return False


def inspect_pdf(data: bytes | str | Path, include_values: bool = False) -> DocumentInventory:
    raw = Path(data).read_bytes() if not isinstance(data, (bytes, bytearray)) else bytes(data)
    doc = pymupdf.open("pdf", raw)
    with pikepdf.open(io.BytesIO(raw)) as pk:
        geoms = [geometry_from_pikepdf_page(p) for p in pk.pages]
        meta = {k: (v if include_values else None) for k, v in (doc.metadata or {}).items()
                if v and k not in ("format", "encryption")}
        xmp = doc.get_xml_metadata() or ""
        toc = doc.get_toc(simple=True)
        fields: dict[str, str | None] = {}
        for page in doc:
            for wdg in page.widgets() or []:
                fields[wdg.field_name or f"field{len(fields)}"] = (str(wdg.field_value) if include_values else None)
        reachable = _reachable_objgens(pk)
        # Indirect scalars (e.g. stream lengths) come back as plain Python values and
        # cannot hide content; only containers and streams are counted.
        containers = (pikepdf.Dictionary, pikepdf.Stream, pikepdf.Array)
        unref = [o for o in pk.objects
                 if isinstance(o, containers) and o.is_indirect and o.objgen not in reachable]
        alt = _struct_alt_texts(pk)
        ocgs = doc.get_ocgs() or {}
        inv = DocumentInventory(
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            pdf_version=pk.pdf_version,
            page_count=doc.page_count,
            encrypted=pk.is_encrypted,
            producer=(doc.metadata or {}).get("producer") or None,
            metadata=meta,
            has_xmp=bool(xmp.strip()),
            xmp=xmp if include_values and xmp.strip() else None,
            outline_entries=len(toc),
            outline_titles=[t[1] for t in toc] if include_values else None,
            embedded_files=list(doc.embfile_names()) if include_values else [f"<{n} file(s)>" for n in
                                                                              [doc.embfile_count()] if n],
            form_fields=fields,
            has_javascript=_has_javascript(pk),
            optional_content_groups=[v.get("name", str(k)) for k, v in ocgs.items()] if include_values
            else [f"<{len(ocgs)} group(s)>"] if ocgs else [],
            struct_tree="/StructTreeRoot" in pk.Root,
            alt_texts=alt if include_values else None,
            alt_text_count=len(alt),
            eof_markers=len(re.findall(rb"%%EOF", raw)),
            mupdf_versions=doc.version_count,
            linearized=bool(doc.is_fast_webaccess),
            unreferenced_objects=len(unref),
            unreferenced_streams=sum(1 for o in unref if isinstance(o, pikepdf.Stream)),
        )
        for page, geom, pk_page in zip(doc, geoms, pk.pages):
            inv.pages.append(_page_inventory(doc, page, geom, pk_page, include_values))
    return inv


def revision_prefixes(raw: bytes) -> list[bytes]:
    """Earlier revisions of an incrementally-updated file (each ends at a %%EOF).

    The last prefix equals the whole file and is excluded. Prefixes that do not open
    as a PDF (e.g. the first-page section of a linearised file) are skipped.
    """
    out = []
    for m in re.finditer(rb"%%EOF[ \t]*(\r\n|\r|\n)?", raw):
        end = m.end()
        if end >= len(raw.rstrip()):
            continue
        prefix = raw[:end]
        try:
            d = pymupdf.open("pdf", prefix)
            if d.page_count > 0:
                out.append(prefix)
        except Exception:  # noqa: BLE001 - malformed prefix: not a usable revision
            continue
    return out
