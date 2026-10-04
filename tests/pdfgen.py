"""Tiny helpers that build PDFs byte-by-byte with pikepdf, so tests control every coordinate."""

from __future__ import annotations

import io

import pikepdf


def page_pdf(
    content: bytes,
    mediabox=(0, 0, 600, 800),
    cropbox=None,
    rotate: int = 0,
    images: dict[str, tuple[int, int, bytes]] | None = None,
    rotate_on_parent: bool = False,
) -> bytes:
    """One-page PDF with Helvetica as /F1 and optional 8-bit grey images.

    ``images`` maps a resource name to ``(width, height, raw_gray_bytes)``.
    """
    pdf = pikepdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica)
    )
    xobjects = pikepdf.Dictionary()
    for name, (w, h, data) in (images or {}).items():
        xobjects[f"/{name}"] = pdf.make_stream(
            data,
            Type=pikepdf.Name.XObject,
            Subtype=pikepdf.Name.Image,
            Width=w,
            Height=h,
            ColorSpace=pikepdf.Name.DeviceGray,
            BitsPerComponent=8,
        )
    page = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font), XObject=xobjects),
        Contents=pdf.make_stream(content),
    )
    if not rotate_on_parent:
        page.MediaBox = list(mediabox)
        if cropbox is not None:
            page.CropBox = list(cropbox)
        if rotate:
            page.Rotate = rotate
    pdf.pages.append(pikepdf.Page(page))
    if rotate_on_parent:
        # Put the inheritable attributes on the /Pages node instead of the page.
        root_pages = pdf.Root.Pages
        root_pages.MediaBox = list(mediabox)
        if cropbox is not None:
            root_pages.CropBox = list(cropbox)
        if rotate:
            root_pages.Rotate = rotate
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()
