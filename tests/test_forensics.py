import pymupdf

from benchmark.dataset import classify_encoding
from benchmark.forensics import inspect_pdf, revision_prefixes
from benchmark.schemas import PageEncoding
from benchmark.synth import digital, mrc, raster_hidden_text, vector


def test_digital_text_is_visible_and_values_hidden_by_default():
    fx = digital()
    doc = pymupdf.open("pdf", fx.pdf)
    doc.set_metadata({"author": "Jane Elizabeth Smithers", "title": "Report"})
    inv = inspect_pdf(doc.tobytes())
    assert inv.pages[0].visible_chars > 0 and inv.pages[0].invisible_chars == 0
    assert "author" in inv.metadata and inv.metadata["author"] is None  # key only
    shown = inspect_pdf(doc.tobytes(), include_values=True)
    assert shown.metadata["author"] == "Jane Elizabeth Smithers"
    assert classify_encoding(inv.pages[0])[0] is PageEncoding.DIGITAL_TEXT


def test_hidden_ocr_layer_is_reported_as_invisible_text():
    inv = inspect_pdf(raster_hidden_text().pdf)
    p = inv.pages[0]
    assert p.visible_chars == 0 and p.invisible_chars > 300
    assert classify_encoding(p)[0] is PageEncoding.RASTER_HIDDEN_TEXT


def test_mrc_layers_are_inventoried():
    inv = inspect_pdf(mrc().pdf)
    p = inv.pages[0]
    masks = [im for im in p.images if im.is_mask]
    full = [im for im in p.images if im.coverage > 0.85]
    assert len(full) == 1 and abs(full[0].dpi_x - 150) < 1
    assert len(masks) >= 5 and all(abs(m.dpi_x - 300) < 2 for m in masks)
    enc, dpi = classify_encoding(p)
    assert enc is PageEncoding.RASTER_MRC and abs(dpi - 150) < 1


def test_vector_outline_page():
    inv = inspect_pdf(vector().pdf)
    p = inv.pages[0]
    assert p.visible_chars == 0 and p.filled_paths >= 200
    assert classify_encoding(p)[0] is PageEncoding.VECTOR_OUTLINE


def test_incremental_save_keeps_the_previous_revision(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(digital().pdf)
    doc = pymupdf.open(str(path))
    page = doc[0]
    word = next(w for w in page.get_text("words") if w[4] == "Elizabeth")
    page.add_redact_annot(pymupdf.Rect(word[:4]))
    page.apply_redactions()
    doc.saveIncr()
    doc.close()
    raw = path.read_bytes()
    inv = inspect_pdf(raw)
    assert inv.eof_markers == 2 and inv.mupdf_versions == 2
    prefixes = revision_prefixes(raw)
    assert len(prefixes) == 1
    old_text = pymupdf.open("pdf", prefixes[0])[0].get_text()
    assert "Elizabeth" in old_text  # the "redacted" word survives in the earlier revision
    assert "Elizabeth" not in pymupdf.open("pdf", raw)[0].get_text()


def test_unreferenced_streams_are_counted():
    doc = pymupdf.open("pdf", digital().pdf)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, b"BT /F1 12 Tf (orphaned text) Tj ET")
    inv = inspect_pdf(doc.tobytes(garbage=0))
    assert inv.unreferenced_streams >= 1
