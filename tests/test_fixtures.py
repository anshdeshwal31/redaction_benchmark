import numpy as np
import pymupdf
import pytest

from benchmark.geometry import as_poly, dilate, rasterize
from benchmark.gt import validate_gt
from benchmark.schemas import WordSource
from benchmark.synth import ALL_VARIANTS


@pytest.fixture(scope="module", params=sorted(ALL_VARIANTS))
def fixture(request):
    return ALL_VARIANTS[request.param]()


def test_gt_is_valid_against_its_pdf(fixture, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(fixture.pdf)
    issues = validate_gt(fixture.gt, source_pdf=path)
    assert [str(i) for i in issues] == []


def test_gt_polygons_sit_on_the_rendered_ink(fixture):
    dpi = 100
    s = dpi / 72
    doc = pymupdf.open("pdf", fixture.pdf)
    for page in fixture.gt.pages:
        pix = doc[page.index].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        ink = gray < 128
        polys = [dilate(as_poly(w.poly), 1.0) for w in page.words]
        polys += [dilate(as_poly(r.poly), 1.0) for r in page.regions]
        covered = rasterize(polys, gray.shape, s)
        assert (ink & covered).sum() / ink.sum() > 0.995
        for w in page.words:
            assert (ink & rasterize([as_poly(w.poly)], gray.shape, s)).any(), w.id


def test_word_sources_match_encoding(fixture):
    sources = {w.source for p in fixture.gt.pages for w in p.words}
    name = fixture.name
    if name.endswith(("-digital", "-cropbox")) or "-rotate" in name and "rotated" not in name:
        assert sources == {WordSource.TEXT_LAYER}
    elif name.endswith("-vector"):
        assert sources == {WordSource.VECTOR}
    else:
        assert sources == {WordSource.RASTER}


@pytest.mark.parametrize("name", sorted(ALL_VARIANTS))
def test_generation_is_byte_deterministic(name):
    a, b = ALL_VARIANTS[name](), ALL_VARIANTS[name]()
    assert a.sha256 == b.sha256
    assert a.gt == b.gt


def test_rotated_content_reads_bottom_to_top():
    fx = ALL_VARIANTS["rotated_content"]()
    assert {w.angle for p in fx.gt.pages for w in p.words} == {270.0}
    page = fx.gt.pages[0]
    assert page.width_pt < page.height_pt  # portrait page carrying landscape content
