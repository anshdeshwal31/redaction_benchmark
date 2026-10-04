import itertools

import numpy as np
import pikepdf
import pymupdf
import pypdfium2 as pdfium
import pytest
import shapely
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import Polygon as ShapelyPolygon

from benchmark.geometry import (
    Frame,
    PageGeometry,
    check_against_pymupdf,
    geometry_from_pikepdf_page,
    oriented_cells,
    rasterize,
    rect_poly,
    to_shapely,
)
from tests.pdfgen import page_pdf

ROTATIONS = (0, 90, 180, 270)
CROPS = (None, (50, 100, 550, 750))

coord = st.floats(min_value=-500, max_value=2000, allow_nan=False, allow_infinity=False)


@st.composite
def geometries(draw):
    x0 = draw(st.floats(-200, 200))
    y0 = draw(st.floats(-200, 200))
    w = draw(st.floats(50, 2000))
    h = draw(st.floats(50, 2000))
    mb = (x0, y0, x0 + w, y0 + h)
    cx0 = draw(st.floats(x0, x0 + w * 0.4))
    cy0 = draw(st.floats(y0, y0 + h * 0.4))
    cx1 = draw(st.floats(cx0 + 10, x0 + w))
    cy1 = draw(st.floats(cy0 + 10, y0 + h))
    return PageGeometry(mb, (cx0, cy0, cx1, cy1), draw(st.sampled_from(ROTATIONS)))


@settings(max_examples=300, deadline=None)
@given(geometries(), st.lists(st.tuples(coord, coord), min_size=1, max_size=8), st.floats(36, 600))
def test_round_trip_every_frame_pair(g, pts, dpi):
    pts = np.array(pts)
    for src, dst in itertools.permutations(list(Frame), 2):
        there = g.transform(pts, src, dst, dpi)
        back = g.transform(there, dst, src, dpi)
        assert np.allclose(back, pts, atol=1e-6), (src, dst)


@settings(max_examples=200, deadline=None)
@given(geometries())
def test_display_corners_map_to_cropbox_corners(g):
    cx0, cy0, cx1, cy1 = g.cropbox
    corners_user = np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]])
    disp = g.transform(corners_user, Frame.PDF_USER, Frame.DISPLAY_PT)
    w, h = g.display_size
    assert np.allclose(disp.min(axis=0), [0, 0], atol=1e-6)
    assert np.allclose(disp.max(axis=0), [w, h], atol=1e-6)


def test_display_size_swaps_for_quarter_turns():
    g = PageGeometry((0, 0, 600, 800), (0, 0, 600, 800), 90)
    assert g.display_size == (800, 600)
    assert PageGeometry((0, 0, 600, 800), (0, 0, 600, 800), 180).display_size == (600, 800)


def test_cropbox_is_clipped_and_rotation_normalised():
    g = PageGeometry((0, 0, 100, 100), (-10, -10, 200, 50), -90)
    assert g.cropbox == (0, 0, 100, 50)
    assert g.rotation == 270
    with pytest.raises(ValueError):
        PageGeometry((0, 0, 100, 100), (0, 0, 100, 100), 45)


def _dark_bbox(gray: np.ndarray, thresh: int = 60):
    ys, xs = np.nonzero(gray < thresh)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


@pytest.mark.parametrize("rotate", ROTATIONS)
@pytest.mark.parametrize("cropbox", CROPS)
@pytest.mark.parametrize("dpi", (72, 150))
def test_frames_match_mupdf_and_pdfium_renders(rotate, cropbox, dpi):
    # A filled square at known user-space coordinates must land where our
    # conversion says it does, in both renderers.
    data = page_pdf(b"0 g 300 300 20 10 re f", cropbox=cropbox, rotate=rotate)
    with pikepdf.open(__import__("io").BytesIO(data)) as pdf:
        g = geometry_from_pikepdf_page(pdf.pages[0])
    expected = g.transform([[300, 300], [320, 310]], Frame.PDF_USER, Frame.IMAGE_PX, dpi)
    ex0, ey0 = expected.min(axis=0)
    ex1, ey1 = expected.max(axis=0)

    doc = pymupdf.open("pdf", data)
    assert check_against_pymupdf([g], doc) == []
    pix = doc[0].get_pixmap(dpi=dpi)
    mu = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., 0]
    assert (pix.w, pix.h) == g.pixel_size(dpi)
    pd = pdfium.PdfDocument(data)
    pf = pd[0].render(scale=dpi / 72).to_numpy()[..., 0]

    for gray in (mu, pf):
        x0, y0, x1, y1 = _dark_bbox(gray)
        assert abs(x0 - ex0) <= 1 and abs(x1 - ex1) <= 1
        assert abs(y0 - ey0) <= 1 and abs(y1 - ey1) <= 1


@pytest.mark.parametrize("rotate", ROTATIONS)
@pytest.mark.parametrize("cropbox", CROPS)
def test_pymupdf_extraction_is_in_unrotated_frame(rotate, cropbox):
    # Drawings, words and image placements reported by PyMuPDF are CropBox-relative
    # and unrotated; converting them with our matrices must give display coordinates.
    img = (40, 20, bytes([0] * 400 + [255] * 400))  # top half black, bottom half white
    content = b"0 g 300 300 20 10 re f BT /F1 12 Tf 200 400 Td (MARK) Tj ET q 80 0 0 40 100 150 cm /Im0 Do Q"
    data = page_pdf(content, cropbox=cropbox, rotate=rotate, images={"Im0": img})
    with pikepdf.open(__import__("io").BytesIO(data)) as pdf:
        g = geometry_from_pikepdf_page(pdf.pages[0])
    page = pymupdf.open("pdf", data)[0]

    rect = next(d["rect"] for d in page.get_drawings() if d.get("fill") is not None)
    got = g.transform([[rect.x0, rect.y0], [rect.x1, rect.y1]], Frame.UNROTATED_PT, Frame.DISPLAY_PT)
    want = g.transform([[300, 300], [320, 310]], Frame.PDF_USER, Frame.DISPLAY_PT)
    assert np.allclose(np.sort(got, axis=0), np.sort(want, axis=0), atol=0.01)

    word = next(w for w in page.get_text("words") if w[4] == "MARK")
    left_baseline = g.transform([[word[0], word[3]]], Frame.UNROTATED_PT, Frame.PDF_USER)[0]
    assert abs(left_baseline[0] - 200) < 0.5  # x of the text origin
    assert 395 < left_baseline[1] < 400.5  # bbox bottom sits at/just below the baseline

    info = page.get_image_info(xrefs=True)[0]
    a, b, c, d, e, f = info["transform"]
    # Unit-square corner (0, 0) is the first image row (top), in the unrotated frame.
    top_left_unrot = np.array([e, f])
    top_left_user = g.transform([top_left_unrot], Frame.UNROTATED_PT, Frame.PDF_USER)[0]
    assert np.allclose(top_left_user, [100, 190], atol=0.01)  # image spans y 150..190, top at 190
    assert np.allclose([a, d], [80, 40], atol=0.01)


def test_inherited_attributes_on_pages_node():
    data = page_pdf(b"0 g 10 10 5 5 re f", cropbox=(10, 20, 410, 520), rotate=90, rotate_on_parent=True)
    with pikepdf.open(__import__("io").BytesIO(data)) as pdf:
        g = geometry_from_pikepdf_page(pdf.pages[0])
    assert g.rotation == 90
    assert g.cropbox == (10, 20, 410, 520)
    assert check_against_pymupdf([g], pymupdf.open("pdf", data)) == []


def test_rasterize_axis_aligned_rect_is_pixel_exact():
    # Rect covering points [10, 20] x [5, 15] at 2 px/pt -> columns 20..39, rows 10..29.
    mask = rasterize([rect_poly(10, 5, 20, 15)], (60, 60), 2.0)
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 39, 10, 29)
    assert mask.sum() == 20 * 20


def test_rasterize_respects_origin():
    mask = rasterize([rect_poly(110, 205, 120, 215)], (40, 40), 2.0, origin=(100, 200))
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 39, 10, 29)


@pytest.mark.parametrize("angle", (0, 90, 270))
def test_oriented_cells_follow_reading_direction(angle):
    if angle == 0:
        word = rect_poly(0, 0, 100, 10)
    else:
        word = rect_poly(0, 0, 10, 100)
    cells = oriented_cells(word, 4, angle)
    assert len(cells) == 4
    centres = np.array([c.mean(axis=0) for c in cells])
    if angle == 0:
        assert np.all(np.diff(centres[:, 0]) > 0)  # left to right
    elif angle == 90:
        assert np.all(np.diff(centres[:, 1]) > 0)  # top to bottom
    else:
        assert np.all(np.diff(centres[:, 1]) < 0)  # bottom to top
    total = sum(to_shapely(c).area for c in cells)
    assert total == pytest.approx(1000, rel=1e-6)


@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.tuples(st.floats(0, 40), st.floats(0, 30)), min_size=3, max_size=7),
    st.floats(0.5, 4.0),
)
def test_rasterize_matches_point_in_polygon_at_pixel_centres(pts, scale):
    poly = np.array(pts)
    geom = to_shapely(poly)
    if geom.is_empty or geom.area < 1e-3 or not ShapelyPolygon(poly).is_valid:
        return  # self-intersecting inputs follow the even-odd rule; shapely repairs them differently
    shape = (int(30 * scale) + 3, int(40 * scale) + 3)
    mask = rasterize([poly], shape, scale)
    jj, ii = np.mgrid[0 : shape[0], 0 : shape[1]]
    cx, cy = (ii + 0.5) / scale, (jj + 0.5) / scale
    inside = shapely.contains_xy(ShapelyPolygon(poly), cx, cy)
    # Centres exactly on an edge may legitimately go either way (half-open edges).
    boundary = shapely.distance(ShapelyPolygon(poly).boundary, shapely.points(cx, cy)) < 1e-9
    assert np.array_equal(mask[~boundary], inside[~boundary])
