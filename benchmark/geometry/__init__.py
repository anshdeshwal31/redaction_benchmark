"""Coordinate frames and polygon geometry shared by every evaluator module."""

from .frames import (
    Frame,
    PageGeometry,
    check_against_pymupdf,
    geometry_from_pikepdf_page,
    load_page_geometries,
    normalize_rect,
)
from .polygons import (
    area,
    as_poly,
    bbox,
    dilate,
    geometry_polygons,
    intersection_area,
    oriented_cells,
    rasterize,
    rect_poly,
    to_shapely,
    union,
)

__all__ = [
    "Frame",
    "PageGeometry",
    "check_against_pymupdf",
    "geometry_from_pikepdf_page",
    "load_page_geometries",
    "normalize_rect",
    "area",
    "as_poly",
    "bbox",
    "dilate",
    "geometry_polygons",
    "intersection_area",
    "oriented_cells",
    "rasterize",
    "rect_poly",
    "to_shapely",
    "union",
]
