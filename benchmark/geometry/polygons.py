"""Polygon helpers: construction, set operations, rasterisation, oriented cells.

Polygons are ``(N, 2)`` float arrays. Unless stated otherwise they are in the
canonical ``display_pt`` frame (see :mod:`benchmark.geometry.frames`).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry


def as_poly(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(arr) < 3:
        raise ValueError(f"a polygon needs at least 3 points, got {len(arr)}")
    return arr


def rect_poly(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Axis-aligned rectangle as a polygon (TL, TR, BR, BL in a y-down frame)."""
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def bbox(poly: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(poly).reshape(-1, 2)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def to_shapely(poly: np.ndarray) -> BaseGeometry:
    geom = ShapelyPolygon(np.asarray(poly).reshape(-1, 2))
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return geom


def union(polys: Iterable[np.ndarray]) -> BaseGeometry:
    geoms = [to_shapely(p) for p in polys]
    return shapely.union_all(geoms) if geoms else ShapelyPolygon()


def area(poly: np.ndarray) -> float:
    return float(to_shapely(poly).area)


def intersection_area(a: np.ndarray | BaseGeometry, b: np.ndarray | BaseGeometry) -> float:
    ga = a if isinstance(a, BaseGeometry) else to_shapely(a)
    gb = b if isinstance(b, BaseGeometry) else to_shapely(b)
    return float(ga.intersection(gb).area)


def dilate(poly: np.ndarray, distance: float) -> np.ndarray:
    """Grow a polygon by ``distance`` (mitred joins keep rectangles rectangular)."""
    grown = to_shapely(poly).buffer(distance, join_style="mitre")
    if grown.is_empty:
        return np.asarray(poly, dtype=np.float64)
    if grown.geom_type != "Polygon":
        grown = grown.convex_hull
    return np.asarray(grown.exterior.coords[:-1], dtype=np.float64)


def geometry_polygons(geom: BaseGeometry) -> list[np.ndarray]:
    """Exterior rings of a (multi)polygon as arrays (holes are dropped)."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [np.asarray(geom.exterior.coords[:-1], dtype=np.float64)]
    out: list[np.ndarray] = []
    for g in getattr(geom, "geoms", []):
        out.extend(geometry_polygons(g))
    return out


def rasterize(
    polys: Iterable[np.ndarray],
    shape: tuple[int, int],
    scale: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Boolean mask of ``shape`` (rows, cols) covered by ``polys``.

    Polygons are in points; pixel coordinates are ``(point - origin) * scale`` with
    pixel ``(i, j)`` spanning ``[i, i+1) x [j, j+1)``. A pixel is set exactly when its
    centre lies inside a polygon (even-odd rule, half-open edges), so adjacent
    polygons never double-count a pixel. ``cv2.fillPoly`` is deliberately not used:
    it fills boundary pixels inclusively and over-counts by up to one pixel per side.
    """
    mask = np.zeros(shape, dtype=bool)
    ox, oy = origin
    for p in polys:
        q = (np.asarray(p, dtype=np.float64).reshape(-1, 2) - (ox, oy)) * scale
        _fill_exact(mask, q)
    return mask


def _fill_exact(mask: np.ndarray, poly_px: np.ndarray) -> None:
    rows, cols = mask.shape
    xs, ys = poly_px[:, 0], poly_px[:, 1]
    j0 = max(int(math.ceil(ys.min() - 0.5)), 0)
    j1 = min(int(math.ceil(ys.max() - 0.5)) - 1, rows - 1)
    if j1 < j0:
        return
    x0, y0 = xs, ys
    x1, y1 = np.roll(xs, -1), np.roll(ys, -1)
    yc = (np.arange(j0, j1 + 1) + 0.5)[:, None]
    crosses = ((y0 <= yc) & (yc < y1)) | ((y1 <= yc) & (yc < y0))
    dy = np.where(y1 == y0, 1.0, y1 - y0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # Near-horizontal edges never cross a scanline centre; their values are masked out.
        xi = np.where(crosses, x0 + (yc - y0) / dy * (x1 - x0), np.nan)
    xi.sort(axis=1)
    for k, row in enumerate(xi):
        row = row[~np.isnan(row)]
        for a, b in zip(row[0::2], row[1::2]):
            i0 = max(int(math.ceil(a - 0.5)), 0)
            i1 = min(int(math.ceil(b - 0.5)) - 1, cols - 1)
            if i1 >= i0:
                mask[j0 + k, i0 : i1 + 1] = True


def oriented_cells(poly: np.ndarray, n: int, angle_deg: float = 0.0) -> list[np.ndarray]:
    """Split ``poly`` into ``n`` equal cells along the reading direction.

    ``angle_deg`` is the direction of the text baseline measured clockwise from the
    +x axis in the y-down display frame (0 = normal text, 90 = reads top-to-bottom,
    270 = reads bottom-to-top). Cells are clipped to the polygon.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    poly = as_poly(poly)
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)])
    v = np.array([-math.sin(t), math.cos(t)])
    tu = poly @ u
    sv = poly @ v
    t0, t1 = float(tu.min()), float(tu.max())
    s0, s1 = float(sv.min()), float(sv.max())
    shape = to_shapely(poly)
    cells: list[np.ndarray] = []
    step = (t1 - t0) / n
    for k in range(n):
        a, b = t0 + k * step, t0 + (k + 1) * step
        corners = np.array([a * u + s0 * v, b * u + s0 * v, b * u + s1 * v, a * u + s1 * v])
        clipped = shape.intersection(ShapelyPolygon(corners))
        rings = geometry_polygons(clipped)
        cells.append(max(rings, key=lambda r: to_shapely(r).area) if rings else corners)
    return cells
