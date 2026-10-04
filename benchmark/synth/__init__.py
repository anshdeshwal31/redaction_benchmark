"""Synthetic documents with exact ground truth (fictional content only)."""

from .fixtures import (
    ALL_VARIANTS,
    Fixture,
    cropbox,
    digital,
    mrc,
    raster,
    raster_hidden_text,
    rotate_attr,
    rotated_content,
    transform_gt,
    vector,
)
from .spec import DocSpec, medico_legal_spec, narrative_spec

__all__ = [
    "ALL_VARIANTS",
    "DocSpec",
    "Fixture",
    "cropbox",
    "digital",
    "medico_legal_spec",
    "mrc",
    "narrative_spec",
    "raster",
    "raster_hidden_text",
    "rotate_attr",
    "rotated_content",
    "transform_gt",
    "vector",
]
