"""Dataset manifest (PLAN.md §8). Contains hashes and measured conditions, never PII.

Metadata *values* (Title, Author, ...) and original filenames can be personal data,
so the manifest records only which metadata keys are present.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field

from .common import Sha256, StrictModel

DATASET_SCHEMA = "rb.dataset/0.1"


class PageEncoding(str, Enum):
    DIGITAL_TEXT = "digital_text"  # visible text layer, no full-page raster
    RASTER_SINGLE = "raster_single"  # one full-page image, no text
    RASTER_MRC = "raster_mrc"  # full-page background + bilevel foreground layers
    RASTER_HIDDEN_TEXT = "raster_hidden_text"  # full-page raster + invisible OCR text layer
    RASTER_VISIBLE_TEXT = "raster_visible_text"  # full-page raster + visible text on top
    VECTOR_OUTLINE = "vector_outline"  # glyphs drawn as filled paths, no text layer
    MIXED = "mixed"
    BLANK = "blank"


class PageConditions(StrictModel):
    index: int = Field(ge=0)
    display_w: float
    display_h: float
    rotation_attr: Literal[0, 90, 180, 270]
    cropbox_offset: bool
    encoding: PageEncoding
    text_chars: int = Field(ge=0)
    invisible_text_chars: int = Field(ge=0)
    image_count: int = Field(ge=0)
    bilevel_image_count: int = Field(ge=0)
    image_coverage: float = Field(ge=0, description="Fraction of the page covered by image placements")
    full_page_image_dpi: float | None = None
    vector_paths: int = Field(ge=0)
    skew_deg: float | None = Field(None, description="Estimated text skew of raster content")
    text_axis: Literal["horizontal", "vertical", "unknown"] | None = None
    tags: list[str] = Field([], description="Annotated conditions (handwriting, form, table, ...)")


class Datasheet(StrictModel):
    document_type: str | None = None
    provenance: str | None = None
    authorisation: str | None = None
    notes: str | None = None


class DocumentEntry(StrictModel):
    doc_id: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    split: Literal["dev", "test", "regression", "synthetic"]
    matter: str | None = None
    gt_version: str | None = None
    pdf_version: str | None = None
    producer: str | None = None
    metadata_keys: list[str] = []
    has_xmp: bool = False
    encrypted: bool = False
    revisions: int = Field(1, ge=1, description="Number of %%EOF-terminated revisions")
    page_conditions: list[PageConditions]
    datasheet: Datasheet = Datasheet()


class DatasetManifest(StrictModel):
    schema_id: Literal["rb.dataset/0.1"] = Field(DATASET_SCHEMA, alias="schema")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    created: str
    measured_with: dict[str, str] = {}
    documents: list[DocumentEntry]

    def document(self, doc_id: str) -> DocumentEntry:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(doc_id)
