"""Ground-truth document schema (PLAN.md §7).

Principles encoded here:
- facts, not policy: spans record what they *are*; a policy profile decides what to do;
- visual-first, geometry-anchored: every word has a polygon in ``display_pt``;
- entity-centric: mentions link to entities with stable IDs across documents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import (
    ClassifiedClass,
    EntityKind,
    Legibility,
    NameComponent,
    PiiType,
    Polygon,
    RegionKind,
    Role,
    Sha256,
    StrictModel,
    WordSource,
)

GT_SCHEMA = "rb.gt/0.1"


class GTWord(StrictModel):
    id: str = Field(min_length=1)
    text: str
    poly: Polygon
    order: int = Field(ge=0, description="Reading order within the page")
    line: str | None = None
    block: str | None = None
    angle: float = Field(0.0, description="Baseline direction, degrees clockwise from +x in display_pt")
    source: WordSource
    legibility: Legibility = Legibility.CLEAR
    char_polys: list[Polygon] | None = Field(
        None, description="Exact character cells when known (born-digital); one per character of `text`"
    )

    @model_validator(mode="after")
    def _char_polys_match_text(self) -> "GTWord":
        if self.char_polys is not None and len(self.char_polys) != len(self.text):
            raise ValueError(f"word {self.id}: {len(self.char_polys)} char_polys for {len(self.text)} characters")
        return self


class GTRegion(StrictModel):
    """Non-text visual content: signatures, initials, avatars, photos, logos, stamps."""

    id: str = Field(min_length=1)
    kind: RegionKind
    poly: Polygon
    transcription: str | None = None


class GTPage(StrictModel):
    index: int = Field(ge=0)
    width_pt: float = Field(gt=0, description="Display width (after /Rotate)")
    height_pt: float = Field(gt=0, description="Display height (after /Rotate)")
    rotation: Literal[0, 90, 180, 270] = 0
    conditions: list[str] = []
    measured: dict[str, float | int | str | bool | None] = {}
    words: list[GTWord] = []
    regions: list[GTRegion] = []


class Segment(StrictModel):
    """A contiguous piece of a mention on one page.

    ``char_start``/``char_end`` select a sub-word span (e.g. the name inside an email
    address) and are only valid when the segment has exactly one word.
    """

    page: int = Field(ge=0)
    words: list[str] = []
    regions: list[str] = []
    char_start: int | None = Field(None, ge=0)
    char_end: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "Segment":
        if not self.words and not self.regions:
            raise ValueError("a segment needs at least one word or region")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be given together")
        if self.char_start is not None:
            if len(self.words) != 1:
                raise ValueError("a sub-word span needs exactly one word")
            if self.char_end <= self.char_start:
                raise ValueError("char_end must be greater than char_start")
        return self


class GTMention(StrictModel):
    id: str = Field(min_length=1)
    type: PiiType
    subtype: str | None = None
    entity: str | None = None
    text: str = Field(description="The mention as it appears, segments joined by single spaces")
    segments: list[Segment] = Field(min_length=1)
    optional_words: list[str] = Field([], description="Words whose redaction is neither required nor penalised")
    components: dict[NameComponent, list[str]] = {}
    form: str | None = None
    parent: str | None = Field(None, description="Enclosing mention for nested PII")
    conditions: list[str] = []


class AttributeValue(StrictModel):
    value: str
    evidence: list[str] = []


class GTEntity(StrictModel):
    id: str = Field(min_length=1)
    kind: EntityKind
    role: Role = Role.UNKNOWN
    canonical_name: str | None = None
    attributes: dict[str, AttributeValue] = {}
    matter: str | None = None


class GTRelation(StrictModel):
    type: str
    source: str
    target: str


class GTClassified(StrictModel):
    id: str = Field(min_length=1)
    cls: ClassifiedClass
    segments: list[Segment] = Field(min_length=1)
    text: str | None = None
    note: str | None = None


class SurfaceItem(StrictModel):
    """PII outside the page content: metadata fields, filename, outlines, ..."""

    id: str = Field(min_length=1)
    location: str = Field(description='e.g. "info:/Author", "xmp", "filename", "outline"')
    text: str
    type: PiiType
    entity: str | None = None


class GTDocument(StrictModel):
    schema_id: Literal["rb.gt/0.1"] = Field(GT_SCHEMA, alias="schema")
    doc_id: str = Field(min_length=1)
    source_sha256: Sha256
    gt_version: str
    matter: str | None = None
    pages: list[GTPage]
    mentions: list[GTMention] = []
    entities: list[GTEntity] = []
    relations: list[GTRelation] = []
    classified: list[GTClassified] = []
    surface: list[SurfaceItem] = []
    annotators: list[str] = []
    notes: str | None = None

    # -- convenience lookups (not serialised) ---------------------------------

    def page(self, index: int) -> GTPage:
        for p in self.pages:
            if p.index == index:
                return p
        raise KeyError(index)

    def word_index(self) -> dict[str, tuple[int, GTWord]]:
        return {w.id: (p.index, w) for p in self.pages for w in p.words}

    def region_index(self) -> dict[str, tuple[int, GTRegion]]:
        return {r.id: (p.index, r) for p in self.pages for r in p.regions}
