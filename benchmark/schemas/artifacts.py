"""Canonical stage artifacts a candidate *may* emit next to its output PDF (PLAN.md §9).

Every artifact declares its coordinate frame. ``display_pt`` is canonical; ``pdf_user``
and ``image_px`` (with ``dpi``) are accepted and converted by the evaluator. The
output PDF itself is always required and always evaluated, whatever these claim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..geometry.frames import Frame
from .common import Polygon, StrictModel

TEXT_SCHEMA = "rb.text/0.1"
MENTIONS_SCHEMA = "rb.mentions/0.1"
ENTITIES_SCHEMA = "rb.entities/0.1"
PLAN_SCHEMA = "rb.plan/0.1"
PSEUDONYMS_SCHEMA = "rb.pseudonyms/0.1"
RUNINFO_SCHEMA = "rb.runinfo/0.1"


class _Framed(StrictModel):
    frame: Frame = Frame.DISPLAY_PT
    dpi: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _dpi_for_pixels(self):
        if self.frame is Frame.IMAGE_PX and self.dpi is None:
            raise ValueError("frame image_px requires dpi")
        return self


# -- [T] text acquisition ------------------------------------------------------


class TextToken(StrictModel):
    id: str = Field(min_length=1)
    text: str
    poly: Polygon
    conf: float | None = Field(None, ge=0, le=1)
    line: str | None = None
    order: int | None = Field(None, ge=0)


class TextPage(StrictModel):
    index: int = Field(ge=0)
    tokens: list[TextToken] = []
    detected_rotation: Literal[0, 90, 180, 270] | None = None
    skew_deg: float | None = None


class TextArtifact(_Framed):
    schema_id: Literal["rb.text/0.1"] = Field(TEXT_SCHEMA, alias="schema")
    source: Literal["ocr", "text_layer", "hybrid"]
    pages: list[TextPage]


# -- [D] detection -------------------------------------------------------------


class RegionAnchor(StrictModel):
    page: int = Field(ge=0)
    poly: Polygon


class TextSpanAnchor(StrictModel):
    page: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class PredMention(StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(description="The candidate's own label; mapped via the manifest's label_map")
    text: str | None = None
    tokens: list[str] | None = Field(None, description="Token ids from text.json")
    regions: list[RegionAnchor] | None = None
    span: TextSpanAnchor | None = Field(None, description="Offsets into page_texts[page]")
    conf: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def _anchored(self) -> "PredMention":
        if not (self.tokens or self.regions or self.span):
            raise ValueError(f"mention {self.id} needs tokens, regions or span")
        return self


class MentionsArtifact(_Framed):
    schema_id: Literal["rb.mentions/0.1"] = Field(MENTIONS_SCHEMA, alias="schema")
    page_texts: dict[int, str] | None = None
    mentions: list[PredMention]


# -- [R] resolution ------------------------------------------------------------


class Cluster(StrictModel):
    id: str = Field(min_length=1)
    mentions: list[str] = Field(min_length=1)
    attributes: dict[str, str] = {}


class EntitiesArtifact(StrictModel):
    schema_id: Literal["rb.entities/0.1"] = Field(ENTITIES_SCHEMA, alias="schema")
    clusters: list[Cluster]


# -- [L] localization / plan ---------------------------------------------------


class PlanRegion(StrictModel):
    id: str = Field(min_length=1)
    page: int = Field(ge=0)
    poly: Polygon
    action: Literal["remove", "replace"]
    replacement: str | None = None
    mentions: list[str] = []


class PlanArtifact(_Framed):
    schema_id: Literal["rb.plan/0.1"] = Field(PLAN_SCHEMA, alias="schema")
    regions: list[PlanRegion]


# -- [P] replacement -----------------------------------------------------------


class PseudonymEntry(StrictModel):
    cluster: str | None = None
    original: str
    replacement: str


class MentionReplacement(StrictModel):
    mention: str
    replacement: str


class PseudonymsArtifact(StrictModel):
    schema_id: Literal["rb.pseudonyms/0.1"] = Field(PSEUDONYMS_SCHEMA, alias="schema")
    entries: list[PseudonymEntry] = []
    mentions: list[MentionReplacement] = []


# -- run information -----------------------------------------------------------


class ModelInfo(StrictModel):
    name: str
    version: str | None = None
    extra: dict[str, str] = {}


class ReleaseInfo(StrictModel):
    """The tool's own safety verdict. Tools always hand over their output; this records whether
    they would have released it, so the benchmark can grade the safety check itself (false
    alarms, unsafe releases)."""

    released: bool
    checks_failed: list[str] = Field([], description="The tool's own reason codes for holding the document back")


class RunInfo(StrictModel):
    schema_id: Literal["rb.runinfo/0.1"] = Field(RUNINFO_SCHEMA, alias="schema")
    candidate: str
    candidate_version: str | None = None
    config: dict = {}
    seed: int | None = None
    models: list[ModelInfo] = []
    started_at: str | None = None
    finished_at: str | None = None
    notes: str | None = None
    release: ReleaseInfo | None = None
