"""Result records of an output evaluation (PLAN.md §5.8). All are plain dataclasses so
they serialise to tidy tables for the results store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .document import DocFinding, SurfaceOutcome


class MentionStatus(str, Enum):
    SAFE_REMOVED = "SAFE_REMOVED"
    SAFE_REPLACED = "SAFE_REPLACED"
    PARTIAL = "PARTIAL"
    LEAK_VISIBLE = "LEAK_VISIBLE"
    LEAK_RECOVERABLE = "LEAK_RECOVERABLE"
    LEAK_LATENT = "LEAK_LATENT"
    NOT_EVALUATED = "NOT_EVALUATED"


SAFE = {MentionStatus.SAFE_REMOVED, MentionStatus.SAFE_REPLACED}
L1_L2_LEAKS = {MentionStatus.PARTIAL, MentionStatus.LEAK_VISIBLE, MentionStatus.LEAK_RECOVERABLE}


class WordStatus(str, Enum):
    PRESERVED = "PRESERVED"
    DAMAGED = "DAMAGED"
    REMOVED = "REMOVED"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class MentionOutcome:
    mention_id: str
    type: str
    subtype: str | None
    entity: str | None
    role: str | None
    requirement: str
    pending: str | None
    pages: list[int]
    conditions: list[str]
    cells: int
    visible_residual: float
    worst_config: str | None
    latent_residual: float
    text_residual: float
    image_residual: float
    image_latent_residual: float
    vector_residual: float
    revision_residual: float
    replaced: bool
    l2_sources: list[str]
    status: MentionStatus


@dataclass
class WordOutcome:
    word_id: str
    page: int
    requirement: str
    visible_fraction: float | None
    status: WordStatus


@dataclass
class PageOutcome:
    index: int
    evaluated: bool
    geometry_changed: bool
    registration_shift_px: tuple[float, float] | None
    changed_area_pt2: float
    spill_area_pt2: float
    spill_fraction: float
    orphan_regions: list[tuple[float, float, float, float]]
    blank_output: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Integrity:
    opens: bool
    page_count_equal: bool
    problems: list[str]
    validator_warnings: list[str]

    @property
    def ok(self) -> bool:
        return self.opens and self.page_count_equal and not self.problems


@dataclass
class DocumentEvaluation:
    doc_id: str
    eval_version: str
    eval_digest: str
    policy: str
    source_sha256: str
    output_sha256: str
    render_configs: list[str]
    integrity: Integrity
    mentions: list[MentionOutcome]
    words: list[WordOutcome]
    pages: list[PageOutcome]
    findings: list[DocFinding]
    surfaces: list[SurfaceOutcome]
    revisions_checked: int
    summary: dict

    def mention(self, mention_id: str) -> MentionOutcome:
        return next(m for m in self.mentions if m.mention_id == mention_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["integrity"]["ok"] = self.integrity.ok
        return d
