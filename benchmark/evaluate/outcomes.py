"""Result records of an output evaluation (PLAN.md §5.8). All are plain dataclasses so
they serialise to tidy tables for the results store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from .document import DocFinding, SurfaceOutcome


class MentionStatus(str, Enum):
    SAFE_REMOVED = "SAFE_REMOVED"
    SAFE_REPLACED = "SAFE_REPLACED"
    # Hidden but not replaced although the policy requires replacement (client Q7): not a leak,
    # not a pass; reported on its own.
    NOT_REPLACED = "NOT_REPLACED"
    PARTIAL = "PARTIAL"
    LEAK_VISIBLE = "LEAK_VISIBLE"
    LEAK_RECOVERABLE = "LEAK_RECOVERABLE"
    LEAK_LATENT = "LEAK_LATENT"
    NOT_EVALUATED = "NOT_EVALUATED"


SAFE = {MentionStatus.SAFE_REMOVED, MentionStatus.SAFE_REPLACED}
HIDDEN = SAFE | {MentionStatus.NOT_REPLACED}  # original content gone: not a leak
L1_L2_LEAKS = {MentionStatus.PARTIAL, MentionStatus.LEAK_VISIBLE, MentionStatus.LEAK_RECOVERABLE}
# String forms for the results tables (reports read statuses back as text).
SAFE_VALUES = frozenset(s.value for s in SAFE)
HIDDEN_VALUES = frozenset(s.value for s in HIDDEN)


class WordStatus(str, Enum):
    PRESERVED = "PRESERVED"
    DAMAGED = "DAMAGED"
    REMOVED = "REMOVED"
    # Content intact but displaced or re-typeset (eval-0.4): not damage. For a MUST word, a
    # displaced original, which counts as visible for its mention.
    MOVED = "MOVED"
    NOT_EVALUATED = "NOT_EVALUATED"


# Word statuses whose content is present in the output (not collateral, not protected damage).
PRESENT = frozenset({WordStatus.PRESERVED, WordStatus.MOVED})
PRESENT_VALUES = frozenset(s.value for s in PRESENT)


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
    # eval-0.4 (defaults keep older constructors and stored tables readable):
    displaced_residual: float = 0.0  # share of the units seen only as a displaced original
    shared_words: list[str] = field(default_factory=list)  # word ids whose whole word holds a shared role
    shared_chars: list[list] = field(default_factory=list)  # [word id, character index] of shared characters
    shared_visible: float = 0.0  # share of the shared units still visible (reported, never a leak)
    shared_pairs: list[list] = field(default_factory=list)  # [state word id, excused] of each state/postcode pair
    address_unit: str | None = None  # the address unit (street line and locality) the mention belongs to
    locality_shape: str | None = None  # what the read-back of a replaced locality looks like
    review_flags: list[str] = field(default_factory=list)  # review aids (OCR), never a status change


@dataclass
class WordOutcome:
    word_id: str
    page: int
    requirement: str
    visible_fraction: float | None  # for a MOVED word: at its displayed place
    status: WordStatus
    # eval-0.4, all optional:
    in_place_fraction: float | None = None  # visible fraction at the GT place (MOVED words)
    dx_pt: float | None = None  # displacement of a MOVED word (display pt)
    dy_pt: float | None = None
    scale: float | None = None  # size of a MOVED word relative to the original
    match_score: float | None = None  # template score of the MOVED match
    text_offset_pt: float | None = None  # distance of its text-layer occurrence from its displayed place
    restyled: bool | None = None  # found re-typeset in a substitute font


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
    # What the verdicts rest on (eval-0.4): 'pdf', the output PDF's pixels, text layer and file; or 'text', the
    # redacted text of a text-only tool (benchmark.evaluate.textlevel).
    evidence: str = "pdf"

    def mention(self, mention_id: str) -> MentionOutcome:
        return next(m for m in self.mentions if m.mention_id == mention_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["integrity"]["ok"] = self.integrity.ok
        return d
