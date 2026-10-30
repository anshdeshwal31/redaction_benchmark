"""Candidate manifest: how the runner invokes a tool and what the tool exposes (PLAN.md §9)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from ..geometry.frames import Frame
from .common import PiiType, StrictModel

CANDIDATE_SCHEMA = "rb.candidate/0.1"


class ArtifactKind(str, Enum):
    OUTPUT_PDF = "output_pdf"
    TEXT = "text"
    MENTIONS = "mentions"
    ENTITIES = "entities"
    PLAN = "plan"
    PSEUDONYMS = "pseudonyms"
    RUN_INFO = "run_info"
    REDACTED_TEXT = "redacted_text"


ARTIFACT_FILENAMES = {
    ArtifactKind.OUTPUT_PDF: "output.pdf",
    ArtifactKind.TEXT: "text.json",
    ArtifactKind.MENTIONS: "mentions.json",
    ArtifactKind.ENTITIES: "entities.json",
    ArtifactKind.PLAN: "plan.json",
    ArtifactKind.PSEUDONYMS: "pseudonyms.json",
    ArtifactKind.RUN_INFO: "run_info.json",
    ArtifactKind.REDACTED_TEXT: "redacted_text.json",
}
# A candidate's primary output, graded end to end: the output PDF, or the redacted text of a tool that
# emits text instead (eval-0.4, the text-level path). Exactly one of them.
PRIMARY_OUTPUTS = (ArtifactKind.OUTPUT_PDF, ArtifactKind.REDACTED_TEXT)


class InjectionKind(str, Enum):
    TEXT = "text"
    MENTIONS = "mentions"
    ENTITIES = "entities"
    REGIONS = "regions"


class CandidateManifest(StrictModel):
    schema_id: Literal["rb.candidate/0.1"] = Field(CANDIDATE_SCHEMA, alias="schema")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str | None = None
    version_command: str | None = Field(None, description="Shell command printing the version on its first line; "
                                                          "placeholders: {python} {repo} {env:NAME}")
    command: str = Field(
        description="Template, run through the shell; placeholders: {input} {workdir} {mode} {inject_dir} {seed} "
                    "{policy} {matter} {python} {repo} {env:NAME}. Paths and values are quoted by the runner and "
                    "reach the tool unchanged, one argument each; on Windows (cmd.exe) a value containing a double "
                    "quote, % or ! or a line break is refused instead. {env:NAME} is the value of an environment "
                    "variable such as RB_MLREDACT_PYTHON and must be set",
    )
    provides: list[ArtifactKind]
    accepts_injection: list[InjectionKind] = []
    label_map: dict[str, PiiType] = Field({}, description="Candidate label -> benchmark PiiType")
    frame: Frame = Frame.DISPLAY_PT
    dpi: float | None = Field(None, gt=0)
    seed_flag: str | None = None
    timeout_s: int = Field(900, gt=0)
    env: dict[str, str] = {}

    @property
    def primary_output(self) -> ArtifactKind:
        """The output graded end to end: redacted_text for a text-only tool, else output_pdf."""
        return ArtifactKind.REDACTED_TEXT if ArtifactKind.REDACTED_TEXT in self.provides else ArtifactKind.OUTPUT_PDF

    @model_validator(mode="after")
    def _check(self) -> "CandidateManifest":
        if sum(k in self.provides for k in PRIMARY_OUTPUTS) != 1:
            raise ValueError("every candidate must provide exactly one primary output: output_pdf, or "
                             "redacted_text for a tool that emits redacted text")
        if self.frame is Frame.IMAGE_PX and self.dpi is None:
            raise ValueError("frame image_px requires dpi")
        if not (self.version or self.version_command):
            raise ValueError("give a version or a version_command so runs are reproducible")
        return self
