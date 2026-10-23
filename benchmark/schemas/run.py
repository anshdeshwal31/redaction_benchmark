"""Experiment specs and run manifests (PLAN.md §4, §14.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .candidate import InjectionKind
from .common import Sha256, StrictModel

EXPERIMENT_SCHEMA = "rb.experiment/0.1"
RUN_SCHEMA = "rb.run/0.1"

Mode = Literal["e2e", "oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions"]

MODE_INJECTS: dict[str, list[InjectionKind]] = {
    "e2e": [],
    "oracle_text": [InjectionKind.TEXT],
    "oracle_mentions": [InjectionKind.TEXT, InjectionKind.MENTIONS],
    "oracle_entities": [InjectionKind.TEXT, InjectionKind.MENTIONS, InjectionKind.ENTITIES],
    # Everything upstream is perfect, so only application (and replacement) is tested: the
    # mentions tell the tool what kind of fake each region needs.
    "oracle_regions": [InjectionKind.TEXT, InjectionKind.MENTIONS, InjectionKind.ENTITIES, InjectionKind.REGIONS],
}


class ExperimentSpec(StrictModel):
    schema_id: Literal["rb.experiment/0.1"] = Field(EXPERIMENT_SCHEMA, alias="schema")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    mode: Mode = "e2e"
    repeats: int = Field(1, ge=1, le=50)
    vary_seed: bool = Field(False, description="Pass a different seed to each repeat (seed sensitivity)")
    inject_from: str = Field("gt", description='Source of injected stage inputs: "gt", or "run:<run_id>" to '
                                               "inject another candidate's artifacts (stage swapping)")
    description: str = ""


class RunItem(StrictModel):
    doc_id: str
    repeat: int
    workdir: str
    input_sha256: str
    injected: dict[str, str] = {}  # kind -> sha256
    seed: int | None = None
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float | None = None
    artifacts: dict[str, str] = {}  # kind -> sha256 of the file the candidate wrote
    invalid_artifacts: dict[str, str] = {}  # kind -> validation error
    error: str | None = None


class RunPolicy(StrictModel):
    """The policy file a run was made with. It reaches the tool only through ``{policy}`` in the
    tool's command or through oracle inputs built under it; when it did, scoring warns if it scores
    by another policy."""

    name: str
    version: str
    sha256: Sha256 = Field(description="SHA-256 of the policy file's bytes")
    path: str = Field(description="Absolute path of the policy file when the run was made")
    given_to_tool: bool | None = Field(None, description="Whether the tool received this policy: its command passes "
                                                         "{policy}, or the experiment injected mentions, entities or "
                                                         "regions built from GT under it. Absent in runs recorded "
                                                         "before it was kept; scoring then works it out from the "
                                                         "recorded command and experiment")


class RunManifest(StrictModel):
    schema_id: Literal["rb.run/0.1"] = Field(RUN_SCHEMA, alias="schema")
    run_id: str
    created: str
    benchmark: dict
    dataset: dict
    candidate: dict
    experiment: ExperimentSpec
    environment: dict
    policy: RunPolicy | None = Field(None, description="Absent in runs made before the run's policy was recorded")
    items: list[RunItem] = []
