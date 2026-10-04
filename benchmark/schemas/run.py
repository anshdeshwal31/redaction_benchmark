"""Experiment specs and run manifests (PLAN.md §4, §14.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .candidate import InjectionKind
from .common import StrictModel

EXPERIMENT_SCHEMA = "rb.experiment/0.1"
RUN_SCHEMA = "rb.run/0.1"

Mode = Literal["e2e", "oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions"]

MODE_INJECTS: dict[str, list[InjectionKind]] = {
    "e2e": [],
    "oracle_text": [InjectionKind.TEXT],
    "oracle_mentions": [InjectionKind.TEXT, InjectionKind.MENTIONS],
    "oracle_entities": [InjectionKind.TEXT, InjectionKind.MENTIONS, InjectionKind.ENTITIES],
    "oracle_regions": [InjectionKind.REGIONS],
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


class RunManifest(StrictModel):
    schema_id: Literal["rb.run/0.1"] = Field(RUN_SCHEMA, alias="schema")
    run_id: str
    created: str
    benchmark: dict
    dataset: dict
    candidate: dict
    experiment: ExperimentSpec
    environment: dict
    items: list[RunItem] = []
