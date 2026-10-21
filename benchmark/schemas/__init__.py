"""Versioned schemas (pydantic models; JSON Schema exported to /schemas)."""

from .artifacts import (
    Cluster,
    EntitiesArtifact,
    MentionReplacement,
    MentionsArtifact,
    PlanArtifact,
    PlanRegion,
    PredMention,
    PseudonymEntry,
    PseudonymsArtifact,
    RegionAnchor,
    ReleaseInfo,
    RunInfo,
    TextArtifact,
    TextPage,
    TextSpanAnchor,
    TextToken,
)
from .candidate import ARTIFACT_FILENAMES, ArtifactKind, CandidateManifest, InjectionKind
from .common import (
    Action,
    ClassifiedClass,
    EntityKind,
    Legibility,
    NameComponent,
    PiiType,
    RegionKind,
    Requirement,
    Role,
    WordSource,
)
from .dataset import DatasetManifest, Datasheet, DerivedFrom, DocumentEntry, PageConditions, PageEncoding
from .gt import (
    AttributeValue,
    GTClassified,
    GTDocument,
    GTEntity,
    GTMention,
    GTPage,
    GTRegion,
    GTRelation,
    GTWord,
    Segment,
    SurfaceItem,
)
from .policy import PolicyMatch, PolicyProfile, PolicyRule
from .run import ExperimentSpec, RunItem, RunManifest

# Exported JSON Schemas: file name -> model.
EXPORTED = {
    "gt.schema.json": GTDocument,
    "policy.schema.json": PolicyProfile,
    "dataset.schema.json": DatasetManifest,
    "candidate.schema.json": CandidateManifest,
    "text.schema.json": TextArtifact,
    "mentions.schema.json": MentionsArtifact,
    "entities.schema.json": EntitiesArtifact,
    "plan.schema.json": PlanArtifact,
    "pseudonyms.schema.json": PseudonymsArtifact,
    "run_info.schema.json": RunInfo,
    "experiment.schema.json": ExperimentSpec,
    "run.schema.json": RunManifest,
}

__all__ = [name for name in dir() if not name.startswith("_")]
