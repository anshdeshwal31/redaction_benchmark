"""Component metrics (PLAN.md §3, §6, §10): detection, text acquisition, resolution,
localization, plan fidelity, pseudonymisation and determinism."""

from .align import DEFAULT_THETA, InkIndex
from .detection import DetectionResult, MentionMatch, ProjectedMention, evaluate_detection, project_mentions
from .determinism import DeterminismResult, compare_runs
from .localization import FidelityResult, LocalizationResult, TransformDiagnosis, evaluate_plan, plan_fidelity
from .pseudonyms import PseudonymResult, evaluate_pseudonyms, replacements_by_mention
from .resolution import (
    ResolutionResult,
    b_cubed,
    ceaf_e,
    evaluate_resolution,
    lea,
    muc,
    response_from_artifacts,
    response_from_replacements,
)
from .text_acq import TextAcqResult, evaluate_text

__all__ = [name for name in dir() if not name.startswith("_")]
