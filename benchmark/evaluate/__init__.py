"""Output evaluation: what the candidate actually produced (PLAN.md §5)."""

from .config import EvalConfig
from .evaluator import check_integrity, estimate_shift, evaluate_output
from .outcomes import (
    L1_L2_LEAKS,
    SAFE,
    DocumentEvaluation,
    Integrity,
    MentionOutcome,
    MentionStatus,
    PageOutcome,
    WordOutcome,
    WordStatus,
)

__all__ = [
    "EvalConfig",
    "check_integrity",
    "estimate_shift",
    "evaluate_output",
    "DocumentEvaluation",
    "Integrity",
    "L1_L2_LEAKS",
    "MentionOutcome",
    "MentionStatus",
    "PageOutcome",
    "SAFE",
    "WordOutcome",
    "WordStatus",
]
