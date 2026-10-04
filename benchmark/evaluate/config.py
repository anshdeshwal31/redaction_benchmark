"""Evaluation configuration: every threshold in one versioned, hashable place (PLAN.md §14.3).

Defaults are calibrated on the synthetic fixtures (tests/test_evaluator.py) and must be
fixed *before* candidates are run. Changing any value creates a new evaluation version.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "eval-0.1"
    dpi: float = Field(200, ge=150, le=600, description="Rendering resolution for pixel evidence. Calibrated "
                                                          "for 150-300 DPI; below ~120 DPI the character bands of "
                                                          "8 pt text are too coarse and produce false leaks")
    renderers: tuple[Literal["mupdf", "pdfium"], ...] = ("mupdf", "pdfium")

    # -- per-character-cell pattern survival (L1 / L3) --------------------------
    cell_min_orig_std: float = Field(8.0, description="Cells with less original contrast carry no information")
    cell_ncc: float = Field(0.6, description="|NCC| at or above which the original pattern is present "
                                              "(calibrated: untouched 100%, replaced glyphs ~1%)")
    cell_amp_visible: float = Field(0.35, description="Output/original contrast ratio needed to be *visible*")
    cell_min_visible_std: float = Field(6.0, description="Absolute output contrast (grey levels) to be visible")
    cell_latent_noise_std: float = Field(1.0, description="Output contrast below this is noise, not latent content")

    # -- mention decisions --------------------------------------------------------
    safe_max_residual: float = Field(0.02, description="SAFE when at most this fraction of cells survives")
    leak_min_residual: float = Field(0.90, description="LEAK_VISIBLE when at least this fraction survives")

    # -- words (collateral / protected damage) -------------------------------------
    word_preserved_fraction: float = Field(0.8, description="Word preserved if this fraction of cells is visible")
    word_removed_fraction: float = Field(0.05, description="Word removed if at most this fraction is visible")

    # -- file-level survival (L2) ---------------------------------------------------
    layer_ncc: float = Field(0.6, description="Image-layer cell correlates with the original content")
    layer_amp: float = Field(0.25, description="Image-layer contrast ratio for recoverable (else latent)")
    layer_min_inside: float = Field(0.5, description="Fraction of a cell that must fall inside a layer")
    text_ratio: float = Field(90.0, description="rapidfuzz ratio for a text-layer token match")
    vector_survival: float = Field(0.5, description="Fraction of original glyph paths that must survive")

    # -- page-level ----------------------------------------------------------------
    tolerance_pt: float = Field(2.0, description="Dilation of required/optional spans before spill counts")
    change_threshold: int = Field(40, description="Grey-level difference for a changed pixel (after blur)")
    registration_min_shift_px: float = Field(1.5)
    check_revisions: bool = True

    def digest(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
