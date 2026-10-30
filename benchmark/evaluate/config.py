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

    # eval-0.3: NOT_REPLACED verdict, stand-in detection for regions, searchable-text and
    # collateral pass marks, page size changes no longer an integrity problem.
    # eval-0.3.1 (2026-10-22): the pen strokes of a drawn signature or mark that are still in the file,
    # under a box or a stand-in, are recoverable like glyph paths. A stroke survives only as the very
    # same path; form rules and the glyphs of kept words drawn as outlines whose whole box lies inside
    # the word are not counted (a typed "____" line in the text layer has no such glyphs). Results
    # labelled eval-0.3 either ignored drawn strokes (scored before the check was added, early on
    # 2026-10-22) or matched them by bounding box alone; rescore them.
    # eval-0.4 (2026-10-30; docs/metrics.md): band evidence passes a character identity gate (edge and
    # run clauses, near-black ink variant) in place and in image layers resampled to the render grid;
    # kept and leaked text is looked for at its displayed place; words are found MOVED, displaced
    # originals and identifying slices are leaks, and words re-typeset in a substitute font are looked
    # for (the restyled search); shared roles the policy enables are excused, the state with its
    # postcode prefix only as a checked pair; addresses are judged as units (E1 replacement per part,
    # E3 token survival, E4 short words). Results labelled eval-0.3, eval-0.3.1 or eval-0.4-dev must be
    # rescored; no re-run is needed except where a mock's default changed (the ceiling,
    # reference-replace, now draws its fakes at the original size).
    version: str = "eval-0.4"
    dpi: float = Field(200, ge=150, le=600, description="Rendering resolution for pixel evidence. Calibrated "
                                                          "for 150-300 DPI; below ~120 DPI the character bands of "
                                                          "8 pt text are too coarse and produce false leaks")
    renderers: tuple[Literal["mupdf", "pdfium"], ...] = ("mupdf", "pdfium")

    # -- per-character-cell pattern survival (L1 / L3) --------------------------
    bands: int = Field(4, ge=1, le=8, description="Sub-cells across the text line per character (eval-0.1 used 3, "
                                                 "which missed boxes offset by half a line on rotated text)")
    cell_min_orig_std: float = Field(8.0, description="Cells with less original contrast carry no information")
    cell_ncc: float = Field(0.6, description="|NCC| at or above which the original pattern is present "
                                              "(calibrated: untouched 100%, replaced glyphs ~1%)")
    cell_amp_visible: float = Field(0.35, description="Output/original contrast ratio needed to be *visible*")
    cell_min_visible_std: float = Field(6.0, description="Absolute output contrast (grey levels) to be visible")
    cell_latent_noise_std: float = Field(1.0, description="Output contrast below this is noise, not latent content")

    # -- character identity (eval-0.4; docs/metrics.md, "Character identity") ----------------------
    identity_subbands: int = Field(16, ge=1, le=64, description="Sub-bands across the line per character for "
                                                               "identity: each band cell's pixels split into "
                                                               "identity_subbands/bands strips (docs/metrics.md)")
    identity_min: float = Field(0.80, description="Identity a character of a MUST or MAY word needs for its band "
                                                 "evidence to stand (tool 2's fakes reach at most 0.636; "
                                                 "docs/metrics.md)")
    identity_min_kept: float = Field(0.70, description="Identity a character of a kept word needs (untouched kept "
                                                      "words score at least 0.821; docs/metrics.md)")
    identity_digit_extra: float = Field(0.05, description="Added to every identity threshold for a digit: tabular "
                                                         "look-alikes 3/8, 5/6, 6/8 (docs/metrics.md)")
    identity_offset_px: int = Field(1, ge=0, le=3, description="Identity is also taken at the word's best "
                                                               "whole-pixel offset within this many pixels "
                                                               "(docs/metrics.md)")
    identity_box_pt: float = Field(2.5, description="Opening square (pt) that finds box pixels, left out in one "
                                                    "identity variant (docs/metrics.md)")
    identity_blur_px: tuple[float, ...] = Field((0.7, 1.4), description="Gaussian sigmas (px at the evaluation "
                                                                        "DPI) of the blurred originals identity "
                                                                        "also compares with; the output is never "
                                                                        "blurred (docs/metrics.md)")
    identity_ink_level: float = Field(0.3, description="Output pixels darker than this times the paper level are "
                                                       "near-black ink (a fake drawn over the original), left out "
                                                       "in one identity variant and in the second band test of MUST "
                                                       "and MAY characters (veils 0.3/0.6/0.9: 18/18/17 of 20 "
                                                       "caught; docs/metrics.md)")
    identity_ink_dilate_px: int = Field(1, ge=0, le=3, description="Dilation (px) of the near-black ink mask: a "
                                                                   "fake's anti-aliased fringe is ink too "
                                                                   "(docs/metrics.md)")
    identity_edge_min: float = Field(0.92, description="Edge clause, MUST and MAY characters: identity over two or "
                                                       "more flagged bands touching the top or bottom band (a box "
                                                       "shifted half a line); 0 of 724 same-font fake characters "
                                                       "pass (docs/metrics.md)")
    identity_run_min: float = Field(0.94, description="Run clause, MUST and MAY characters: identity of one flagged "
                                                      "band pooled along a run of characters (docs/metrics.md)")
    overprint_presence_min: float = Field(0.85, description="Presence (share of the original's ink still in the "
                                                            "output) that keeps a character's evidence when it "
                                                            "lies in a run of overprint_run such characters: an "
                                                            "original under a fake drawn over it (docs/metrics.md)")
    overprint_run: int = Field(3, ge=1, description="Run length (the whole word if shorter) for the presence "
                                                    "clause (docs/metrics.md)")

    # -- mention decisions --------------------------------------------------------
    safe_max_residual: float = Field(0.02, description="SAFE when at most this fraction of cells survives")
    leak_min_residual: float = Field(0.90, description="LEAK_VISIBLE when at least this fraction survives")

    region_replaced_min_std: float = Field(12.0, description="A hidden region counts as replaced by a stand-in "
                                                            "(signature, avatar, silhouette) when the output "
                                                            "inside it varies at least this much (grey levels); "
                                                            "boxes and blank paper are near-uniform")

    # -- words (collateral / protected damage) -------------------------------------
    word_preserved_fraction: float = Field(0.8, description="Word preserved if this fraction of cells is visible")
    word_removed_fraction: float = Field(0.05, description="Word removed if at most this fraction is visible")
    collateral_fail_rate: float = Field(0.01, description="A document fails when more than this share of its "
                                                          "ordinary words (2+ letters or digits) is damaged or "
                                                          "removed (client Q16: only identifying words may change)")
    collateral_min_chars: int = Field(2, description="Ordinary words with fewer letters/digits (punctuation, "
                                                    "single characters) are reported but never fail a document")

    # -- searchable text (client Q11) --------------------------------------------------
    searchable_min_rate: float = Field(0.95, description="Share of kept words (preserved ordinary and protected "
                                                        "words with letters or digits) that must be findable in "
                                                        "the output's text layer at their place on the page")
    searchable_band: float = Field(0.5, description="A word's text must lie within this many line heights of "
                                                    "its displayed place across the line (the next line, 1.32 "
                                                    "heights away, never counts; docs/metrics.md)")
    searchable_margin: float = Field(0.25, description="...and within half the word's length plus this many line "
                                                       "heights along it, about one word space (a layer drifting "
                                                       "up to 4 pt along the line stays searchable; "
                                                       "docs/metrics.md)")
    text_place_leak: bool = Field(True, description="A MUST or MAY word's text found at its displayed place with "
                                                   "the searchable tolerance is a text-layer leak, besides the "
                                                   "1 pt zone of its GT box (docs/metrics.md)")

    # -- moved words and displaced originals (eval-0.4; docs/metrics.md, "Moved words") ----------
    moved_match: float = Field(0.83, description="Template score a word's original image needs where it is found "
                                                 "away from its place (true moves score at least 0.845, different "
                                                 "words at most 0.814; docs/metrics.md)")
    moved_anchor: float = Field(0.85, description="Template score within 2 px of its place that makes a word found "
                                                  "in place an anchor: it bounds its neighbours' search and claims "
                                                  "its ink")
    moved_across: float = Field(0.5, description="Search window across the line, in word heights")
    moved_along: float = Field(8.0, description="Search window along the line, in word heights (clipped to the "
                                                "anchors on the word's GT line)")
    moved_gap: float = Field(0.15, description="Anchors bound the window this many word heights beyond their box")
    moved_scale_min: float = Field(0.86, description="Smallest size of a moved word relative to the original")
    moved_scale_max: float = Field(1.18, description="Largest size of a moved word relative to the original")
    moved_scale_step: float = Field(0.02, description="Scale step of the search")
    moved_overlap: float = Field(0.25, description="A found box may overlap an anchor or a taken box by at most "
                                                   "this share of its area")
    moved_lambda: float = Field(0.02, description="Assignment order: score minus this times the displacement in "
                                                  "word heights")
    moved_candidates: int = Field(4, ge=1, description="Positions kept per scale and per word after suppression")
    moved_floor: float = Field(0.5, description="Positions scoring less are not candidates")
    moved_nms: float = Field(0.5, description="Overlap at which two positions are the same one")
    moved_margin_along: float = Field(0.15, description="The template is the GT box grown this many word heights "
                                                        "along the line...")
    moved_margin_across: float = Field(0.10, description="...and this many across it")
    moved_zero_px: float = Field(1.5, description="A match at scale 1 within this many pixels of the GT place must "
                                                  "pass the character identity gate there (an altered word in "
                                                  "place is not moved; docs/metrics.md)")
    moved_side_ink: float = Field(0.02, description="A kept word is moved only as a whole word: at most this share "
                                                    "of the pixels beside the found box may be ink "
                                                    "(docs/metrics.md)")
    moved_side_strip: tuple[float, float] = Field((0.02, 0.15), description="The strips beside the found box, "
                                                                           "from and to this many word heights "
                                                                           "along the reading direction")
    # words re-typeset in a substitute font (docs/metrics.md, "Moved words")
    restyle_match: float = Field(0.90, description="Template score of a word's GT text typeset in a candidate font "
                                                   "where it is found (the right font scores at least 0.90 for "
                                                   "Times and Helvetica-Bold words; any negative at most 0.827; "
                                                   "docs/metrics.md)")
    restyle_margin: float = Field(0.05, description="...and by how much it must beat the original's own image at "
                                                    "the same transform (the original font is the moved search's)")
    restyle_chars: float = Field(0.8, description="Share of the typeset characters that must correlate with the "
                                                  "output there")
    restyle_fonts: tuple[str, ...] = Field(("embedded", "Helvetica", "Times-Roman", "Courier", "Helvetica-Bold"),
                                           description="Candidate fonts, in order: 'embedded' is the output page's "
                                                       "own embedded fonts (when it has a text layer), the others "
                                                       "base-14 families")

    # -- address units (eval-0.4; docs/metrics.md, "Address units") ------------------------------
    address_new_ink_fraction: float = Field(0.5, description="E1: a street line is replaced when this share of an "
                                                             "identifying word's informative bands carry new ink "
                                                             "(original glyphs left out), or the read-back holds new "
                                                             "text at the word")
    address_adjacent_lines: int = Field(1, ge=0, description="E3: GT lines before and after the unit's lines, in "
                                                             "the same block, that its region also covers")
    address_short_chars: int = Field(2, ge=0, description="E4: a word of an address part with at most this many "
                                                          "letters or digits counts in a channel only when all its "
                                                          "characters do there")

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
