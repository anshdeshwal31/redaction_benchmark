# Metric definitions (evaluation version eval-0.1)

All thresholds live in `benchmark/evaluate/config.py` (`EvalConfig`). Every result
records the configuration digest; changing a threshold creates a new evaluation version.

## Units

- **GT word**: a word of the visible page, with a polygon in `display_pt`.
- **Character cell**: one character of a GT word (exact box for born-digital words,
  otherwise an equal split of the word polygon along its reading direction).
- **Band**: a third of a character cell across the text line (top, middle, bottom).
- **Unit**: one character (or one slice of a non-text region such as a signature).

## End-to-end verdict per required mention

For each band the original and output renderings (same engine, same DPI) are compared:

- `ncc` = normalised cross-correlation of the two pixel patterns; `amp` = output
  contrast / original contrast.
- A band is **visible** if `|ncc| ≥ 0.6`, `amp ≥ 0.35` and output contrast ≥ 6 grey levels;
  **latent** if correlated but fainter (above noise).
- Bands aggregate to units: a character counts when ≥ 2 of its bands are flagged, or one
  band together with the same band of a neighbour, *and* it is part of a run of ≥ 2 such
  characters (single characters count only in words of ≤ 2 characters, e.g. initials).

The rule was calibrated on synthetic fixtures across five encodings (digital, single
raster, MRC, vector outline, rotated content) at 150 and 200 DPI:

| Situation | Characters judged visible |
|---|---|
| Untouched text | 100% |
| Removed (true redaction or rasterised black box) | 0% |
| Replaced by a different name | ~1% at band level, 0% after the run rule |
| Gaussian blur (σ 1.2) | ≥ 99% (blur is not redaction) |
| Box shifted down half a line | ~34% (tops of letters legible) |

The evaluator is valid between 150 and 300 DPI (`EvalConfig.dpi` enforces ≥ 150); below
~120 DPI, bands of 8 pt text are too coarse and produce false leaks.

Each required mention then gets exactly one status, checked in this order:

| Status | Rule |
|---|---|
| `LEAK_VISIBLE` | ≥ 90% of units visible in any render configuration (MuPDF, PDFium; with and without annotations if the output has any) |
| `PARTIAL` | > 2% of units visible |
| `LEAK_RECOVERABLE` | not visible, but the original survives in the file: text layer at the location (any render mode), an image layer on its own, glyph vector paths, or an earlier incremental revision |
| `LEAK_LATENT` | > 2% of units latent (e.g. translucent box, MRC ghost) |
| `SAFE_REPLACED` | none of the above, and the region carries new text-like ink or new text |
| `SAFE_REMOVED` | none of the above |
| `NOT_EVALUATED` | output unreadable, or the page's displayed aspect ratio changed |

Document-level findings (separate from mentions): distinctive terms of required mentions
found in metadata, XMP, outlines, annotation contents, form values, attachment names,
structure-tree alternate text or unreferenced streams.

**Clean document**: integrity ok, no `PARTIAL`/`LEAK_VISIBLE`/`LEAK_RECOVERABLE`, no
document finding, no protected word damaged, every required mention evaluated.

## Over-redaction

Each GT word is `PRESERVED` (≥ 80% of characters visible), `REMOVED` (≤ 5%) or
`DAMAGED`. Collateral rate = non-preserved ORDINARY words / ORDINARY words. Protected
damage = non-preserved MUST_NOT words (clinical terms, medications, eponyms, citations,
homograph surnames, boilerplate, dates under the current policy). Words a policy marks MAY
(e.g. titles, undecided categories) are never penalised.

**Effective token precision / recall** come from the output alone: a word counts as
redacted when < 50% of its characters remain visible.

## Component metrics

| Stage | Metrics |
|---|---|
| Text acquisition | CER/WER (raw, case-folded, alphanumeric) on GT words matched by ≥ 50% token-area overlap; missed and hallucinated rates; box IoU; ≥ 95% coverage rate; Kendall τ reading order; line splices; orientation accuracy; all repeated on required PII and per PII type; **mention recognisability** = all words of a required mention recognised (ratio ≥ 80) and located |
| Detection | Predictions projected onto GT words (a word is selected when ≥ 50% of its ink is covered). Token P/R/F1 (MAY words neutral); mention recall under strict / exact-boundary / tolerant / partial matching; boundary classes `exact`, `over_benign`, `over_harmful`, `merged`, `split`, `under` (naming the missing name components), `missed`; false positives by protected class; type confusion; entity completeness |
| Resolution | MUC, B³, CEAF-e, LEA, pairwise links, explicit false merges and false splits. Response clusters come from `entities.json`, or are inferred from shared pseudonym values (mentions carrying a given name or surname only) |
| Localization | Ink coverage of each required mention by planned regions; full-coverage rate; spill area outside the tolerance zone (2 pt); collateral words; median signed edge offsets; per-page **transform diagnosis** by per-region vote among: aligned, oversized, DPI scale (72/96/150/200/300 ratios), y-flip, x-flip, unrotated frame, MediaBox origin, page offset ±1, translation |
| Application | Share of each planned region whose pixels changed in the output (applied ≥ 50%, partial ≥ 10%, else not applied); plan-vs-change IoU per page |
| Pseudonyms | Replacement read from the output text layer at each GT name component: survived originals, value and entity consistency, collisions, collision with a real name in the document, near-identical replacements, case style, honorific/gender mismatch. Image-only outputs need the measurement OCR |
| Determinism | D0 bytes; D1 canonical artifacts (mentions as GT-word sets, clusters as partitions, plan as rasterised masks, pseudonym map, tokens); D2 rendered pages exact / tolerant and change-mask IoU; D3 per-mention verdicts, flip rate and unstable mentions |

## Attribution

For each failing required mention, the first failing stage in causal order:
`TEXT_MISSING`/`TEXT_CORRUPTED` → `DETECTION_MISS`/`DETECTION_ALIAS_PROPAGATION`/
`DETECTION_BOUNDARY`/`DETECTION_TYPE` → `LOCALIZATION_SYSTEMATIC`/`WRONG_PAGE`/
`LOCALIZATION_LOCAL` → `APPLICATION_NOT_APPLIED`/`APPLICATION_OFFSET` →
`REMOVAL_INCOMPLETE`. Tags: `PSEUDONYM_*`, `NONDETERMINISTIC`.

`confirmed` attributions rest on exposed artifacts or oracle runs; `inferred` ones on
black-box heuristics (an orphan redaction next to a missed mention, the same entity
protected elsewhere). They are reported separately and never pooled.

## Statistics

Rates pool required mentions; 95% intervals are document-cluster bootstraps (mentions
within a document are correlated). Comparisons use a paired bootstrap over shared
mentions. With few documents the intervals are wide; the report shows n everywhere.
