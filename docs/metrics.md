# Metric definitions (evaluation version eval-0.2)

All thresholds live in `benchmark/evaluate/config.py` (`EvalConfig`). Every result
records the configuration digest; changing a threshold creates a new evaluation version.

## Units

- **GT word**: a word of the visible page, with a polygon in `display_pt`.
- **Character cell**: one character of a GT word (exact box for born-digital words,
  otherwise an equal split of the word polygon along its reading direction).
- **Band**: a quarter of a character cell across the text line (`EvalConfig.bands`, default 4).
- **Unit**: one character (or one slice of a non-text region such as a signature).

## End-to-end verdict per required mention

For each band the original and output renderings (same engine, same DPI) are compared:

- `ncc` = normalised cross-correlation of the two pixel patterns; `amp` = output
  contrast / original contrast.
- A band is **visible** if `|ncc| ≥ 0.6`, `amp ≥ 0.35` and output contrast ≥ 6 grey levels;
  **latent** if correlated but fainter (above noise).
- Bands aggregate to units: a character counts when at least two of its informative bands
  are flagged (its only band, for glyphs such as a dash), or one band together with the same
  band of a neighbour; *and* it belongs to a run of visible characters of at least 3 (words of
  6+ characters), 2 (shorter words) or 1 (initials).

Calibration (`EvalConfig` eval-0.2), synthetic fixtures in six encodings (digital, single
raster, raster skewed 1.2°, MRC, vector outline, content rotated 90°), 72 required mentions
each, MuPDF renderer:

| Situation | Judged not safe, 150 DPI | 200 DPI | 300 DPI |
|---|---|---|---|
| True redaction / rasterised black box | 0 / 72 | 0 / 72 | 0 / 72 |
| Box offset by half a text line (perpendicular to the reading direction) | 72 / 72 | 72 / 72 | 72 / 72 |
| Box covering half of each word | 72 / 72 | 72 / 72 | 72 / 72 |
| Gaussian blur (σ 1.2) | 72 / 72 | 72 / 72 | 72 / 72 |
| Names and contact details correctly pseudonymised (text and rasterised) | 0 / 26 | 0 / 26 | 0 / 26 |

eval-0.1 used three bands and a run of 2 for every word. It missed 17 of 59 half-line
offsets (the uncovered top half of rotated lowercase text sits in half-uncovered bands whose
correlation stays below the threshold), and four bands with the old run rule flagged two
correct replacements of an all-capitals surname (adjacent capitals with similar strokes).
Both were found by running the evaluator self-check on more layouts.

Real documents (machine-drafted GT, so names without titles were not annotated): every
self-check passes on the two born-digital seed documents (TRN 4 pages, PSY 23 pages, 118
required mentions). Under eval-0.1, blur on 9-pt bold running headers was judged `PARTIAL`;
under eval-0.2 all 118 blurred mentions are `LEAK_VISIBLE`. The scanned seed documents need GT
first.

The evaluator is valid between 150 and 300 DPI (`EvalConfig.dpi` enforces ≥ 150).

Each required mention then gets exactly one status, checked in this order:

| Status | Rule |
|---|---|
| `LEAK_VISIBLE` | ≥ 90% of units visible in any render configuration (MuPDF, PDFium; with and without annotations if the output has any) |
| `PARTIAL` | > 2% of units visible |
| `NO_OUTPUT` | the candidate produced no output for the document (failed run); counted as unprotected in the strict view, excluded from the completed-only view |
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
| Pseudonyms | Replacement read from the output text layer at each GT name component: survived originals, value and entity consistency, collisions, collision with a real name in the document, near-identical replacements, case style, honorific/gender mismatch. Replaced contact details and identifiers: Australian phone formats (mobile, landline, 13/1300/1800, +61), email syntax, email local part derived from the new name and not the old one, postcode inside the state's Australia Post ranges, Medicare check digit. Image-only outputs need the measurement OCR |
| Determinism | D0 bytes; D1 canonical artifacts (mentions as GT-word sets, clusters as partitions, plan as rasterised masks, pseudonym map, tokens); D2 rendered pages exact / tolerant and change-mask IoU; D3 per-mention verdicts, flip rate and unstable mentions |

## Attribution

For each failing required mention, the first failing stage in causal order:
`TEXT_MISSING`/`TEXT_CORRUPTED` → `DETECTION_MISS`/`DETECTION_ALIAS_PROPAGATION`/
`DETECTION_BOUNDARY`/`DETECTION_TYPE` → `LOCALIZATION_SYSTEMATIC`/`WRONG_PAGE`/
`LOCALIZATION_LOCAL` → `APPLICATION_NOT_APPLIED`/`APPLICATION_OFFSET` →
`REMOVAL_INCOMPLETE`. A failed run gives `CANDIDATE_ERROR` for every required mention.

`INTERACTION` (confirmed): every exposed stage looks correct for the mention, but the
mention is safe when perfect upstream input is injected (an oracle run), or the tool's own
plan covers it and GT regions are applied correctly while its own regions are not.

Tags on otherwise safe mentions: `PSEUDONYM_INCONSISTENT`, `PSEUDONYM_COLLISION`,
`PSEUDONYM_INVALID`, `PSEUDONYM_ATTRIBUTE`, `PSEUDONYM_COMPONENT_SURVIVED`,
`NONDETERMINISTIC`. An inconsistency becomes `RESOLUTION_SPLIT`, and a collision
`RESOLUTION_MERGE`, when the tool's own entity clusters split or merged that entity
(clusters inferred from pseudonyms are not used for this, which would be circular).

`confirmed` attributions rest on exposed artifacts or oracle runs; `inferred` ones on
black-box heuristics (an orphan redaction next to a missed mention, the same entity
protected elsewhere). They are reported separately and never pooled.

## Statistics

Rates pool required mentions; 95% intervals are document-cluster bootstraps (mentions
within a document are correlated). Comparisons use a paired bootstrap over shared
mentions. With few documents the intervals are wide; the report shows n everywhere.
