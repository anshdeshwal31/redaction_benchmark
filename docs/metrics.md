# Metric definitions (evaluation version eval-0.4)

All thresholds live in `benchmark/evaluate/config.py` (`EvalConfig`). Every result
records the configuration digest; changing a threshold creates a new evaluation version.

**Rescore results labelled eval-0.3, eval-0.3.1 or eval-0.4-dev.** eval-0.4 (2026-10-30) changes
what counts as a surviving original, so earlier results are not comparable. Rescoring needs no new
run (`rb score <run_id>`), except for the ceiling mock (`mock-reference-replace`), whose default
changed: it now draws its fakes at the original size, in the same state where the policy shares the
state, so its runs are made again. The board names each run's evaluation version and warns about runs
of another version or other settings, and scoring refuses an oracle run scored under another version
for attribution. eval-0.4 differs from eval-0.3.1 in these parts, each described
below:

- **Character identity** (in place, in every render configuration and earlier revision, and in image
  layers resampled to the render grid): a character's band evidence stands only when the character is
  still the original glyph, so a correct fake typeset in the original's font, size and place is no
  longer a leak.
- **Text at its displayed place**: kept words are looked for in the text layer at their displayed
  place, and so is the original of a required word; address words match only as whole tokens.
- **Moved words and displaced originals**: a kept word found intact away from its place is `MOVED`,
  not damage; a required word found so is a displaced original and counts as visible; an identifying
  slice (an email's local part, an identifier's digits) is looked for on its own; words re-typeset in a
  substitute font are looked for too (the restyled search).
- **Shared words**: the structure a realistic fake may keep, as the policy lists it (policy 1.2.0),
  is left out of every residual; the state and its postcode prefix only as a checked pair.
- **Address units**: each address part is judged with its partner (E1 replacement, E3 token survival
  in the text layer, E4 short words).
- **Text-level evaluation**: a tool that emits redacted text instead of a PDF is graded on that text.

Paragraph reflow (a longer fake wrapping a line's last words onto the next line) is eval-0.4's one
documented limit; the reading-order search that would find such words (F2) follows in eval-0.4.1.
The other limits are listed at the end ("Known limits of eval-0.4").

eval-0.3.1 (2026-10-22) differed from eval-0.3 in one place: the pen strokes of a drawn signature
or mark that are still in the output file make it `LEAK_RECOVERABLE` (see the table below).

## Units

- **GT word**: a word of the visible page, with a polygon in `display_pt`.
- **Character cell**: one character of a GT word (exact box for born-digital words,
  otherwise an equal split of the word polygon along its reading direction).
- **Band**: a quarter of a character cell across the text line (`EvalConfig.bands`, default 4).
- **Identity sub-band**: a sixteenth of a character cell across the line
  (`identity_subbands` 16, four per band), used only for character identity.
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

### Character identity

The band test alone can be fooled by a different glyph whose strokes line up with the original's in
two bands: under eval-0.3.1 a correct fake 'SA' typeset over 'QLD' counted the top bands of Q and L.
Since eval-0.4 the visible and latent bands of a word character stand only when the character as a
whole is still the original glyph.

- **Identity** of a character: the contrast-weighted correlation between original and output over
  its identity sub-bands (those with original contrast of at least `cell_min_orig_std`; a sub-band
  the output leaves uniform drops out), the largest absolute value over 18 variants: 2 offsets (none,
  and the word's best whole-pixel offset within `identity_offset_px` 1) × 3 pixel sets (all pixels;
  box pixels left out: output darker than half the paper level, opened with an `identity_box_pt`
  2.5 pt square; near-black ink left out: output darker than `identity_ink_level` 0.3 × the paper
  level, dilated by `identity_ink_dilate_px` 1) × 3 originals (as rendered, and blurred by
  `identity_blur_px` 0.7 and 1.4 px; the output is never blurred).
- **Presence**: the share of the original's ink still in the output, over the character's bands with
  output contrast.
- A character's evidence stands when (a) its identity reaches `identity_min` 0.80 (characters of
  MUST and MAY words) or `identity_min_kept` 0.70 (other words), plus `identity_digit_extra` 0.05
  for a digit (tabular look-alikes 3/8, 5/6, 6/8); or (b) it lies in a run of `overprint_run` 3
  characters of its word (the whole word if shorter) whose presence reaches
  `overprint_presence_min` 0.85: an original under a fake drawn over it. Characters of MUST and MAY
  words also stand through two clauses, both with the digit extra: (c) the **edge clause**, two or
  more flagged bands, contiguous and touching the top or bottom band, whose pooled identity reaches
  `identity_edge_min` 0.92 (a box shifted half a line leaves the original's top or bottom intact);
  (d) the **run clause**, a flagged band shared by a run of at least two consecutive characters (as
  many as the run rule asks for the word) whose identity pooled over that band along the run reaches
  `identity_run_min` 0.94. A run holds only characters of shared structure ("Shared words") or only
  identifying ones: a kept postcode prefix, the original's own glyph, would otherwise carry a fake
  digit drawn beside it (measured on the MRC fixture). Bands the gate drops count as new ink.
- **The ink variant and the second band test.** A fake typeset over a translucent white box (a veil)
  over the original covers part of it with the fake's own black ink, which spoils the comparison. For
  characters of MUST and MAY words the band test therefore looks again at the bands it did not flag,
  with the output's near-black ink left out (when at least half the band remains), and identity has
  the matching pixel set. Without them most veiled originals read safe (the 22 October plan caught 4,
  2 and 2 of 20 at opacities 0.3, 0.6 and 0.9).
- **Where it applies**: every render configuration, every earlier revision, and image layers, each
  layer resampled bilinearly onto the render grid over the words it flags and gated the same way
  (sampled at the nearest pixel, a 72-85 DPI layer came out blocky and its recoverable originals
  failed the gate). Signature and
  mark slices are not gated.

Calibration of eval-0.4 (fictional synthetic fixtures in Helvetica, 150-300 DPI, MuPDF and PDFium;
`tests/test_calibration.py`, `tests/test_identity_gate.py`):

| Case | eval-0.3.1 | eval-0.4 |
|---|---|---|
| Every MUST word replaced at its place by a same-font fake (P1: vector, 150-DPI and 300-DPI JPEG; policy 1.2.0) | 12-13 of 20 safe | 20 of 20 safe, no image-layer evidence |
| White box shifted ±0.5 line with the fake inside (X1 300-DPI raster, V1 vector and raster) | | 18-20 of 20 not safe |
| Fake over a white veil of opacity 0.3, 0.6, 0.9 (X2) | | 20, 20 and 19 of 20 not safe |
| White box shifted 0.3 or 0.4 line with the fake inside | 14 of 20 safe | 14 of 20 safe (known limit) |
| The original re-typeset in Times or Helvetica-Bold in a 300-DPI raster (X3) | | 20 of 20 not safe; Courier 16 or more |
| Kept words of a document re-typeset in another font (W4), collateral words | 43 | 0 |
| Tool 2's fresh synthetic run (`t2-e2e-synth-lx`), MUST mentions not safe on its 7 readable documents | 25 | 0 (policy 1.2.0) |

### Earlier calibration and the seed documents

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

Real documents. The self-check has been run on all five seed documents against the AI
annotated GT (`gt_version 0.1-ai`, not yet human-reviewed), **under the draft policy
(`au-medicolegal-v0`) and eval-0.2**, before the client's policy existed; it has not been re-run
under the client's policy (1.1.0 to 1.2.0), GT 0.2-ai, eval-0.3.1 or eval-0.4, which require far more
mentions (TRN 91, DPI 100, IME 230, PSY 275, NOC 497 under 1.1.0) and add the `replace-all` and
`remove-only` checks and, in eval-0.4, ten checks of fakes drawn the way tools draw them
(`replace-in-place`, `retypeset`, `retypeset-keep`, `overprint`, `overprint-veil`, `shift-white`,
`text-next-line`, `whole-address`, `keeps-locality-moved`, `erases-locality`). The
earlier run on machine-drafted GT for the two born-digital documents is superseded. Under
eval-0.1, blur on 9-pt bold running headers was judged `PARTIAL`; under eval-0.2 all blurred
mentions are `LEAK_VISIBLE`.

| Document | Encoding | Required mentions (draft policy) | Result (eval-0.2) |
|---|---|---|---|
| TRN | born-digital, text layer | 4 | 10 pass, 1 fail |
| PSY | born-digital, text layer | 113 | 11 pass, 0 fail |
| DPI | vector outlines, no text layer | 33 | 10 pass, 0 fail |
| IME | 200-DPI colour scan | 59 | 10 pass, 0 fail |
| NOC | MRC scan with forms and rotated pages | 303 | 8 pass, 2 fail |

Open calibration issues found by these runs. The GT was not changed to make any of them pass.

- **NOC `reference-true` and `reference-raster`, protected damage.** Three words of grey
  `DD / MM / YYYY` form placeholders (ink at about 241–247 of 255) are counted as damaged
  because the reference mock re-encodes the page image, which washes out near-white ink. The
  damage measure needs a minimum-contrast floor.
- **NOC `reference-raster`, latent leak.** The mentions "Helen" and "Joanne" on a rotated
  statement page are judged `LEAK_LATENT`. The latent check picks up the unredacted ruled-table
  border 1–2 pt above the line. It needs a margin or rule suppression.
- **TRN `partial-half`.** 3 of 4 required mentions are not safe, below the 80% gate. The one
  safe mention is a name nested in an email address, and the partial mock happens to black out
  exactly that half of the word. The verdict is correct; the check should be measured per word,
  or skip mentions whose character range is about half the word.
The fourth issue of eval-0.3 and 0.3.1, correct fakes in the original's font, size and place judged
`PARTIAL` ("SA" written over "QLD"), is closed by character identity; what remains of it is listed
under "Known limits of eval-0.4". The three issues above are still open in eval-0.4; changing the
evaluator for any of them needs a new evaluation version and a rescore. The self-check has not yet
been re-run on the seed documents under eval-0.4 and policy 1.2.0; it must be before any real
scoring.

The evaluator is valid between 150 and 300 DPI (`EvalConfig.dpi` enforces ≥ 150).

### Statuses

Each required mention then gets exactly one status. `NO_OUTPUT` is assigned when scoring, to
every required mention of a document whose run produced no output; the evaluator never sees
it. For the other documents the evaluator checks in this order:

| Status | Rule |
|---|---|
| `NOT_EVALUATED` | output unreadable, the page's displayed aspect ratio changed, or nothing in the mention can be measured (text-level path: a word of the mention is missing from the tool's original text) |
| `LEAK_VISIBLE` | ≥ 90% of units visible in any render configuration (MuPDF, PDFium; with and without annotations if the output has any). A unit is visible in place, or as part of a displaced original (below); units of shared structure the policy excuses are left out of the count, as they are for every status |
| `PARTIAL` | > 2% of units visible, or, in an address part, one identifying word visible whole (E4) |
| `LEAK_RECOVERABLE` | not visible, but the original survives in the file: text layer at the word (its 1 pt zone or its displayed place, any render mode; for an address part a whole identifying word, and E3's token survival), an image layer on its own, glyph vector paths, the pen strokes of a drawn signature or mark, or an earlier incremental revision. A pen stroke survives only when the output still has the very same path (the same items through the same points), so a new stand-in drawn to the original's size is not taken for it; straight rules and boxes in the region (the form) and the glyphs of kept words drawn as outlines whose whole box lies inside the word (a "Signed:" label the signature crosses) are not counted. A kept word is drawn as outlines when it has letters or digits and the source page has at least half as many paths wholly inside its box (1 pt around) as it has; a typed `______` or `.....` line has none, so pen strokes across it, or wholly inside its box, count. Not covered: a mark drawn only with horizontal and vertical lines or rectangles, which counts as form, and other kept curved art inside a region's box, which counts as its strokes |
| `LEAK_LATENT` | > 2% of units latent (e.g. translucent box, MRC ghost) |
| `SAFE_REPLACED` | none of the above, and the region carries new text-like ink or new text (an address part: E1, "Address units"); for a region-only mention (signature, mark, avatar, photo), the region holds new picture content: grey-level spread ≥ 12 inside it, eroded by 2 px, where a box or blank paper is near-uniform |
| `NOT_REPLACED` | none of the above, and the policy requires a replacement (`action: replace`): hidden, so not a leak, but not a pass either (client Q7) |
| `SAFE_REMOVED` | none of the above, and the policy allows removal |
| `NO_OUTPUT` | (assigned by scoring) the candidate produced no output for the document (failed run), or its primary output (`output.pdf`, or `redacted_text.json` for a text-only tool) was written or changed after the run recorded it (a tool still writing after its timeout); counted as unprotected in the strict view, excluded from the completed-only view |

Document-level findings (separate from mentions): distinctive terms of required mentions
found in metadata, XMP, outlines, annotation contents, form values, attachment names,
structure-tree alternate text or unreferenced streams.

**Leak rate** counts required mentions that are not hidden (`NOT_REPLACED` is hidden).
**Failure rate** also counts `NOT_REPLACED`: it is the rate the client's bar is measured by.

### Documents and runs

**Clean document**: integrity ok, no `PARTIAL`/`LEAK_VISIBLE`/`LEAK_RECOVERABLE`, no
`NOT_REPLACED`, no document finding, no protected word damaged, every required mention
evaluated, and both pass marks below met.

- **Collateral pass mark** (client Q16): at most 1% of the document's ordinary words with
  two or more letters or digits damaged or removed (`DAMAGED` or `REMOVED`; a `MOVED` word is intact
  and never counts). Punctuation-only words are reported but never fail a document.
- **Searchable text** (client Q11): at least 95% of the kept words (`PRESERVED` and `MOVED` ordinary
  and protected words with a letter or digit) are found in the output's text layer at their displayed
  place (substring or fuzzy ratio ≥ 90): within `searchable_band` 0.5 line heights across the line
  and half the word's length plus `searchable_margin` 0.25 line heights along it, after the page's
  registration shift and any change of page size; a `MOVED` word is looked for at the place it was
  found (owner, 2026-10-29). An image-only output fails, and so does a scanned source returned
  without an OCR text layer. `searchable_preserved_rate` and `searchable_moved_rate` report the two
  kinds apart; a word found only further along on its own line is counted as misplaced, else missing.
  The text layer's alignment (`text_offset_*`, the width ratio) and its word breaks (glued and
  out-of-order pairs of neighbouring words) are reported and never fail a document. When nothing is
  counted the rate is n/a (`searchable_rate` null), and the mark is met.

A changed page size is a warning, not an integrity problem (client Q11); the page is rescaled,
and a changed aspect ratio still makes its pixel evidence unreliable. A changed page count
remains an integrity problem, because pages could no longer be matched.

**The tool's own safety check.** A tool reports in `run_info.json` whether it would have
released the document. Scoring compares that with the clean-document verdict: correct
release, correct hold, false alarm (held back but clean) or unsafe release (released but not
clean).

**Same fake across the matter** (client Q8): for every person replaced in two or more
documents of one matter, each name component must get one replacement value in all of them
(`matters` table, repeat 0). Documents are grouped by the GT's matter; the tool is given the
dataset's (`{matter}`), and a document where the two differ, or only one of them has a matter, is
listed (`matter_mismatch` in `scoring.json`, and the run report). A document whose GT has no matter
is left out of this check.

**Departures from the client's answers.** A policy can depart from the client's answers on
purpose; each departure is recorded under `deviations` in the policy file (question, the client's
answer, what is applied instead, why, and the effect on scores). Scoring copies the policy id and
its departures into `scoring.json`, and the run report and the comparison board show them in a
marked section, "Departures from the client's answers", above the scorecard, so a score is never
read as the client's own rules when it is not. Policy 1.2.0, like 1.1.0 and 1.1.1, has two: the whole address
is replaced (Q4; its locality is MUST: the project owner decided on 2026-10-22 that tools 1 and 2
replace whole addresses, the scores hold every tool scored under the policy to that rule, and only
the owner can lift it) and role words are left as they are (Q10). A board shows
the policy each run was scored under; when its runs were scored under different policies it says so,
and each departure names the runs it applies to.

**The run's policy.** `run.json` records the policy a run was started with (name, version, SHA-256
of the file, rules digest) and whether it reached the tool (`given_to_tool`): it does when the
tool's command passes `{policy}`, or when an oracle experiment injects mentions, entities or regions
built from GT under it. `scoring.json` copies it as `policy_given_to_tool`; for a run recorded
before `given_to_tool` was kept, or before the run's policy was recorded at all, it is worked out
the same way from the recorded command and experiment. `rb score` uses that policy unless
`--policy` names another (or the file is gone, then the client's policy). When the tool was given
the policy and the scoring policy differs in name, version or file content, `scoring.json`
(`policy_warning`), the run report and the board say plainly that the tool was given one policy and
is scored against another. A tool that never receives the policy (no `{policy}`, no oracle inputs,
like tool 2) follows its own rules whatever policy the run was started with: it gets no warning and
can be rescored under any policy without a new run.

**The rules digest.** A policy's rules digest (`rules_sha256` in the run's policy record,
`policy_rules_sha256` for the scoring policy in `scoring.json`) is the SHA-256 of the policy without
its prose: the version, the description, every rule's note, every deviation's client answer,
applied text, reason and effect, and every shared-roles entry's note are left out; the name, the
rules, the defaults, the consistency scope, the deviation ids, the rules each deviation names and the
shared roles stay in. A wording-only release keeps it: 1.1.0 and 1.1.1 have the same digest. 1.2.0
adds rules (the shared roles, and the place-kind and street subtypes in places-kept), so its digest
differs, and a run given 1.1.1 and scored under 1.2.0 is told it is judged against rules it was not
given. When the version or the file differs but both digests are
known and equal, the warning says only the wording changed ("Same rules, wording changed: the tool
was given ... and these scores use ...; the rules digest is the same (...)"), and the board marks
the given policy "same rules"; when the names differ, or a rule changed, the warnings above stand.
For a run recorded before the digest was kept, scoring computes it from the run's policy file when
that file still has the recorded SHA-256 (`run_policy_rules_sha256` in `scoring.json`), and leaves it
unknown otherwise.

## Moved words and displaced originals

A tool that redraws a line holding a fake leaves the line's kept words intact but not at their GT
place: a longer or shorter fake moves the words after it, and the line may be re-typeset a little
larger or smaller. A word the in-place test calls `DAMAGED` or `REMOVED` is looked for near its place
in the output, on the first render configuration of each evaluated page, by its own original image
(`benchmark/evaluate/moved.py`):

- **Template and window.** The original render cropped to the GT box, grown `moved_margin_along`
  0.15 word heights along the line and `moved_margin_across` 0.10 across it, is searched within
  `moved_across` 0.5 word heights across and `moved_along` 8 word heights along the line, clipped to
  the nearest anchors of the word's GT line (words found in place whose whole image is there, a score
  of at least `moved_anchor` 0.85 within 2 px) widened by `moved_gap` 0.15 word heights, so reading
  order is kept. Scales `moved_scale_min` 0.86 to `moved_scale_max` 1.18 in steps of
  `moved_scale_step` 0.02; up to `moved_candidates` 4 positions per scale scoring at least
  `moved_floor` 0.5 after suppression of positions overlapping by `moved_nms` 0.5.
- **A match** scores at least `moved_match` 0.83 (true moves score at least 0.845, different words at
  most 0.814), and the band test at the found transform finds at least `word_preserved_fraction` of the
  word's characters. Matches are assigned one to one per page, greedily by score minus `moved_lambda`
  0.02 × the displacement in word heights, with no found box overlapping an anchor or a taken box by
  more than `moved_overlap` 0.25; identical words of one line keep their order.
- **Two filters.** A match at scale 1 within `moved_zero_px` 1.5 px of the GT place must pass the
  character identity gate there: an altered word in place ('far' typed over 'for') is not a moved
  'for'. A kept word is found only as a whole word: at most `moved_side_ink` 0.02 of the pixels in the
  strips beside the found box (`moved_side_strip`, 0.02 to 0.15 word heights along the line) may be
  ink, so a kept 'on' is not found inside a fake 'Wilson'. The box is also tried half a scale step
  larger and smaller about its centre, since a word re-typeset between two steps of the scale grid
  overhangs the box found at the nearer one.
- **Restyled search.** A word still `DAMAGED` or `REMOVED` is looked for re-typeset in a substitute
  font: its GT text typeset, at the source text layer's size and baseline (else the word height /
  1.374), in the output page's embedded fonts (when it has a text layer; a font is tried only when it
  holds every glyph of the word) and the twelve base-14 text fonts: Helvetica, Times-Roman, Courier,
  Helvetica-Bold and their italic, oblique and bold styles (`restyle_fonts`), searched in the same
  window and scales. The identity gate takes an original drawn in another style for new ink, so a
  style not tried here would read safe where eval-0.3.1 flagged it (with four fonts, 15 to 20 of 20
  Times-Italic, Times-Bold or Courier-Bold originals in a 300-DPI raster did). It is found where the
  typeset text scores at least `restyle_match` 0.90, beats the original's own image at the same
  transform by `restyle_margin` 0.05, and `restyle_chars` 0.8 of its characters correlate with the
  output there. The right font scores at least 0.90 for 139 of 139 Times and 41 of 41 Helvetica-Bold
  words (54 of 82 Courier); the best wrong find scores 0.827.

A kept word found is `MOVED`: its content is intact, so it is neither collateral nor protected damage,
and it is searched for in the text layer at the place it was found. Its word row reports
`in_place_fraction`, `dx_pt`, `dy_pt`, `scale`, `match_score` and `restyled`.

A MUST or MAY word found is a **displaced original**: its characters count as visible for its
mention, its found box's ink is never taken for replacement ink, and the text layer is also searched
at the found place. `displaced_residual` reports the share of a mention's units seen only so. A word
with shared characters (a mobile's '04', a postcode's prefix, an identifier's prefix, an email domain;
"Shared words") is a displaced original only where its glyphs also pass the identity gate: a fake
keeping those characters is not the original moved.

**Identifying slices.** A MUST word with shared characters is also looked for by its identifying
slice alone (an email's local part, an identifier's or a phone number's other digits), so a changed
domain or prefix cannot hide a kept local part or kept digits pushed along the line.

The summary reports `moved_words`, `moved_real_words`, `protected_moved` and
`must_words_displaced`; the board shows "moved kept / displaced required" and the run report the
moved and displaced counts per document. Measured: on tool 2's fresh synthetic run the 21 kept words
of its redrawn lines are `MOVED` on each intact document (collateral 0); every planted kept word of
the redrawn-line cases is `PRESERVED` or `MOVED`, 65 of 65 where moved, within 1 pt; the 20 originals
redrawn beside their fakes (K0c) are 20 of 20 `LEAK_VISIBLE`.

## Text layer at the displayed place

Besides the 1 pt zone of its GT box, a MUST or MAY word's original text found in the output's text
layer at its displayed place (the GT polygon plus the page's registration shift, with the searchable
tolerance above; `text_place_leak`) is a text-layer leak, and so is a displaced original's text at its
found place. Words of addresses and localities match only as whole tokens, a number as a maximal digit
run: '14' is not found in '214' or '142750'; a word of letters also matches the letters of a token
glued to digits ('NSW' in 'NSW2750') and a part of a letter run glued at a change of case ('Example'
and 'Street' in 'ExampleStreet,', as MuPDF extracts words drawn under about 1.5 pt apart), but never
a word inside another ('Park' in 'Parkside') or inside an email or web address. The 1 pt zone
alone is still used for earlier revisions and by the names' pseudonym reader.

## Shared words

The project owner decided on 2026-10-22 that a realistic fake may share a word with the original at
the same place: the state when the fake suburb is in the same state, a street-type word, the 'PO Box'
words, a title ('Dr'). On 2026-10-29 the owner added that a fake organisation may keep its trailing
legal form, a fake may keep an email domain and an identifier's letter prefix, and a fake phone number
its national structure prefix. A policy enables these roles per mention type in `shared_roles`
(policy 1.2.0 enables all of them; the draft policy none). Roles are read from GT alone, by the word's
place in the mention (`benchmark/gt/shared_roles.py`, the address role reader
`address_split.address_roles`), never from a tool's output or report:

- ADDRESS street line: `box_kind` (PO Box, GPO Box, Locked Bag, Private Bag, RMB, ... before their
  number), `dx` ('DX'), `designator` (unit, level and building designators before a number or letter),
  `corner` (Cnr, Crn, '&', and 'Corner', 'of', 'and' in a corner form), `via`, `capital_city_phrase`
  ('IN YOUR CAPITAL CITY'), `country` (last), `street_type` (the street's last word after a name word
  that is neither a number nor 'The'; never a building word, never in a corner form).
- LOCATION locality: the `state`, in any written form at the end of the address (never 'Victoria' in
  Mount Victoria, Port Victoria or Victoria Point), with the postcode's prefix digits (`pc_prefix`:
  '2' NSW and ACT, '3' VIC, '4' QLD, '5' SA, '6' WA, '7' TAS, '08' NT, matched from the left on the
  original postcode), always as a pair; the `country` ('Australia', or a foreign country, last). A
  glued 'NSW2750' shares its letters and prefix digits. A postcode with no state word shares the prefix
  of the state its range gives as plain characters, with no pair.
- ORGANISATION `legal_form`: the trailing Pty, Ltd, Limited, Proprietary, Inc or Incorporated after a
  name word.
- EMAIL `email_domain`: '@' and the domain after a non-empty local part.
- ID_* `id_prefix`: one to four capitals and their separator when digits follow ('CLM-').
- PHONE `phone_prefix`: a mobile's '04', '+61', the '13' of a 6-digit number, '1300' and '1800', a
  bracketed area code '(0X)'.
- Titles need no role: GT holds them as optional (MAY) words.

**Treatment.** Excused units leave the numerator and the denominator of every residual of their
mention (visible in place or displaced, latent, image layer, earlier revision, text layer, vector
paths), E3's searched words, and detection's and the effective-token metrics' token counts (they are
neutral: neither required nor an extra). A word with shared characters is searched in the text layer
by its identifying slice, with the text's own copy of each shared part cut out first and a slice of
digits matched only as a whole digit run ('04' of '0404' is not found in a fake '0498', nor 'smith' in
a kept '@smithlaw.com.au'). In pixels a template match of the slice is the fake's own shared part only
where the slice's text occurs in the shared part ('smith' in '@smithlaw') and the rest of that shared
part is the original's glyphs around the match at the same transform (identity gate), wherever the
fake put it; with nothing else shared ('04' of '0404' on a kept '04') the match must lie for the most
part on that occurrence's place. A slice that never occurs in its shared part ('48213' in 'CLM-') is
never excused, so kept digits slid onto a dropped prefix, or a kept local part pushed onto the old
domain's place, are found in an image-only output. Shared units are never the replacement: a mention
left with only a domain, a prefix, a state or a street type is `NOT_REPLACED`, also when a line
re-typeset after the deletion slides them (or the line's other kept words, or the locality's fake)
into an erased word's place: their text and ink there are left out of the replacement tests, and new
text is looked for at the identifying words only. The line's kept words are the words no MUST rule
decides and the shared words of any mention (another mention's MUST words must change and never
slide). A word counts as slid only when the output's lines hold it no more often than the input's
lines kept it (a fake 'Will' for 'Jane' on a line keeping 'will' adds one and is a replacement), and
the text at an erased word is slid only when it holds at least one whole kept word: pieces of kept
words alone ('Sign' where 'Signed' is kept) are new text. A mention whose every letter and digit would be
shared is judged in full.

**The pair.** The state and the postcode's prefix digits are excused only together: when the prefix
digits are the original's glyphs where the state word is seen (in place, or at its `MOVED` transform,
refined within half a character along and a quarter height across), judged per character at the MUST
identity threshold with no run rule; or, when the state is not seen in pixels, when the text layer at
its displayed place holds the state followed by a 4-digit token starting with the prefix. A same-state
fake passes; a kept state beside another state's fake counts (a fictional case reads `PARTIAL` 0.2).
A state with no postcode after it cannot be checked and is excused.

**Why it cannot excuse a real survival.** (1) The roles are closed lists read by position in GT: never
a suburb, street name, building word, house, unit, level, box or DX number, a postcode digit after the
prefix, a name word, an email's local part or an identifier's digits; the never-list (Victoria Road,
Mount Victoria, Victoria Point, Queensland Avenue, Tweed Heads South, Box Hill, Old Bells Line of Road,
'Australia' in a building name) is tested on the address case table. (2) Every identifying character
keeps the full test in every channel; its survival alone makes the part `PARTIAL` or worse. (3) Shared
units alone are no replacement. (4) Shared characters still take part in the run rule, so a kept
identifying character next to a kept shared part counts. (5) The pair stops a kept state next to
another state's fake, and the slice search finds kept digits or a kept local part pushed out of place
by a changed prefix or domain.

Each mention reports `shared_words`, `shared_chars` ([word id, index]), `shared_visible` (the share of
its shared units still visible; never a leak) and `shared_pairs` ([state word id, excused]); the run
report has a "Shared structure" table and the board a "shared structure / pairs failed" column.

## Address units

An address unit is a street line (ADDRESS, `address_street_line`) read with its `<id>-loc` locality, a
locality on its own, or an unsplit address. Each part is judged with three rules
(`benchmark/evaluate/addresses.py`):

- **E1, replacement per part.** The unit's read-back is the output's text-layer tokens on the unit's
  lines, between the nearest kept words before and after the unit at their displayed place (else
  `moved_along` word heights). The read-back is cut into street line and locality where the splitter
  cuts an address (after a street type, a comma, a PO box number or a line change, else by the suburb
  rules), so a suburb of several words ('North Rocks NSW 2151') re-typeset onto a deleted street line
  is the locality's, never the street line's new text. A street line is replaced when an identifying word (the part's words
  less its shared ones) has new text in the read-back at its place, or new ink on at least
  `address_new_ink_fraction` 0.5 of its informative bands (visible original bands and displaced
  originals' boxes left out) where that ink looks like text (the edges of a plain box make a word's
  edge bands new ink, and a box is no fake). A locality is replaced, when the read-back has any token, exactly when
  the read-back ends in a locality shape: a suburb word right before the state (or the postcode), and
  a state and a 4-digit postcode where the original had them (any written form of the state; a state
  the pair check kept need not be in the text of a scan). With no token, every identifying word needs
  new ink that looks like text. `locality_shape` reports the parts found ('suburb+state+postcode', ..., 'none', or
  'no_text_at_unit' when judged on pixels).
- **E3, token survival.** An identifying word of the unit survives in the text layer when the
  output's whole tokens equal to it in the unit's region (its GT lines and `address_adjacent_lines` 1
  line before and after in the same block, ±0.5 line height across, the block's width plus 0.25 line
  heights) outnumber the input's ordinary words equal to it there that the output keeps. A survival is
  a text-layer leak (`LEAK_RECOVERABLE` unless pixels show more), whatever its share.
- **E4, short words.** A word of an address part with at most `address_short_chars` 2 letters or
  digits counts in a channel only when all its characters do there ('14' with only its '1' seen counts
  nothing), and a fully visible identifying word makes the part at least `PARTIAL`.

On a page with no text layer the measurement OCR, when installed, may add review flags
(`LEAK_SUSPECTED` for E3, `LOCALITY_SHAPE_UNCONFIRMED` for E1) in `review_flags`; it never changes a
status. The run report has an "Address units" table (both parts side by side, their statuses, what the
text at the unit holds, ids and shapes only).

## Over-redaction

Each GT word is `PRESERVED` (≥ 80% of characters visible in place), `REMOVED` (≤ 5%),
`DAMAGED`, or `MOVED` (found intact away from its place, or re-typeset; "Moved words and displaced
originals"). Collateral rate = `DAMAGED` and `REMOVED` ORDINARY words / ORDINARY words. Protected
damage = `DAMAGED` and `REMOVED` MUST_NOT words (clinical terms, medications, eponyms, citations,
homograph surnames, boilerplate, dates under the current policy). Words a policy marks MAY
(e.g. titles, undecided categories) are never penalised.

**Effective token precision / recall** come from the output alone: a word counts as
redacted when < 50% of its characters remain visible; a `MOVED` word is present, and words of shared
structure excused in their mention are neutral.

## Component metrics

| Stage | Metrics |
|---|---|
| Text acquisition | CER/WER (raw, case-folded, alphanumeric) on GT words matched by ≥ 50% token-area overlap; missed and hallucinated rates; box IoU; ≥ 95% coverage rate; Kendall τ reading order; line splices; orientation accuracy; all repeated on required PII and per PII type; **mention recognisability** = all words of a required mention recognised (ratio ≥ 80) and located |
| Detection | Predictions projected onto GT words (a word is selected when ≥ 50% of its ink is covered). Token P/R/F1 (MAY words neutral); mention recall under strict / exact-boundary / tolerant / partial matching; boundary classes `exact`, `over_benign`, `over_harmful`, `merged`, `split`, `under` (naming the missing name components), `missed`; false positives by protected class; type confusion; entity completeness. A split address (the street line and its locality `<id>-loc`) is one unit when the policy scores both parts: one prediction over the whole address is `exact` for both parts (`over_benign` when it also takes the address's optional words), a locality predicted as ADDRESS has the right type, and the whole-address span counts as an exact prediction for precision. A span that also takes words outside the address is `over_harmful` or `merged` as before. Shared structure the evaluation excused (after the pair check) is neutral: neither required nor an extra for a mention's boundary, and left out of the token counts |
| Resolution | MUC, B³, CEAF-e, LEA, pairwise links, explicit false merges and false splits. Response clusters come from `entities.json`, or are inferred from shared pseudonym values (mentions carrying a given name or surname only). A detected mention whose prediction `entities.json` puts in no cluster is a cluster of its own (two GT mentions matched to the same unclustered prediction share it), so a tool that fails to link its names scores as splitting them rather than being left out; mentions with no prediction are left out |
| Localization | Ink coverage of each required mention by planned regions; full-coverage rate; spill area outside the tolerance zone (2 pt); collateral words; median signed edge offsets; per-page **transform diagnosis** by per-region vote among: aligned, oversized, DPI scale (72/96/150/200/300 ratios), y-flip, x-flip, unrotated frame, MediaBox origin, page offset ±1, translation |
| Application | Share of each planned region whose pixels changed in the output (applied ≥ 50%, partial ≥ 10%, else not applied); plan-vs-change IoU per page |
| Pseudonyms | Replacement read from the output text layer at each GT name component: survived originals, value and entity consistency, collisions (one fake for a name component of two people with different names; one person written two ways, or two people who share a name such as a married couple's surname, may share a fake), collision with a real name in the document, near-identical replacements, case style, honorific/gender mismatch. Replaced contact details and identifiers: Australian phone formats (mobile, landline, 13/1300/1800, +61), email syntax, email local part derived from the new name and not the old one, postcode inside the state's Australia Post ranges (any written form of the state), Medicare check digit. Addresses (E6) are read per address unit: the unit's read-back at its displayed place (between its kept words), cut into street line and locality by the GT role reader. Reported per address: the postcode keeps no original digit after the state's prefix (`postcode_shared_digits`), no fake word repeats an original address word of the document in any role (`fake_reuses_original`; shared structure such as a street type left out), and one address keeps one fake within the document (`address_value_consistency_rate`) and across the matter (`matters` rows with component `address`). An address is known by its identity (number, building, street name, canonical street type, box or DX parts, suburb, state; unit and level beneath), never by its GT entity, and identities and fakes are stored hashed. These checks are reported and never gate a document. Image-only outputs need the measurement OCR |
| Determinism | D0 bytes; D1 canonical artifacts (mentions as GT-word sets, clusters as partitions, plan as rasterised masks, pseudonym map, tokens); D2 rendered pages exact / tolerant and change-mask IoU; D3 per-mention verdicts, flip rate and unstable mentions |

## Text-level evaluation

A tool that emits redacted text instead of a PDF (the client's tool is graded so, on its own
redacted text) declares `provides: [redacted_text]` and writes `redacted_text.json`
(`schemas/redacted_text.schema.json`, `rb.redacted_text/0.1`): `original_text`, the text it redacted
(as it extracted it, or as the benchmark injected it), `redacted_text`, `page_break` (a form feed by
default) and `mappings`, the replacements it says it made (reported, never evidence). It is graded
by `benchmark/evaluate/textlevel.py`; the results carry `evidence` 'text' (the PDF path's is 'pdf'),
the same evaluation version and digest, and the same summary keys:

1. **Alignment.** GT words, in reading order, are aligned to the original text's whitespace tokens,
   case folded and stripped of punctuation (difflib, autojunk off), page by page when both texts hold
   one page break per GT page break, else as one text (with an integrity warning). A GT word with no
   token is `NOT_EVALUATED`, and so is every required mention holding such a word.
2. **Diff.** The original text is diffed against the redacted text in two steps: whitespace tokens
   first, line breaks among them (difflib), then characters (Levenshtein opcodes) only between the
   tokens matched. Each character of an aligned GT word is kept in place or changed. A character diff
   of the whole text could merge a deleted mention with the next line's new text and credit it with
   that fake.
3. **Words.** `PRESERVED`, `DAMAGED` or `REMOVED` by the share of kept characters, with the PDF path's
   fractions. `MOVED` cannot occur.
4. **Mentions.** The PDF path's units, shared roles and cut-offs. A character is visible when it is
   kept and lies in a run of kept characters as long as the run rule asks for its word; E4 holds for
   address parts. The pair holds when the state is changed, or kept with its postcode's prefix digits
   (for a postcode such as NSW 1xxx, with a postcode of that state after it).
   A **displaced original** is an identifying whole token of the mention (a number as a maximal digit
   run; an email's local part also before an '@') that the redacted text holds more often than the
   original text holds it outside the required mentions plus the required words kept in place: its
   characters are visible in every mention holding it, since the text does not say which one moved. A
   mention is **replaced** when the diff puts new text with a letter or digit at it (read as the whole
   tokens of the redacted text over the mention) that is not a placeholder: a bracketed tag ('[REDACTED]', '<PERSON>', '{NAME_1}', '[Claimant]': only words for a PII type, a role or a redaction in any case, or letters in capitals with no digit), never a fake written in brackets ('(Ruth)', '<ruth.dunbar@example.com>', an identifier '(CLM-70596)'); a run of X, asterisks
   or block characters, a tag in capitals naming a PII type or a redaction ('PERSON_2', 'REDACTED'), or
   one text standing for every replaced required mention with different originals; an address's
   locality also needs E1's locality shape on the new text over the unit. Otherwise the status follows
   the PDF path's order. `LEAK_RECOVERABLE` and `LEAK_LATENT` cannot occur: the text is the whole
   output. Signatures, marks and images have no text: their mentions are left out.
5. **Components.** Detection takes the changed spans as predictions: runs of consecutive changed GT
   words on a page, carried through shared words left as they were (so a whole-address fake is one
   prediction, exact for both parts) and ended at a GT line change unless the next word shares a
   mention or an address unit with the last (two mentions with only a line break between them are two
   predictions); a prediction's label comes from the tool's mappings when one's
   original equals the run's text, its type from the label map, else none. The pseudonym checks read
   the new text the diff puts at each GT word (a mention's new text split on white space when it has
   one token per word) and each address unit's new text as its read-back; resolution is inferred from
   the fakes. Collateral, protected damage, the effective-token metrics and the clean-document verdict
   are as for a PDF; searchable text is n/a, since the text is the whole output.

The tool's own extraction decides what can be aligned: on an OCR'd scan its original text may differ
from GT enough to leave mentions `NOT_EVALUATED`, which the strict view (the board's and the report's
strict leak rate) counts against it, like `NO_OUTPUT`; the completed-only view leaves them out. Run the
tool on the benchmark's injected text as well, to tell extraction from redaction. The board and the
report say what each run is graded on (the evidence column), and a text-only tool's failed documents
list the same required mentions as its scored ones (no signatures, marks or images).

## Known limits of eval-0.4

Measured on fictional synthetic fixtures (Helvetica; other fonts and real scans untested) and reported
for review; none lowers a pass mark.

- **Paragraph reflow** (owner, 2026-10-29: eval-0.4's one documented limit). A longer fake that wraps
  a line's last words onto the next line, and any line shift beyond `moved_along` 8 line heights, is
  not found: those kept words read `DAMAGED` or `REMOVED`. The reading-order search that would find
  them (F2) follows in eval-0.4.1; until then tools are asked to keep line breaks
  (`docs/adapter_guide.md`). A fixture with reflowed lines keeps 8 of 54 words as collateral.
- **Glued address words in one case.** The text-layer search splits a glued token at a change of case
  ('ExampleStreet' gives 'Example' and 'Street') or at digits ('NSW2750'). Address words glued
  together in all capitals or all lower case ('EXAMPLESTREET', 'examplestreet') cannot be split, so
  a hidden original written that way in the text layer is not found as a text-layer leak.
- **Moves.** Vertical moves over half a line (an 8 pt drop is found at most 4 of 10 times), scales
  outside 0.86-1.18, and the same word twice on a line with no anchor between (one false `MOVED` in
  the trap case) are outside the search. Words are looked for on the first render configuration only;
  latent displaced originals and displaced originals inside image layers are not looked for, and the
  restyled search skips vertical words. Same-font digit groups can be taken for a neighbouring original by the moved search,
  which is why the ceiling mock draws numbers that keep no shared digit centred at 80% of the line.
  One-letter look-alike garbles of kept words can pass as `MOVED` (9 and 21 words in two fictional
  cases).
- **Substitute fonts** neither embedded in the output nor one of the twelve base-14 text fonts are not
  looked for (an italic or monospace TrueType substitute can read safe where eval-0.3.1 flagged it),
  and Courier substitutes in 300-DPI rasters are caught in 16 or more of 20 mentions, not all.
- **Character identity.** Look-alike digits in protected words (3/8, 5/6, 6/8) can still read as
  intact (pinned by a test). Random same-font fakes can still match in 1-4 of 554 digit words and 3-4
  of 384 name words; the run clause admits a few same-font fake letters (two letters where 'Pty' was
  at 150 DPI, one letter of a JPEG under PDFium): each only a false `PARTIAL`. The self-check is
  pinned at 150 DPI with MuPDF, the configuration the tests run; at 200 DPI under MuPDF one correct
  re-typeset fake name on the rotated-scan fixture reads `PARTIAL` (2 of its 13 characters), a false
  alarm of the same kind (it is safe under PDFium and at 150 DPI). White fake text on a black box can pass the presence clause. A white box
  shifted 0.3-0.4 of a line with the fake inside leaves under half the x-height and reads safe in 14
  of 20 cases, as in eval-0.3.1.
- **Shared words.** A state kept as the original's glyphs beside a same-state postcode drawn at another
  size or place pairs only through the text layer (the state at its displayed place, then a 4-digit
  postcode starting with the prefix digit), since the prefix digit is not the original's glyph where
  the state is seen; so does a state before a postcode whose prefix matches no state digit (NSW 1xxx,
  VIC 8xxx, QLD 9xxx, ACT 02xx), with a postcode of that state after it. On an image-only output
  neither can pair (OCR never decides): the locality reads `PARTIAL` (0.2, about 0.16). A state with no
  postcode after it is excused unchecked. A street type in a corner form is not
  shared, so a correct fake keeping 'Streets' in 'Corner of Smith and King Streets' reads `PARTIAL`.
  'Mount Victoria 2786' shares the postcode's '2' as a plain character (no state word). A kept
  identifying slice whose text and surroundings equal the shared part's ('smith' pushed onto the
  domain of 'smith@smith.com.au' with the domain kept) cannot be told from it in pixels.
- **Re-typeset lines.** A word erased from a line typeset again is `NOT_REPLACED` only when its place
  holds at least one whole kept word: where only pieces of kept words land there (a short erased word),
  they read as new text and the mention as replaced. On an image-only output nothing says which word
  slid there, so a kept word's ink at an erased place reads as replacement ink.
- **Address units.** The rules are checked on the fictional address case table and synthetic
  addresses; no tool output has yet exercised box, DX, corner, multi-line or form addresses. The text
  layer decides E1 and E3: a correct locality drawn only in pixels under a text layer that holds no
  locality reads `NOT_REPLACED`. A kept locality pushed to the next line of an image-only output gets
  an OCR review flag only.
- **Text in the 1 pt zone.** The names' pseudonym reader still reads the 1 pt zone of each GT word,
  not the displayed place.
- **Runtime.** eval-0.4 takes about 2-13 s per two-page synthetic document (eval-0.3.1: 1-5 s); the
  restyled search tries every embedded font with no cap and the twelve base-14 fonts (about twice the
  time of four on an image-only output). Measure on the seed documents before real scoring.
- **Memory.** Every page's cells stay in memory until the document's moved, restyled and slice
  searches are done. Their pixel indices are kept in 32 bits, the originals' blurred renders for the
  last few pages only, and a page's other renders are dropped once it is evaluated. Peak memory on a
  fictional dense document (about 415 words a page): 575 MB for 3 pages and 1.13 GB for 12, about
  62 MB a page (eval-0.3.1: 373 MB and 779 MB, about 45 MB a page), so roughly 3 GB for 40 pages. Score
  long documents one at a time.

## Attribution

For each failing required mention, the first failing stage in causal order:
`TEXT_MISSING`/`TEXT_CORRUPTED` → `DETECTION_MISS`/`DETECTION_ALIAS_PROPAGATION`/
`DETECTION_BOUNDARY`/`DETECTION_TYPE` → `LOCALIZATION_SYSTEMATIC`/`WRONG_PAGE`/
`LOCALIZATION_LOCAL` → `APPLICATION_NOT_APPLIED`/`APPLICATION_OFFSET` →
`REMOVAL_INCOMPLETE`. A hidden mention without its required replacement is `NOT_REPLACED`
at the `REPLACEMENT` stage. A failed run gives `CANDIDATE_ERROR` for every required mention.

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

## Failure decomposition

For a candidate with an end-to-end run and oracle runs on the same dataset, on the
required mentions common to all of them:

| Stage | Share of the failure rate (points) |
|---|---|
| Text acquisition | e2e − oracle_text |
| Detection | oracle_text − oracle_mentions |
| Resolution | oracle_mentions − oracle_entities |
| Localization | oracle_entities − oracle_regions |
| Application (incl. replacement) | oracle_regions |

The failure rate counts leaks and required replacements that were not made. Missing modes
merge the neighbouring stages. The parts sum to the end-to-end failure rate. A
negative part means a stage does better on its own upstream output than on perfect input
(an interaction).

## Inter-annotator agreement

`rb gt agree` matches the two annotators' words one-to-one by polygon IoU ≥ 0.5, then:
strict mention F1 (same words and type), tolerant mention F1 (same words up to optional
words, any type), Cohen's κ over word labels (PII type or O) and over protected classes,
B³ over entity links of matched mentions, and word-box IoU. Release gates: tolerant F1 ≥
0.95 and link B³ F1 ≥ 0.95.

## Statistics

Rates pool required mentions; 95% intervals are document-cluster bootstraps (mentions
within a document are correlated). Comparisons use a paired bootstrap over shared
mentions. With few documents the intervals are wide; the report shows n everywhere.
