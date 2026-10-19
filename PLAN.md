# Redaction Benchmark — Technical Plan (v0)

> **Status (2026-10-18).** This is the original design, kept as the roadmap. Phases 1 and 3–7
> of §15 are built. Phase 2 has AI-annotated ground truth (`gt_version 0.1-ai`) that no human
> has reviewed. The §16 decisions are still open, apart from part of Q12. Phases 8–9 have not
> started: no real candidate tool has been run. [ARCHITECTURE.md](ARCHITECTURE.md) describes
> the code as built and lists where it differs from this plan.

## Context

You are building several independent PII-redaction tools for Australian medico-legal documents (mostly scans), each in its own codebase and not yet ready to integrate. `redaction_benchmark` must be a **neutral, independent evaluation framework**. It must answer which tool is better, where it is better, which stage a failure starts in, and whether the tool's behaviour can be reproduced. When this plan was written the repo held only a README and 5 representative PDFs. The plan is grounded in:
- a visual, page-by-page review of all 88 pages;
- structural analysis of the PDFs: text layers, image layers and their resolution and compression, vector content, metadata/XMP, annotations, measured skew, and what each image layer contains on its own.

To avoid repeating names that are themselves PII, documents are referred to by short IDs. Examples below describe roles, not people.

| ID | File (truncated) | Pages | How the content is physically encoded |
|---|---|---|---|
| DPI | `2. Degree of Permanent Impairment Report…` | 9 | "Microsoft Print to PDF". Text is drawn as **vector glyph outlines** (up to ~2,600 filled paths per page) with **no text layer**. Raster letterhead and raster signature. |
| NOC | `20250801_ Notice of Claim for Damages…` | 30 | Konica Minolta scan with **MRC compression**: a 150-DPI colour/grey background plus **2–71 separate 300-DPI 1-bit CCITT text masks per page** |
| PSY | `3. Report of … (Psychiatrist)…` | 23 | Word → PDF with a full text layer. Letterhead/footer text exists **only as images**. Raster signature. Appended CV page. |
| IME | `5. Independent Medico-Legal Report…` | 22 | Desktop scanner: one 200-DPI colour JPEG per page. Page size varies per page (591–600 × 838–846 pt). |
| TRN | `6. Plaintiff Training Transcript` | 4 | HTML → PDF (wkhtmltopdf) with a text layer. The person's initials exist **only as an avatar image**. |

---

## 1. Understanding of the 5 representative PDFs

### 1.1 Findings → implications

1. **There are five different encodings, not "scanned vs digital".**
   - Vector outlines (DPI): OCR is easy, but there is no text layer.
   - MRC layered scan (NOC).
   - Single-JPEG scan (IME).
   - Word-generated (PSY) and HTML-generated (TRN) PDFs.
   → The top axis of the condition taxonomy is the *encoding pathway*. It determines both how hard it is to get the text and *how* PII must be physically removed.
2. **PII lives outside the visible text.**
   - **NOC:** each page is a stack of up to 72 image layers. The colour background layer *on its own* contains both signatures from the declaration page and part of a handwritten place name (the rest of the word is in a 1-bit mask). It also holds faint "ghost" word shapes of the typed text that become visible with contrast stretching.
   - **DPI:** PII is vector paths, so a box drawn on top leaves them in the content stream.
   - **PSY:** the text layer survives any overlay. PSY and TRN also carry PII-adjacent text that exists only as images.
   - **Metadata:** DPI's `Author` field is a person named in the report. PSY's Title/Keywords contain the expert's name. Input filenames contain names and a plaintiff number.
   → Evaluating the rendered page is **necessary but not sufficient**. The benchmark must also check whether the original content survives *anywhere in the file* (§5).
3. **Geometry is hostile.**
   - NOC pp. 13–23 (11 pages) are landscape tables **rotated 90° inside the image**, while the page `/Rotate` is 0.
   - Measured skew is ≈0.4–1.3° on NOC (at 1.3°, a 500-pt line drifts ≈11 pt, about one line height) and ≈−0.5–+0.2° on IME.
   - IME page sizes vary from page to page.
   - NOC forms split one address across 5 boxes and one name across separate surname and given-name rows.
   - A form ID runs vertically in the margin.
   → Spatial ground truth must be **polygons in one canonical page frame**. Scoring must be **ink-based**, because axis-aligned box IoU misbehaves on skewed or rotated text. Coordinate bugs (rotation, DPI scale, y-flip, crop offset, wrong page) must be **diagnosed automatically**.
4. **Text-layer order ≠ visual order.** PSY's text layer emits the running header/footer before the body, and different tools will serialise differently. → Ground truth cannot be anchored to character offsets in any one text serialisation. It must be anchored to **positioned words**.
5. **Handwriting and marks.**
   - NOC: a block-capital handwritten name of the Justice of the Peace, a handwritten place and date, 2 signatures, and handwritten initials on every annexure page.
   - IME: a blue-ink signature and a handwritten date.
   - PSY and DPI: raster signatures.
   → Ground truth needs **region-only PII** (signatures, initials) as well as transcribed handwriting. OCR metrics must slice out handwriting.
6. **Identities are dense and ambiguous.**
   - NOC's annexure names more than 25 people (colleagues, managers, an investigator with a mobile number, clinicians, children), mostly by **first name only**. Their full names appear once, on p. 5.
   - One colleague's name is spelled **3 ways**. The GP's name is spelled **2 ways**, plus a short form.
   - **Two different people share a first name** in the same narrative.
   - **Spouses share a surname**, both in NOC and in IME.
   - The claimant appears as a full name, title + surname, a **nickname**, a **former (née) name**, "the Claimant", and as the first-person narrator.
   - PSY/IME/DPI use "SURNAME Given", "SURNAME, Given" and all-caps forms. PSY also gives a previous name.
   - Names appear **inside a lawyer's reference code** (`Name:initials:number`), **inside an email address**, and **inside a business name**. A claim number sits inside a file-path footer.
   → Entity resolution must be scored explicitly (false merges and false splits). Pseudonym consistency must be checked **per name component**.
7. **Hard negatives are everywhere.**
   - Surnames that are ordinary English words (an adjective, "low", a month name) appear in reports that also use those words normally ("…a good capacity…", "low back pain", "she may require…").
   - Eponyms (Hoover's sign, Potter syndrome), a case citation (*Fox v Wood*), and buildings or facilities named after people.
   - A form's example address, and already-pseudonymised "Student A/B/C".
   → Ground truth must mark **protected content**, so that over-redaction and over-pseudonymisation are measured, not just recall.
8. **Repetition skews counts.** Running headers/footers repeat name, DOB and reference on 18 pages (IME), the name on 21 pages (PSY) and "SURNAME, Given" on 8 pages (DPI). → Report **entity- and document-level** metrics alongside mention-level ones. One leaked occurrence is enough to re-identify someone.
9. **Content that must survive is valuable.**
   - Diagnoses, medications, impairment percentages, dates (current requirement), legal citations.
   - Sensitive attributes that are central to the claims: Indigenous heritage in a racism claim, religion, mental health.
   - The source itself contains a gender-pronoun error (DPI once uses "his" for a female claimant).
   → Damage to clinical or legal content is a first-class failure. Gender and nationality ground truth must come from authoritative cues (form checkbox, title, explicit statements), not from pronouns.
10. **Identity crosses documents.** DPI and IME concern the same claimant (IME's document index lists the DPI report). PSY and TRN concern a second claimant, and NOC a third. One expert appears in three documents. → Entity IDs must be stable across documents, and the scope of pseudonym consistency is a policy decision.
11. **Classic leak vectors are absent.** None of the five PDFs has annotations, AcroForm fields, links, attachments, JavaScript, `/Rotate`, or a hidden OCR text layer. → These must be added to the dataset deliberately (§8).

### 1.2 Document-condition taxonomy (derived from the PDFs)

Conditions vary *within* a document: NOC mixes clean typed form pages, rotated narrative tables and a handwritten block. Tags are therefore applied at document, page, region and mention level. Page-level tags are **measured automatically** where possible (encoding, effective DPI, skew, rotation); the rest are annotated.

| Level | Axis | Values (✓ observed, ○ needed) |
|---|---|---|
| Page | Encoding | text-layer digital ✓ · HTML digital ✓ · vector outline ✓ · single-JPEG scan ✓ · MRC scan ✓ · scan + hidden OCR layer ○ · mixed bundle ○ · fax ○ · phone photo ○ |
| Page | Geometry | upright ✓ · content rotated 90° in image ✓ · `/Rotate` attribute ○ · skew bins (<0.5° ✓, 0.5–1.5° ✓, >1.5° ○) · variable page size ✓ · CropBox offset ○ |
| Page | Image quality | 150/200/300 DPI/vector ✓ · JPEG artefacts ✓ · MRC ghosts ✓ · edge shadows/borders ✓ · speckle ✓ · faint/grey text ✓ · <150 DPI ○ · blur ○ · bleed-through ○ |
| Region | Layout | prose · running header/footer ✓ · form label-value ✓ · multi-box field ✓ · checkbox ✓ · bordered table ✓ · table across pages ✓ · multi-line cell ✓ · letterhead ✓ · two-column CV ✓ · rotated margin text ✓ |
| Region | Marks | typed · handwritten block capitals ✓ · signature ✓ · initials ✓ · handwritten cursive prose ○ · stamp ○ · margin annotation ○ · highlighter ○ |
| Mention | Difficulty | **form:** full / given / surname / initials / nickname / former / UPPER / inverted / possessive / hyphenated / titled · **embedded:** in code, email, org name, path · **split:** across line, cell, field or page · **placement:** header/footer, inside an image · **ambiguity:** homograph, spelling variant, shared first name or surname · **OCR risk:** small, faint, touching a border · repeat count |

### 1.3 Are these five documents sufficient?

- **Sufficient to build and prove the framework.** They exercise every stage: 5 encodings, rotation, skew, handwriting, forms, tables, layered images, metadata leakage, heavy aliasing and hard negatives.
- **Not sufficient to rank tools.** There are 88 pages, 3 claimants, and **one document per encoding**, so condition effects are confounded with document identity. Because mentions within a document are correlated, the effective sample size is close to the number of documents, so confidence intervals will be wide. These documents also shaped the design, so they **cannot serve as a blind test set**.
- **Gaps to fill (§8):**
  - Scans with a hidden OCR text layer (the top real-world leak risk).
  - Handwritten GP notes and WorkCover certificates; stamps and margin notes; faxes and phone photos.
  - `/Rotate` and CropBox oddities; AcroForm fields, annotations, attachments, incremental saves; pre-redacted documents.
  - Long mixed bundles (IME's index references a ~1,050-page brief).
  - Australian identifiers absent here: Medicare card, TFN, IHI, DVA, CRN, licence, BSB/account.
  - Broader name diversity; printed emails and SMS.

---

## 2. Benchmark objectives

Each objective is measured per candidate and per slice (encoding, condition, PII type, role, document):

| Objective | Question | Primary measures |
|---|---|---|
| **Safety** | Is any required PII still exposed? | Residual PII by leak level: **L1** visible · **L2** recoverable from the file · **L3** latent (recoverable with enhancement). Reported per mention, entity and document. |
| **Utility** | Was everything else preserved? | Collateral damage to ordinary words and to *protected* words; document integrity |
| **Component quality** | Which stage is weak? | Text acquisition, detection, resolution, localization, application and pseudonymisation metrics |
| **Reliability** | Does the same input give the same meaningful result? | Determinism at the bytes / structure / visual / decision levels |
| **Diagnosis** | Why did each failure happen? | Per-mention root cause, marked *confirmed* or *inferred* |
| **Robustness** | Where does performance degrade? | Condition-sliced metrics plus controlled degradation ladders |

**Out of scope (v1):** speed, cost, resource use, scalability, code quality, architecture, deployment, and the candidates' technology choices.

---

## 3. Evaluation methodology

### 3.1 Reference stage model: evaluation interfaces, not an imposed architecture

```
input PDF
 [T] Text acquisition (OCR or text-layer extraction) → positioned tokens (text, polygon, order, conf)
 [D] PII detection + typing                          → mentions (type; anchored to tokens | text spans | regions)
 [R] Entity resolution                               → mention clusters (+ inferred attributes)
 [L] Localization / redaction planning               → page-space regions + action (remove | replace)
 [P] Replacement generation                          → entity→pseudonym map, mention→replacement text
 [A] Application / rendering                         → output PDF   ← always required, always evaluated
```

Why each boundary exists, based on what the documents show:
- **T** is the most variable stage: 3 of 5 documents have no usable text layer, there are five encodings, and there is rotation and handwriting.
- **D vs L:** finding a name is a *text* problem, while placing it on a rotated, skewed or multi-field page is a *geometry* problem.
- **R:** aliasing is pervasive, and it drives both pseudonym consistency and alias propagation.
- **A:** correct coordinates do not guarantee removal from MRC layers, vector paths or text layers.
- **P:** consistency, validity and attribute preservation are orthogonal to detection.

Each interface is both:
- a **claim** that can be scored against ground truth (GT), and
- an **injection point** where the benchmark can substitute GT ("oracle" inputs).

Tools may merge stages, for example a vision model that emits regions and types directly. Such tools simply expose fewer interfaces.

### 3.2 Common evaluation space: ground-truth words

Every candidate output is projected onto the GT's positioned words, each of which has a polygon and an ink mask. This makes tools with different OCR, tokenisation and reading order directly comparable.
- **Spatial projection (preferred):** a candidate region or token box *selects* GT word *w* if it covers ≥ θ (default 50%) of *w*'s ink.
- **Text projection (fallback for text-only outputs):** character-level alignment of the candidate's page text against the GT transcript, using anchors, then mapping spans to GT words.
- **Discontinuous mentions** (multi-box addresses, names split across lines, cells or fields) are just sets of GT words.

### 3.3 Text acquisition (OCR / extraction)

Candidate words are matched to GT words by spatial overlap, using one-to-one assignment with explicit split and merge handling. Metrics on the matched words:
- **Recognition:**
  - CER/WER on matched words, under raw, case-folded and punctuation-normalised variants.
  - Missed-text rate (GT words with no match).
  - Hallucinated-text rate (candidate words with no match).
- **Localization:**
  - Word-box IoU distribution.
  - % of GT words with ≥95% of their ink inside the candidate box.
  - Signed edge offsets.
- **Reading order:**
  - Kendall τ against GT order.
  - **Line-splice errors:** tokens that are adjacent in the candidate's order but not in reality. This is how table cells fuse into false entities.
- **Orientation:** accuracy of detecting rotated pages and regions.
- **Layout:** evaluated through its effects (order, splices, orientation), not as a standalone layout benchmark.

**PII-weighted variants** are restricted to GT tokens inside required mentions, by type:
- PII-CER.
- PII-token exact-recognition rate.
- **Mention recognisability:** all tokens recognised (edit similarity ≥ τ) and localised. This is the upper bound on what *any* detector could find from this OCR.

These sit next to general CER so that "excellent OCR overall, but names fail" is visible. Handwriting, digits inside identifiers (one wrong digit leaves the rest of an ID unredacted) and image-only text are reported as separate slices. Text-layer extraction is scored with the same metrics, with `source = text_layer`.

### 3.4 PII detection

All matching happens in GT-word space. Several mention-level schemes are reported side by side:
- **Strict:** exact required-word set and correct type.
- **Exact-boundary:** exact required-word set, type ignored.
- **Policy-tolerant:** all required words present, plus only words from the mention's *optional* set (titles, possessive 's, punctuation).
- **Partial:** any overlap, with type-aware and type-unaware variants.

Each matched mention also gets a **boundary class**:
- exact;
- over-extended, benign (optional tokens only, e.g. "Mr");
- over-extended, harmful (takes in adjacent ordinary or protected words);
- **under-extended, naming the leaked component** (e.g. "surname missing");
- split;
- merged (e.g. name and address detected as one span).

So for a GT mention "John Smith": detecting "Mr John Smith" is tolerant-correct and strict-incorrect (benign over-extension), while detecting "John" is under-extended with the surname leaked.

Further detection metrics:
- **Token-level P/R/F1** over PII words, by type. This is robust to boundary conventions.
- **Character coverage** within mentions, for partial credit.
- **False positives classified by what they hit:** a protected token (eponym, homograph, medical term, date, boilerplate), an ordinary word, or an optional token (no penalty).
- **Type confusion matrix** in the benchmark's taxonomy. Each candidate declares a fixed label mapping, which is published in reports.
- **Entity-level:** % of entities with *every* mention detected, and % with at least one miss.
- **Repeat consistency:** detection rate across repeats of the same surface form, e.g. header occurrences.

### 3.5 Entity resolution

- **Scope:** PII-bearing mentions only (names, initials, nicknames, former names, names inside emails and codes). Pronouns and role nouns are excluded in v1 (open question Q10).
- **Metrics:** B³, MUC, CEAF-e, LEA, pairwise link precision/recall, and **explicit lists** of false merges and false splits:
  - merges: two people with the same first name; spouses sharing a surname;
  - splits: nickname, former name, spelling variants, "SURNAME, Given" headers.
- **Modes:** run on GT mentions if the tool accepts injected mentions, and on predicted mentions after alignment (unmatched mentions count as singletons).
- **Black-box pseudonymisers:** resolution is **inferred from their replacements**. Mentions given the same pseudonym component are treated as one cluster, so even final-PDF-only tools get resolution diagnostics.
- **Cross-document linking** is scored only if the policy requires matter-level consistency (Q8).

### 3.6 Localization: spatial accuracy of planned regions

Scored on the regions a tool exposes. For black-box tools it is scored on "effective regions" derived by differencing the output against the original.
- **Ink coverage** per mention: the fraction of required ink inside the regions. This is the safety-relevant quantity, not IoU.
- **Spill:** area outside the tolerance zone (required ∪ optional ink, dilated by 2 pt by default), plus collateral and protected GT words selected.
- **Edge offsets:** signed, per side.
- **Transform diagnosis per page:**
  - fit a similarity/affine transform between matched candidate regions and GT;
  - test known bug hypotheses: DPI scale (72↔96/150/200/300), y-axis flip, 90/180/270° mapping, CropBox offset, page index ±1;
  - a good non-identity fit is reported as one systematic bug (e.g. *"scale ≈ 0.24: px@300 DPI treated as points"*) instead of dozens of unrelated misses.
- **Orphan regions** (not near any GT mention) are matched by shape to nearby missed mentions, flagging a *probable* mislocalisation.

### 3.7 Application fidelity

- When a plan is exposed, planned regions are compared with the pixels actually changed in the output:
  - IoU between plan and change;
  - regions planned but not applied;
  - shifted or scaled application;
  - redaction applied on the wrong page.
- **Removal** forensics are covered in §5.3.

### 3.8 Pseudonymisation / synthetic replacement (only for tools that replace)

Replacement text is read from the output at each GT mention: from the text layer if there is one, otherwise with the benchmark's measurement OCR. It is then parsed using the mention's GT **name-component structure** (title, given, middle, surname, nickname, initials, order, case). Checks:
- **Coverage:** every required mention is replaced or removed, and no original component survives (e.g. given name replaced but surname kept counts as a leak).
- **Consistency within the policy scope:**
  - for each entity, each original component maps to exactly one replacement component;
  - this holds across forms: full name, "Ms X", "SURNAME, Given", upper case, initials, possessive, and the local part of an email address.
- **Distinctness:**
  - different entities never share a replacement;
  - no replacement equals another real name in the document;
  - related entities (spouses sharing a surname) are handled according to policy.
- **Validity and format:**
  - plausible names, addresses, phones and emails;
  - formats preserved: AU mobile/landline patterns, postcode–state consistency, an email address derived from the *new* name;
  - checksum validity of identifiers is configurable.
- **Attribute preservation:**
  - gender, from GT evidence plus honorific and pronoun agreement with the unchanged text;
  - "nationality", per the definition to be confirmed (Q9);
  - reported with instrument uncertainty and an optional human-review sample.
- **Rendering:** the replacement fits its region; overflow onto neighbouring text counts as collateral damage.
- **Determinism:** the mapping is stable across runs (§6).

---

## 4. Component isolation

Three complementary techniques are used. Every experiment is a declarative spec, and it is **reported as N/A** when a candidate lacks the capability it needs.

### 4.1 Evidence trace (observational)

When a tool exposes intermediates, each GT mention gets a stage-by-stage record. Was it recognised by T? Detected by D, and with which boundary and type? Clustered correctly by R? Covered by L? Applied by A? Truly removed? Together these records form each candidate's **failure funnel**.

### 4.2 Interface injection (oracle experiments, needs tool support)

| Experiment | Injected from GT | Isolates |
|---|---|---|
| `e2e` | — | the whole system |
| `oracle_text` | tokens | D→A given perfect text (how good the detector is with perfect OCR) |
| `oracle_mentions` | mentions + types | R, L, P, A |
| `oracle_entities` | mentions + clusters | L, P, A (pseudonymiser in isolation) |
| `oracle_regions` | regions (+ clusters) | A only (does the engine redact correctly given correct coordinates?) |

Leak rates from these runs are decomposed by subtraction (the terms telescope):

| Stage | Share of the leak rate |
|---|---|
| Text acquisition | e2e − oracle_text |
| Detection | oracle_text − oracle_mentions |
| Resolution + localization | oracle_mentions − oracle_regions |
| Application | oracle_regions |

Per mention, "leaks in e2e but safe under `oracle_text`" is a *confirmed* OCR-caused failure. If a tool supports every injection, a full factorial design (with Shapley-style attribution) reveals **non-additive interaction effects**. The same mechanism later supports **cross-candidate stage swapping**, e.g. feeding tool A's OCR into tool B's detector.

### 4.3 Input manipulation (black-box, no tool support needed)

- **Clean twins:**
  - for every scanned or vector page, the benchmark typesets a born-digital twin with the same words at the same positions;
  - handwriting and signatures are copied in as image crops;
  - content and layout stay identical while image quality becomes perfect, so the scan-vs-twin gap isolates scan/OCR difficulty for *any* tool, including final-PDF-only ones.
- **Degradation ladders:**
  - single-factor transforms of twins and born-digital pages: DPI 300→100, skew 0→3°, rotation inside the image vs via `/Rotate`, JPEG quality, noise/blur, MRC-style layering, fax simulation, an added hidden OCR layer;
  - GT transforms exactly with the geometry, giving causal curves such as "recall vs DPI".
- **Invariance probes (later):** re-saved, object-reordered, linearised or renamed PDFs should not change any decision.

### 4.4 How your isolation questions are answered

| Question | Method |
|---|---|
| How good is the detector when OCR is perfect? | `oracle_text`; for black-box tools, clean twins |
| How much does OCR quality affect detection? | Detection recall vs mention-recognisability curve; e2e − oracle_text |
| How good is the engine with correct coordinates? | `oracle_regions` |
| What error is added when detector output feeds the engine? | Planned-region coverage vs output; oracle_mentions vs oracle_regions |
| What share of final failures does each component cause? | Funnel (observational) + telescoping decomposition (counterfactual) |

---

## 5. End-to-end evaluation (the actual output, never the tool's claims)

Every output PDF goes through the same candidate-independent pipeline.

### 5.1 Integrity

- The PDF parses and validates (qpdf).
- Page count, page size and orientation are preserved. If they are not, pages are registered by image alignment and the change is flagged.
- There are no blank or corrupted pages.
- Non-PII text stays legible, and stays extractable if a text layer is required (Q11).

### 5.2 Visible leakage (L1)

- **Rendering:** original and output are rendered at 300 DPI with **two independent renderers** (MuPDF, PDFium), with annotations and optional-content layers in both view and print states. PII visible in *any* configuration counts, which catches annotation-only "redactions" and hidden-layer tricks.
- **Pattern survival per character cell:** normalised cross-correlation of the original glyph pattern (exact cells for digital pages, estimated for scans). A black box over black text counts as obscured, unlike a naive dark-pixel test; the measure is also robust to recompression.
- **Reader OCR:** a pinned OCR engine tries to recover any required token from the output region.
- **Default thresholds** (versioned, with sensitivity analysis):
  - **SAFE:** ≤ 2% of the glyph pattern survives and nothing is recovered;
  - **LEAK_VISIBLE:** ≥ 90% survives;
  - **PARTIAL:** anything in between.

### 5.3 Recoverable leakage (L2): content-survival forensics

The question is whether the *original* PII still exists in the file, visible or not:
- **Text:** all text objects, including invisible render mode, near GT regions.
- **Images:** every image XObject (including those inside Form XObjects, stencil masks and soft masks) is mapped into GT coordinates and compared with the original crop. This catches boxes drawn over untouched image layers and partially edited MRC stacks.
- **Vector paths:** glyph paths that survive under overlays (the DPI-type case).
- **Document level:**
  - Info and XMP metadata, outlines, annotation `/Contents`, form values, attachments, page thumbnails;
  - structure-tree `/Alt` and `/ActualText`;
  - earlier revisions from incremental saves and unreferenced objects (raw decompressed scan);
  - the output filename.

### 5.4 Latent leakage (L3)

Contrast-enhanced reader OCR on each image layer, and on boxes with less than 100% opacity. This covers MRC ghosts, translucent overlays and near-black text on black.

### 5.5 Over-redaction

- Changed pixels outside the tolerance zone.
- Collateral **ordinary** words damaged.
- **Protected** words damaged: clinical terms, medications, dates (while dates are kept), citations, eponyms, homographs.
- Spill area.

### 5.6 Pseudonym checks

The checks in §3.8, applied to the actual output.

### 5.7 Effective detection

Changed-pixel components are projected onto GT words. This gives token-level precision/recall for **every** tool, including final-PDF-only ones.

### 5.8 Outcomes and roll-ups

- Each required mention: **SAFE_REMOVED / SAFE_REPLACED / PARTIAL / LEAK_VISIBLE / LEAK_RECOVERABLE / LEAK_LATENT**.
- Each protected word: **PRESERVED / DAMAGED / REMOVED**.
- **Per entity:** any-leak rate.
- **Per document:** **clean-document rate**, i.e. no L1/L2 leak of required PII and no protected damage.
- **Crashes and timeouts:** completion rate is reported separately. A missing output counts as unprotected in the "strict" view and is excluded in the "completed-only" view; both are shown.

### 5.9 The measurement OCR is an instrument

- It is pinned and versioned, and runs locally.
- It is **calibrated** on the unredacted originals: its recall on original PII is measured per condition. "Nothing recovered" is then weighed by its sensitivity on that condition (it is weak, for example, on handwriting).
- Pixel-pattern evidence is primary; OCR is corroboration.

---

## 6. Determinism

- **Protocol:**
  - N repeats (default 5) with identical inputs, configuration and seed;
  - optionally in fresh processes and on a second machine;
  - if the tool exposes a seed, also across seeds ("deterministic given a seed" vs "stable across seeds");
  - the environment is recorded for every run.
- **Canonicalisation before comparison:**
  - Unicode NFC and whitespace normalisation;
  - coordinates rounded to 0.1 pt in the canonical frame;
  - sets sorted;
  - clusters compared as **partitions** (invariant to label permutation);
  - volatile PDF fields stripped (dates, `/ID`, producer, XMP instance IDs);
  - regions rasterised to 150-DPI masks.
- **Levels, reported separately:**

  | Level | Name | Identical across runs |
  |---|---|---|
  | D0 | exact | byte-identical files (informational only, never the headline) |
  | D1 | structural | canonical stage artifacts: tokens; mentions in GT-word space; clusters; regions; pseudonym map |
  | D2 | visual | rendered output pages, exact and anti-aliasing-tolerant, plus redaction-mask IoU across runs |
  | D3 | decision | per-mention outcomes (safe / leak / damaged) and pseudonym assignments |

- **Metrics:**
  - reproducibility rate per level and per stage;
  - **flip rate**: mentions whose outcome changes between runs, with the list of unstable mentions (the most dangerous kind of failure);
  - metric variance across runs.
- **Pseudonyms:** *within-run consistency* and *across-run stability* are distinct properties and are both reported.

---

## 7. Ground truth

### 7.1 Principles

1. **Facts, not policy.**
   - GT records what each span *is*: type, subtype, entity, role, attributes.
   - A separate, versioned **policy profile** decides what must be redacted, replaced or kept.
   - When requirements change (dates, doctors' names), no re-annotation is needed.
2. **Visual-first.** The transcription is what a human sees, including letterhead text, avatars and handwriting. Each word is tagged with its source: text layer, raster, vector or handwriting.
3. **Geometry-anchored.** Every word has a polygon in the canonical frame. Ink masks are *derived automatically* from the source PDF.
4. **Entity-centric**, with IDs stable across documents. Matters are recorded explicitly: M1 = DPI + IME, M2 = PSY + TRN, M3 = NOC.
5. **Independent of candidates.** Pre-annotation uses only benchmark-owned tools; candidate outputs never seed GT.

### 7.2 Representation: one JSON file per document, schema-versioned

The layers:
- document metadata and conditions;
- **pages:** size, rotation, measured skew and DPI, conditions;
- **words:** id, text, polygon, line/block, order, source, legibility;
- **image regions:** signature, initials, avatar, logo;
- **mentions:** type/subtype, entity, required word segments, optional words, name components, conditions, nested children (e.g. a person inside an org name or a reference code);
- **entities:** role, canonical name, aliases, attributes with evidence;
- **relations:** spouse, parent, employer;
- **classified non-PII spans:** medical, medication, legal citation, eponym/named facility, homograph, boilerplate, date;
- **document-level PII surface:** metadata fields, filename.

```json
{ "schema": "rb.gt/0.1", "doc_id": "PSY", "matter": "M2", "source_sha256": "…", "gt_version": "0.1.0",
  "pages": [{ "index": 18, "size_pt": [595.3, 841.9], "rotation": 0, "conditions": ["table"],
    "words": [{ "id": "p18w51", "text": "…", "poly": [[93.1,142.0],[150.2,142.0],[150.2,154.1],[93.1,154.1]],
                "line": "p18l7", "order": 51, "source": "text_layer" }] }],
  "mentions": [{ "id": "m0612", "entity": "E1", "type": "PERSON_NAME", "form": "given+middle / SURNAME",
    "segments": [{ "page": 18, "words": ["p18w51","p18w52"] }, { "page": 18, "words": ["p18w53"] }],
    "optional_words": [], "components": { "given": ["p18w51"], "middle": ["p18w52"], "surname": ["p18w53"] },
    "conditions": ["table_cell", "line_split"] }],
  "entities": [{ "id": "E1", "kind": "person", "role": "claimant",
    "attributes": { "gender": { "value": "F", "evidence": ["title", "form_checkbox"] } } }],
  "classified": [{ "words": ["p13w40","p13w41"], "class": "EPONYM" }]
}
```

This is illustrative; it will be finalised in Phase 0.

**Type taxonomy v0 (owned by the benchmark):**
- PERSON_NAME, with components;
- SIGNATURE / HANDWRITTEN_MARK and IMAGE_PII;
- ADDRESS (residential / postal / organisational; multi-field) and LOCATION;
- PHONE, EMAIL, URL;
- DATE (DOB / event / document) and AGE;
- identifiers: ID_CLAIM, ID_LEGAL_REF, ID_PROVIDER, ID_GOVERNMENT (Medicare / TFN / IHI / DVA / CRN / licence / passport), ID_BUSINESS (ABN / ACN), ID_FINANCIAL;
- FILE_PATH, ORGANISATION, QUASI (occupation, physical measurements, family composition).

**Roles:** claimant, family, treating clinician, examining expert, legal, insurer staff, employer staff, witness, alleged perpetrator, official (JP), admin staff (e.g. typist initials in reference codes).

### 7.3 Policy profiles (versioned YAML)

Profiles map (type, subtype, role, component, class) → **MUST / MAY / MUST_NOT** and an expected action (remove / replace / either). Examples:
- private persons' names: MUST, replace;
- honorifics: MAY;
- dates: MUST_NOT (current requirement);
- DOB, professionals' names, organisations and suburbs: TBD (§16).

Every result records the policy version, and the same GT can be scored under several policies.

### 7.4 Creation workflow

1. **Words:**
   - from the text layer for PSY and TRN bodies;
   - from the local measurement OCR for DPI, IME, NOC and raster parts;
   - then human correction, including rotated pages and handwriting.
2. **Pre-labelling:** benchmark-owned regex, gazetteers and generic NER propose mentions. Humans then fix mentions, boundaries, optional words and types.
3. **Entity linking and attributes:** done by humans, assisted by string-similarity suggestions, with the evidence recorded.
4. **Classified non-PII spans:** assisted by lexicons (medications, eponyms, homograph surnames, dates).
5. **Conditions:**
   - auto-measured: encoding, DPI, skew, rotation, header/footer, line split, repeat count;
   - annotated: handwriting, forms/tables, stamps.
6. **Generated automatically:** ink masks, tolerance zones and per-page **review sheets** (GT overlays as images).

Tooling: Label Studio (OCR/box/relation templates) with converters. Fall back to a minimal custom annotator if discontinuous mentions prove awkward in Label Studio.

### 7.5 Quality control

- **Annotation guidelines** with worked examples taken from these five documents: boundary rules, multi-box forms, names inside codes and emails, signatures, homographs.
- **Double annotation** of ≥ 20% of pages, always including NOC Annexure A (the hardest pages), followed by adjudication.
- **Agreement metrics:** tolerant and strict mention F1, token κ, entity-link B³, polygon IoU. Proposed release gates: tolerant mention F1 ≥ 0.95 and link B³ F1 ≥ 0.95.
- **Automated validators in CI:**
  - schema and referential integrity;
  - polygons inside the page;
  - each mention's text equals its words;
  - every entity has at least one mention;
  - **unannotated-repeat detector:** a surface form annotated as PII on one page but not elsewhere;
  - **homograph sanity:** every occurrence of a homograph surname is classified one way or the other;
  - attribute evidence present.
- **GT dispute queue:**
  - false positives that several tools agree on, or that look like real PII, are adjudicated by the benchmark owner;
  - fixes create a new GT version with the rationale logged;
  - old results stay tied to the old version and can be re-scored.

### 7.6 Versioning and storage

- GT JSON lives in git, so changes are reviewable as diffs.
- PDFs live outside git, in an access-controlled store referenced by SHA-256 in the manifests (Q12).
- GT itself contains PII and needs the same access controls as the PDFs.

---

## 8. Dataset strategy

- **Splits:**
  - `dev`: shared with tool developers. All five seed documents go here, because they shaped the design.
  - `test`: sealed. GT is not shared; results are reported in aggregate plus approved examples.
  - `regression`: grows from failures.
  - `synthetic`: unrestricted and shareable.
- **Releases:**
  - immutable, semantic-versioned manifests: document hashes, split, GT version, policy compatibility, conditions;
  - a **datasheet per document**: provenance, authorisation, encoding, PII density, matter.
- **Growth sources, in priority order:**
  1. Seed set (these 5) → `dataset v0.1.0`, used to prove the framework.
  2. Clean twins and degradation ladders of the seed set: exact GT under controlled conditions.
  3. More **real** documents, stratified by encoding and type:
     - IME, DPI and psychiatric reports; NOCs and other forms;
     - GP notes and WorkCover certificates (often handwritten); discharge summaries; radiology;
     - Centrelink and financial records; statutory declarations; printed emails and SMS;
     - long mixed bundles; scans with hidden OCR layers; faxes; phone photos.
  4. **Synthetic** medico-legal documents built from templates with synthetic identities:
     - difficulty controlled deliberately: homograph surnames, nicknames, former names, shared names, diverse name origins, every AU identifier type;
     - optionally printed, hand-annotated and physically re-scanned for realism.
  5. **Adversarial PDF structure:** `/Rotate`, CropBox offsets, AcroForm values, annotations, attachments, thumbnails, incremental-save history, unapplied redaction annotations, white, tiny or off-page text, PII in bookmarks.
  6. **Regression cases** from real failures (§13).
- **Size guidance:**
  - plan on tens of documents per major encoding stratum in the sealed test set, i.e. ≥ 30–50 documents for credible v1 comparisons;
  - always report document-cluster bootstrap confidence intervals and n;
  - the seed set is for diagnosis, not for significance.

---

## 9. Candidate integration

### 9.1 The contract is files, not APIs

Each candidate provides:
- **A manifest** (`candidates/<name>.yaml`) declaring:
  - a version command and a run command template;
  - `provides` (which artifacts it emits) and `accepts_injection` (text / mentions / entities / regions);
  - a fixed label mapping to the benchmark taxonomy;
  - its coordinate frame and seed flag.
- **A thin adapter** (in the candidate's repo or in `adapters/`) that turns `input.pdf` (plus any injected JSON) into:
  - `output.pdf` (**required**);
  - optional canonical artifacts: `text.json`, `mentions.json`, `entities.json`, `plan.json`, `pseudonyms.json`;
  - `run_info.json`: versions, model identifiers, seeds, configuration.

### 9.2 Capability tiers

- **T0** output PDF only → **T1** + plan → **T2** + mentions → **T3** + text → **T4** + entities/pseudonyms. Each tier also records which injections are supported.
- **Missing stages are N/A**: never imputed and never scored as zero.
- Black-box tools still get output-level metrics, effective detection, inferred resolution, and twin/ladder diagnostics.

### 9.3 Execution modes

- **Runner mode:** a subprocess per document × experiment × repeat, with timeouts, logs and hashes.
- **Offline submission:** a team drops artifacts in the standard layout, and the benchmark validates and scores them.
- Inputs are passed under **neutral filenames**, because the original names contain names and a plaintiff number. GT is never reachable from the candidate's working directory.

### 9.4 Validation

`rb candidate check` validates artifacts against the schemas and declared frames. Malformed artifacts are rejected with reasons, never silently repaired.

### 9.5 Canonical coordinate frame

- PDF points of the *displayed* page (CropBox after `/Rotate`), origin top-left, y downwards.
- Artifacts may instead declare `pdf_user_space` or `image_px@dpi`; the geometry library converts.

### 9.6 Neutrality rules

- The evaluator contains no candidate-specific branches.
- Adapters only translate formats.
- Label mappings are declared once per candidate version and published.
- Thresholds and policy are fixed per evaluation version *before* runs.
- GT and pre-annotation never use candidate output.
- Disputes are adjudicated by the benchmark owner and logged.

---

## 10. Metrics

| Family | Metrics | Why |
|---|---|---|
| Safety | Required-mention leak rate by L1/L2/L3; entity any-leak rate; **clean-document rate**; residual-pattern distribution; leaked-character count | Privacy outcome at every granularity; one leak is enough to re-identify |
| Utility | Collateral-word rate; protected-word damage (by class); spill area; integrity checks; text-layer preservation | Over-redaction and document usefulness |
| Text | CER/WER on matched words; missed/hallucinated text; **PII-CER**, PII-token exact rate, **mention recognisability**; word-box IoU, ink-in-box rate; reading-order τ, line splices; orientation accuracy | General OCR vs OCR where it matters; spatial OCR feeds redaction |
| Detection | Strict / exact-boundary / tolerant / partial mention P-R-F1; token P-R-F1; character coverage; boundary-class distribution; FP-class distribution; type confusion; entity-complete rate | Standard and boundary-aware scoring; diagnosis |
| Resolution | B³, MUC, CEAF-e, LEA; pairwise link P/R; merge/split lists | Identity and alias correctness |
| Localization | Ink coverage; spill; edge offsets; transform fit and diagnosis | "Knew *what*, but not *where*" |
| Application | Plan-fidelity IoU; unapplied/shifted regions; content survival by layer | Engine correctness given its own plan |
| Pseudonym | Coverage; component-level consistency; distinctness; validity/format; attribute preservation; overflow; stability | Replacement quality |
| Determinism | D0–D3 rates per stage; flip rate; unstable-mention list | Reliability |
| Robustness | Metric vs condition (ladder curves); invariance pass rate | Degradation behaviour |

**Statistics:**
- Micro (mention) and macro (entity, document) averages.
- **Document-cluster bootstrap** confidence intervals.
- Paired bootstrap for comparing candidates on identical mentions.
- n shown in every cell; cells below a minimum n are flagged.
- Threshold-sensitivity analysis, to show whether rankings flip under different θ.

---

## 11. Failure attribution

- **Units:** required GT mention (for leaks), GT word (for over-redaction), entity, document.
- **Evidence per mention** (whatever is available):
  - conditions;
  - T: recognition and localisation;
  - D: match, boundary class, type;
  - R: cluster correctness;
  - L: coverage and transform diagnosis;
  - plan fidelity;
  - L1/L2/L3 outcome;
  - pseudonym checks;
  - outcomes in oracle runs and on the clean twin.
- **Root cause = the first stage in causal order that fails given correct upstream input.** Categories:
  - **Text acquisition:**
    - `TEXT_MISSING`;
    - `TEXT_CORRUPTED`, confirmed if the mention is safe under `oracle_text` or on the twin.
  - **Detection:**
    - `DETECTION_MISS`;
    - `DETECTION_ALIAS_PROPAGATION`: an alias of an entity the tool found elsewhere;
    - `DETECTION_BOUNDARY`;
    - `DETECTION_TYPE`: a wrong type led to the wrong policy or action.
  - **Resolution:** `RESOLUTION_MERGE`, `RESOLUTION_SPLIT`.
  - **Localization:**
    - `LOCALIZATION_LOCAL`;
    - `LOCALIZATION_SYSTEMATIC(transform)`;
    - `WRONG_PAGE`.
  - **Application and removal:**
    - `APPLICATION_NOT_APPLIED`, `APPLICATION_OFFSET`;
    - `REMOVAL_INCOMPLETE(layer: text | image | vector | metadata | revision | annotation)`.
  - **Pseudonymisation:** `PSEUDONYM_INCONSISTENT | INVALID | ATTRIBUTE | COLLISION`.
  - **Composition:** `INTERACTION`. Each stage is correct on its own inputs but the composition fails. Example: span→box offsets drift after Unicode normalisation of "½" (present in IME) or after de-hyphenation.
  - **Run-level:** `NONDETERMINISTIC`, `CANDIDATE_ERROR`, `UNKNOWN_STAGE`.
- **Over-redaction causes:**
  - a false-positive detection, broken down by FP class;
  - localization spill (e.g. line-height spill on skewed lines);
  - application spill;
  - replacement overflow.
- **Confidence labelling:** every attribution is marked **confirmed** (direct artifact or counterfactual evidence) or **inferred** (black-box heuristic, e.g. an orphan box beside a missed mention). The two are never mixed.
- **Aggregations:**
  - funnel per candidate;
  - counterfactual decomposition;
  - attribution × condition and attribution × PII-type heatmaps;
  - condition "lift" per failure category.

---

## 12. Reporting

- **Results store:** tidy long-format tables in Parquet, queryable with DuckDB. Tables cover:
  - mention outcomes, word outcomes, OCR word matches, entity outcomes;
  - page and document metrics, determinism comparisons, pseudonym checks, forensic findings, attributions.
  - Any new question is a new group-by, not new code.
- **Run report** (static HTML, offline):
  - scorecard matrix (no composite) with CIs and n;
  - per-stage pages;
  - condition × candidate heatmaps; PII-type and role tables;
  - funnel and decomposition;
  - leakage report by level and layer; over-redaction report; determinism; pseudonyms; integrity;
  - **failure gallery**: original crop, output crop, GT overlay, candidate boxes and the attribution chain.
- **Comparison report** (A vs B, or version N vs N+1):
  - per-mention transitions: fixed / regressed / unchanged / category changed;
  - paired bootstrap on the differences;
  - **propagation analysis**: did a component's gains reach the final output?
- **How the reports answer your questions:**

| Question | View |
|---|---|
| Best OCR? | Text scorecard (general and PII-weighted, by encoding) |
| Best PII recall / precision / boundary accuracy? | Detection scorecard (tolerant + strict, token-level), boundary-class chart |
| Best spatial redaction? | Localization coverage/spill and output residual pattern |
| Least residual PII? | Leakage report L1/L2/L3 per mention / entity / document |
| Least over-redaction? | Collateral and protected-damage report |
| Most deterministic? | Determinism D0–D3, flip rate |
| Best on difficult scans? | Condition × candidate heatmap; ladder curves |
| Which conditions cause most failures? | Attribution × condition lift |
| Which PII types are missed most? | Type × outcome table |
| Which failures come from OCR / detection / spatial? | Funnel + counterfactual decomposition |
| Did a component improvement help the final result? | Comparison report, propagation analysis |

- **Composite score: deferred.**
  - Later, it can be added as a derived, versioned "decision profile" over the stored results: hard gates (e.g. zero L2 leaks, decision-level determinism), then lexicographic or weighted ordering, or a **Pareto front** of residual PII vs collateral damage.
  - It would always be shown next to the full scorecard and never replace it.
- **Report privacy:**
  - galleries contain real PII, so reports are sensitive by default;
  - a "shareable" mode masks GT text in the crops (type labels and geometry only).

---

## 13. Regression strategy

- **Failure fingerprint:** (document, GT mention or word ID, category, stage). It is keyed on GT, so it is stable across runs and candidate versions.
- **`rb compare`** reports fixed failures, new regressions, persistent failures, and category changes (e.g. `DETECTION_MISS` → `LOCALIZATION_LOCAL` signals partial progress).
- **`rb regress promote <fingerprint>`** adds a case to the regression suite: a reference to the original mention plus metadata (origin run, candidate, category, description). It can also auto-generate:
  - a **minimal repro**: a single page or crop, with transformed GT;
  - **synthetic siblings**: the same difficulty pattern with new identities, to avoid overfitting to one instance.
- **Operation:**
  - the regression suite runs as a fast gate for every candidate version;
  - per-candidate history is tracked as trends;
  - cases from the sealed test set are shared with developers only as descriptions and synthetic siblings.

---

## 14. Architecture

### 14.1 Shape

An **offline, artifact-centric evaluation pipeline**.
- Candidates produce files.
- The evaluator is a pure function of (dataset version, GT version, policy version, evaluation config, candidate artifacts) → results.
- Running and scoring are decoupled, so artifacts can be **re-scored without re-running tools** when GT is corrected or a metric is added.

```
dataset vX ─┬──────────────── oracle inputs (from GT) ───────┐
(PDFs, GT,  │                                                ▼
 conditions)▼            ┌────────┐ command ┌───────────┐  output.pdf + optional stage JSON
       [ runner ]───────▶│adapter │────────▶│ candidate │──────────────┐
                         └────────┘         └───────────┘              ▼
   offline submission ───────────────────────────────────────▶ [ validate + hash ]
                                                                       ▼
 GT + policy + eval config ─▶ align ─▶ forensics ─▶ metrics ─▶ attribution ─▶ results store ─▶ report / compare / regress
```

### 14.2 Modules: one Python package with a CLI (`rb`)

| Module | Responsibility |
|---|---|
| `schemas` | Versioned JSON Schemas + pydantic models for GT, policy, candidate manifest, stage artifacts, run manifest and results. |
| `geometry` | The **single** implementation of coordinate frames (display vs user space, `/Rotate`, CropBox, px@DPI, rotated content), polygons, ink masks and projection to GT words. Property-tested heavily, because this is the classic source of evaluator bugs. |
| `dataset` | Registry, manifests, hash checks, splits, automatic condition measurement. |
| `gt` | Annotation import/export, validators, ink-mask derivation, review sheets, agreement metrics, policy resolution. |
| `forensics` | Candidate-independent PDF inspector: two renderers; text including invisible text; image layers and masks; vector paths; annotations and optional content (OCG); metadata/XMP; attachments; thumbnails; structure tree; incremental revisions; raw stream scan. Also produces a **PII-surface inventory** of the *input* documents, which is useful to tool developers right away. |
| `align` | Spatial and text alignment, Hungarian assignment, cluster alignment. |
| `metrics` | Pure functions per metric family, with documented definitions and hand-computed fixtures. |
| `attribution` | Evidence assembly, root-cause rules, counterfactual decomposition. |
| `runner` | YAML experiment specs, oracle-input materialisation, repeats, timeouts, logs, run manifests, offline submission. |
| `mocks`, `reference` | Fault-injecting mock candidates; a reference (oracle) redactor; a reference OCR baseline. |
| `synth` | Clean twins, degradation ladders, and later a synthetic document generator. |
| `report` | Results store, HTML report, gallery, compare, regression. |

### 14.3 Reproducibility record

Each run writes a manifest containing:
- run ID;
- benchmark version and git commit;
- dataset version and manifest hash;
- GT and policy versions; evaluation-config hash; experiment spec;
- candidate name, version (commit/tag), manifest hash, configuration, seeds, model IDs (from `run_info.json`);
- environment: OS, Python, pinned library, renderer and OCR versions;
- SHA-256 of every input and artifact;
- timestamps and logs.

Re-scoring is exactly reproducible, because the evaluator's own determinism is tested. Re-running a tool is reproducible to the extent the tool is deterministic, which §6 measures.

### 14.4 Technology for the benchmark itself (not for candidates)

- Python 3.11.
- PDF handling: PyMuPDF (already installed), pypdfium2 (second renderer), pikepdf/qpdf (raw objects, revisions, validation).
- Imaging and geometry: numpy + OpenCV (masks, registration), shapely (polygons).
- Matching: rapidfuzz (alignment).
- Schemas: pydantic + JSON Schema.
- Results: DuckDB + Parquet.
- Reports: Jinja2 + Vega-Lite/Plotly (static HTML).
- Tests: pytest + hypothesis.
- One pinned **local** measurement OCR: PP-OCRv6 (PaddleOCR on ONNX Runtime, own venv locked by `requirements-ocr.txt`), chosen over Tesseract after a comparison on the seed documents; Tesseract stays selectable (`RB_OCR_ENGINE=tesseract`). Everything runs locally, and no document leaves the machine unless you approve otherwise (Q12).
- PyMuPDF is AGPL. That is fine for internal use, but needs review if the benchmark is ever distributed.

### 14.5 Alternatives considered

| Alternative | Decision |
|---|---|
| **In-process Python plugin API** | Rejected: it couples tools to the benchmark's runtime, dependencies and language. Files work for any stack and for teams running their own tools. |
| **REST/service API** | Not now: it forces every tool to run a server. It can be added later as one adapter type. |
| **End-to-end-only harness** | Kept only as the minimum tier (T0): on its own it cannot explain failures. |
| **Box-IoU spatial scoring** | Replaced by ink coverage + spill: IoU penalises safe over-coverage and ignores which direction a box is wrong in. |
| **Character-offset-anchored GT** | Replaced by word polygons: offsets break across tools' reading orders and tokenisations. |
| **Database / web app / orchestration framework** | Unnecessary at this scale. |

### 14.6 Deliberate simplicity

- No server, queue or custom UI beyond static reports; annotation uses an existing tool.
- Experiments are YAML; storage is the filesystem.
- One package, one CLI.

---

## 15. Development plan

Each phase ends with a usable deliverable and an exit check.

| Phase | Content | Exit criteria |
|---|---|---|
| **0. Decisions & specs** | Resolve §16; methodology document; metric definitions; annotation guidelines v0; schemas v0; policy profile v0 | Policy v0 and the data-governance decision signed off |
| **1. Foundations** | Package + CLI skeleton; schemas and validators; dataset registry for the 5 PDFs (hashes, auto-measured page conditions); geometry library + property tests; forensics inspector run on the originals (PII-surface inventory) | `rb dataset validate` and `rb inspect` work; geometry tests green |
| **2. Ground truth v0.1** (runs alongside phase 1) | Words, incl. rotated pages and handwriting; mentions, entities, attributes, classified spans, conditions; validators; review sheets; double annotation of ≥ 20% + adjudication | Validators pass; agreement gates met; `dataset v0.1.0` and `gt v0.1.0` tagged |
| **3. Output evaluator + mocks** | Rendering/registration; L1 pattern survival + reader OCR; L2 forensics; L3 enhancement; over-redaction; integrity; effective detection. Mocks: perfect; null; fault injectors (see list below) | Every injected fault is detected with the **expected category**; perfect mock = all SAFE and zero collateral; null mock = all LEAK |
| **4. Component metrics** | Text acquisition, detection, resolution, localization + transform diagnosis, plan fidelity, pseudonymisation, determinism | Fixture tests + stage-level mock acceptance tests pass |
| **5. Attribution & reporting** | Evidence chain, categories, funnel, decomposition; results store; HTML report + gallery; compare; regression promote | The report answers every question in §12 for the mock suite |
| **6. Runner & integration** | Manifest + adapter contract; experiment specs (e2e, oracle_*, determinism repeats); oracle inputs; offline submission; `rb candidate check`; adapter guide; reference redactor | Mocks run as external processes and score identically to in-process |
| **7. Controlled inputs** | Clean twins and degradation ladders for the seed set | Mock and baseline results move monotonically along the ladders as designed |
| **8. Real candidates** | Onboard each tool as it becomes ready; baseline runs; GT dispute loop | — |
| **9. Dataset growth** | Stratified real documents; synthetic generator; sealed test split; regression operations | — |

Fault injectors for the phase-3 mocks:
- **Detection:** dropped types, boundary shrink/extend, type swap, homograph false positives.
- **Geometry:** box shift, scale, y-flip, rotation-unaware coordinates, page offset; partial and dilated boxes.
- **Removal:** overlay-only, text layer kept, metadata kept, incremental-save leak.
- **Pseudonyms:** inconsistent, gender-flipped and colliding pseudonyms.
- **Run-level:** seeded nondeterminism, corrupted output.

---

## 16. Open questions (need your decision before or during Phase 0)

These change the *policy profile*, not the architecture: GT records facts, so none of them blocks building the framework.

1. **Whose PII must go?**
   - All private persons: claimant, family, witnesses, colleagues and alleged perpetrators, the JP.
   - Or *also* professionals: examining experts, treating clinicians, lawyers, insurer staff, typists' initials.
2. **Organisations** (employer, clinics, schools, law firms, insurers) and their addresses, phones and ABNs: keep, remove or replace? The employer is a strong quasi-identifier.
3. **Dates:** does "dates are not redacted" include **date of birth** and ages? DOB is one of the strongest identifiers.
4. **Quasi-identifiers:** suburbs and towns, birthplace, children's ages, occupation, height and weight. In scope?
5. **Identifiers:** claim numbers, legal references, provider numbers, file paths, filenames. All MUST?
6. **Non-text PII:** signatures, handwritten initials, avatar and photo images. Must they be removed?
7. **Action per type:** which types must be pseudonymised vs removed, or may a tool choose?
8. **Pseudonym consistency scope:** per document, per matter (DPI and IME concern the same claimant), or global? Must mappings be stable across runs?
9. **"Nationality":** which attribute exactly (nationality, ethnicity, cultural origin of the name), and who determines GT for it? Must Indigenous identity (central to one claim) and family relationships (shared surnames) be preserved?
10. **Coreference scope:** PII-bearing mentions only (proposed), or also role nouns and pronouns?
11. **Output requirements:** must outputs keep a searchable text layer? Is image-only output acceptable? Must page count and size be preserved?
12. **Data governance:**
    - Are these real people's documents?
    - May the PDFs and GT live in this git repo or any remote?
    - Who may see GT and reports?
    - Should the test set be sealed from tool developers (are they different people)?
    - May any cloud service (e.g. OCR for pre-annotation) ever see documents?
    - **Answer so far (2026-10-05, from the project owner):** the client has clearly allowed the five representative PDFs to be sent to any third party and to be published publicly (the form of the permission is not recorded here). So the PDFs may be sent to external services, including cloud OCR and language models, and may live in a public remote. Still open: which artefacts are published (the PDFs, the machine drafts, the ground truth), who may see reports, and whether the test set is sealed from tool developers. Until those are decided the ignore rules in `.gitignore` stay as they are.
13. **Annotation resources:** who annotates and who adjudicates? Is double annotation feasible?
14. **More data:** which additional real documents can be obtained, how many, and under what authorisation?
15. **Integration:** may we require each tool to provide a CLI adapter, and ideally to accept injected text, mentions or regions?
16. **Over-redaction tolerance:**
    - Is damaging clinical or legal content a hard failure?
    - Is any margin acceptable, e.g. redacting a whole line?

---

## Verification (how we prove the benchmark itself is correct)

- **Metrics:** unit tests with hand-computed fixtures for every metric.
- **Geometry:** property-based round-trip tests across all frames × rotations × scales.
- **Mock acceptance suite:**
  - every injected fault is detected *and* attributed to the expected category;
  - the perfect and null mocks bound the scale.
- **Evaluator determinism:** scoring the same artifacts twice produces byte-identical results tables.
- **GT releases:** gated by the validators and agreement thresholds.
- **Measurement OCR:** calibrated on the unredacted originals.
- **End-to-end smoke test on the five PDFs:**
  - with the perfect mock, the null mock and a reference redactor (true removal + rasterisation);
  - expected: reference = all SAFE, with no L2/L3 findings.

## Proposed repository layout

```
redaction_benchmark/
  benchmark/      # package: schemas geometry dataset gt forensics align metrics attribution runner mocks reference synth report cli
  schemas/        # versioned JSON Schemas
  policies/       # redaction policy profiles (YAML)
  datasets/       # manifests + datasheets (no PDFs)
  gt/             # per-document GT JSON + annotation guidelines
  experiments/    # e2e, oracle_*, determinism, ladders
  candidates/     # candidate manifests
  adapters/       # optional in-repo adapters
  docs/           # methodology, metric definitions, adapter guide
  tests/          # unit, property, mock-acceptance
  runs/ results/  # generated, git-ignored
```

Source PDFs stay in an external, access-controlled data directory referenced by hash. `representative_pdfs/` is moved out of git once the storage decision (Q12) is made.
