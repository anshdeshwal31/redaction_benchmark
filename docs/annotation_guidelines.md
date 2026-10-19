# Ground-truth annotation guidelines (v0)

Ground truth records **facts**: what each span is, who it refers to, and where it is. It
never records whether something should be redacted; the policy profile decides that. So
annotate every person name, including doctors and lawyers, every date, every organisation.

Examples refer to roles and are paraphrased; never copy real names into this document.

## 1. Words

- Transcribe what a careful human reads on the page, including letterhead text that exists
  only as an image, avatar initials, and handwriting. Mark `source` (`text_layer`,
  `raster`, `vector`, `handwriting`) and `legibility`.
- One word = one whitespace-separated token as printed. Keep attached punctuation in the
  word ("Smith," stays one word); mentions use a character range to exclude it.
- Polygons follow the visible ink tightly. On skewed or rotated content use rotated
  quadrilaterals and set `angle` (degrees clockwise from left-to-right; text reading
  bottom-to-top is 270).
- Reading order follows how a person reads the page; running headers and footers come
  first and last respectively.

## 2. Mentions

A mention is one occurrence of a PII fact. Annotate **every** occurrence, including
repeats in running headers and footers, tables and annexures.

| Situation | Rule |
|---|---|
| Titles (Mr, Ms, Dr, A/Prof) | Not part of the name. Add them as `optional_words`. |
| Possessive ("Smith's") | Required span is the name only (character range); the `'s` is excluded. |
| "SURNAME, Given" / all capitals | One mention; components tagged (`surname`, `given`). |
| Nickname, former name ("née X") | Separate mentions of the same entity; components `nickname` / `former_surname`. |
| Initials ("J. S.", typist initials in a reference) | Mention with component `initials`. |
| Name inside a reference code (`Name:abc:123`) | Nested mention (`parent` = the reference) with a character range. |
| Name inside an email address | Nested PERSON_NAME mention on the local part, parent = the EMAIL. |
| Name inside an organisation name | Nested PERSON_NAME mention, parent = the ORGANISATION. |
| Multi-box form address (unit / street / suburb / state / postcode) | One ADDRESS mention with several segments; field labels are not part of it. |
| Name split across lines, table cells or form rows | One mention, several segments. |
| Signature, handwritten initials, photo or avatar | Region (`signature`, `initials`, `avatar`, ...) plus a region-only mention (SIGNATURE / HANDWRITTEN_MARK / IMAGE_PII) linked to the entity. |
| Handwritten name or place in block capitals | Words with `source: handwriting`, annotated like typed text. |
| Dates | Annotate with subtype `dob`, `event` or `document` (the policy keeps dates for now, but DOB is an open question). |

## 3. Entities

- One entity per real-world person or organisation, with a **stable id across documents
  of the same matter** (e.g. the same claimant in two reports).
- `role` from the fixed list (claimant, family_member, treating_clinician, examining_expert,
  legal_representative, insurer_staff, employer_staff, witness, alleged_perpetrator,
  official, admin_staff, ...).
- Attributes need evidence: `gender` from a form checkbox, honorific or explicit statement,
  not from pronouns alone (source documents contain pronoun errors). Nationality / cultural
  background only when stated in the document, pending policy question Q9.
- People who share a first name or a surname are different entities; relate spouses and
  family with `relations`.

## 4. Classified (protected) spans

Annotate content whose loss is harmful even though it is not PII:

- medical terms and medications, medical eponyms ("Hoover's sign"), legal citations;
- **homographs**: ordinary words equal to a surname in the same document ("good", "low",
  a month name); classify *every* such occurrence that is not the name;
- form boilerplate, example addresses printed on forms, already-anonymised placeholders.

## 5. Document surface

Record PII outside the page content as `surface` items: metadata fields (Author, Title,
Keywords), XMP, outline titles, and the source filename.

## 6. Process and quality control

1. `rb gt draft` creates a machine draft (text layer / measurement OCR / regex). Every
   pre-labelled mention is tagged `prelabel` and has no entity: confirm, fix or delete it.
2. Complete words, mentions, entities, regions, classified spans and surface items.
3. `rb gt validate <file> --source <pdf> --policy policies/au-medicolegal-v0.yaml` must show
   no errors; resolve or justify every warning (`unannotated-repeat`,
   `homograph-unclassified`, `no-components`).
4. `rb gt review` renders overlay sheets; a second annotator checks them.
5. At least 20% of pages are annotated twice independently (always including the hardest
   pages, e.g. narrative annexures), then adjudicated. Measure agreement with
   `rb gt agree a.gt.json b.gt.json` (words are matched geometrically, so the two
   annotators may box and number words differently). Release gates: tolerant mention F1
   ≥ 0.95 and entity-link B³ F1 ≥ 0.95. The command lists mentions only one annotator
   found and mentions typed differently, which is the adjudication agenda.
6. Run `rb selfcheck --dataset <name>@<version> --docs <doc>` on every newly annotated
   document. It runs the fault-injecting mocks on the real document and checks that the
   evaluator reaches the expected verdicts; a failure means a calibration problem to
   investigate before the document's results are trusted.
7. Disputes raised by benchmark results are adjudicated by the benchmark owner; fixes
   create a new GT version with the reason recorded.

The ground truth for the five seed documents (`gt_version 0.1-ai`) was made by an AI
annotator (claude-opus-5-5) that corrected the machine drafts under these guidelines. Steps 4
and 5 have not happened yet, so it is not a release.

Real-document GT contains PII. Keep it under `$RB_DATA_DIR/gt` (git-ignored). The client has
allowed the five representative PDFs to be shared and published, but whether their GT and
drafts may be published is still open (PLAN.md §16 Q12), so the ignore rules stay.
