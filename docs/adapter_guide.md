# Connecting a redaction tool to the benchmark

This guide is for developers of a candidate redaction tool. The benchmark never imports
your code and never sees your internals. It runs one shell command per document and
reads the files that command writes.

## 1. The contract

The benchmark calls your adapter with:

| Placeholder | Meaning |
|---|---|
| `{input}` | Path to `input.pdf`. The name is always neutral; the original filename is never passed. |
| `{workdir}` | Empty directory to write your outputs into. |
| `{mode}` | `e2e`, `oracle_text`, `oracle_mentions`, `oracle_entities` or `oracle_regions`. |
| `{inject_dir}` | For oracle modes: a directory with benchmark-supplied stage inputs (see §4). `""` otherwise. |
| `{seed}` | Expands to `<seed_flag> <n>` if your manifest declares a `seed_flag`, otherwise empty. |
| `{python}`, `{repo}` | The benchmark's Python interpreter and repository root (convenience for Python adapters). |

Your adapter must write **`output.pdf`** (the redacted document). Everything else is
optional. Write only what your tool really produces:

| File | Schema | Stage it lets the benchmark score |
|---|---|---|
| `output.pdf` | — | End-to-end safety, over-redaction, integrity (always) |
| `text.json` | `schemas/text.schema.json` | OCR / text extraction quality |
| `mentions.json` | `schemas/mentions.schema.json` | PII detection: recall, precision, boundaries, types |
| `entities.json` | `schemas/entities.schema.json` | Entity / alias resolution |
| `plan.json` | `schemas/plan.schema.json` | Spatial accuracy, coordinate-bug diagnosis, plan-vs-output fidelity |
| `pseudonyms.json` | `schemas/pseudonyms.schema.json` | Replacement mapping (consistency is checked on the output itself) |
| `run_info.json` | `schemas/run_info.schema.json` | Version, configuration, model ids, seed (recorded for reproducibility) |

Stages you do not expose are reported as **N/A**, never as zero. A tool that only writes
`output.pdf` still gets the full end-to-end evaluation, effective detection measured from
the output pixels, and resolution inferred from its pseudonyms where it replaces names.

## 2. The manifest

Create `candidates/<name>.yaml`:

```yaml
schema: rb.candidate/0.1
name: my-tool
version_command: "git -C /path/to/my-tool rev-parse --short HEAD"   # or: version: "1.4.0"
command: 'python /path/to/my-tool/adapter.py --in {input} --out {workdir} --mode {mode} --inject {inject_dir} {seed}'
provides: [output_pdf, mentions, plan, run_info]
accepts_injection: []            # any of: text, mentions, entities, regions
label_map:                       # your labels -> benchmark types (published with results)
  PER: PERSON_NAME
  ADDR: ADDRESS
  PHONE_NUMBER: PHONE
frame: display_pt                # or pdf_user, or image_px (then also: dpi: 300)
seed_flag: --seed
timeout_s: 900
```

`label_map` is fixed per tool version. Labels missing from it make `mentions.json` invalid.
Artifacts not listed in `provides` are ignored, even if present.

## 3. Coordinates

All polygons are lists of `[x, y]` points. The canonical frame `display_pt` is PDF points
of the page **as displayed**: the CropBox after applying `/Rotate`, origin top-left, y
downwards. You may instead declare `pdf_user` (raw PDF user space, y upwards) or
`image_px` with a `dpi` (pixels of a rendering of the displayed page). Declare the frame
honestly: the benchmark converts it, and also checks whether a known coordinate bug
(DPI scale, y-flip, unrotated frame, MediaBox origin, page offset) explains misplaced
regions.

Mentions can be anchored by `regions` (preferred), by `tokens` (ids from your
`text.json`), or by a character `span` into `page_texts[page]`. For spans, `page_texts[page]`
must equal your page's token texts joined by single spaces, in `text.json` order.

## 4. Oracle modes (optional, strongly recommended)

If your tool can accept stage inputs, list them in `accepts_injection`. The benchmark can
then isolate your stages by giving them perfect upstream input from ground truth:

| Mode | Files in `{inject_dir}` | Your tool should |
|---|---|---|
| `oracle_text` | `text.json` | skip OCR and use these tokens |
| `oracle_mentions` | `text.json`, `mentions.json` | skip detection and redact exactly these mentions |
| `oracle_entities` | + `entities.json` | also use these clusters for pseudonyms |
| `oracle_regions` | `plan.json` | apply exactly these regions |

Injected files use the same schemas, in `display_pt`. Only mentions required by the
current policy are injected.

## 5. Rules

- Ground truth is never placed in your working directory, and you must not try to obtain it.
- Do not special-case benchmark documents.
- Never network-send documents unless the project owner has approved it in writing.
- Exit non-zero on failure; the benchmark records the error and scores the document as
  unprotected in the strict view.

## 6. Checking your adapter

```bash
rb candidate check path/to/workdir --manifest candidates/my-tool.yaml
rb run --candidate candidates/my-tool.yaml --dataset synthetic@0.1.0
rb score <run_id> && rb report <run_id>
```

The synthetic dataset is fictional and safe to share; use it to debug the adapter before
running on real documents.
