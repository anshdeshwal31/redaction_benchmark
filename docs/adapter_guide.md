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
| `{policy}` | Absolute path of the redaction policy the run was started with (`rb run --policy`, `policies/*.yaml`). Optional: use it if your tool can follow a policy file. `run.json` records the policy's name, version and file hash, and whether your tool received it (through `{policy}`, or oracle inputs built under it); `rb score` uses the same policy unless told otherwise. If your tool received it, the results, report and board warn when the run is scored under another policy or an edited file. If it did not, your tool is judged by whichever policy the scores use, with no warning, and can be rescored under another policy without a new run. |
| `{matter}` | The document's matter id from the dataset manifest, `""` when it has none. Documents of one matter belong to the same case (see §2.1). A matter id is a plain code: letters, digits, `_`, `.` and `-`. |
| `{env:NAME}` | The value of environment variable `NAME` (`[A-Z][A-Z0-9_]*`), e.g. your tool's own interpreter (see §2.1). The run stops before anything starts if it is not set. |
| `{python}`, `{repo}` | The benchmark's Python interpreter and repository root (convenience for Python adapters). |

The command runs through the shell (`cmd.exe` on Windows). Every path and value reaches your tool
unchanged, as one argument: the runner quotes it, so spaces, apostrophes and characters such as
`&`, `|` or `^` are safe, and you must not add quotes of your own around a placeholder. On Windows
`cmd.exe` would still change a value containing a double quote, `%`, `!` or a line break even inside
quotes, so such a value is refused with an error that names it: a shared value (the policy path, a
`{env:NAME}` value, the runs directory) stops the run before anything starts, and a document whose
own paths hold one is recorded as not run.

Your adapter must write **`output.pdf`** (the redacted document) **every time**, even when your
tool's own safety check would hold the document back: the benchmark grades every output, bad ones
included, and grades the safety check separately (see `release` below). Everything else is
optional, but each stage file you leave out is a stage the benchmark cannot grade, so a bottleneck
there stays invisible. Write only what your tool really produces:

| File | Schema | Stage it lets the benchmark score |
|---|---|---|
| `output.pdf` | — | End-to-end safety, over-redaction, integrity (always) |
| `text.json` | `schemas/text.schema.json` | OCR / text extraction quality |
| `mentions.json` | `schemas/mentions.schema.json` | PII detection: recall, precision, boundaries, types |
| `entities.json` | `schemas/entities.schema.json` | Entity / alias resolution |
| `plan.json` | `schemas/plan.schema.json` | Spatial accuracy, coordinate-bug diagnosis, plan-vs-output fidelity |
| `pseudonyms.json` | `schemas/pseudonyms.schema.json` | Replacement mapping (consistency is checked on the output itself) |
| `run_info.json` | `schemas/run_info.schema.json` | Version, configuration, model ids, seed (recorded for reproducibility), and `release`: your own safety verdict |

`run_info.json` → `release` is `{"released": true|false, "checks_failed": ["<your reason codes>"]}`. The
benchmark compares it with its own verdict on the output: correct release, correct hold, false
alarm (held back but clean) or unsafe release (released but not clean).

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
frame: display_pt                # or unrotated_pt, pdf_user, or image_px (then also: dpi: 300)
seed_flag: --seed
timeout_s: 900
```

`label_map` is fixed per tool version. Labels missing from it are reported as invalid
(`rb candidate check`, and `invalid_artifacts` in the run), but `mentions.json` is still scored:
those mentions count for detection, with no type, so never as the right type. Artifacts not listed
in `provides` are ignored, even if present, except `run_info.json`: it is always read, validated
and its `release` verdict graded. `version_command` runs through the shell like `command` and
understands `{python}`, `{repo}` and `{env:NAME}` (double any other brace, as in `command`); the
first line it prints is the version.

### 2.1 Your tool's interpreter, and one fake per person across a matter

Each tool runs in its own environment, which the benchmark does not know. Point the manifest at
it through an environment variable named `RB_<TOOL>_PYTHON`, as the measurement OCR already
does with `RB_PPOCR_PYTHON`:

| Variable | Tool |
|---|---|
| `RB_MLREDACT_PYTHON` | mlredact (tool 2), `candidates/mlredact-client.yaml` |
| `RB_REDACTOR_PYTHON` | tool 1 (name reserved; no manifest yet) |
| `RB_PRIVATEGPT_PYTHON` | the client's tool (name reserved; no manifest yet) |

```yaml
version_command: '{env:RB_MLREDACT_PYTHON} -m mlredact.cli.main bench-version'
command: '{env:RB_MLREDACT_PYTHON} -m mlredact.cli.main bench --input {input} --workdir {workdir} --scope-id {matter} {seed}'
```

If the variable is not set, `rb run` stops before running anything and names the variable; set it
to the tool's interpreter or path (for example `export RB_MLREDACT_PYTHON=/path/to/backend/.venv/bin/python`).

The client wants one fake per person across all documents of a matter (Q8). The benchmark runs
documents one at a time, so pass `{matter}` to your tool and derive each person's fake from it
(for example a keyed hash of the matter and the person), or keep a store per matter id. The
benchmark checks this across the documents of each matter recorded in the GT (the `matters`
table, `docs/metrics.md`); a document whose dataset matter differs from its GT's, or that has a
matter on one side only, is listed in the report.

How the fakes are drawn matters too. The evaluator compares the output with the original glyph by
glyph, and a correct fake written in the original's font, at its size and position, can line up
with the original's strokes closely enough to be judged `PARTIAL`: an open calibration issue
(`docs/metrics.md`). The benchmark's own ceiling mock draws its fakes a little smaller than the
line for this reason.

## 3. Coordinates

All polygons are lists of `[x, y]` points. The canonical frame `display_pt` is PDF points
of the page **as displayed**: the CropBox after applying `/Rotate`, origin top-left, y
downwards. You may instead declare `unrotated_pt` (the CropBox before `/Rotate`, origin
top-left, y downwards; the frame PyMuPDF reports words and drawings in), `pdf_user` (raw PDF
user space, y upwards) or `image_px` with a `dpi` (pixels of a rendering of the displayed
page). Declare the frame
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
| `oracle_regions` | + `plan.json` | apply exactly these regions; each names its mention, so you know what kind of fake to draw |

In `plan.json` each region's `action` is the policy's: `replace` where a fake is required (the
client's rules require it for every redacted item), `remove` where removal is allowed. Covering a
`replace` region with a box or blank paper is scored as `NOT_REPLACED`.

Injected files use the same schemas, in `display_pt`. Only mentions required by the
current policy are injected.

Injected inputs do not always come from ground truth. An experiment with
`inject_from: "run:<run_id>"` feeds you another candidate's artifacts instead (stage
swapping: for example tool A's OCR into your detector). They arrive in the same format
(`display_pt`, benchmark labels), but they may be wrong: use them as you would your own
upstream output, without second-guessing them from the page.

## 5. Rules

- Ground truth is never placed in your working directory, and you must not try to obtain it.
- Do not special-case benchmark documents.
- Never network-send documents unless the project owner has approved it in writing.
- Exit non-zero on failure; the benchmark records the exit code. Without an `output.pdf` every
  required mention is `NO_OUTPUT`, unprotected in the strict view; an `output.pdf` you did write is
  graded whatever the exit code, like every output.
- A run longer than `timeout_s` is stopped together with every process it started, and anything
  written after that is not graded.

## 6. Checking your adapter

```bash
rb candidate check path/to/workdir --manifest candidates/my-tool.yaml
rb run --candidate candidates/my-tool.yaml --dataset synthetic@0.1.0
rb score <run_id> && rb report <run_id>
```

The synthetic dataset is fictional and safe to share; use it to debug the adapter before
running on real documents.
