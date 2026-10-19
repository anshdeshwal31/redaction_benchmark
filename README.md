# redaction_benchmark

A neutral evaluation framework for PII redaction tools working on Australian medico-legal
PDFs (mostly scans). It answers which tool is better, where it is better, which stage a
failure started in, and whether the tool behaves reproducibly. The design is in
[PLAN.md](PLAN.md).

The benchmark judges the **actual output document**, never a tool's claims. A required
mention is only safe if its original content cannot be seen in any viewer, recovered from
the file (text layer, image layers, vector paths, earlier revisions, metadata) or
recovered by contrast enhancement. When a tool also exposes intermediate results (OCR,
detections, entity clusters, planned regions), each stage is scored and failures are
traced to the stage that caused them.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest          # 271 tests, about 18 minutes
```

Optional: the measurement OCR, used to draft GT words on pages without a text layer. The
default engine is PP-OCRv6 (PaddleOCR on ONNX Runtime, CPU). PaddleOCR pins numpy, PyYAML
and OpenCV to versions that conflict with this package's pins, so it gets its own venv:

```bash
py -3.11 -m venv .venv-ocr
.venv-ocr/Scripts/python -m pip install -r requirements-ocr.txt   # model weights download on first use
```

Set `RB_PPOCR_PYTHON` if that interpreter lives elsewhere. `RB_OCR_ENGINE=tesseract` selects
Tesseract instead (set `RB_TESSERACT` if it is not on PATH). On the five seed documents
PP-OCRv6 read text that Tesseract lost (light-coloured running headers, scan artefacts,
pages rotated inside the image) at about a tenth of the speed (~30 s per page).

## Workflow

```bash
rb synth dataset --version 0.1.0                 # fictional fixtures in 9 encodings, with exact GT
rb run --candidate candidates/my-tool.yaml --dataset synthetic@0.1.0 [--experiment experiments/oracle-regions.yaml]
rb score <run_id> [--oracle oracle_regions=<oracle_run_id>]
rb report <run_id> [--shareable]                 # static HTML: scorecard, components, slices, funnel, gallery
rb query <run_id> "SELECT type, status, count(*) FROM mentions WHERE requirement='MUST' GROUP BY ALL"
rb compare <run_a> <run_b>                       # mention-level fixed / regressed / category changed
rb regress promote <run_id> <doc_id> <mention_id> ; rb regress check <run_id>
rb synth ladder --factor dpi --version 0.1.0 --source synthetic@0.1.0   # controlled degradation ladder
rb board <run_id> <run_id> ...                   # comparison board: scorecard, heatmaps, decomposition, ladders
```

Isolating stages: run the same candidate with `experiments/oracle-text.yaml`,
`oracle-mentions.yaml` and `oracle-regions.yaml` (perfect upstream input from GT), score
the end-to-end run with `--oracle oracle_text=<run>` etc., and put all of them on one
board to see what share of the leak rate each stage causes. Set
`inject_from: "run:<run_id>"` in an experiment to feed one candidate's artifacts into
another (stage swapping).

Real documents:

```bash
rb dataset ingest --name seed --version 0.1.0 --doc DPI=path.pdf ... --matter DPI=M1
rb inspect path.pdf                              # every content surface: text, layers, paths, metadata, revisions
rb gt draft --dataset seed@0.1.0                 # machine draft for annotators (written to data/, git-ignored)
rb gt validate data/gt/DPI.gt.json --source path.pdf --policy policies/au-medicolegal-v0.yaml
rb gt agree annotator_a.gt.json annotator_b.gt.json   # inter-annotator agreement and release gates
rb selfcheck --dataset seed@0.1.0 --docs DPI     # does the evaluator reach the right verdicts on this document?
```

## Layout

| Path | Contents |
|---|---|
| `benchmark/geometry` | Coordinate frames (`display_pt` canonical), polygons, exact rasterisation |
| `benchmark/schemas` | Versioned contracts (exported to `schemas/*.schema.json`) |
| `benchmark/dataset`, `benchmark/forensics` | Registry with automatic condition measurement; PDF inventory |
| `benchmark/gt` | GT IO, policy resolution, validators, drafts and review sheets |
| `benchmark/evaluate` | Output evaluation: visible / recoverable / latent leakage, over-redaction, integrity |
| `benchmark/metrics` | Text acquisition, detection, resolution, localization, fidelity, pseudonyms, determinism |
| `benchmark/attribution.py`, `benchmark/score.py` | Root-cause attribution; per-document scoring |
| `benchmark/runner`, `benchmark/report` | Running adapters, oracle inputs, scoring runs; Parquet store, HTML report, compare, regression |
| `benchmark/mocks`, `benchmark/synth` | Fault-injecting mock candidates; synthetic documents, clean twins, degradation ladders |
| `policies/` | Redaction policy profiles (draft; open questions are scored as MAY) |
| `candidates/`, `experiments/` | Candidate manifests (mocks for now); experiment specs |
| `docs/` | [Adapter guide](docs/adapter_guide.md), [metric definitions](docs/metrics.md), [annotation guidelines](docs/annotation_guidelines.md) |

## Data handling

Source PDFs, real-document GT, run outputs and reports can contain personal information.
`data/`, `runs/`, `results/`, `representative_pdfs/` and all PDFs are git-ignored. PDFs are
stored by hash under `data/store/`; dataset manifests contain hashes and measured
conditions only. Reports of real documents include image crops; use `--shareable` to mask
the PII spans before circulating one.

## Status

Built: the full evaluation pipeline, validated against fault-injecting mocks on synthetic
documents in every encoding seen in the seed set.

Ground truth for the five seed documents exists as `gt_version 0.1-ai`: an AI annotator
(claude-opus-5-5) corrected the machine drafts. All five validate with 0 errors and each has
a review sheet in `data/gt/<DOC>.review.md`. It is **not yet human-reviewed**, so it is not a
release. It lives in the git-ignored `data/` folder. The evaluator self-check has been run on
all five (details in [docs/metrics.md](docs/metrics.md)):

| Doc | Required mentions | Self-check |
|---|---|---|
| TRN | 4 | 10 pass, 1 fail (partial-half) |
| DPI | 33 | 10 pass |
| IME | 59 | 10 pass |
| PSY | 113 | 11 pass |
| NOC | 303 | 8 pass, 2 fail (reference-true, reference-raster) |

The three failures are evaluator calibration issues on real scans and are open.

**Not yet done**: human review and double annotation of the seed GT, the three calibration
fixes (the first two change the evaluator, so they need a new evaluation version, eval-0.3,
and a rescore), decisions on the open policy questions (PLAN.md §16; the client
questionnaire for them is [docs/client_policy_decisions.docx](docs/client_policy_decisions.docx)),
and real candidate tools.
