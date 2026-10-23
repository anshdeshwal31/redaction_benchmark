# redaction_benchmark

A neutral evaluation framework for PII redaction tools working on Australian medico-legal
PDFs (mostly scans). It answers which tool is better, where it is better, which stage a
failure started in, and whether the tool behaves reproducibly. The design is in
[PLAN.md](PLAN.md); how the code is put together, and where it differs from the plan, is in
[ARCHITECTURE.md](ARCHITECTURE.md).

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
.venv/Scripts/python -m pytest          # 403 tests, about 20 minutes
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
rb score <run_id> [--oracle oracle_regions=<oracle_run_id>]   # under the run's own policy unless --policy
rb report <run_id> [--shareable]                 # static HTML: scorecard, components, slices, funnel, gallery
rb query <run_id> "SELECT type, status, count(*) FROM mentions WHERE requirement='MUST' GROUP BY ALL"
rb compare <run_a> <run_b>                       # mention-level fixed / regressed / category changed
rb regress promote <run_id> <doc_id> <mention_id> ; rb regress check <run_id>
rb synth ladder --factor dpi --version 0.1.0 --source synthetic@0.1.0   # controlled degradation ladder
rb board <run_id> <run_id> ...                   # comparison board: scorecard, heatmaps, decomposition, ladders
```

Isolating stages: run the same candidate with `experiments/oracle-text.yaml`,
`oracle-mentions.yaml`, `oracle-entities.yaml` and `oracle-regions.yaml` (perfect upstream
input from GT), score the end-to-end run with `--oracle oracle_text=<run>` etc., and put all
of them on one board to see what share of the failure rate (leaks plus replacements not made)
each stage causes. Set `inject_from: "run:<run_id>"` in an experiment to feed one candidate's
artifacts into another (stage swapping).

The ceiling: `candidates/mock-reference-replace.yaml` is a fake tool that does everything the
client's rules ask, under whatever policy the run uses: a fake of the right kind for every
required item (one fake per person across a matter, its own fake for each phone and Medicare
number), a drawn stand-in for every signature or image, and a searchable text layer. It is clean
on all nine synthetic encodings; put it on the board next to the real tools as the best result
reachable. Every mock manifest passes `--policy {policy}`, so the mocks follow the policy of the
run they are part of. `run.json` records that policy (name, version, file hash) and whether it
reached the tool; `rb score` uses it unless `--policy` says otherwise. When the tool was given the
policy (its command passes `{policy}`, or an oracle experiment built its inputs under it), the
results, report and board warn if the scores use another policy or an edited file. A tool that
never reads the policy, such as tool 2, follows its own rules whatever the run was started with: it
gets no warning and can be rescored under another policy without a new run.

Real tools run in their own environments. Their manifests reach the interpreter through an
environment variable named `RB_<TOOL>_PYTHON`, as the measurement OCR does with `RB_PPOCR_PYTHON`:
`RB_MLREDACT_PYTHON` for tool 2 (`candidates/mlredact-client.yaml`; run end to end on the synthetic
dataset on 2026-10-22, results kept outside the repository, not yet on the seed documents);
`RB_REDACTOR_PYTHON` (tool 1) and `RB_PRIVATEGPT_PYTHON` (the client's tool) are reserved for
their manifests, which do not exist yet. `rb run` stops before running anything, naming the
variable, when one is not set. The command placeholders (`{policy}`, `{matter}` for one fake per
person across a matter, `{env:NAME}`) and how values reach the tool are in the
[adapter guide](docs/adapter_guide.md).

Real documents:

```bash
rb dataset ingest --name seed --version 0.1.0 --doc DPI=path.pdf ... --matter DPI=M1
rb inspect path.pdf                              # every content surface: text, layers, paths, metadata, revisions
rb gt draft --dataset seed@0.1.0                 # machine draft for annotators (written to data/, git-ignored)
rb gt validate data/gt/DPI.gt.json --source path.pdf --policy policies/au-medicolegal-v1.yaml
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
| `benchmark/ocr.py`, `benchmark/ocr_split.py`, `benchmark/_ppocr_worker.py` | Measurement OCR: engine selection, glued-word splitting, the PP-OCRv6 worker that runs in `.venv-ocr` |
| `benchmark/attribution.py`, `benchmark/score.py` | Root-cause attribution; per-document scoring |
| `benchmark/selfcheck.py` | Evaluator self-check: fault mocks run on a document with GT |
| `benchmark/runner`, `benchmark/report` | Running adapters, oracle inputs, scoring runs; Parquet store, HTML report, compare, regression |
| `benchmark/mocks`, `benchmark/synth` | Fault-injecting mock candidates; synthetic documents, clean twins, degradation ladders |
| `policies/` | Redaction policy profiles: the client's decisions (`au-medicolegal-v1.yaml`, the default) and the earlier draft (`au-medicolegal-v0.yaml`, open questions scored as MAY) |
| `candidates/`, `experiments/` | Candidate manifests (the fault mocks, the ceiling mock and tool 2's `mlredact-client.yaml`); experiment specs |
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

**Policy.** The default policy is the client's decisions, `policies/au-medicolegal-v1.yaml`
(version 1.1.0): every person, organisation, contact detail, identifier, signature and image of a
person is replaced with a fake; dates, ages, occupation, clinical or legal content, and suburbs
and towns mentioned on their own are kept. Hiding a required item without replacing it is its own
verdict (`NOT_REPLACED`); a document also has to stay searchable and keep 99% of its ordinary
words. The draft (`au-medicolegal-v0.yaml`) stays for re-scoring old runs. Definitions:
[docs/metrics.md](docs/metrics.md).

**Policy 1.1.0 departs from the client's answers in two places, on purpose.** Both are recorded
under `deviations` in the policy file, and every run report and comparison board shows them in a
marked section, "Departures from the client's answers", above the scorecard:

| Question | The client's answer | Applied instead | Why |
|---|---|---|---|
| Q4 | Suburbs and towns are kept, and postcodes too | The whole address is replaced: street line, suburb, state and postcode. A suburb or town mentioned on its own is still kept. | The project owner decided on 2026-10-22 that every tool replaces whole addresses, and the scores hold every tool to it. It began when GT 0.2 split each address into street line and locality by a rule no person had checked; checking the splits does not end it, only the owner can. A tool that keeps the suburb and postcode of an address, as the client asked, is scored as leaking them. |
| Q10 | Role words such as "the claimant" also count | Role words are left as they are | They are not in the ground truth, so changing them only counts as damage to ordinary words. Decided by the project owner on 2026-10-21; to be confirmed with the client. |

Ground truth for the five seed documents exists as `gt_version 0.2-ai`: an AI annotator
(claude-opus-5-5) corrected the machine drafts (0.1-ai), then addresses were split into street
line and locality (`rb gt split-addresses`; 0.1-ai is archived under `data/gt-archive/`). All five validate with 0 errors and each has
a review sheet in `data/gt/<DOC>.review.md`. It is **not yet human-reviewed**, so it is not a
release. It lives in the git-ignored `data/` folder. The evaluator self-check was run on all
five **before the client's policy existed**: under the draft policy (`au-medicolegal-v0`), on GT
0.1-ai, with eval-0.2 (details in [docs/metrics.md](docs/metrics.md)):

| Doc | Required mentions (draft policy) | Self-check (eval-0.2) |
|---|---|---|
| TRN | 4 | 10 pass, 1 fail (partial-half) |
| DPI | 33 | 10 pass |
| IME | 59 | 10 pass |
| PSY | 113 | 11 pass |
| NOC | 303 | 8 pass, 2 fail (reference-true, reference-raster) |

The three failures are evaluator calibration issues on real scans and are open. The table is not
the current state: under the default policy 1.1.0 the same documents have 91, 100, 230, 275 and
497 required mentions (TRN, DPI, IME, PSY, NOC), and the self-check has two more checks for the
client's rules (`replace-all`: the ceiling mock replaces every required mention and keeps the text
searchable; `remove-only`: removal without a fake is `NOT_REPLACED` where a fake is required and
`SAFE_REMOVED` where removal is allowed). `rb selfcheck --dataset seed@0.1.0` runs every check
under the default policy; that has not been done yet. A check that breaks is reported as an error
(by its exception's class only, never its message) and the remaining checks and documents still
run.

**Not yet done**: human review and double annotation of the seed GT, including a check of the
address splits; the self-check of the seed documents under the client's policy; the calibration
fixes (they change the evaluator, so they need a new evaluation version, eval-0.4, and a rescore);
a decision on the quasi-identifiers the questionnaire did not ask about (religion, education,
qualifications, marital, benefit or employment status, accreditation, courses), which stay MAY;
and runs of the real tools on the seed documents (tool 2 has run end to end on the synthetic
dataset; tool 1's and the client's tool's manifests do not exist yet). The client's filled-in
questionnaire is `docs/client_policy_decisions_filled_by_joel.docx`, kept out of git.
