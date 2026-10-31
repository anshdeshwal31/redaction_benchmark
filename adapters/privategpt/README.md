# PrivateGPT runner: the client's redaction, graded as built

This folder connects the client's own redaction tool (the `privategpt-redaction-testing` repository) to the
benchmark without changing it. The runner calls his redaction code exactly as his upload endpoint does and
hands the benchmark his redacted text, which the text-level path grades (`docs/adapter_guide.md`, section 8;
`docs/metrics.md`, "Text-level evaluation").

What is graded is the automatic output of an upload: his redacted text as his upload endpoint returns it. His
workspace also has an optional second pass, which the runner does not run (see "What the scores cover").

| File | What it is |
|---|---|
| `run_privategpt.py` | The runner. It runs in his Python environment and imports his code from his repository. |
| `fake_ollama.py` | A fake Ollama server for plumbing checks on a machine without a model. |
| `requirements.txt` | His environment: his own requirements for the redaction path. |
| `../../candidates/privategpt-text.yaml` | The candidate manifest. |
| `../../tests/test_privategpt_adapter.py` | The tests. None needs a model. |

## What it does per document

For each document, the benchmark runs the runner once, in the dataset's order:

1. It checks the model endpoint: the endpoint answers, it lists the model, and the model answers a short chat
   call that holds no document text.
2. It imports his `redaction.redaction_service.RedactionService` from his repository. It does not copy the
   code and does not write bytecode next to it.
3. It finds his project for the document's matter, or creates it, through his store's own calls
   (`list_projects`, `create_project`). One benchmark matter is one of his projects, which is how his tool
   keeps one fake per person across a case. A document without a matter gets a project of its own, as an
   upload without a project does in his tool.
4. It calls `redact_document(input, filename, session_id=<the project's session>)` and then
   `add_document_to_project` and `update_project_mapping_count`, the same calls and order as his upload
   endpoint (`server.py`, `handle_redact_upload`). Inside, his code extracts the text with docling (with
   its OCR fallback for scanned pages), finds personal data with his regexes and the model, reuses the
   project's stored mappings and aliases, and swaps in his synthetic values.
5. It writes his result unchanged as `redacted_text.json`:
   - `original_text` is the text he redacted (his docling Markdown, after his OCR clean-up);
   - `redacted_text` is his redacted text;
   - `mappings` are his own `{pii_type, original, synthetic}` entries, one for one, in his order, as
     `{label, original, replacement}`. Nothing is split, merged or added. The manifest's `label_map` gives
     his types benchmark types (`address` is `ADDRESS`).
6. It writes `run_info.json`: version, model, counts (his findings by type, his verifier's confidence and
   issue counts, his model calls) and the plumbing list below. Counts only, never document values.

His docling Markdown has no page breaks, so the benchmark aligns each document as a whole.

Each benchmark run starts with an empty store, so earlier runs never leak into a new one. Within a run, a
document run again (an experiment with repeats) goes into the same project, like a second upload in his
tool: his stored mappings are reused.

## Plumbing only

The runner changes none of his behaviour. Everything it does around his code, and each place where it
points his code somewhere other than his defaults, is listed here.

| What | His default | In the runner | Why |
|---|---|---|---|
| Database | `redaction_data.db` in the folder the server starts in | `<run>/privategpt-store/redaction_data.db`, one per benchmark run, passed as `db_path` | Never his repository or his real data; documents of a matter share it. |
| Redacted text files | `shared_uploads/redacted`, or `/app/shared_uploads/redacted` when `/app` exists | `<run>/privategpt-store/redacted/`, passed as `redacted_dir`; set again after construction if `/app` exists | Same. |
| Working folder | where the server starts | the document's benchmark working directory | Any relative path his code writes lands in the run. |
| Model endpoint and model | `OLLAMA_HOST` or `http://localhost:11434`; `qwen2.5:14b` | `RB_PRIVATEGPT_OLLAMA_URL` (default `http://localhost:11434`), also set as `OLLAMA_HOST` in the runner's process; `RB_PRIVATEGPT_MODEL` (default `qwen2.5:14b`) | Set per machine. |
| Printed output | the server's console | `privategpt.log` in the working directory | His code prints document values; the runner's own output holds counts only. |
| Project session id | a random UUID per project (`server.py`) | a UUID derived from the matter | His fakes are seeded by the session id, so a re-run makes the same fakes. His own baseline harness fixes session ids the same way. |
| File name | the uploaded file's name | `document-<first 12 hex of the input's SHA-256>.pdf` | The benchmark passes a neutral `input.pdf`; his store names redacted files after the upload, so equal names would overwrite each other. Detection never reads the name. |
| docling device | CUDA (his converters ask for it) | unchanged; only for a plumbing check on a machine without CUDA, `RB_PRIVATEGPT_DOCLING_DEVICE=cpu` sets the device his converters use, and `run_info.json` says so | docling refuses to convert when CUDA is asked for and missing (below). Never set on the GPU server. |
| Model call counting | none | a pass-through wrapper around `requests` inside his `pii_detector`, and around its `_ollama_chat_json` | To report a document whose model calls failed (below). Each call and answer is passed on unchanged. |

The encryption key for his stored originals is his own default (`REDACTION_ENCRYPTION_KEY`, else his
built-in key); it changes nothing in the output.

## Failures are loud

A document his tool cannot process gets a non-zero exit code, `error.json` (exit code, reason, model call
counts) and `run_info.json` in its working directory. The benchmark records the exit code; without
`redacted_text.json` every required mention of the document is `NO_OUTPUT`.

| Exit | Meaning |
|---|---|
| 0 | Done. |
| 2 | His code raised (traceback in `privategpt.log`), or the mode is not `e2e`. |
| 3 | The model endpoint failed during the document (connection error or an HTTP error on one of his calls). His code would carry on with regexes only for that text, so the output would not be his tool as built: the runner withholds `redacted_text.json`. Fix the endpoint and run again. |
| 4 | The set-up check failed: the endpoint does not answer, does not have the model, is on another machine without the owner's go-ahead, docling is missing (his code would fall back to PyPDF2), or CUDA is missing. |

His code needs CUDA. It asks docling for the CUDA device and expects a CPU fallback, but docling (2.129, the
version current at his last commit on 2026-09-18, and 2.145) refuses to convert instead. On a machine without
CUDA every document would fail, so the runner stops with exit 4 and says so. Laptop plumbing checks set
`RB_PRIVATEGPT_DOCLING_DEVICE=cpu`; their results are recorded as not the tool as built.

A model answer with no valid JSON after his own three attempts is his tool's behaviour, not a failure: his
code skips that chunk, and `run_info.json` counts it (`llm.detector_calls_without_json`).

## What the scores cover

- Graded: every required mention found in his extracted text, with the same rules as for a PDF; collateral
  damage; detection from the spans he changed; the pseudonym checks on his fakes, across each matter.
- Not graded, by the text-level path's design: anything a PDF would show (layers, metadata, pixels),
  signatures and images, searchability. His tool produces no PDF.
- Text his extraction leaves out (docling's Markdown drops page headers and footers; letterhead images are
  not read) has no place in `original_text`: those mentions are `NOT_EVALUATED`. They never reach his output,
  so nothing leaks from it, but his tool never handled them either. Where each number puts them:
  - the board's and the demo's **strict leak rate counts them as leaked**, as it counts a document with no
    output (`docs/metrics.md`, "Text-level evaluation"); on the fake-model run of three synthetic documents
    that was 21 of 60 required mentions (58.3% strict, against 35.9% on the 39 he handled);
  - the board's "leak (completed)" and each document's own summary (`must_leak_rate`) leave them out;
  - the board and the demo show how many there were next to the strict rate ("not in its text"), and the
    run report lists them per document.

  Tool 2 is graded on the same headers and footers in its PDF. The owner's decision (2026-10-31): the demo's
  headline for his tool is the leak rate on the items he handled, with the count of items that never reached
  his output and the strict rate shown under it.
- His optional second pass is not run. After an upload his workspace offers "Check Now" (it also runs by
  itself when a project is opened): his verifier and the model look for values his first pass missed
  (`get_fix_suggestions`, `POST /api/redact/<session>/suggestions`), and "Fix All" applies the suggested
  replacements on a click (`POST /fix`). The runner grades only what `redact_document` returns, before any
  such fix: the output a user gets without reviewing the suggestions. A second, labelled mode that applies all
  his suggestions automatically could be added; it would grade a choice his interface leaves to the user.
- No release verdict: his verifier's result is advisory (his interface shows it; nothing holds a document
  back), so `run_info.json` has no `release` and the safety-check grade is N/A.

## Running it on the GPU server

The model runs on the GPU server. Run the runner there too, so documents never leave the machine.

Before you start, the server needs the same code and data as the laptop. A plain clone has neither:

- **The code at known commits.** This runner, its tests, the candidate manifest and the demo (`demo/`) are not
  committed yet, and neither is tool 2's sideways and upside-down hold. A clone of either repository therefore
  lacks them (and a clone of tool 2 releases sideways pages). Commit and push both with the owner's go-ahead, or
  copy the working trees, and record the two commits (or `+dirty`) next to the results.
- **The documents and the answers.** `data/`, `datasets/seed/` and every `*.pdf` are git-ignored, and the
  benchmark finds each document by its hash in `data/store`. Copy them from the laptop, then check them:

  ```bash
  rsync -avR data/store data/gt datasets/seed gpu:~/redaction_benchmark/      # on the laptop, from the repo root
  rb dataset validate --name synthetic --version 0.1.0                         # on the server: 0 problem(s)
  rb dataset validate --name seed --version 0.1.0
  for d in TRN DPI IME PSY NOC; do rb gt validate data/gt/$d.gt.json --policy policies/au-medicolegal-v1.yaml | tail -2; done
  ```

1. His repository, read only, for example at `~/privategpt-redaction-testing` (a clone at the commit to be
   graded). Never run his server from it during the benchmark.
2. Ollama with his model: `ollama pull qwen2.5:14b`, then `curl -s localhost:11434/api/tags`.
3. His environment, outside his repository (Python 3.11, as his Docker image). The owner's decision
   (2026-10-31): "as built" is **his Docker image's pins**, since that is how he deploys his tool. His
   repository describes two environments:
   - **his local set-up** (`DEVELOPER_HANDOVER.md`: `requirements-local.txt`, then torch, torchvision and
     docling unpinned). The recipe below follows it, with the versions current at his last commit;
   - **his Docker image** (`Dockerfile.backend` installs `requirements-docker.txt`), which pins
     `torch==2.4.1`, `torchvision==0.19.1` and `transformers==4.51.0` and leaves docling and easyocr unpinned.
     Its backend service has no GPU reserved in `docker-compose.yml` (only Ollama and the RAG service have),
     so it is not shown to run his redaction on a GPU. **This is the one to use**: the recipe below installs
     `torch==2.4.1 torchvision==0.19.1 transformers==4.51.0` (CUDA wheels) on the first install line.

   The two can give different docling, torch and transformers versions, and so possibly a different text
   extraction. Neither recipe has been run yet: the laptop's environment is CPU-only and newer (docling 2.145,
   torch 2.15). Run the chosen one once on the server with `--check` (step 4) before any scored run, and keep
   the versions `run_info.json` records with the results. The recipe, with his Docker pins and `--exclude-newer`
   taking the other versions current at his last commit (docling 2.129):

   ```bash
   AS_BUILT=2026-09-18T23:59:59Z
   uv venv --python 3.11 ~/privategpt-venv
   uv pip install --python ~/privategpt-venv/bin/python torch==2.4.1 torchvision==0.19.1 transformers==4.51.0   # his Docker pins, CUDA wheels
   uv pip install --python ~/privategpt-venv/bin/python --exclude-newer $AS_BUILT -r adapters/privategpt/requirements.txt
   uv pip install --python ~/privategpt-venv/bin/python pip
   ~/privategpt-venv/bin/python -m spacy download en_core_web_sm
   ```

   With plain `pip` instead of `uv`, the same packages in the same order. If the PyTorch index gives no upload
   dates for `--exclude-newer`, install torch without it. `run_info.json` records the versions used (docling,
   easyocr, torch, spaCy and its model).
4. The environment variables, then the set-up check (endpoint, model, his code, docling, CUDA; docling
   fetches its layout models on the first document):

   ```bash
   export RB_PRIVATEGPT_PYTHON=~/privategpt-venv/bin/python
   export RB_PRIVATEGPT_REPO=~/privategpt-redaction-testing
   export RB_PRIVATEGPT_OLLAMA_URL=http://localhost:11434     # the default
   export RB_PRIVATEGPT_MODEL=qwen2.5:14b                      # the default
   "$RB_PRIVATEGPT_PYTHON" adapters/privategpt/run_privategpt.py --repo "$RB_PRIVATEGPT_REPO" --check
   ```

5. The runs, from the benchmark's repository root, synthetic first:

   ```bash
   rb run --candidate candidates/privategpt-text.yaml --dataset synthetic@0.1.0 --run-id priv-e2e-synth
   rb score priv-e2e-synth && rb report priv-e2e-synth
   rb run --candidate candidates/privategpt-text.yaml --dataset seed@0.1.0 --run-id priv-e2e-seed
   rb score priv-e2e-seed && rb report priv-e2e-seed
   rb board priv-e2e-seed <tool 2's seed run> --name client-vs-tool2
   ```

   Check `run.json` for items with an exit code; a document that exited 3 or 4 is re-run after the endpoint is
   fixed, never reported as his result.

If the runner must run on another machine than the model, set `RB_PRIVATEGPT_OLLAMA_URL` to the server and
`RB_PRIVATEGPT_ALLOW_REMOTE=1`, only with the project owner's written approval: every document's text is sent
to that endpoint.

Timing: his architecture notes give about 10 seconds of model time per document on the server GPU; long
documents take longer (one model call per 3,000 characters), and scanned pages add docling's OCR. The manifest
allows 4 hours per document.

## Plumbing check without a model

On a machine without Ollama, the fake server answers his calls in his format. Its "model" is a few fixed
rules (names after a title, plus values from an optional entities file), so its results are plumbing checks
only and never the client's scores.

```bash
python adapters/privategpt/fake_ollama.py --port 11500 [--entities entities.json] &
export RB_PRIVATEGPT_OLLAMA_URL=http://127.0.0.1:11500
export RB_PRIVATEGPT_DOCLING_DEVICE=cpu        # only on a machine without CUDA
RB_RUNS_DIR=/tmp/rb-runs RB_RESULTS_DIR=/tmp/rb-results \
  rb run --candidate candidates/privategpt-text.yaml --dataset synthetic@0.1.0 --docs syn-digital syn-vector --run-id priv-fake-1
```

Such a run's version says so: the runner adds `[plumbing check: fake model, docling on cpu; not the tool as
built]` when the endpoint is the fake server (its model list says so) or `RB_PRIVATEGPT_DOCLING_DEVICE` is set.
The board, the demo's Benchmark page and its job page show that mark. Keep these results out of every folder a
client board or the demo reads (`/tmp/rb-results` above), or list the demo's runs (`runs` in `demo/demo.yaml`).
Runs made before the mark existed (priv-fake-1 and priv-fake-2 in `~/rbwork/results-deliver`) do not carry it.

## Tests

```bash
python -m pytest -q -n 0 tests/test_privategpt_adapter.py                 # fast: no model, no docling
RB_PRIVATEGPT_PYTHON=... RB_PRIVATEGPT_REPO=... [RB_PRIVATEGPT_DOCLING_DEVICE=cpu] python -m pytest -q -n 0 tests/test_privategpt_adapter.py
```

The second line also runs his code in his environment against the fake server, on fictional text: his own
detector reading the fake's answers, a two-document matter sharing one project, and a model failure
withholding the output. These load docling and take a few minutes. Every test checks that his repository is
left as it was.

## Privacy

- `redacted_text.json`, `privategpt.log` and the store hold document text and values. They stay in the run
  folder like every other output; never paste or print them for real documents.
- The runner prints counts, ids and paths only. `run_info.json` and `error.json` hold counts and reasons, never
  values; his verifier's report is reduced to counts because it quotes originals.
- The fake server's request log holds paths, model names, options and prompt lengths, never prompt text.
