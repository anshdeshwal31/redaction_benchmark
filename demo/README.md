# Side-by-side demo

A small web app for showing the redaction tools to the client:

- **Try a document**: upload a PDF; every configured tool redacts it, one after the other, and the results are
  shown side by side. For a tool that writes a PDF: its page images, a download, its own release verdict
  (released, or held back and why) and what it replaced (original → fake, from its `pseudonyms.json`). For a tool
  that writes text: its redacted text, a download, and its replacement list, marked clearly as text, not a PDF.
  An output the tool would not release is offered as "the held-back output (not safe to share)", after a warning
  and the tool's own note. The text download (.txt or .json) holds the redacted text only: the tool's own
  `redacted_text.json` also holds the original text, so it is never handed out. A run whose version carries the
  runner's plumbing mark (a fake model, a laptop setting) is marked "not the tool as built".
- **Benchmark results**: the comparison boards and scored runs found in the results folders, with a short
  summary per tool (required items leaked, not replaced, over-redaction, clean documents, PDF- or text-graded,
  searchable, the tool's own release check), a plain-language explanation of each number, and links to the
  full boards.

The app knows a tool only through its candidate manifest (`docs/adapter_guide.md`): it fills in the manifest's
command with the benchmark's own runner, exactly as `rb run` does, on a copy of the upload named `input.pdf` in a
folder of its own, and reads the files the adapter contract defines. Each tool is run end to end (`{mode}` is
`e2e`), without a matter (`{matter}` is empty), with seed 0, and is given the configured policy if its command
takes `{policy}`.

## Install

On the server, in the benchmark's environment:

```bash
cd redaction_benchmark
.venv/bin/python -m pip install -e '.[demo]'     # Flask, Werkzeug and Waitress, pinned in pyproject.toml
```

## Configure

Settings are in `demo/demo.yaml` (or a copy named by `RB_DEMO_CONFIG`); environment variables override them.
Relative paths are taken from the benchmark root.

| Variable | Setting | Default |
|---|---|---|
| `RB_DEMO_CONFIG` | the settings file | `demo/demo.yaml` |
| `RB_DEMO_CANDIDATES` | the tools: candidate manifests separated by `:` (the column is named after the tool) | `candidates/mlredact-client.yaml`, `candidates/privategpt-text.yaml` |
| `RB_DEMO_RESULTS_DIRS` | results folders for the Benchmark page, separated by `:` | `$RB_RESULTS_DIR`, else `results/` |
| `RB_DEMO_RUNS` | the scored runs the Benchmark page lists, separated by commas (`runs` in demo.yaml) | every scored run in the folders |
| `RB_DEMO_WORK_DIR` | where uploads and tool outputs are kept | `runs/demo` (git-ignored) |
| `RB_DEMO_KEEP_MINUTES` | minutes a finished job is kept | 120 |
| `RB_DEMO_TIMEOUT_S` | a tool still running after this is stopped (or sooner, at its manifest's `timeout_s`) | 1800 |
| `RB_DEMO_MAX_UPLOAD_MB` | upload limit | 50 |
| `RB_DEMO_POLICY` | the policy given to tools that take `{policy}` | `policies/au-medicolegal-v1.yaml` |
| `RB_DEMO_HOST`, `RB_DEMO_PORT` | where the app listens | `127.0.0.1`, `8765` |

The tools' own variables are read from the environment the app is started in, as with `rb run`:

| Variable | For |
|---|---|
| `RB_MLREDACT_PYTHON` | Tool 2: its interpreter, e.g. `.../redaction_tool_from_scratch_2/backend/.venv/bin/python`. Required. |
| `MLREDACT_CONFIG`, `MLREDACT_SURROGATE_KEY` | Tool 2, optional: see the comment at the top of `candidates/mlredact-client.yaml`. |
| `RB_PRIVATEGPT_PYTHON` | The client's tool: the interpreter of his environment, built outside his repository (`adapters/privategpt/README.md`). Required. |
| `RB_PRIVATEGPT_REPO` | The client's tool: his repository, read only. Required. |
| `RB_PRIVATEGPT_OLLAMA_URL`, `RB_PRIVATEGPT_MODEL` | The client's tool: the Ollama endpoint (default `http://localhost:11434`) and model (default `qwen2.5:14b`, pulled with `ollama pull qwen2.5:14b`). Keep Ollama on the same server. |

`candidates/privategpt-text.yaml` and `adapters/privategpt/README.md` describe the client's tool's variables in
full; run its set-up check (`--check`) before the demo. In the demo each upload is a project of its own in his
tool, with a store inside the job's folder, so it is deleted with the job. The upload page shows each tool as **ready** or **not ready**, with the reason (a manifest not found, a
variable not set), before anything runs; a tool that is not ready is skipped and its column says why.

Each tool in `demo.yaml` can have a `note`, shown under its name on the upload page and on its column: what the
demo runs of the tool and what it leaves out. The client's tool's note says that only the automatic output of an
upload is run, not his optional "Check Now" and "Fix All" pass (`adapters/privategpt/README.md`, "What the scores
cover").

The app starts tools without a shell. A manifest whose command needs shell syntax (`&&`, a pipe, `$VAR`,
`VAR=value cmd`) is shown as not ready: move that into a wrapper script, or the variables into the manifest's `env`.

## Results for the Benchmark page

The page reads the top level of each results folder: every board (`board-*.html`, with the scorecard JSON that
`rb board` writes next to it) and every scored run (a folder with `scoring.json` and `tables/`). Its numbers are
the benchmark's own scorecard, so they match the boards. To produce them, for each tool:

```bash
export RB_RUNS_DIR=/data/rb/runs RB_RESULTS_DIR=/data/rb/results
rb run --candidate candidates/mlredact-client.yaml --dataset synthetic@0.1.0 --run-id t2-e2e-synth
rb run --candidate candidates/privategpt-text.yaml --dataset synthetic@0.1.0 --run-id pg-e2e-synth
rb score t2-e2e-synth --policy policies/au-medicolegal-v1.yaml
rb score pg-e2e-synth --policy policies/au-medicolegal-v1.yaml
rb board t2-e2e-synth pg-e2e-synth --name demo --title "Tool 2 and PrivateGPT, synthetic set"
```

Then start the app with `RB_DEMO_RESULTS_DIRS=/data/rb/results` (several folders: separate them with `:`).

The page lists every scored run in those folders. Keep plumbing runs (the fake model, `RB_PRIVATEGPT_DOCLING_DEVICE`)
out of them, or name the runs to show in `runs` (or `RB_DEMO_RUNS=t2-e2e-synth,pg-e2e-synth`): a run made before the
runner's plumbing mark existed, such as `priv-fake-1` and `priv-fake-2` in `~/rbwork/results-deliver`, looks like a
real run of the client's tool. A run with the mark gets a "plumbing check: not the tool as built" badge.

For a text-graded tool the leak column's headline is the rate on the items it handled (the owner's choice,
2026-10-31), with how many required items its own reader never extracted (running headers and footers, for the
client's tool) and the strict rate, which counts them as leaked, under it. The Benchmark page shows the scores of the
fictional synthetic set only: the evaluator is not calibrated on the client's real scans yet, so real documents are
shown live, side by side, but not scored on the page. The page's "What the numbers mean" explains both; `docs/metrics.md` has the rule.

## Start

On a fresh server, first get the code and the data the laptop has (`adapters/privategpt/README.md`, "Before you
start"): the demo, the client's runner and tool 2's sideways and upside-down hold are not committed yet, and the
documents (`data/`, `datasets/seed/`) are git-ignored.

From the benchmark root, with the benchmark on the Python path:

```bash
cd redaction_benchmark
export PYTHONPATH="$PWD"
export RB_MLREDACT_PYTHON=... RB_PRIVATEGPT_PYTHON=... RB_PRIVATEGPT_REPO=...
export RB_DEMO_RESULTS_DIRS=/data/rb/results
.venv/bin/python -m demo                          # or: --host 127.0.0.1 --port 8765 --config my-demo.yaml
```

The log lists each tool as ready or not, and the results folders. It names jobs by id only; it never holds a file
name or document text.

## Reach it through an SSH tunnel

The app listens on `127.0.0.1` only, so nobody else on the network can reach it. From your laptop:

```bash
ssh -N -L 8765:127.0.0.1:8765 you@gpu-server
```

and open <http://localhost:8765>. The app has no login: do not bind it to another address (`--host 0.0.0.0`)
unless the network is private, and the app warns when you do.

## What happens to an upload

- The file is checked (a PDF that opens, not password-protected, within the size limit) and saved as `input.pdf`
  in a new folder named by a random id, readable only by the server's user. Its original name is never kept,
  shown, logged or passed to a tool.
- Jobs wait in one queue; the tools of a job run one after another, so the GPU and the memory are never shared
  between two tools. The job page shows each tool's progress and refreshes itself while the job runs.
- A tool that fails, writes no output or runs past the time limit is stopped together with everything it started,
  and its column says what happened (for a failure, with the end of the tool's own error output). A tool that
  exits with an error but wrote its output is shown, with a note.
- The job's folder (upload, outputs, page images, logs) is deleted `keep_minutes` after the job finishes, at once
  with the job page's **Delete** button (a running tool is stopped first), when the app stops (Ctrl-C, SIGTERM or
  SIGHUP: a running tool is stopped first, with everything it started) and when the app starts again. Only a
  `kill -9` or a crash leaves the folders until the next start. The app only ever deletes folders it created.

## Tests and a local run without the real tools

```bash
pytest -q -n 0 tests/test_demo.py
```

The tests use the benchmark's mock (`candidates/mock-reference-replace.yaml`) and a stand-in text tool
(`demo/candidates/standin-text.yaml`, `demo/standin_text.py`: a crude pattern replacer, **not** the client's tool),
so no model is needed. The same pair gives a local run:

```bash
RB_DEMO_CANDIDATES=candidates/mock-reference-replace.yaml:demo/candidates/standin-text.yaml \
RB_DEMO_WORK_DIR=/tmp/rb-demo .venv/bin/python -m demo
```

and upload a fictional synthetic PDF: the mock redacts only documents whose ground truth is in `gt/`, such as the
`synthetic@0.1.0` documents.

## Limits

- Linux or macOS only (the tools are started without a shell; on Windows every tool reads as not ready).
- Job status lives in the app's memory: a restart forgets the jobs and deletes their folders.
- One app process per work folder.
