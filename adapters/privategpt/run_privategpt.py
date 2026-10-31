"""Benchmark runner for the client's PrivateGPT redaction, graded exactly as built.

The client's tool redacts text, not PDFs: it extracts a document's text (docling Markdown), finds personal
data (regex, then an Ollama model), swaps in synthetic values and keeps the mappings in a SQLite store per
project. This runner calls that code unchanged, from the client's own repository, and hands the benchmark
``redacted_text.json`` for the text-level scoring path (docs/adapter_guide.md, section 8).

It runs in the client's Python environment (``RB_PRIVATEGPT_PYTHON``), never in the benchmark's, and
imports nothing from the benchmark. It never copies or edits the client's code. What it does around his
code is plumbing only (README.md, "Plumbing"):

- one benchmark matter is one of his projects, with the project's shared session, as his upload endpoint
  does; the store is one SQLite file per benchmark run, next to the run's documents;
- his database, redacted files and printed log go to the run folder, never to his repository;
- the Ollama endpoint and model come from environment variables (default: his own, qwen2.5:14b on
  localhost);
- his model calls are counted, not changed, so that a document whose model calls failed is reported as a
  failure instead of being graded on a silently degraded result.

Exit codes: 0 done; 2 his code raised, or the mode is not supported; 3 the model endpoint failed during the
document (output withheld); 4 the set-up check failed (endpoint, model, docling, CUDA). Every failure writes
``error.json`` in the working directory.

The runner's own output (stdout) holds counts, ids and paths only. His code prints document values; those
lines go to ``privategpt.log`` in the working directory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

CANDIDATE = "privategpt-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:14b"  # the client's own default (RedactionService, pii_detector)
ENV_URL = "RB_PRIVATEGPT_OLLAMA_URL"
ENV_MODEL = "RB_PRIVATEGPT_MODEL"
ENV_ALLOW_REMOTE = "RB_PRIVATEGPT_ALLOW_REMOTE"
ENV_DEVICE = "RB_PRIVATEGPT_DOCLING_DEVICE"  # plumbing checks on a machine without CUDA only (README.md)
# The version line of a run that is not his tool as built (fake model, docling device switched) carries this mark, so the
# board and the demo show it next to every score of that run.
PLUMBING_MARK = "plumbing check"
FAKE_TAG = "rb_fake_ollama"  # fake_ollama.py's /api/tags answer says so with this key
STORE_DIRNAME = "privategpt-store"
DB_NAME = "redaction_data.db"  # the file name his server uses
SESSION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://redaction-benchmark/privategpt/session")

EXIT_OK, EXIT_TOOL_ERROR, EXIT_MODEL_FAILED, EXIT_SETUP = 0, 2, 3, 4


class SetupError(RuntimeError):
    """The environment cannot run the client's tool as built (endpoint, model, docling)."""


# -- pure helpers (importable without the client's dependencies; the tests use them) -------------------------


def settings(env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    return {"ollama_url": (env.get(ENV_URL) or DEFAULT_OLLAMA_URL).rstrip("/"),
            "model": env.get(ENV_MODEL) or DEFAULT_MODEL,
            "allow_remote": env.get(ENV_ALLOW_REMOTE) == "1"}


def is_local_endpoint(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_endpoint_allowed(url: str, allow_remote: bool) -> None:
    """Documents go to the model endpoint. One on another machine needs the owner's go-ahead, given as
    RB_PRIVATEGPT_ALLOW_REMOTE=1 (the benchmark's rule: never network-send documents without approval)."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise SetupError(f"{ENV_URL} must be an http(s) URL")
    if not is_local_endpoint(url) and not allow_remote:
        raise SetupError(f"the model endpoint is not on this machine; documents would be sent to it. "
                         f"Set {ENV_ALLOW_REMOTE}=1 only with the project owner's approval")


def project_for(matter: str, input_sha256: str) -> tuple[str, str]:
    """(project name, session id) for a document. One benchmark matter is one of his projects; a document
    without a matter is a project of its own, as an upload without a project is in his tool. The session id
    seeds his fakes; it is derived from the matter so that a re-run makes the same fakes (his server draws a
    random one per project, his own baseline harness fixes them the same way)."""
    name = f"matter-{matter}" if matter else f"document-{input_sha256[:16]}"
    return name, str(uuid.uuid5(SESSION_NAMESPACE, name))


def default_store_dir(workdir: Path) -> tuple[Path, str]:
    """The project store for this document. Under a benchmark run (<run>/<doc>/r<n>/out) it is one store
    per run, shared by every document of the run, so a matter's documents share his project as they would
    in his tool. Elsewhere it is a store of its own in the working directory."""
    w = workdir.resolve()
    if w.name == "out" and re.fullmatch(r"r\d+", w.parent.name):
        return w.parents[2] / STORE_DIRNAME, "run"
    return w / STORE_DIRNAME, "workdir"


def to_text_mappings(mappings: list[dict]) -> list[dict]:
    """His mappings ({pii_type, original, synthetic, source}) as the benchmark's, in his order, one for one."""
    return [{"label": str(m.get("pii_type", "")), "original": str(m.get("original", "")),
             "replacement": str(m.get("synthetic", ""))} for m in mappings]


def redacted_text_artifact(result: dict) -> dict:
    """``redacted_text.json`` from his ``redact_document`` result, unchanged. His docling Markdown has no page
    breaks, so the texts hold none and the benchmark aligns the document as a whole."""
    return {"schema": "rb.redacted_text/0.1",
            "original_text": result["original_text"],
            "redacted_text": result["redacted_text"],
            "page_break": "\f",
            "mappings": to_text_mappings(result.get("mappings") or [])}


def verification_summary(verification: dict | None) -> dict:
    """Counts from his verifier's report; no values (his report quotes originals for critical issues)."""
    if not verification:
        return {}
    by_sev: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for issue in verification.get("issues") or []:
        by_sev[issue.get("severity", "?")] = by_sev.get(issue.get("severity", "?"), 0) + 1
        by_type[issue.get("type", "?")] = by_type.get(issue.get("type", "?"), 0) + 1
    return {"overall_confidence": verification.get("overall_confidence"),
            "checks_passed": verification.get("checks_passed"),
            "checks_total": verification.get("checks_total"),
            "issues_by_severity": dict(sorted(by_sev.items())),
            "issues_by_type": dict(sorted(by_type.items()))}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repo_version(repo: Path) -> str:
    """His commit, '+modified' when tracked files differ from it (untracked files are ignored)."""
    try:
        sha = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=30, check=True).stdout.strip()
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return sha + ("+modified" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def runner_version() -> str:
    return sha256_file(Path(__file__))[:12]


def is_fake_endpoint(url: str, allow_remote: bool) -> bool:
    """Is the endpoint the fake server (fake_ollama.py)?  Asked only of an endpoint the runner may use; the model
    list holds no document text."""
    if not (is_local_endpoint(url) or allow_remote):
        return False
    try:
        return bool(_get_json(f"{url}/api/tags", timeout=5).get(FAKE_TAG))
    except (OSError, ValueError, AttributeError):
        return False


def plumbing_reasons(env: dict | None = None) -> list[str]:
    """Why a run would not be his tool as built: the fake model, or docling's device switched for a laptop."""
    env = os.environ if env is None else env
    cfg = settings(env)
    reasons = []
    if is_fake_endpoint(cfg["ollama_url"], cfg["allow_remote"]):
        reasons.append("fake model")
    if env.get(ENV_DEVICE):
        reasons.append(f"docling on {env[ENV_DEVICE]}")
    return reasons


def version_line(repo: Path, env: dict | None = None) -> str:
    """The candidate version: his commit, the model, the runner; and the plumbing mark when the run is not his tool
    as built, so every score of such a run says so."""
    line = f"privategpt@{repo_version(repo)} model={settings(env)['model']} runner={runner_version()}"
    reasons = plumbing_reasons(env)
    if reasons:
        line += f" [{PLUMBING_MARK}: {', '.join(reasons)}; not the tool as built]"
    return line


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- the model endpoint ------------------------------------------------------------------------------------------


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_model(url: str, model: str) -> None:
    """The endpoint answers, lists the model, and the model answers a chat call (no document text)."""
    try:
        tags = _get_json(f"{url}/api/tags", timeout=15)
    except (OSError, ValueError) as exc:
        raise SetupError(f"the Ollama endpoint does not answer ({type(exc).__name__})") from None
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise SetupError(f"model {model} is not on the endpoint (ollama pull {model})")
    try:
        reply = _post_json(f"{url}/api/chat", {"model": model, "stream": False, "think": False,
                                               "messages": [{"role": "user", "content": "Reply with []"}],
                                               "options": {"temperature": 0, "num_predict": 8}}, timeout=600)
        reply["message"]["content"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SetupError(f"model {model} does not answer a chat call ({type(exc).__name__})") from None


class LlmCounter:
    """Counts his model calls without changing them: every HTTP call his detector makes (failures are
    connection errors and non-200 answers), and every detector call whose answer held no valid JSON after
    his own retries (his code then skips that chunk; that is the tool's behaviour, reported, not a failure)."""

    def __init__(self):
        self.http_calls = 0
        self.http_failures = 0
        self.calls = 0
        self.calls_without_json = 0

    def as_dict(self) -> dict:
        return {"http_calls": self.http_calls, "http_failures": self.http_failures,
                "detector_calls": self.calls, "detector_calls_without_json": self.calls_without_json}


class _CountingRequests:
    """Stands in for the ``requests`` module inside his pii_detector only; passes everything through."""

    def __init__(self, real, counter: LlmCounter):
        self._real = real
        self._counter = counter

    def __getattr__(self, name):
        return getattr(self._real, name)

    def post(self, *args, **kwargs):
        self._counter.http_calls += 1
        try:
            response = self._real.post(*args, **kwargs)
        except self._real.RequestException:
            self._counter.http_failures += 1
            raise
        if response.status_code != 200:
            self._counter.http_failures += 1
        return response


def install_counters(pii_detector, counter: LlmCounter) -> None:
    pii_detector.requests = _CountingRequests(pii_detector.requests, counter)
    original = pii_detector._ollama_chat_json

    def counted(*args, **kwargs):
        counter.calls += 1
        try:
            return original(*args, **kwargs)
        except pii_detector.OllamaJSONError:
            counter.calls_without_json += 1
            raise

    pii_detector._ollama_chat_json = counted


# -- running his code ---------------------------------------------------------------------------------------


def import_his_code(repo: Path):
    """His modules, from his repository. No bytecode is written: his repository stays as it is."""
    backend = repo / "backend"
    if not (backend / "redaction" / "redaction_service.py").is_file():
        raise SetupError("--repo does not hold backend/redaction/redaction_service.py")
    sys.dont_write_bytecode = True
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from redaction import pii_detector  # noqa: E402  (his code)
    from redaction.redaction_service import RedactionService  # noqa: E402  (his code)
    return pii_detector, RedactionService


def spacy_model_available() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


NO_CUDA = (f"CUDA is not available: his code asks docling for CUDA and docling then refuses to convert. Run on the "
           f"GPU server with CUDA wheels of torch; for a plumbing check on a machine without CUDA set {ENV_DEVICE}=cpu")


def package_versions() -> dict:
    """The versions his result depends on (his requirements are unpinned)."""
    from importlib import metadata

    out = {}
    for name in ("docling", "docling-slim", "docling-ibm-models", "easyocr", "torch", "spacy", "en-core-web-sm",
                 "requests", "cryptography"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def set_docling_device(service, device: str) -> None:
    """His converters ask docling for CUDA, and docling refuses to run without it. On a machine without CUDA
    (the laptop), a plumbing check sets the device his converters will use; nothing else changes. Real runs
    (the GPU server) never set it."""
    for converter in (service._docling_converter, service._docling_converter_ocr):
        for option in (converter.format_to_options.values() if converter else ()):
            pipeline = getattr(option, "pipeline_options", None)
            if pipeline is not None and getattr(pipeline, "accelerator_options", None) is not None:
                pipeline.accelerator_options.device = device


def find_or_create_project(store, name: str, session_id: str) -> tuple[str, str, bool]:
    """(project id, session id, created) for a matter, through his store's own project calls."""
    for project in store.list_projects():
        if project.get("name") == name:
            return project["id"], project["session_id"], False
    return store.create_project(name, "benchmark matter", session_id), session_id, True


def redact_one(args, cfg: dict, workdir: Path, log, config: dict) -> int:
    """Run his pipeline on one document, filling ``config`` (run_info.json) as it goes. Returns the exit code."""
    repo = Path(args.repo).resolve()
    check_endpoint_allowed(cfg["ollama_url"], cfg["allow_remote"])
    check_model(cfg["ollama_url"], cfg["model"])
    input_path = Path(args.input).resolve()
    input_sha = sha256_file(input_path)
    store_dir, store_scope = (Path(args.store_dir).resolve(), "given") if args.store_dir \
        else default_store_dir(workdir)
    project_name, session_id = project_for(args.matter or "", input_sha)
    plumbing = [
        f"database: {DB_NAME} in the run's store folder (his default: redaction_data.db in the working folder)",
        "redacted files: redacted/ in the run's store folder (his default: shared_uploads/redacted)",
        "working folder: the document's benchmark working directory",
        f"model endpoint: {ENV_URL} (also set as OLLAMA_HOST in this process); model: {ENV_MODEL}",
        "printed output: privategpt.log in the working directory",
        "model calls counted through a pass-through wrapper (pii_detector.requests, _ollama_chat_json)",
        "session id: derived from the matter (his server draws a random one per project)",
        "file name: document-<input hash>.pdf (the benchmark's input is always input.pdf)",
    ]
    config.update({"project": project_name, "session_id": session_id,
                   "store_scope": store_scope, "input_sha256": input_sha, "ollama_model": cfg["model"],
                   "ollama_endpoint_local": is_local_endpoint(cfg["ollama_url"]), "plumbing": plumbing})

    os.environ["OLLAMA_HOST"] = cfg["ollama_url"]
    pii_detector, RedactionService = import_his_code(repo)
    counter = LlmCounter()
    install_counters(pii_detector, counter)

    store_dir.mkdir(parents=True, exist_ok=True)
    redacted_dir = store_dir / "redacted"
    redacted_dir.mkdir(exist_ok=True)
    os.chdir(workdir)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        service = RedactionService(model=cfg["model"], ollama_url=cfg["ollama_url"], use_llm=True,
                                   db_path=str(store_dir / DB_NAME), redacted_dir=str(redacted_dir))
    if Path(service.redacted_dir).resolve() != redacted_dir.resolve():
        # His constructor ignores redacted_dir when /app exists (his Docker image); keep his files here.
        service.redacted_dir = str(redacted_dir)
        plumbing.append("redacted_dir set again after construction: /app exists on this machine")
    if service._docling_converter is None:
        raise SetupError("docling is not installed in RB_PRIVATEGPT_PYTHON; his code would fall back to PyPDF2, "
                         "which is not the tool as built")
    device = os.environ.get(ENV_DEVICE)
    if not device and not cuda_available():
        raise SetupError(NO_CUDA)
    if device:
        set_docling_device(service, device)
        config["docling_device_override"] = device
        plumbing.append(f"docling device set to {device} ({ENV_DEVICE}); his code asks for CUDA: a plumbing check on "
                        f"a machine without CUDA, not the tool as built")
    config["spacy_model"] = spacy_model_available()
    config["versions"] = package_versions()

    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        project_id, session_id, created = find_or_create_project(service.store, project_name, session_id)
        config.update(session_id=session_id, project_created=created,
                      project_documents_before=len(service.store.get_project_documents(project_id)),
                      project_mappings_before=len(service.store.get_mappings(session_id, decrypt=False)))
    filename = f"document-{input_sha[:12]}{input_path.suffix.lower() or '.pdf'}"
    t0 = time.time()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = service.redact_document(str(input_path), filename, session_id=session_id)
        service.store.add_document_to_project(project_id, filename,
                                              redacted_text_path=result.get("redacted_text_path"),
                                              pii_count=result["pii_count"])
        service.store.update_project_mapping_count(project_id)
    config.update(duration_s=round(time.time() - t0, 1), llm=counter.as_dict(),
                  original_chars=result["original_length"], redacted_chars=result["redacted_length"],
                  pii_count=result["pii_count"], pii_types=dict(sorted(result["pii_types"].items())),
                  mapping_sources=_count(m.get("source", "?") for m in result.get("mappings") or []),
                  verification=verification_summary(result.get("verification")))
    if counter.http_failures:
        config["error"] = "model endpoint failed during the document; output withheld"
        return EXIT_MODEL_FAILED
    write_json(workdir / "redacted_text.json", redacted_text_artifact(result))
    return EXIT_OK


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def run_info(config: dict, started: str, version: str, cfg: dict) -> dict:
    return {"schema": "rb.runinfo/0.1", "candidate": CANDIDATE, "candidate_version": version,
            "config": config, "seed": None,
            "models": [{"name": cfg["model"], "extra": {"provider": "ollama"}}],
            "started_at": started, "finished_at": now(),
            "notes": "The client's PrivateGPT redaction, called unchanged from his repository; text output, graded "
                     "by the text-level path. His tool has no release gate (its verifier's result is advisory), so "
                     "no release verdict is reported."}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="the client's repository (privategpt-redaction-testing), read only")
    ap.add_argument("--input", help="the document ({input})")
    ap.add_argument("--workdir", help="the benchmark's working directory ({workdir})")
    ap.add_argument("--mode", default="e2e", help="only e2e: his tool takes no injected stage")
    ap.add_argument("--matter", default="", help="the benchmark matter ({matter}); one matter = one project")
    ap.add_argument("--store-dir", help="project store folder (default: one per benchmark run)")
    ap.add_argument("--version", action="store_true", help="print the version line and exit")
    ap.add_argument("--check", action="store_true", help="check endpoint, model and his code, then exit")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    cfg = settings()

    if args.version:
        print(version_line(repo))
        return EXIT_OK
    if args.check:
        try:
            check_endpoint_allowed(cfg["ollama_url"], cfg["allow_remote"])
            check_model(cfg["ollama_url"], cfg["model"])
            import_his_code(repo)
            import docling  # noqa: F401
            if not os.environ.get(ENV_DEVICE) and not cuda_available():
                raise SetupError(NO_CUDA)
        except (SetupError, ImportError) as exc:
            print(f"[privategpt] check failed: {exc}")
            return EXIT_SETUP
        print(f"[privategpt] check passed: endpoint local={is_local_endpoint(cfg['ollama_url'])}, "
              f"model={cfg['model']}, spaCy model={spacy_model_available()}, CUDA={cuda_available()}, "
              f"docling device override={os.environ.get(ENV_DEVICE) or 'none'}")
        return EXIT_OK
    if not (args.input and args.workdir):
        ap.error("--input and --workdir are required")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    started = now()
    version = version_line(repo)
    config: dict = {"mode": args.mode, "matter": args.matter or ""}
    code = EXIT_OK
    with open(workdir / "privategpt.log", "a", encoding="utf-8") as log:
        try:
            if args.mode != "e2e":
                code, config["error"] = EXIT_TOOL_ERROR, f"mode {args.mode} is not supported: his tool takes no " \
                                                         "injected stage"
            else:
                code = redact_one(args, cfg, workdir, log, config)
        except SetupError as exc:
            code, config["error"] = EXIT_SETUP, str(exc)
        except Exception as exc:  # his code raised: recorded, never skipped
            traceback.print_exc(file=log)
            code, config["error"] = EXIT_TOOL_ERROR, f"{type(exc).__name__} (traceback in privategpt.log)"
    write_json(workdir / "run_info.json", run_info(config, started, version, cfg))
    if code != EXIT_OK:
        write_json(workdir / "error.json", {"exit_code": code, "error": config.get("error"),
                                            "llm": config.get("llm"), "at": now()})
        print(f"[privategpt] FAILED exit={code}: {config.get('error')} (error.json, privategpt.log)")
    else:
        print(f"[privategpt] done: pii_count={config.get('pii_count')} project={config.get('project')} "
              f"documents_before={config.get('project_documents_before')} llm={config.get('llm')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
