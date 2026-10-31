"""Running the configured tools on an upload, one job at a time.

Each tool is called the way ``rb run`` calls it: its manifest's command, filled in by the benchmark's
own ``build_command`` (docs/adapter_guide.md, section 1), on a copy of the upload named ``input.pdf``
in a folder of its own. Unlike ``rb run``, the command is not given to a shell: it is split into its
arguments and started directly, so a tool whose command needs shell syntax (``&&``, a pipe,
``$VAR``, ``VAR=value cmd``) is refused with a message saying so. The uploaded file's name is never
kept, shown or passed on.

Jobs wait in one queue and the tools of a job run one after another, because they share the GPU and
the memory. Status lives in memory; the job folder holds the upload and every output, and is deleted
``keep_minutes`` after the job finishes, on request, and when the app starts again. The app's log
names jobs by id and never holds document text.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import yaml
from pydantic import ValidationError

from benchmark.runner.artifacts import load_artifacts
from benchmark.runner.runner import (
    ROOT,
    CandidateEnvError,
    CommandValueError,
    _kill_tree,
    build_command,
    check_env,
    load_candidate,
)
from benchmark.schemas import ArtifactKind, CandidateManifest, PseudonymsArtifact, RedactedTextArtifact, RunInfo

from .config import CandidateSlot, DemoConfig

log = logging.getLogger("demo")

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
MARKER = ".rb-demo-job"  # only folders holding this file are ever deleted by the app
MAX_WAITING = 10
# PyMuPDF must not be used from two threads at once: upload checks run in the request threads, page images
# and output checks in the worker.
_MUPDF = threading.Lock()

# Job states
QUEUED, RUNNING, FINISHED = "queued", "running", "finished"
# Tool states
WAITING, DONE, FAILED, TIMED_OUT, NOT_RUN, STOPPED = "waiting", "done", "failed", "timed out", "not run", "stopped"


class UploadError(ValueError):
    """An upload the demo cannot run; the message is shown to the user."""


# -- tools ------------------------------------------------------------------------------------------

# Shell syntax outside quotes. Without a shell these would reach the tool as literal arguments.
_SHELL_ANYWHERE = set("|&;<>()$`*?[")
_SHELL_WORD_START = set("#~")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def shell_syntax(template: str) -> list[str]:
    """The shell features a command template uses outside quotes (placeholders are left out: the
    runner quotes every value it fills in). Empty when the command runs the same without a shell."""
    text = _PLACEHOLDER.sub("X", template.replace("{{", "").replace("}}", ""))
    found: list[str] = []
    quote, escaped, word_start = None, False, True
    for c in text:
        if escaped:
            escaped, word_start = False, False
        elif quote == "'":
            quote = None if c == "'" else quote
        elif c == "\\":
            escaped = True
        elif quote == '"':
            if c == '"':
                quote = None
            elif c in "$`":
                found.append(c)
        elif c in "'\"":
            quote, word_start = c, False
        elif c.isspace():
            word_start = True
        else:
            if c in _SHELL_ANYWHERE or (word_start and c in _SHELL_WORD_START):
                found.append(c)
            word_start = False
    try:
        first = shlex.split(text)[:1]
    except ValueError:
        return ["unbalanced quotes"]
    if first and _ASSIGNMENT.match(first[0]):
        found.append("VAR=value before the command")
    return list(dict.fromkeys(found))


def command_argv(manifest: CandidateManifest, input_pdf: Path, out_dir: Path, policy: Path) -> list[str]:
    """The tool's argument list: the manifest command filled in by the benchmark's runner (every path
    quoted there), then split back into arguments for a start without a shell."""
    cmd = build_command(manifest, input_pdf, out_dir, "e2e", None, 0, policy=policy, matter=None)
    return shlex.split(cmd)


@dataclass
class ToolSpec:
    slot: CandidateSlot
    manifest: CandidateManifest | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.slot.label

    @property
    def name(self) -> str:
        return self.manifest.name if self.manifest else self.slot.manifest.stem

    @property
    def kind(self) -> str:
        """'text' for a tool that outputs redacted text (adapter guide, section 8), else 'pdf'."""
        if self.manifest and ArtifactKind.REDACTED_TEXT in self.manifest.provides:
            return "text"
        return "pdf"


def _shown(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_spec(slot: CandidateSlot) -> ToolSpec:
    """The tool's manifest, and every reason it cannot run here (each one a plain message)."""
    spec = ToolSpec(slot)
    if not slot.manifest.exists():
        spec.problems.append(f"candidate manifest not found: {_shown(slot.manifest)}")
        return spec
    try:
        spec.manifest = load_candidate(slot.manifest)
    except (ValidationError, ValueError, OSError, yaml.YAMLError) as exc:
        spec.problems.append(f"candidate manifest {_shown(slot.manifest)} is invalid: "
                             f"{(str(exc).splitlines() or [type(exc).__name__])[0]}")
        return spec
    if os.name == "nt":
        spec.problems.append("the demo runs tools on Linux or macOS only")
    try:
        check_env(spec.manifest)
    except CandidateEnvError as exc:
        spec.problems.append(str(exc))
    features = shell_syntax(spec.manifest.command)
    if features:
        spec.problems.append(f"its command uses shell syntax ({' '.join(features)}), and the demo starts tools "
                             f"without a shell: put that in a wrapper script, or the variables in the manifest's env")
    return spec


# -- jobs -------------------------------------------------------------------------------------------


@dataclass(eq=False)
class ToolRun:
    index: int
    spec: ToolSpec
    state: str = WAITING
    message: str = ""
    note: str = ""
    started: float | None = None
    finished: float | None = None
    timeout_s: int | None = None
    exit_code: int | None = None
    version: str | None = None
    release: dict | None = None
    tool_note: str = ""  # the tool's own notes (run_info.json), shown with an output it did not release
    plumbing: str = ""  # why this run is not the tool as built (a plumbing check), when it says so
    preview_pages: int = 0
    page_count: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started


@dataclass(eq=False)
class Job:
    id: str
    dir: Path
    created: float
    page_count: int
    size: int
    runs: list[ToolRun]
    state: str = QUEUED
    finished: float | None = None
    original_pages: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    deleted: bool = False


def check_pdf(data: bytes, max_bytes: int) -> int:
    """The page count of an uploaded PDF, or an UploadError saying why it cannot be used."""
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadError(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")
    if b"%PDF-" not in data[:1024]:
        raise UploadError("This is not a PDF file.")
    with _MUPDF:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception:  # noqa: BLE001 - any parser failure means the same to the user
            raise UploadError("The PDF could not be opened.") from None
        with doc:
            if doc.needs_pass:
                raise UploadError("The PDF is password-protected.")
            if doc.page_count == 0:
                raise UploadError("The PDF has no pages.")
            return doc.page_count


def render_pages(pdf: Path, dest: Path, dpi: int, limit: int) -> tuple[int, int]:
    """Write PNG images of the first ``limit`` pages; return (pages rendered, pages in the PDF)."""
    dest.mkdir(exist_ok=True)
    with _MUPDF, pymupdf.open(str(pdf)) as doc:
        n = min(doc.page_count, limit)
        for i in range(n):
            doc[i].get_pixmap(dpi=dpi).save(str(dest / f"p{i + 1:03d}.png"))
        return n, doc.page_count


class JobQueue:
    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.jobs: dict[str, Job] = {}
        self._waiting: collections.deque[Job] = collections.deque()
        self._lock = threading.Condition()
        self._stop = threading.Event()
        cfg.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        purge_leftovers(cfg.work_dir)
        self._threads = [threading.Thread(target=self._worker, name="demo-worker", daemon=True),
                         threading.Thread(target=self._cleaner, name="demo-cleaner", daemon=True)]
        for t in self._threads:
            t.start()

    # -- public ------------------------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        return [load_spec(slot) for slot in self.cfg.candidates]

    def submit(self, data: bytes) -> Job:
        pages = check_pdf(data, int(self.cfg.max_upload_mb * 1024 * 1024))
        with self._lock:
            if len(self._waiting) >= MAX_WAITING:
                raise UploadError(f"{MAX_WAITING} jobs are already waiting; try again when one has finished.")
            job_id = secrets.token_hex(16)
            job_dir = self.cfg.work_dir / job_id
            job_dir.mkdir(mode=0o700)
            (job_dir / MARKER).write_text("rb demo job folder; deleted by the demo app\n", encoding="utf-8")
            (job_dir / "input.pdf").write_bytes(data)
            runs = [ToolRun(i, spec) for i, spec in enumerate(self.specs())]
            job = Job(job_id, job_dir, time.time(), pages, len(data), runs)
            self.jobs[job_id] = job
            self._waiting.append(job)
            self._lock.notify_all()
        log.info("job %s queued: %d page(s), %d tool(s)", job_id, pages, len(runs))
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id) if JOB_ID.match(job_id or "") else None

    def ahead(self, job: Job) -> int:
        """Jobs that will run before this one (0 once it runs)."""
        with self._lock:
            if job.state != QUEUED or job not in self._waiting:
                return 0
            running = any(j.state == RUNNING for j in self.jobs.values())
            return list(self._waiting).index(job) + running

    def delete(self, job_id: str) -> bool:
        """Delete a job and its folder; a running tool is stopped first (by the worker)."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return False
            self.jobs.pop(job.id, None)
            job.deleted = True
            job.cancel.set()
            if job in self._waiting:
                self._waiting.remove(job)
            if job.state != RUNNING:
                _remove(job.dir)
        log.info("job %s deleted on request", job.id)
        return True

    def shutdown(self) -> None:
        """Stop the app's work: a running tool is stopped (its whole process group), then every job folder is
        deleted, as on the Delete button. Called when the app stops (Ctrl-C, SIGTERM, exit); safe to call twice."""
        self._stop.set()
        with self._lock:
            for job in self.jobs.values():
                job.cancel.set()
            self._lock.notify_all()
        for t in self._threads:
            t.join(timeout=30)
        with self._lock:
            jobs = list(self.jobs.values())
            self.jobs.clear()
            self._waiting.clear()
        for job in jobs:
            job.deleted = True
            _remove(job.dir)
        if jobs:
            log.info("app stopped: %d job folder(s) deleted", len(jobs))

    # -- worker ------------------------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._waiting and not self._stop.is_set():
                    self._lock.wait(timeout=1)
                if self._stop.is_set():
                    return
                job = self._waiting.popleft()
                job.state = RUNNING
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 - keep the queue alive; the type only, never text
                log.error("job %s: internal error (%s)", job.id, type(exc).__name__)
                for run in job.runs:
                    if run.state in (WAITING, RUNNING):
                        run.state, run.message = FAILED, f"The demo hit an internal error ({type(exc).__name__})."
            finally:
                with self._lock:
                    job.state, job.finished = FINISHED, time.time()
                    if job.deleted:
                        _remove(job.dir)

    def _process(self, job: Job) -> None:
        if not job.cancel.is_set():
            try:
                job.original_pages, _ = render_pages(job.dir / "input.pdf", job.dir / "original",
                                                     self.cfg.preview_dpi, self.cfg.max_preview_pages)
            except Exception as exc:  # noqa: BLE001 - the tools still run without the preview
                log.warning("job %s: no page images of the upload (%s)", job.id, type(exc).__name__)
        for run in job.runs:
            if job.cancel.is_set():
                run.state, run.message = STOPPED, "Not run: the job was deleted."
                continue
            self._run_tool(job, run)

    def _run_tool(self, job: Job, run: ToolRun) -> None:
        spec = load_spec(run.spec.slot)  # read again: a manifest may have been fixed since the upload
        run.spec = spec
        if spec.problems:
            run.state, run.message = NOT_RUN, "Not run: " + "; ".join(spec.problems)
            log.info("job %s: %s not run (configuration)", job.id, spec.name)
            return
        manifest = spec.manifest
        tool_dir = job.dir / f"tool-{run.index}"
        out_dir = tool_dir / "out"
        out_dir.mkdir(parents=True)
        input_pdf = tool_dir / "input.pdf"
        shutil.copyfile(job.dir / "input.pdf", input_pdf)
        try:
            argv = command_argv(manifest, input_pdf, out_dir, self.cfg.policy)
        except (CommandValueError, CandidateEnvError, ValueError) as exc:
            run.state, run.message = NOT_RUN, f"Not run: {exc}"
            return
        env = dict(os.environ)
        env.update({k: v.format(repo=str(ROOT)) for k, v in manifest.env.items()})
        run.timeout_s = min(manifest.timeout_s, self.cfg.job_timeout_s)
        run.state, run.started = RUNNING, time.time()
        log.info("job %s: %s started", job.id, spec.name)
        with open(tool_dir / "stdout.log", "wb") as so, open(tool_dir / "stderr.log", "wb") as se:
            try:
                proc = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, env=env,
                                        cwd=str(tool_dir), start_new_session=True)
            except OSError as exc:
                run.state, run.finished = FAILED, time.time()
                run.message = f"The tool could not be started: {exc.strerror or type(exc).__name__}."
                return
            deadline = time.monotonic() + run.timeout_s
            while True:
                try:
                    run.exit_code = proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if job.cancel.is_set() or self._stop.is_set():
                    _kill_tree(proc)
                    run.state, run.finished = STOPPED, time.time()
                    run.message = "Stopped: the job was deleted." if job.cancel.is_set() else "Stopped: the app shut down."
                    return
                if time.monotonic() >= deadline:
                    _kill_tree(proc)
                    run.state, run.finished = TIMED_OUT, time.time()
                    run.message = (f"Stopped after {duration(run.timeout_s)}: the tool did not finish in time. "
                                   f"Anything it wrote is not shown.")
                    log.info("job %s: %s timed out after %ds", job.id, spec.name, run.timeout_s)
                    return
        run.finished = time.time()
        self._collect(run, out_dir, tool_dir)
        log.info("job %s: %s %s in %.0fs (exit code %s)", job.id, spec.name, run.state, run.elapsed, run.exit_code)

    def _collect(self, run: ToolRun, out_dir: Path, tool_dir: Path) -> None:
        manifest = run.spec.manifest
        with _MUPDF:  # it opens output.pdf
            artifacts, _, errors = load_artifacts(out_dir, manifest)
        run.problems = [f"{k}: {v}" for k, v in sorted(errors.items())]
        info = _read_model(out_dir / "run_info.json", RunInfo)
        if info is not None:
            run.version = info.candidate_version
            if info.release is not None:
                run.release = {"released": info.release.released, "checks_failed": list(info.release.checks_failed)}
                if not info.release.released:
                    run.tool_note = info.notes or ""
            run.plumbing = plumbing_reason(info.candidate_version, info.config)
        run.version = run.version or manifest.version
        if run.spec.kind == "text":
            ok = "redacted_text" in artifacts
        else:
            ok = (out_dir / "output.pdf").exists() and "output_pdf" not in errors
            if ok:
                try:
                    run.preview_pages, run.page_count = render_pages(out_dir / "output.pdf", tool_dir / "pages",
                                                                     self.cfg.preview_dpi, self.cfg.max_preview_pages)
                except Exception as exc:  # noqa: BLE001 - the PDF is still offered for download
                    run.note = f"Its pages could not be shown here ({type(exc).__name__}); download the PDF to see it."
        if ok:
            run.state = DONE
            if run.exit_code:
                run.note = " ".join(filter(None, [f"The tool exited with code {run.exit_code} but wrote its output; "
                                                  f"it is shown as written.", run.note]))
        else:
            run.state = FAILED
            primary = "redacted_text" if run.spec.kind == "text" else "output_pdf"
            reason = errors.get(primary) or "it wrote no output"
            run.message = f"No usable output ({reason}); the tool exited with code {run.exit_code}."

    # -- retention ---------------------------------------------------------------------------------

    def _cleaner(self) -> None:
        while not self._stop.wait(timeout=30):
            self.expire()

    def expire(self, now: float | None = None) -> list[str]:
        """Delete finished jobs older than keep_minutes; return their ids."""
        now = time.time() if now is None else now
        limit = self.cfg.keep_minutes * 60
        with self._lock:
            old = [j for j in self.jobs.values() if j.state == FINISHED and j.finished and now - j.finished >= limit]
            for job in old:
                self.jobs.pop(job.id, None)
                job.deleted = True
                _remove(job.dir)
        for job in old:
            log.info("job %s deleted after %g minutes", job.id, self.cfg.keep_minutes)
        return [j.id for j in old]


def _remove(path: Path) -> None:
    if path.is_dir() and (path / MARKER).exists():
        shutil.rmtree(path, ignore_errors=True)


def purge_leftovers(work_dir: Path) -> int:
    """Delete job folders left by an earlier start of the app (their status was in its memory)."""
    n = 0
    for child in work_dir.iterdir():
        if child.is_dir() and JOB_ID.match(child.name) and (child / MARKER).exists():
            shutil.rmtree(child, ignore_errors=True)
            n += 1
    if n:
        log.info("deleted %d job folder(s) left by an earlier start", n)
    return n


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d}s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


# -- reading a finished tool's output for display ---------------------------------------------------


def _read_model(path: Path, model):
    if not path.exists():
        return None
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValidationError, ValueError, OSError):
        return None


def replacements(job: Job, run: ToolRun) -> list[tuple[str, str, str]]:
    """(kind, original, replacement) rows: pseudonyms.json entries and the mappings in
    redacted_text.json, each pair once, in the tool's order."""
    out_dir = job.dir / f"tool-{run.index}" / "out"
    provides = set(run.spec.manifest.provides) if run.spec.manifest else set()
    rows: list[tuple[str, str, str]] = []
    if ArtifactKind.PSEUDONYMS in provides:
        art = _read_model(out_dir / "pseudonyms.json", PseudonymsArtifact)
        rows += [("", e.original, e.replacement) for e in (art.entries if art else [])]
    if ArtifactKind.REDACTED_TEXT in provides:
        art = _read_model(out_dir / "redacted_text.json", RedactedTextArtifact)
        rows += [(m.label, m.original, m.replacement) for m in (art.mappings if art else [])]
    seen, unique = set(), []
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


def redacted_text(job: Job, run: ToolRun) -> RedactedTextArtifact | None:
    return _read_model(job.dir / f"tool-{run.index}" / "out" / "redacted_text.json", RedactedTextArtifact)


def text_pages(art: RedactedTextArtifact) -> list[str]:
    return art.redacted_text.split(art.page_break) if art.page_break else [art.redacted_text]


def redacted_json(art: RedactedTextArtifact) -> bytes:
    """The redacted text as a .json download: the redacted text and its page separator only. The tool's own
    redacted_text.json also holds the original text and every original value, so it is never handed out."""
    return json.dumps({"redacted_text": art.redacted_text, "page_break": art.page_break}, ensure_ascii=False,
                      indent=1).encode("utf-8")


PLUMBING_MARK = "plumbing check"  # adapters/privategpt/run_privategpt.py puts it in the version of such a run


def plumbing_reason(version: str | None, config: dict | None = None) -> str:
    """Why a tool's run is not the tool as built, from its version line (the runner's plumbing mark) or its
    recorded settings (docling's device switched for a laptop); empty when it is the tool as built."""
    if version and PLUMBING_MARK in version:
        return version[version.index(PLUMBING_MARK):].strip(" []")
    device = (config or {}).get("docling_device_override")
    if device:
        return f"{PLUMBING_MARK}: docling on {device}; not the tool as built"
    return ""


def plain_text(art: RedactedTextArtifact) -> str:
    """The redacted text as a .txt download: pages separated by a marked line, not a form feed."""
    pages = text_pages(art)
    return "".join((f"\n\n----- page {i + 1} -----\n\n" if i else "") + p for i, p in enumerate(pages))


def output_tail(job: Job, run: ToolRun, lines: int = 20) -> str:
    """The end of a failed tool's error output, or of its standard output when it wrote no errors (a tool
    may print its reason there). Shown only on the tool's own result card, never logged."""
    for name in ("stderr.log", "stdout.log"):
        path = job.dir / f"tool-{run.index}" / name
        data = path.read_bytes()[-20000:].decode("utf-8", errors="replace").strip() if path.exists() else ""
        if data:
            return "\n".join(line[:300] for line in data.splitlines()[-lines:])
    return ""
