"""Run a candidate over a dataset through its adapter command (PLAN.md §9, §14).

Each (document, repeat) gets its own working directory containing only a neutrally named
``input.pdf`` and, for oracle experiments, the injected GT-derived artifacts. GT itself is
never placed there. Every input and output is hashed into the run manifest.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .. import __version__
from ..dataset import document_pdf, load_manifest, manifest_path
from ..gt.io import find_gt, load_policy, read_model, write_model
from ..schemas import CandidateManifest, InjectionKind
from ..schemas.run import MODE_INJECTS, ExperimentSpec, RunItem, RunManifest
from .artifacts import load_artifacts, sha256_file
from .oracle import materialise, materialise_from_run

ROOT = Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return Path(os.environ.get("RB_RUNS_DIR", ROOT / "runs"))


class CommandValueError(ValueError):
    """A value cannot be passed to a candidate command unchanged through the shell."""


# Commands run through the shell: cmd.exe on Windows. Inside double quotes cmd.exe leaves & | < > ^ ( )
# alone, but it still expands %NAME% (and !NAME! with delayed expansion), a double quote ends the quoting,
# and a line break ends the command.
_CMD_UNSAFE = re.compile(r'["%!\r\n]')


def _quote(s: str, what: str = "a value") -> str:
    """Quote one substituted value so the tool receives it unchanged as one argument. POSIX shells
    take any value in single quotes; on Windows the value goes in double quotes, and a value cmd.exe
    would still change is refused with an error naming it, never passed on altered."""
    if os.name != "nt":
        return shlex.quote(s)
    bad = sorted(set(_CMD_UNSAFE.findall(s)))
    if bad:
        shown = ", ".join("a line break" if c in "\r\n" else repr(c) for c in bad)
        raise CommandValueError(f"{what} contains {shown}, which cmd.exe would change or act on; "
                                f"rename it or move it to a path without these characters: {s!r}")
    # Backslashes right before the closing quote are doubled so the program's own argument parsing
    # (the C runtime's rules) does not read the quote as part of the value.
    return f'"{s}{chr(92) * (len(s) - len(s.rstrip(chr(92))))}"'


# {env:NAME} reaches a tool's own interpreter or install (RB_MLREDACT_PYTHON, RB_REDACTOR_PYTHON, ...).
_ENV_PLACEHOLDER = re.compile(r"\{env:([^{}]*)\}")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class CandidateEnvError(RuntimeError):
    """A candidate command names an environment variable that is not set, not a valid name, or holds
    a value that cannot be passed through the shell."""


def _env_value(name: str) -> str:
    if not _ENV_NAME.match(name):
        raise CandidateEnvError(f"invalid placeholder {{env:{name}}}: the variable name must match [A-Z][A-Z0-9_]*")
    value = os.environ.get(name, "")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]  # cmd's `set X="C:\a b\python.exe"` keeps the quotes; they are not part of the path
    if not value:
        raise CandidateEnvError(f"environment variable {name} is not set; set it to the tool's interpreter or path")
    return value


def _env_quoted(name: str) -> str:
    value = _env_value(name)
    try:
        return _quote(value, f"environment variable {name}")
    except CommandValueError as exc:
        raise CandidateEnvError(str(exc)) from None


def check_env(manifest: CandidateManifest) -> None:
    """Fail before anything runs when the command or version_command needs a variable that is not
    set, or one whose value the shell would change."""
    for template in (manifest.command, manifest.version_command):
        for name in _ENV_PLACEHOLDER.findall(template or ""):
            try:
                _env_quoted(name)
            except CandidateEnvError as exc:
                raise CandidateEnvError(f"candidate {manifest.name}: {exc}") from None


def expand(template: str, values: dict[str, str]) -> str:
    """Fill a command template: {env:NAME} first (str.format cannot parse it), then the named
    placeholders. Substituted paths and values arrive at the tool unchanged, one argument each:
    ``_quote`` quotes them for the shell, or refuses a value the shell would change."""
    def env(match: re.Match) -> str:
        return _env_quoted(match.group(1)).replace("{", "{{").replace("}", "}}")

    try:
        return _ENV_PLACEHOLDER.sub(env, template).format(**values)
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}} in {template!r}") from None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop a command and every process it started. On Windows the shell is all that Popen.kill
    stops, and the tool would keep running beside the next document."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=60)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the command runs in a session of its own
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_command(cmd: str, stdout, stderr, env: dict, cwd: str, timeout: float) -> int:
    """Run a candidate command through the shell. On a timeout (or Ctrl+C) the whole process tree
    is stopped before the exception propagates, so no tool keeps writing after its item is recorded."""
    extra = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr, env=env, cwd=cwd, **extra)
    try:
        return proc.wait(timeout=timeout)
    except BaseException:
        _kill_tree(proc)
        raise


def _git_state() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment() -> dict:
    import importlib.metadata as md

    pkgs = {}
    for name in ("pymupdf", "pypdfium2", "pikepdf", "numpy", "opencv-python-headless", "shapely", "rapidfuzz"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {"os": platform.platform(), "python": sys.version.split()[0], "packages": pkgs}


def load_candidate(path: str | Path) -> CandidateManifest:
    return read_model(CandidateManifest, path)


def load_experiment(path: str | Path) -> ExperimentSpec:
    return read_model(ExperimentSpec, path)


def _common_values() -> dict[str, str]:
    return {"python": _quote(sys.executable, "the benchmark's Python interpreter ({python})"),
            "repo": _quote(str(ROOT), "the repository path ({repo})")}


def resolve_version(manifest: CandidateManifest) -> str:
    if manifest.version:
        return manifest.version
    cmd = expand(manifest.version_command, _common_values())
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    first = (out.stdout or "").strip().splitlines()
    if out.returncode != 0 or not first:
        raise RuntimeError(f"version_command {manifest.version_command!r} failed (exit code {out.returncode}): "
                           f"{(out.stderr or '').strip() or 'it printed no version'}")
    return first[0]


def build_command(manifest: CandidateManifest, input_pdf: Path, out_dir: Path, mode: str,
                  inject_dir: Path | None, seed: int, policy: str | Path | None = None,
                  matter: str | None = None) -> str:
    seed_arg = f"{manifest.seed_flag} {seed}" if manifest.seed_flag else ""
    values = {
        **_common_values(),
        "input": _quote(str(input_pdf), "the input path ({input})"),
        "workdir": _quote(str(out_dir), "the working directory ({workdir})"),
        "mode": mode,
        "inject_dir": _quote(str(inject_dir), "the inject directory ({inject_dir})") if inject_dir else '""',
        "seed": seed_arg,
        "policy": _quote(str(Path(policy).resolve()), "the policy path ({policy})") if policy else '""',
        "matter": _quote(matter, "the matter id ({matter})") if matter else '""',
    }
    return expand(manifest.command, values)


def run_candidate(
    candidate_path: str | Path,
    dataset: str,
    experiment_path: str | Path,
    policy_path: str | Path,
    docs: list[str] | None = None,
    run_id: str | None = None,
) -> RunManifest:
    manifest = load_candidate(candidate_path)
    check_env(manifest)
    exp = load_experiment(experiment_path)
    needed = MODE_INJECTS[exp.mode]
    missing = [k.value for k in needed if k not in manifest.accepts_injection]
    if missing:
        raise ValueError(f"candidate {manifest.name} does not accept injected {missing}; "
                         f"experiment {exp.name} is not applicable (report N/A)")
    if needed and "{inject_dir}" not in manifest.command:
        raise ValueError("experiment injects artifacts but the command template has no {inject_dir}")
    name, version = dataset.split("@")
    ds = load_manifest(name, version)
    policy = load_policy(policy_path)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{stamp}-{manifest.name}-{exp.name}"
    root = runs_root() / run_id
    # Values shared by every item: refused here, before anything runs, if the shell would change them.
    _common_values()
    _quote(str(Path(policy_path).resolve()), "the policy path ({policy})")
    _quote(str(root), "the run directory")
    cand_version = resolve_version(manifest)
    if root.exists():
        raise FileExistsError(root)
    cand_file = Path(candidate_path).read_bytes()
    run = RunManifest(
        run_id=run_id,
        created=stamp,
        benchmark={"version": __version__, **_git_state()},
        dataset={"name": name, "version": version,
                 "manifest_sha256": hashlib.sha256(manifest_path(name, version).read_bytes()).hexdigest()},
        candidate={"name": manifest.name, "version": cand_version,
                   "manifest_sha256": hashlib.sha256(cand_file).hexdigest(),
                   "manifest": manifest.model_dump(mode="json", by_alias=True)},
        experiment=exp,
        environment=environment(),
    )
    env = dict(os.environ)
    env.update({k: v.format(repo=str(ROOT)) for k, v in manifest.env.items()})
    for doc in ds.documents:
        if docs and doc.doc_id not in docs:
            continue
        src = document_pdf(doc)
        for rep in range(exp.repeats):
            work = root / doc.doc_id / f"r{rep}"
            out_dir = work / "out"
            out_dir.mkdir(parents=True)
            input_pdf = work / "input.pdf"
            shutil.copyfile(src, input_pdf)
            seed = rep if exp.vary_seed else 0
            item = RunItem(doc_id=doc.doc_id, repeat=rep, workdir=str(work.relative_to(root)),
                           input_sha256=sha256_file(input_pdf), seed=seed)
            inject_dir = None
            if needed and exp.inject_from.startswith("run:"):
                inject_dir = work / "inject"
                try:
                    paths = materialise_from_run(needed, runs_root() / exp.inject_from[4:], doc.doc_id, input_pdf,
                                                 inject_dir)
                except FileNotFoundError as exc:
                    item.error = f"stage swap: {exc}"
                    run.items.append(item)
                    continue
                item.injected = {k: sha256_file(p) for k, p in paths.items()}
            elif needed:
                gt = find_gt(doc.doc_id, doc.sha256)
                if gt is None:
                    item.error = "no GT available for oracle injection"
                    run.items.append(item)
                    continue
                inject_dir = work / "inject"
                paths = materialise(needed, gt, policy, inject_dir)
                item.injected = {k: sha256_file(p) for k, p in paths.items()}
            try:
                cmd = build_command(manifest, input_pdf, out_dir, exp.mode, inject_dir, seed, policy=policy_path,
                                    matter=doc.matter)
            except CommandValueError as exc:
                item.error = f"not run: {exc}"
                run.items.append(item)
                continue
            t0 = time.time()
            try:
                with open(work / "stdout.log", "wb") as so, open(work / "stderr.log", "wb") as se:
                    item.exit_code = run_command(cmd, so, se, env, str(work), manifest.timeout_s)
            except subprocess.TimeoutExpired:
                item.timed_out = True
                item.error = f"timed out after {manifest.timeout_s}s"
            item.duration_s = round(time.time() - t0, 3)
            _, hashes, errors = load_artifacts(out_dir, manifest)
            item.artifacts = hashes
            item.invalid_artifacts = errors
            if item.exit_code not in (0, None) and not item.error:
                item.error = f"exit code {item.exit_code}"
            run.items.append(item)
    write_model(run, root / "run.json")
    return run


def load_run(run_id: str) -> RunManifest:
    return read_model(RunManifest, runs_root() / run_id / "run.json")
