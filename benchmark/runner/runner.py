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
import shlex
import shutil
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


def _quote(s: str) -> str:
    return subprocess.list2cmdline([s]) if os.name == "nt" else shlex.quote(s)


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


def resolve_version(manifest: CandidateManifest) -> str:
    if manifest.version:
        return manifest.version
    out = subprocess.run(manifest.version_command, shell=True, capture_output=True, text=True, timeout=60)
    first = (out.stdout or "").strip().splitlines()
    if out.returncode != 0 or not first:
        raise RuntimeError(f"version_command failed: {out.stderr.strip()}")
    return first[0]


def build_command(manifest: CandidateManifest, input_pdf: Path, out_dir: Path, mode: str,
                  inject_dir: Path | None, seed: int) -> str:
    seed_arg = f"{manifest.seed_flag} {seed}" if manifest.seed_flag else ""
    values = {
        "python": _quote(sys.executable),
        "repo": _quote(str(ROOT)),
        "input": _quote(str(input_pdf)),
        "workdir": _quote(str(out_dir)),
        "mode": mode,
        "inject_dir": _quote(str(inject_dir)) if inject_dir else '""',
        "seed": seed_arg,
    }
    return manifest.command.format(**values)


def run_candidate(
    candidate_path: str | Path,
    dataset: str,
    experiment_path: str | Path,
    policy_path: str | Path,
    docs: list[str] | None = None,
    run_id: str | None = None,
) -> RunManifest:
    manifest = load_candidate(candidate_path)
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
    cand_version = resolve_version(manifest)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{stamp}-{manifest.name}-{exp.name}"
    root = runs_root() / run_id
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
            cmd = build_command(manifest, input_pdf, out_dir, exp.mode, inject_dir, seed)
            t0 = time.time()
            try:
                with open(work / "stdout.log", "wb") as so, open(work / "stderr.log", "wb") as se:
                    proc = subprocess.run(cmd, shell=True, stdout=so, stderr=se, env=env, cwd=str(work),
                                          timeout=manifest.timeout_s)
                item.exit_code = proc.returncode
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
