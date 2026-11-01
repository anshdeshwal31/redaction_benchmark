"""Demo settings: a YAML file (demo/demo.yaml by default, or $RB_DEMO_CONFIG), then environment
variables, which win. Relative paths are taken from the benchmark repository root.

| Setting | Variable | Default |
|---|---|---|
| candidates | RB_DEMO_CANDIDATES (manifest paths, separated by os.pathsep) | demo.yaml's list |
| results_dirs | RB_DEMO_RESULTS_DIRS (folders, separated by os.pathsep) | $RB_RESULTS_DIR or results/ |
| runs | RB_DEMO_RUNS (run ids, separated by commas) | empty: every scored run in results_dirs |
| work_dir | RB_DEMO_WORK_DIR | runs/demo (git-ignored) |
| policy | RB_DEMO_POLICY | policies/au-medicolegal-v1.yaml |
| keep_minutes | RB_DEMO_KEEP_MINUTES | 120 |
| job_timeout_s | RB_DEMO_TIMEOUT_S | 1800 |
| max_upload_mb | RB_DEMO_MAX_UPLOAD_MB | 50 |
| host, port | RB_DEMO_HOST, RB_DEMO_PORT | 127.0.0.1, 8765 |
| power_off_command | RB_DEMO_POWER_OFF_COMMAND | empty: never switch the machine off (demo/power.py) |
| idle_stop_minutes | RB_DEMO_IDLE_STOP_MINUTES | 0: never switch it off |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "demo.yaml"


@dataclass
class CandidateSlot:
    """One tool column on the upload page: its manifest, the name shown above it, and a note shown under the name
    (what the demo runs of the tool, and what it leaves out)."""

    manifest: Path
    label: str
    note: str = ""


@dataclass
class DemoConfig:
    candidates: list[CandidateSlot] = field(default_factory=list)
    results_dirs: list[Path] = field(default_factory=list)
    # The scored runs the Benchmark page lists (boards are listed whole); empty: every one in results_dirs.
    runs: list[str] = field(default_factory=list)
    work_dir: Path = ROOT / "runs" / "demo"
    policy: Path = ROOT / "policies" / "au-medicolegal-v1.yaml"
    keep_minutes: float = 120
    job_timeout_s: int = 1800
    max_upload_mb: float = 50
    preview_dpi: int = 80
    max_preview_pages: int = 30
    host: str = "127.0.0.1"
    port: int = 8765
    power_off_command: str = ""
    idle_stop_minutes: float = 0


def _path(value: str | Path) -> Path:
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else ROOT / p


def _paths(value: str) -> list[Path]:
    return [_path(v) for v in value.split(os.pathsep) if v.strip()]


def load_config(path: str | Path | None = None, environ: dict | None = None) -> DemoConfig:
    env = os.environ if environ is None else environ
    path = Path(path or env.get("RB_DEMO_CONFIG") or DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"demo config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = DemoConfig()
    for item in raw.get("candidates") or []:
        if isinstance(item, str):
            item = {"manifest": item}
        manifest = _path(item["manifest"])
        cfg.candidates.append(CandidateSlot(manifest, str(item.get("label") or manifest.stem),
                                            " ".join(str(item.get("note") or "").split())))
    cfg.results_dirs = [_path(p) for p in raw.get("results_dirs") or []]
    cfg.runs = [str(r) for r in raw.get("runs") or []]
    for key in ("work_dir", "policy"):
        if raw.get(key):
            setattr(cfg, key, _path(raw[key]))
    for key, cast in (("keep_minutes", float), ("job_timeout_s", int), ("max_upload_mb", float),
                      ("preview_dpi", int), ("max_preview_pages", int), ("host", str), ("port", int),
                      ("power_off_command", str), ("idle_stop_minutes", float)):
        if raw.get(key) is not None:
            setattr(cfg, key, cast(raw[key]))

    if env.get("RB_DEMO_CANDIDATES"):
        # A manifest the file also lists keeps the file's label and note.
        listed = {c.manifest: c for c in cfg.candidates}
        cfg.candidates = [listed.get(p) or CandidateSlot(p, p.stem) for p in _paths(env["RB_DEMO_CANDIDATES"])]
    if env.get("RB_DEMO_RESULTS_DIRS"):
        cfg.results_dirs = _paths(env["RB_DEMO_RESULTS_DIRS"])
    if env.get("RB_DEMO_RUNS"):
        cfg.runs = [r.strip() for r in env["RB_DEMO_RUNS"].split(",") if r.strip()]
    if not cfg.results_dirs:
        cfg.results_dirs = [_path(env.get("RB_RESULTS_DIR") or ROOT / "results")]
    for var, key, cast in (("RB_DEMO_WORK_DIR", "work_dir", _path), ("RB_DEMO_POLICY", "policy", _path),
                           ("RB_DEMO_KEEP_MINUTES", "keep_minutes", float),
                           ("RB_DEMO_TIMEOUT_S", "job_timeout_s", int),
                           ("RB_DEMO_MAX_UPLOAD_MB", "max_upload_mb", float),
                           ("RB_DEMO_HOST", "host", str), ("RB_DEMO_PORT", "port", int),
                           ("RB_DEMO_POWER_OFF_COMMAND", "power_off_command", str),
                           ("RB_DEMO_IDLE_STOP_MINUTES", "idle_stop_minutes", float)):
        if env.get(var):
            setattr(cfg, key, cast(env[var]))
    return cfg
