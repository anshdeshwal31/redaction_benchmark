"""Mock candidate as an external process, speaking the adapter contract (docs/adapter_guide.md).

    python -m benchmark.mocks.adapter --input input.pdf --workdir out --mode e2e \
        [--inject-dir inject] [--seed 0] --spec overlay

Mocks are test doubles and may read GT (located by the input's hash through
$RB_GT_DIRS); real candidates never can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..gt.io import find_gt, load_policy, write_model
from ..schemas import EntitiesArtifact, MentionsArtifact, PlanArtifact, RunInfo, TextArtifact
from .mock import run_mock
from .named import NAMED

ROOT = Path(__file__).resolve().parents[2]
INJECT = {"text.json": ("text", TextArtifact), "mentions.json": ("mentions", MentionsArtifact),
          "entities.json": ("entities", EntitiesArtifact), "plan.json": ("regions", PlanArtifact)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", default="e2e")
    ap.add_argument("--inject-dir", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spec", required=True, choices=sorted(NAMED))
    ap.add_argument("--policy", default=str(ROOT / "policies" / "au-medicolegal-v0.yaml"))
    a = ap.parse_args(argv)
    source = Path(a.input).read_bytes()
    gt = find_gt(sha256=hashlib.sha256(source).hexdigest())
    if gt is None:
        print("mock adapter: no GT for this input (set RB_GT_DIRS)", file=sys.stderr)
        return 2
    spec = NAMED[a.spec]
    if a.seed is not None and spec.seed is not None:
        from dataclasses import replace

        spec = replace(spec, seed=a.seed)
    injected = {}
    if a.inject_dir:
        for fname, (kind, model) in INJECT.items():
            p = Path(a.inject_dir) / fname
            if p.exists():
                injected[kind] = model.model_validate(json.loads(p.read_text(encoding="utf-8")))
    out = run_mock(source, gt, load_policy(a.policy), spec, injected=injected)
    w = Path(a.workdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / "output.pdf").write_bytes(out.pdf)
    for name, art in out.artifacts().items():
        write_model(art, w / f"{name}.json")
    write_model(RunInfo(candidate=f"mock-{spec.name}", candidate_version="0", seed=spec.seed,
                        config={"spec": spec.name, "mode": a.mode, "injected": sorted(injected)}), w / "run_info.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
