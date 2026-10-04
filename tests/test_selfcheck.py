from pathlib import Path

import pytest

from benchmark.evaluate import EvalConfig
from benchmark.gt import load_policy
from benchmark.selfcheck import selfcheck
from benchmark.synth import digital, mrc, rotated_content

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v0.yaml")


@pytest.mark.parametrize("make", [digital, mrc, rotated_content])
def test_selfcheck_passes_on_fixtures(make):
    fx = make()
    results = selfcheck(fx.pdf, fx.gt, POLICY, EvalConfig(dpi=150, renderers=("mupdf",)))
    failed = [(r.name, r.observed) for r in results if r.passed is False]
    assert failed == []
    names = {r.name for r in results}
    assert {"reference-true", "null", "overlay", "incremental", "dpi-scale"} <= names
    if make is not digital:  # no text layer: the pseudonym check needs the measurement OCR
        assert "pseudonyms" not in names
