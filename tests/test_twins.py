"""Controlled inputs: clean twins and degradation ladders (PLAN.md §4.3, phase 7)."""

import tempfile
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from benchmark.dataset import describe_document
from benchmark.evaluate import SAFE, EvalConfig, MentionStatus, evaluate_output
from benchmark.geometry import as_poly, dilate, rasterize
from benchmark.gt import load_policy, validate_gt
from benchmark.mocks import MockSpec, run_mock
from benchmark.synth import digital, raster, rotate_attr, rotated_content
from benchmark.synth.twins import clean_twin, degrade, ladder

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v0.yaml")


def ink_alignment(fx, dpi=100) -> float:
    doc = pymupdf.open("pdf", fx.pdf)
    s = dpi / 72
    tot = cov = 0
    for p in fx.gt.pages:
        pix = doc[p.index].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        ink = g < 128
        zone = rasterize([dilate(as_poly(w.poly), 1) for w in p.words] + [dilate(as_poly(r.poly), 1) for r in p.regions],
                         g.shape, s)
        tot += ink.sum()
        cov += (ink & zone).sum()
    return cov / tot


@pytest.mark.parametrize("make", [raster, rotated_content, rotate_attr])
def test_clean_twin_matches_gt(make):
    fx = make()
    twin = clean_twin(fx.pdf, fx.gt)
    assert [i for i in validate_gt(twin.gt) if i.severity == "error"] == []
    assert ink_alignment(twin) > 0.98
    assert pymupdf.open("pdf", twin.pdf)[0].get_text().strip()  # born-digital: real text layer


@pytest.mark.parametrize("factor", ["dpi", "skew", "jpeg", "noise", "blur"])
def test_ladder_steps_keep_gt_aligned(factor):
    fx = digital()
    for step in ladder(fx.pdf, fx.gt, factor):
        assert ink_alignment(step) > 0.97, step.name


def _measure(fx):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pdf"
        p.write_bytes(fx.pdf)
        return describe_document(p, "x", "synthetic", None).page_conditions[0]


def test_condition_measurement_tracks_the_ladders():
    fx = digital()
    dpis = [_measure(s).full_page_image_dpi for s in ladder(fx.pdf, fx.gt, "dpi")]
    assert dpis == sorted(dpis, reverse=True) and abs(dpis[0] - 300) < 2
    skews = [_measure(s).skew_deg for s in ladder(fx.pdf, fx.gt, "skew")]
    assert all(abs(m - t) <= 0.25 for m, t in zip(skews, (0.0, 0.5, 1.5, 3.0))), skews


@pytest.mark.parametrize("params", [{"dpi": 100}, {"skew_deg": 3.0}, {"jpeg_quality": 10}, {"noise_sigma": 45.0},
                                    {"blur_sigma": 2.5}])
def test_evaluator_verdicts_survive_degradation(params):
    fx = degrade(digital().pdf, digital().gt, **params)
    cfg = EvalConfig(dpi=150, renderers=("mupdf",))
    ref = evaluate_output(fx.pdf, run_mock(fx.pdf, fx.gt, POLICY, MockSpec("r", method="rasterize")).pdf, fx.gt,
                          POLICY, cfg)
    assert all(m.status in SAFE for m in ref.mentions if m.requirement == "MUST"), params
    null = evaluate_output(fx.pdf, fx.pdf, fx.gt, POLICY, cfg)
    assert all(m.status is MentionStatus.LEAK_VISIBLE for m in null.mentions if m.requirement == "MUST"), params
