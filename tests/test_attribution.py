"""Attribution acceptance: each injected fault is traced to the stage that caused it."""

from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from benchmark.evaluate import EvalConfig
from benchmark.gt import load_policy
from benchmark.mocks import MockSpec, run_mock
from benchmark.score import score_document
from benchmark.synth import digital

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v0.yaml")
CFG = EvalConfig(dpi=150)


@lru_cache(maxsize=None)
def scored(spec: MockSpec, expose: bool = True):
    fx = digital()
    out = run_mock(fx.pdf, fx.gt, POLICY, spec)
    return score_document(fx.pdf, out.pdf, fx.gt, POLICY, out.artifacts() if expose else {}, config=CFG)


def categories(s):
    return Counter(a.category for a in s.attribution.mentions if a.category != "SAFE")


def confidences(s):
    return {a.confidence for a in s.attribution.mentions if a.category != "SAFE"}


@pytest.mark.parametrize("spec,expected", [
    (MockSpec("ocr", ocr_drop_types=("PERSON_NAME",)), "TEXT_MISSING"),
    (MockSpec("shrink", boundary="shrink_last"), "DETECTION_BOUNDARY"),
    (MockSpec("miss", drop_conditions=("running_header",)), "DETECTION_ALIAS_PROPAGATION"),
    (MockSpec("scale", scale=0.24), "LOCALIZATION_SYSTEMATIC"),
    (MockSpec("page", page_offset=1), "WRONG_PAGE"),
    (MockSpec("overlay", method="overlay"), "REMOVAL_INCOMPLETE"),
    (MockSpec("ignored", method="null"), "APPLICATION_NOT_APPLIED"),
])
def test_white_box_root_causes_are_confirmed(spec, expected):
    s = scored(spec)
    cats = categories(s)
    assert set(cats) == {expected}, cats
    assert confidences(s) == {"confirmed"}


def test_funnel_shows_where_mentions_are_lost():
    f = scored(MockSpec("ocr", ocr_drop_types=("PERSON_NAME",))).attribution.funnel
    assert f["required"] == 13 and f["recognised"] == 4 and f["truly_safe"] == 4
    f = scored(MockSpec("overlay", method="overlay")).attribution.funnel
    assert f["visually_safe"] == 13 and f["truly_safe"] == 0  # looked perfect, was not


@pytest.mark.parametrize("spec,expected", [
    (MockSpec("bb-shift", method="rasterize", shift_pt=(0, 6)), "LOCALIZATION_LOCAL"),
    (MockSpec("bb-drop", drop_conditions=("running_header",)), "DETECTION_MISS"),
])
def test_black_box_causes_are_only_inferred(spec, expected):
    s = scored(spec, expose=False)
    assert s.stages_exposed == [] and expected in categories(s)
    assert confidences(s) == {"inferred"}


def test_over_redaction_attribution():
    s = scored(MockSpec("dilate", dilate_pt=8))
    over = s.attribution.counts["over_redaction"]
    # Oversized boxes damage neighbours they cover (localization), and the redaction engine
    # additionally removes whole characters whose boxes merely intersect a region (application).
    assert over.get("LOCALIZATION_SPILL", 0) > 0 and set(over) <= {"LOCALIZATION_SPILL", "APPLICATION_SPILL"}
    fp = scored(MockSpec("fp", redact_protected=True)).attribution.counts["over_redaction"]
    assert set(fp) == {"FP_DETECTION"}


def test_pseudonym_inconsistency_is_tagged_and_resolution_inferred():
    s = scored(MockSpec("pi", pseudonyms="inconsistent"), expose=False)
    assert s.resolution is not None and s.resolution.source == "inferred_from_pseudonyms"
    assert s.resolution.pairwise["recall"] < 1.0
    assert any("PSEUDONYM_INCONSISTENT" in a.tags for a in s.attribution.mentions)


def test_missing_stages_are_not_applicable_not_zero():
    s = scored(MockSpec("bb-drop", drop_conditions=("running_header",)), expose=False)
    assert s.text is None and s.detection is None and s.localization is None
    d = s.to_dict()
    assert d["text"] is None and d["detection"] is None
