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


def test_invalid_replacements_are_tagged():
    s = scored(MockSpec("inv", pseudonyms="consistent", replace_other="invalid"))
    tagged = {a.mention_id for a in s.attribution.mentions if "PSEUDONYM_INVALID" in a.tags}
    assert {"ph1", "em1", "gov1", "addr1"} <= tagged
    assert s.attribution.counts["tags"]["PSEUDONYM_INVALID"] >= 4


def test_collision_with_merged_clusters_is_a_resolution_merge():
    # The collision mock gives every person the same pseudonym values; with clusters exposed and
    # correct the root cause stays a pseudonym collision, with merged clusters it is resolution.
    plain = scored(MockSpec("col", pseudonyms="collision"))
    assert "PSEUDONYM_COLLISION" in plain.attribution.counts["tags"]
    assert "RESOLUTION_MERGE" not in plain.attribution.counts["tags"]


def test_split_clusters_with_inconsistent_pseudonyms_is_a_resolution_split():
    # Running headers resolved as a separate person get different pseudonyms from the body text.
    s = scored(MockSpec("split", pseudonyms="consistent", split_conditions=("running_header",)))
    tags = s.attribution.counts["tags"]
    assert s.resolution.false_splits and tags.get("RESOLUTION_SPLIT", 0) > 0
    assert "PSEUDONYM_INCONSISTENT" not in tags
    # Without the clusters artifact the same symptom stays a pseudonym inconsistency.
    blind = scored(MockSpec("split", pseudonyms="consistent", split_conditions=("running_header",)), expose=False)
    assert blind.attribution.counts["tags"].get("PSEUDONYM_INCONSISTENT", 0) > 0


def test_interaction_when_own_plan_is_right_and_gt_regions_work():
    fx = digital()
    out = run_mock(fx.pdf, fx.gt, POLICY, MockSpec("annot", method="annotation"))
    plain = score_document(fx.pdf, out.pdf, fx.gt, POLICY, out.artifacts(), config=CFG)
    assert set(categories(plain)) == {"APPLICATION_OFFSET"}
    # With an oracle_regions run in which GT regions were applied safely, the tool's own correct
    # plan failing in its own engine is an interaction between localization and application.
    oracle = {"oracle_regions": {m.id: "SAFE_REMOVED" for m in fx.gt.mentions}}
    s = score_document(fx.pdf, out.pdf, fx.gt, POLICY, out.artifacts(), config=CFG, oracle=oracle)
    assert set(categories(s)) == {"INTERACTION"}
