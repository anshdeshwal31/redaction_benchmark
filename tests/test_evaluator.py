"""Acceptance tests for the output evaluator (PLAN.md Verification).

Each mock injects one known fault; the evaluator must reach the expected verdict and
attribute it to the right evidence channel. Calibration notes for the thresholds in
EvalConfig live in docs/metrics.md (band |NCC| >= 0.6: untouched text 100% visible,
replaced glyphs ~1%, blur ~99% visible, so blur is never mistaken for redaction).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from benchmark.evaluate import SAFE, EvalConfig, MentionStatus, evaluate_output
from benchmark.evaluate.cells import Cell, PageCells, unit_flags
from benchmark.gt import load_policy
from benchmark.mocks import MockSpec, run_mock
from benchmark.schemas import GTPage
from benchmark.synth import ALL_VARIANTS

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v0.yaml")
CFG = EvalConfig(dpi=150)


@lru_cache(maxsize=None)
def fixture(name):
    return ALL_VARIANTS[name]()


@lru_cache(maxsize=None)
def evaluated(variant: str, spec: MockSpec):
    fx = fixture(variant)
    out = run_mock(fx.pdf, fx.gt, POLICY, spec)
    return evaluate_output(fx.pdf, out.pdf, fx.gt, POLICY, CFG)


def must(ev):
    return [m for m in ev.mentions if m.requirement == "MUST"]


ENCODINGS = ["digital", "raster_skew", "raster_hidden_text", "mrc", "vector", "rotated_content", "rotate_attr",
             "cropbox"]
OVERLAY_CHANNEL = {"digital": "text_layer", "raster_skew": "image_layer", "raster_hidden_text": "text_layer",
                   "mrc": "image_layer", "vector": "vector_paths", "rotated_content": "image_layer",
                   "rotate_attr": "text_layer", "cropbox": "text_layer"}


@pytest.mark.parametrize("variant", ENCODINGS)
@pytest.mark.parametrize("method", ["true", "rasterize"])
def test_reference_redactors_are_clean(variant, method):
    ev = evaluated(variant, MockSpec(f"ref-{method}", method=method))
    assert all(m.status in SAFE for m in must(ev)), [(m.mention_id, m.status) for m in must(ev)]
    assert ev.summary["clean_document"]
    assert ev.summary["collateral_words"] == 0 and ev.summary["protected_damaged"] == []


@pytest.mark.parametrize("variant", ENCODINGS)
def test_null_candidate_leaks_everything(variant):
    ev = evaluated(variant, MockSpec("null", method="null"))
    assert {m.status for m in must(ev)} == {MentionStatus.LEAK_VISIBLE}
    assert ev.summary["effective_token_recall"] == 0.0


@pytest.mark.parametrize("variant", ENCODINGS)
def test_overlay_boxes_hide_but_do_not_remove(variant):
    ev = evaluated(variant, MockSpec("overlay", method="overlay"))
    for m in must(ev):
        assert m.status is MentionStatus.LEAK_RECOVERABLE, (m.mention_id, m.status)
        assert m.visible_residual <= CFG.safe_max_residual
        assert OVERLAY_CHANNEL[variant] in m.l2_sources
    assert ev.summary["effective_token_recall"] == 1.0  # it *looks* perfect


@pytest.mark.parametrize("variant", ["digital", "mrc"])
def test_annotation_boxes_are_visible_when_annotations_are_hidden(variant):
    ev = evaluated(variant, MockSpec("annotation", method="annotation"))
    assert any("noannots" in c for c in ev.render_configs)
    for m in must(ev):
        assert m.status is MentionStatus.LEAK_VISIBLE
        assert "noannots" in m.worst_config


def test_translucent_box_leaves_latent_content():
    ev = evaluated("digital", MockSpec("translucent", method="translucent"))
    for m in must(ev):
        assert m.status is MentionStatus.LEAK_RECOVERABLE  # text layer still there
        assert m.latent_residual > 0.5  # and the glyphs are recoverable by contrast stretching


@pytest.mark.parametrize("variant", ["digital", "raster_skew", "vector"])
def test_blur_is_not_redaction(variant):
    ev = evaluated(variant, MockSpec("blur", method="blur"))
    assert {m.status for m in must(ev)} == {MentionStatus.LEAK_VISIBLE}


@pytest.mark.parametrize("variant", ["digital", "mrc", "vector"])
def test_incremental_save_keeps_original_revision(variant):
    ev = evaluated(variant, MockSpec("incremental", incremental=True))
    assert ev.revisions_checked >= 1
    for m in must(ev):
        assert m.status is MentionStatus.LEAK_RECOVERABLE and "earlier_revision" in m.l2_sources


@pytest.mark.parametrize("variant", ["digital", "raster_hidden_text"])
def test_rasterised_output_with_retained_text_layer(variant):
    ev = evaluated(variant, MockSpec("keep-text", method="rasterize", keep_text_layer=True))
    for m in must(ev):
        assert m.status is MentionStatus.LEAK_RECOVERABLE and m.l2_sources == ["text_layer"]


def test_metadata_leak_is_a_document_finding():
    ev = evaluated("digital", MockSpec("meta", leak_metadata=True))
    assert all(m.status in SAFE for m in must(ev))
    assert any(f.location == "info:author" and "E1" in f.entity_ids for f in ev.findings)
    assert not ev.summary["clean_document"] and "E1" in ev.summary["entities_leaked"]


def test_partial_boxes_are_partial_leaks():
    ev = evaluated("digital", MockSpec("partial", partial=0.5))
    assert {m.status for m in must(ev)} == {MentionStatus.PARTIAL}


@pytest.mark.parametrize("variant", ["digital", "raster_skew", "mrc"])
def test_box_shifted_half_a_line_leaves_legible_tops(variant):
    ev = evaluated(variant, MockSpec("shift", method="rasterize", shift_pt=(0, 6)))
    statuses = [m.status for m in must(ev)]
    assert statuses.count(MentionStatus.PARTIAL) >= len(statuses) // 2


def test_dpi_scale_confusion_misses_everything():
    ev = evaluated("digital", MockSpec("scale", scale=0.24))
    assert not any(m.status in SAFE for m in must(ev))


def test_over_redaction_is_measured():
    ev = evaluated("digital", MockSpec("dilate", dilate_pt=8))
    assert all(m.status in SAFE for m in must(ev))
    assert ev.summary["collateral_words"] > 0 and ev.summary["effective_token_precision"] < 1.0


def test_false_positives_on_protected_content():
    ev = evaluated("digital", MockSpec("fp", redact_protected=True))
    assert len(ev.summary["protected_damaged"]) >= 3  # homograph, eponym, medication


def test_dropped_running_headers_are_the_only_leaks():
    ev = evaluated("digital", MockSpec("drop", drop_conditions=("running_header", "running_footer")))
    leaked = {m.mention_id for m in must(ev) if m.status not in SAFE}
    assert leaked == {"hdr0", "hdr1", "ftr0", "ftr1"}


def test_pseudonymised_names_are_safe_replacements():
    ev = evaluated("digital", MockSpec("pseud", pseudonyms="consistent"))
    names = [m for m in must(ev) if m.type == "PERSON_NAME"]
    assert names and all(m.status is MentionStatus.SAFE_REPLACED for m in names)
    assert all(m.status is MentionStatus.SAFE_REMOVED for m in must(ev) if m.type != "PERSON_NAME")
    assert ev.summary["clean_document"]


def test_corrupt_output_fails_integrity():
    # A truncated file: MuPDF repairs and opens it, PDFium refuses. Integrity fails, the
    # evaluation continues with the engines that can read it, and it is never "clean".
    ev = evaluated("digital", MockSpec("corrupt", method="corrupt"))
    assert not ev.integrity.ok and not ev.summary["clean_document"]
    assert any("repair" in p or "cannot be opened" in p for p in ev.integrity.problems)


def test_rotation_flattened_into_content_is_still_evaluated():
    ev = evaluated("rotate_attr", MockSpec("ref", method="rasterize"))
    assert ev.integrity.ok
    assert any("/Rotate" in w for w in ev.integrity.validator_warnings)
    assert all(m.status in SAFE for m in must(ev))


def test_evaluator_is_deterministic():
    fx = fixture("mrc")
    out = run_mock(fx.pdf, fx.gt, POLICY, MockSpec("shift", shift_pt=(0, 6)))
    a = evaluate_output(fx.pdf, out.pdf, fx.gt, POLICY, CFG).to_dict()
    b = evaluate_output(fx.pdf, out.pdf, fx.gt, POLICY, CFG).to_dict()
    assert a == b


def _toy_cells(pattern: dict[tuple[str, int, int], bool], lengths: dict[str, int]):
    pc = PageCells(page=GTPage(index=0, width_pt=10, height_pt=10), dpi=72, shape=(1, 1))
    flags = []
    for word, n in lengths.items():
        for k in range(n):
            for b in range(3):
                pc.cells.append(Cell(word, None, k, b, np.zeros((3, 2))))
                flags.append(pattern.get((word, k, b), False))
    return pc, np.ones(len(flags), bool), np.array(flags)


def test_unit_rule_ignores_isolated_coincidences_but_keeps_runs_and_top_bands():
    # Isolated single band in one character of a long word: coincidence, ignored.
    pc, inf, fl = _toy_cells({("w", 2, 1): True}, {"w": 6})
    assert not any(unit_flags(pc, inf, fl).values())
    # Top band visible across the whole word (box shifted down half a line): counted.
    pc, inf, fl = _toy_cells({("w", k, 0): True for k in range(6)}, {"w": 6})
    assert all(unit_flags(pc, inf, fl).values())
    # One fully visible character alone in a long word: not a run, ignored.
    pc, inf, fl = _toy_cells({("w", 3, b): True for b in range(3)}, {"w": 6})
    assert not any(unit_flags(pc, inf, fl).values())
    # Initials (two-character word): a single visible character counts.
    pc, inf, fl = _toy_cells({("i", 0, b): True for b in range(3)}, {"i": 2})
    assert unit_flags(pc, inf, fl)[("i", 0)]
