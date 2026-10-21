"""Component metrics: hand-computed coreference fixtures and mock-driven acceptance tests."""

from functools import lru_cache
from pathlib import Path

import pytest

from benchmark.evaluate import EvalConfig, evaluate_output
from benchmark.gt import load_policy, resolve
from benchmark.metrics import (
    InkIndex,
    b_cubed,
    ceaf_e,
    compare_runs,
    evaluate_detection,
    evaluate_plan,
    evaluate_pseudonyms,
    evaluate_resolution,
    evaluate_text,
    lea,
    muc,
    plan_fidelity,
    project_mentions,
    replacements_by_mention,
    response_from_artifacts,
    response_from_replacements,
)
from benchmark.mocks import MockSpec, run_mock
from benchmark.synth import ALL_VARIANTS

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v0.yaml")


@lru_cache(maxsize=None)
def setup(variant):
    fx = ALL_VARIANTS[variant]()
    return fx, resolve(fx.gt, POLICY), InkIndex.build(fx.gt, fx.pdf, 100)


@lru_cache(maxsize=None)
def mock(variant, spec):
    fx, _, _ = setup(variant)
    return run_mock(fx.pdf, fx.gt, POLICY, spec)


def detection(variant, spec):
    fx, res, idx = setup(variant)
    out = mock(variant, spec)
    return evaluate_detection(fx.gt, res, project_mentions(out.mentions, idx, text=out.text))


# -- coreference metrics on a textbook example ---------------------------------------------
# key: {a,b,c} {d,e}; response: {a,b} {c,d,e}

KEY = {"a": "K1", "b": "K1", "c": "K1", "d": "K2", "e": "K2"}
RESP = {"a": "R1", "b": "R1", "c": "R2", "d": "R2", "e": "R2"}


def test_muc_textbook():
    s = muc(KEY, RESP)
    assert s["recall"] == pytest.approx(2 / 3, abs=1e-4)  # (3-2 + 2-1) / (2 + 1)
    assert s["precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_b_cubed_textbook():
    s = b_cubed(KEY, RESP)
    # recall: a 2/3, b 2/3, c 1/3, d 1, e 1 -> 11/15; precision: a 1, b 1, c 1/3, d 2/3, e 2/3 -> 11/15
    assert s["recall"] == pytest.approx(11 / 15, abs=1e-4)
    assert s["precision"] == pytest.approx(11 / 15, abs=1e-4)


def test_ceaf_e_textbook():
    s = ceaf_e(KEY, RESP)
    # best alignment K1-R1 (2*2/5=0.8) + K2-R2 (2*2/5=0.8) = 1.6 over 2 entities each side
    assert s["recall"] == pytest.approx(0.8, abs=1e-4) and s["precision"] == pytest.approx(0.8, abs=1e-4)


def test_lea_and_perfect_scores():
    assert lea(KEY, KEY)["f1"] == 1.0 and muc(KEY, KEY)["f1"] == 1.0 and ceaf_e(KEY, KEY)["f1"] == 1.0
    s = lea(KEY, RESP)
    # recall: K1 importance 3, links 3, resolved links {a,b}=1 -> 1/3; K2: 1 of 1 -> (3*1/3 + 2*1)/5 = 0.6
    assert s["recall"] == pytest.approx(0.6, abs=1e-4)


def test_pairwise_lists_false_merges_and_splits():
    r = evaluate_resolution(KEY, RESP)
    assert r.false_merges == [{"entities": ["K1", "K2"], "pairs": 2}]  # (c,d) and (c,e)
    assert r.false_splits == [{"entity": "K1", "pairs": 2}]  # (a,c) and (b,c)


def test_union_find_inference_from_replacements():
    inferred = response_from_replacements({"m1": {"surname": "Baxter", "given": "Anna"},
                                           "m2": {"surname": "baxter"}, "m3": {"given": "Owen"}})
    assert inferred["m1"] == inferred["m2"] != inferred["m3"]


# -- detection -------------------------------------------------------------------------------


def test_perfect_detection():
    d = detection("digital", MockSpec("perfect"))
    assert d.token["precision"] == 1.0 and d.token["recall"] == 1.0
    assert set(d.boundary_counts) == {"exact"} and d.mention_scores["strict"]["recall"] == 1.0


def test_dropped_last_word_is_under_extension_naming_the_component():
    d = detection("digital", MockSpec("shrink", boundary="shrink_last"))
    m1 = next(x for x in d.matches if x.mention_id == "m1")
    assert m1.boundary == "under" and m1.missing_components == ["surname"]
    assert d.token["recall"] < 1.0 and d.token["precision"] == 1.0


def test_titles_are_benign_over_extension():
    d = detection("digital", MockSpec("title", boundary="extend_title"))
    m1 = next(x for x in d.matches if x.mention_id == "m1")
    assert m1.boundary == "over_benign" and m1.tolerant and not m1.exact
    assert d.token["precision"] == 1.0  # optional words are not false positives


def test_swallowing_the_next_word_is_harmful():
    d = detection("digital", MockSpec("next", boundary="extend_next"))
    assert d.boundary_counts.get("over_harmful", 0) + d.boundary_counts.get("merged", 0) > 0
    assert d.token["precision"] < 1.0 and d.fp_classes


def test_type_swap_breaks_strict_but_not_exact():
    d = detection("digital", MockSpec("swap", type_swap=(("PERSON_NAME", "ORGANISATION"),)))
    assert d.mention_scores["exact_boundary"]["recall"] == 1.0
    assert d.mention_scores["strict"]["recall"] < 0.5
    assert d.type_confusion.get("PERSON_NAME->ORGANISATION") == 9


def test_protected_false_positives_are_classified():
    d = detection("digital", MockSpec("fp", redact_protected=True))
    assert {"HOMOGRAPH_COMMON_WORD", "MEDICATION", "MEDICAL_EPONYM"} <= set(d.fp_classes)


# -- text acquisition ---------------------------------------------------------------------------


def test_perfect_ocr_and_names_missed_by_ocr():
    fx, res, idx = setup("digital")
    perfect = evaluate_text(mock("digital", MockSpec("p")).text, fx.gt, res, idx)
    assert perfect.general["cer_raw"] == 0 and perfect.mention_recognisability["rate"] == 1.0
    names_lost = evaluate_text(mock("digital", MockSpec("o", ocr_drop_types=("PERSON_NAME",))).text, fx.gt, res, idx)
    assert names_lost.pii["missed_rate"] > 0.5
    assert names_lost.pii["missed_rate"] > 2 * names_lost.general["missed_rate"]  # good OCR overall, bad on names
    assert names_lost.mention_recognisability["rate"] < 0.5


def test_ocr_character_noise():
    fx, res, idx = setup("digital")
    r = evaluate_text(mock("digital", MockSpec("n", ocr_char_error=0.1)).text, fx.gt, res, idx)
    assert 0.05 < r.general["cer_raw"] < 0.15 and r.general["missed_rate"] == 0


# -- localization and transform diagnosis -------------------------------------------------------


@pytest.mark.parametrize("variant,spec,expected", [
    ("digital", MockSpec("ok"), "aligned"),
    ("digital", MockSpec("scale", scale=0.24), "systematic: scale 72/300"),
    ("digital", MockSpec("yflip", y_flip=True), "systematic: y-flip"),
    ("digital", MockSpec("page", page_offset=1), "systematic: page offset +1"),
    ("digital", MockSpec("shift", shift_pt=(15, 4)), "systematic: translation"),
    ("cropbox", MockSpec("crop", cropbox_unaware=True), "systematic: mediabox origin used"),
    ("rotate_attr", MockSpec("rot", rotation_unaware=True), "systematic: unrotated frame used"),
])
def test_transform_diagnosis(variant, spec, expected):
    fx, res, idx = setup(variant)
    loc = evaluate_plan(mock(variant, spec).plan, fx.gt, res, idx)
    assert loc.document_diagnosis == expected


def test_translation_parameters_and_oversized_boxes():
    fx, res, idx = setup("digital")
    loc = evaluate_plan(mock("digital", MockSpec("shift", shift_pt=(15, 4))).plan, fx.gt, res, idx)
    d = next(x for x in loc.diagnoses if x.hypothesis == "translation")
    assert d.params == {"dx": 15.0, "dy": 4.0}
    big = evaluate_plan(mock("digital", MockSpec("dilate", dilate_pt=8)).plan, fx.gt, res, idx)
    assert big.document_diagnosis == "aligned" and big.must_full_coverage_rate == 1.0
    assert {d.hypothesis for d in big.diagnoses} == {"oversized regions"}
    assert big.edge_offsets_pt["left"] == -8.0 and big.collateral_words


def test_partial_coverage():
    fx, res, idx = setup("digital")
    loc = evaluate_plan(mock("digital", MockSpec("partial", partial=0.5)).plan, fx.gt, res, idx)
    assert 0.4 < loc.must_mean_coverage < 0.6 and loc.must_full_coverage_rate == 0.0


def test_plan_fidelity_detects_unapplied_plans():
    fx, _, idx = setup("digital")
    applied = mock("digital", MockSpec("ok"))
    ok = plan_fidelity(applied.plan, fx.pdf, applied.pdf, idx)
    assert ok.not_applied == 0
    ignored = mock("digital", MockSpec("null", method="null"))  # plans everything, changes nothing
    bad = plan_fidelity(ignored.plan, fx.pdf, ignored.pdf, idx)
    assert bad.not_applied == len(bad.regions) > 0


# -- resolution and pseudonyms -------------------------------------------------------------------


def _resolution(variant, spec):
    fx, res, idx = setup(variant)
    out = mock(variant, spec)
    det = evaluate_detection(fx.gt, res, project_mentions(out.mentions, idx))
    key = {m.id: m.entity for m in fx.gt.mentions
           if m.type.value == "PERSON_NAME" and m.entity and res.mentions[m.id].requirement.value == "MUST"}
    clusters = {mid: c.id for c in out.entities.clusters for mid in c.mentions}
    return evaluate_resolution(key, response_from_artifacts(
        {x.mention_id: x.best_pred for x in det.matches if x.mention_id in key}, clusters))


def test_resolution_false_merge_of_people_sharing_a_first_name():
    r = _resolution("rotated_content", MockSpec("merge", merge_same_given=True))
    assert r.false_merges == [{"entities": ["N1", "N3"], "pairs": 2}] and r.pairwise["recall"] == 1.0


def test_resolution_false_split_of_aliases():
    r = _resolution("digital", MockSpec("split", split_aliases=True))
    assert r.false_splits and r.false_splits[0]["entity"] == "E1" and r.pairwise["precision"] == 1.0
    assert _resolution("digital", MockSpec("ok")).b_cubed["f1"] == 1.0


@pytest.mark.parametrize("mode,check,expect", [
    ("consistent", "value_consistency_rate", 1.0),
    ("inconsistent", "value_consistency_rate", lambda v: v < 1.0),
    ("gender_flip", "honorific_gender_mismatch", lambda v: v >= 1),
    ("collision", "collisions", lambda v: v >= 1),
])
def test_pseudonym_checks(mode, check, expect):
    fx, res, _ = setup("digital")
    r = evaluate_pseudonyms(fx.gt, res, mock("digital", MockSpec(mode, pseudonyms=mode)).pdf)
    assert r.evaluable and r.checks["survived"] == 0
    value = r.checks[check]
    assert expect(value) if callable(expect) else value == expect


def test_consistent_pseudonyms_infer_a_perfect_resolution():
    fx, res, _ = setup("digital")
    r = evaluate_pseudonyms(fx.gt, res, mock("digital", MockSpec("c", pseudonyms="consistent")).pdf)
    key = {m.id: m.entity for m in fx.gt.mentions if m.type.value == "PERSON_NAME" and m.entity
           and res.mentions[m.id].requirement.value == "MUST"}
    inferred = response_from_replacements({k: v for k, v in replacements_by_mention(r).items() if k in key})
    rr = evaluate_resolution(key, inferred, source="inferred_from_pseudonyms")
    assert rr.b_cubed["f1"] == 1.0 and not rr.false_splits


def test_image_only_output_is_not_evaluable_for_pseudonyms():
    fx, res, _ = setup("digital")
    r = evaluate_pseudonyms(fx.gt, res, mock("digital", MockSpec("r", method="rasterize", pseudonyms="consistent")).pdf)
    assert not r.evaluable and "OCR" in r.reason


# -- determinism ----------------------------------------------------------------------------------


def _runs(spec, n=3):
    fx, _, idx = setup("digital")
    cfg = EvalConfig(dpi=150, renderers=("mupdf",))
    outs = [run_mock(fx.pdf, fx.gt, POLICY, spec) for _ in range(n)]
    evs = [evaluate_output(fx.pdf, o.pdf, fx.gt, POLICY, cfg) for o in outs]
    return compare_runs(fx.pdf, [o.pdf for o in outs], [o.artifacts() for o in outs], evs, idx)


def test_deterministic_tool():
    d = _runs(MockSpec("det", pseudonyms="consistent"))
    assert d.d0_exact and all(d.d1_structural.values()) and d.d2_visual_exact and d.d3_decisions_identical


def test_nondeterministic_tool_is_flagged():
    d = _runs(MockSpec("nondet", seed=None, drop_fraction=0.4), n=4)
    assert not d.d3_decisions_identical and d.unstable_mentions and d.flip_rate > 0
    assert d.d1_structural["mentions"] is False and d.d1_structural["text"] is True


# -- validity of replaced contact details and identifiers ----------------------------------


def test_valid_contact_replacements_pass_format_checks():
    fx, res, _ = setup("digital")
    r = evaluate_pseudonyms(fx.gt, res, mock("digital", MockSpec("v", pseudonyms="consistent",
                                                                  replace_other="valid")).pdf)
    c = r.checks
    assert c["other_replaced"] == 4 and c["invalid_replacements"] == 0
    assert c["phone_valid_rate"] == c["email_valid_rate"] == c["email_from_new_name_rate"] == 1.0
    assert c["medicare_checksum_valid_rate"] == 1.0
    # GT 0.2 addresses are street lines (suburb, state and postcode are kept), so there is no
    # replaced postcode to check against its state.
    assert c["postcode_state_valid_rate"] is None


def test_invalid_contact_replacements_are_flagged():
    fx, res, _ = setup("digital")
    r = evaluate_pseudonyms(fx.gt, res, mock("digital", MockSpec("i", pseudonyms="consistent",
                                                                  replace_other="invalid")).pdf)
    assert set(r.issues["invalid"]) == {"ph1", "em1", "gov1"}
    assert r.checks["postcode_state_valid_rate"] is None  # street line only since GT 0.2


@pytest.mark.parametrize("text,ok", [("0400 123 456", True), ("(07) 5575 2444", True), ("+61 2 5550 1234", True),
                                     ("1300 333 767", True), ("12 34", False), ("0123 456", False)])
def test_phone_formats(text, ok):
    from benchmark.metrics.pseudonyms import phone_valid

    assert phone_valid(text) is ok


@pytest.mark.parametrize("text,ok", [("Mermaid Beach QLD 4218", True), ("Brisbane QLD 4000", True),
                                     ("Sydney NSW 2000", True), ("Canberra ACT 2600", True),
                                     ("Darwin NT 0800", True), ("Sydney NSW 4000", False), ("no postcode", None)])
def test_postcode_state(text, ok):
    from benchmark.metrics.pseudonyms import postcode_state_valid

    assert postcode_state_valid(text) is ok


# -- image-only outputs read with the measurement OCR -------------------------------------------


def _image_only(pdf: bytes, dpi: int = 100) -> bytes:
    import pymupdf

    src, out = pymupdf.open("pdf", pdf), pymupdf.open()
    for page in src:
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=page.get_pixmap(dpi=dpi))
    return out.tobytes()


def _fake_ocr(text_layer_pdf: bytes):
    """Stand-in engine: words of the original text layer, in display_pt, as OcrWords."""
    import pymupdf

    from benchmark import ocr

    doc = pymupdf.open("pdf", text_layer_pdf)

    def ocr_page(pdf, page_index, dpi=300, lang="eng"):
        page, words = doc[page_index], []
        for x0, y0, x1, y1, w, *_ in page.get_text("words"):
            r = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
            words.append(ocr.OcrWord(w, (r.x0, r.y0, r.x1, r.y1), 99.0, "l0"))
        return words

    return ocr_page


def test_pseudonyms_of_an_image_only_output_are_read_with_the_ocr(monkeypatch):
    fx, res, _ = setup("digital")
    text_pdf = mock("digital", MockSpec("c", pseudonyms="consistent")).pdf
    raster_pdf = _image_only(text_pdf)
    # Without an OCR engine the check says so instead of guessing.
    r = evaluate_pseudonyms(fx.gt, res, raster_pdf)
    assert not r.evaluable and "no measurement OCR installed" in r.reason
    monkeypatch.setattr("benchmark.ocr.ocr_page", _fake_ocr(text_pdf))
    monkeypatch.setattr("benchmark.ocr.available", lambda: True)
    via_ocr = evaluate_pseudonyms(fx.gt, res, raster_pdf)
    direct = evaluate_pseudonyms(fx.gt, res, text_pdf)
    assert via_ocr.evaluable and via_ocr.checks == direct.checks
    assert [x.replacement for x in via_ocr.replacements] == [x.replacement for x in direct.replacements]
