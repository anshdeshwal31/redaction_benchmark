"""The client's decisions (policies/au-medicolegal-v1.yaml, eval-0.3).

- Q7: hiding a required item instead of replacing it is NOT_REPLACED: not a leak, not a pass.
- Q6: signatures and other marks count as replaced only when a stand-in is drawn in their place.
- Q11: kept words must stay searchable; a changed page size is accepted.
- Q16: damaging more than 1% of a document's ordinary words fails it.
- The tool's own safety check is graded against the evaluator's verdict.
"""

from functools import lru_cache
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from benchmark.evaluate import HIDDEN, SAFE, EvalConfig, MentionStatus, check_integrity, evaluate_output
from benchmark.gt import load_policy
from benchmark.gt.policy import resolve
from benchmark.mocks import MockSpec, run_mock
from benchmark.runner.oracle import oracle_regions
from benchmark.runner.scoring import matter_consistency, self_check_verdict
from benchmark.schemas import Action, GTMention, PiiType, PolicyProfile, Requirement, Segment
from benchmark.synth import ALL_VARIANTS

ROOT = Path(__file__).resolve().parents[1]
V1 = load_policy(ROOT / "policies" / "au-medicolegal-v1.yaml")
DRAFT = load_policy(ROOT / "policies" / "au-medicolegal-v0.yaml")
CFG = EvalConfig(dpi=150)


@lru_cache(maxsize=None)
def fixture(name):
    return ALL_VARIANTS[name]()


@lru_cache(maxsize=None)
def mocked(variant: str, spec: MockSpec) -> bytes:
    fx = fixture(variant)
    return run_mock(fx.pdf, fx.gt, V1, spec).pdf


def evaluate(variant: str, out: bytes, policy=V1):
    fx = fixture(variant)
    return evaluate_output(fx.pdf, out, fx.gt, policy, CFG)


def must(ev):
    return [m for m in ev.mentions if m.requirement == "MUST"]


def _signature_rect(variant="digital") -> pymupdf.Rect:
    gt = fixture(variant).gt
    sig = next(m for m in gt.mentions if m.type is PiiType.SIGNATURE)
    region = gt.region_index()[sig.segments[0].regions[0]][1]
    xs, ys = [p[0] for p in region.poly], [p[1] for p in region.poly]
    return pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))


def _draw(out: bytes, fn) -> bytes:
    doc = pymupdf.open("pdf", out)
    fn(doc[0])
    return doc.tobytes()


# -- policy -------------------------------------------------------------------------------------


def test_v1_requires_replacement_of_the_whole_address_and_keeps_dates():
    gt = fixture("digital").gt
    res = resolve(gt, V1)
    by_type = {}
    for m in gt.mentions:
        by_type.setdefault((m.type, m.subtype), res.mentions[m.id])
    assert by_type[(PiiType.ADDRESS, "residential")].requirement is Requirement.MUST
    # Policy 1.1.0 (deviation whole-address): the suburb, state and postcode of an address go too.
    locality = by_type[(PiiType.LOCATION, "locality")]
    assert locality.requirement is Requirement.MUST and locality.rule_id == "address-locality-replaced"
    assert by_type[(PiiType.DATE, "dob")].requirement is Requirement.MUST_NOT
    assert by_type[(PiiType.ORGANISATION, None)].requirement is Requirement.MUST
    assert all(r.action is Action.REPLACE for r in res.mentions.values() if r.requirement is Requirement.MUST)


@pytest.mark.parametrize("subtype, requirement", [
    ("locality", Requirement.MUST), ("suburb", Requirement.MUST_NOT), ("city", Requirement.MUST_NOT),
    ("region", Requirement.MUST_NOT), ("state", Requirement.MUST_NOT), ("country", Requirement.MUST_NOT),
    ("postcode", Requirement.MUST_NOT), ("place_of_declaration", Requirement.MUST_NOT),
])
def test_a_place_on_its_own_is_kept_but_the_locality_of_an_address_is_replaced(subtype, requirement):
    # No fixture has a place mentioned on its own, so one is added to an ordinary word.
    gt = fixture("digital").gt
    word = next(w for w in gt.page(1).words if w.text == "six")
    place = GTMention(id="place", type=PiiType.LOCATION, subtype=subtype, text=word.text,
                      segments=[Segment(page=1, words=[word.id])])
    res = resolve(gt.model_copy(update={"mentions": [*gt.mentions, place]}), V1)
    assert res.mentions["place"].requirement is requirement
    assert res.mentions["place"].rule_id == ("address-locality-replaced" if subtype == "locality" else "places-kept")


def test_validation_warns_when_a_place_on_its_own_is_labelled_locality():
    # The subtype decides keep or replace, so `locality` is reserved for the locality of a split address.
    from benchmark.gt.validate import validate_gt

    gt = fixture("digital").gt
    assert not [i for i in validate_gt(gt) if i.code == "locality-outside-address"]
    word = next(w for w in gt.page(1).words if w.text == "six")
    place = GTMention(id="place", type=PiiType.LOCATION, subtype="locality", text=word.text,
                      segments=[Segment(page=1, words=[word.id])])
    issues = [i for i in validate_gt(gt.model_copy(update={"mentions": [*gt.mentions, place]}))
              if i.code == "locality-outside-address"]
    assert [i.severity for i in issues] == ["warning"] and "a place mentioned on its own takes" in issues[0].message


def test_oracle_regions_carry_the_policy_action():
    gt = fixture("digital").gt
    assert {r.action for r in oracle_regions(gt, V1).regions} == {"replace"}
    assert {r.action for r in oracle_regions(gt, DRAFT).regions} == {"remove"}


# -- Q7: removed but not replaced ---------------------------------------------------------------


def test_removal_under_a_replace_policy_is_not_replaced():
    ev = evaluate("digital", mocked("digital", MockSpec("ref-true", method="true")))
    statuses = {m.status for m in must(ev)}
    assert statuses == {MentionStatus.NOT_REPLACED}, statuses
    s = ev.summary
    assert s["must_leak_rate"] == 0.0  # hidden: not a leak
    assert s["must_not_replaced_rate"] == 1.0 and s["must_fail_rate"] == 1.0
    assert not s["clean_document"]


def test_the_same_output_is_safe_under_a_policy_that_allows_removal():
    fx = fixture("digital")
    out = run_mock(fx.pdf, fx.gt, DRAFT, MockSpec("ref-true", method="true")).pdf
    ev = evaluate("digital", out, DRAFT)
    assert all(m.status is MentionStatus.SAFE_REMOVED for m in must(ev))


def test_replaced_names_and_contacts_pass_and_the_rest_is_reported():
    ev = evaluate("digital", mocked("digital", MockSpec("pseud", pseudonyms="consistent", replace_other="valid")))
    names = [m for m in must(ev) if m.type == "PERSON_NAME"]
    assert names and all(m.status is MentionStatus.SAFE_REPLACED for m in names)
    assert all(m.status in HIDDEN for m in must(ev))
    # The mock replaces names and contact details, not organisations or the signature.
    not_replaced = {m.type for m in must(ev) if m.status is MentionStatus.NOT_REPLACED}
    assert "PERSON_NAME" not in not_replaced and "SIGNATURE" in not_replaced


# -- Q6: stand-ins for signatures -----------------------------------------------------------------


def _signature_status(out: bytes) -> MentionStatus:
    ev = evaluate("digital", out)
    return next(m.status for m in ev.mentions if m.type == "SIGNATURE")


def test_a_drawn_stand_in_signature_counts_as_replaced():
    rect = _signature_rect()

    def squiggle(page):
        # Paper first (the mock's removal leaves a black box), then a pen-stroke stand-in.
        page.draw_rect(rect + (-2, -2, 2, 2), color=None, fill=(1, 1, 1))
        pts = [(rect.x0 + 4 + k * (rect.width - 8) / 24, rect.y0 + rect.height * (0.3 if k % 2 else 0.7))
               for k in range(25)]
        page.draw_polyline(pts, color=(0, 0, 0), width=1.2)

    out = _draw(mocked("digital", MockSpec("ref-true", method="true")), squiggle)
    assert _signature_status(out) is MentionStatus.SAFE_REPLACED


@pytest.mark.parametrize("fill", [(0, 0, 0), (1, 1, 1)])
def test_a_box_or_blank_over_a_signature_is_not_replaced(fill):
    rect = _signature_rect()
    out = _draw(mocked("digital", MockSpec("ref-true", method="true")),
                lambda page: page.draw_rect(rect + (-2, -2, 2, 2), color=None, fill=fill))
    assert _signature_status(out) is MentionStatus.NOT_REPLACED


# -- Q11: searchable text and page size -------------------------------------------------------------


def test_image_only_output_is_not_searchable():
    fx = fixture("digital")
    out = run_mock(fx.pdf, fx.gt, DRAFT, MockSpec("ref-raster", method="rasterize")).pdf
    ev = evaluate("digital", out, DRAFT)
    assert all(m.status in SAFE for m in must(ev))  # no leak at all...
    s = ev.summary
    assert s["searchable_rate"] == 0.0 and not s["searchable_ok"]
    assert not s["clean_document"]  # ...but the client needs a searchable file


def test_true_redaction_keeps_the_rest_searchable():
    fx = fixture("digital")
    out = run_mock(fx.pdf, fx.gt, DRAFT, MockSpec("ref-true", method="true")).pdf
    s = evaluate("digital", out, DRAFT).summary
    assert s["searchable_rate"] >= 0.95 and s["searchable_ok"] and s["clean_document"]


def test_a_changed_page_size_is_a_warning_not_a_failure():
    fx = fixture("digital")
    doc = pymupdf.open("pdf", fx.pdf)
    page = doc[0]
    r = page.mediabox
    page.set_mediabox(pymupdf.Rect(r.x0, r.y0, r.x1 + 20, r.y1 + 20))
    integrity, _, _ = check_integrity(fx.pdf, doc.tobytes())
    assert integrity.ok and not integrity.problems
    assert any("display size" in w for w in integrity.validator_warnings)


# -- Q16: collateral pass mark ----------------------------------------------------------------------


def test_heavy_over_redaction_fails_the_collateral_rule():
    fx = fixture("digital")
    out = run_mock(fx.pdf, fx.gt, DRAFT, MockSpec("dilate", dilate_pt=8)).pdf
    s = evaluate("digital", out, DRAFT).summary
    assert s["collateral_real_rate"] > 0.01 and not s["collateral_ok"] and not s["clean_document"]


# -- the tool's own safety check ------------------------------------------------------------------


@pytest.mark.parametrize("released, clean, verdict", [
    (True, True, "correct_release"), (True, False, "unsafe_release"),
    (False, True, "false_alarm"), (False, False, "correct_hold"),
])
def test_self_check_verdicts(released, clean, verdict):
    assert self_check_verdict(released, clean) == verdict


# -- Q8: one fake per person across the matter -------------------------------------------------------


def test_matter_consistency_compares_documents_of_a_matter():
    reps = {("M1", "E1", "surname"): {"DPI": {"jones"}, "IME": {"jones"}},
            ("M1", "E1", "given"): {"DPI": {"amy"}, "IME": {"beth"}},
            ("M1", "E2", "surname"): {"DPI": {"lee"}}}  # only one document: not compared
    rows = matter_consistency(reps, {"run_id": "r"})
    assert [(r["entity"], r["component"], r["consistent"]) for r in rows] == [
        ("E1", "given", False), ("E1", "surname", True)]
    assert rows[0]["documents"] == "DPI,IME" and rows[0]["distinct_replacements"] == 2


# -- departures from the client's answers (policy 1.1.0) -------------------------------------------


def test_v1_records_its_two_departures_from_the_client():
    by_id = {d.id: d for d in V1.deviations}
    assert V1.version == "1.1.0" and set(by_id) == {"whole-address", "role-words"}
    assert by_id["whole-address"].question == "Q4" and by_id["whole-address"].rules == ["address-locality-replaced"]
    assert by_id["role-words"].question == "Q10" and by_id["role-words"].rules == []
    assert all(d.client_answer and d.applied and d.reason for d in V1.deviations)
    assert DRAFT.deviations == []


def _with_deviations(*devs) -> dict:
    data = DRAFT.model_dump(mode="json", by_alias=True)
    data["deviations"] = list(devs)
    return data


def test_a_deviation_must_name_rules_of_its_policy():
    dev = {"id": "d1", "question": "Q4", "client_answer": "a", "applied": "b", "reason": "c"}
    assert PolicyProfile.model_validate(_with_deviations(dict(dev, rules=["dates-kept"]))).deviations[0].effect == ""
    with pytest.raises(ValidationError, match="no-such-rule"):
        PolicyProfile.model_validate(_with_deviations(dict(dev, rules=["no-such-rule"])))
    with pytest.raises(ValidationError, match="listed twice"):
        PolicyProfile.model_validate(_with_deviations(dev, dev))
    with pytest.raises(ValidationError):
        PolicyProfile.model_validate(_with_deviations({"id": "d2", "question": "Q4"}))  # no answer, no reason
