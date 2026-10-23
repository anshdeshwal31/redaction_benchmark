"""The client's decisions (policies/au-medicolegal-v1.yaml, eval-0.3).

- Q7: hiding a required item instead of replacing it is NOT_REPLACED: not a leak, not a pass.
- Q6: signatures and other marks count as replaced only when a stand-in is drawn in their place.
- Q11: kept words must stay searchable; a changed page size is accepted.
- Q16: damaging more than 1% of a document's ordinary words fails it.
- The tool's own safety check is graded against the evaluator's verdict.
- Policy 1.1.0 departs from two answers on purpose (whole addresses replaced, role words kept);
  every report shows the departures.
- A mock that follows the client's rules completely is clean on every encoding: the ceiling. It
  gives each person one fake across a matter, and each phone and Medicare number a fake of its own.
"""

import json
from functools import lru_cache
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from benchmark.evaluate import HIDDEN, SAFE, EvalConfig, MentionStatus, check_integrity, evaluate_output
from benchmark.gt import load_policy
from benchmark.gt.policy import resolve
from benchmark.metrics import evaluate_pseudonyms
from benchmark.mocks import MockSpec, run_mock, target_regions
from benchmark.mocks.named import NAMED
from benchmark.report import deviations
from benchmark.runner.oracle import oracle_entities, oracle_mentions, oracle_regions, oracle_text
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


def test_departures_section_escapes_every_value_and_shows_each_departure_once():
    hostile = {"id": "x", "question": "Q<4>", "client_answer": "<script>alert('a')</script>",
               "applied": "a & b", "reason": '"quoted"', "effect": "", "rules": []}
    found = deviations.collect([{"policy_id": "p@1", "deviations": [hostile]},
                                {"policy_id": "p@1", "deviations": [hostile]}])
    assert len(found) == 1 and found[0]["policy_id"] == "p@1"
    page = deviations.section(found)
    assert deviations.TITLE.replace("'", "&#x27;") in page
    assert "&lt;script&gt;alert(&#x27;a&#x27;)&lt;/script&gt;" in page and "<script>" not in page
    assert "Q&lt;4&gt;" in page and "a &amp; b" in page and "&quot;quoted&quot;" in page
    assert deviations.section([]) == ""
    # The same deviation id in two policies is two departures, with a policy column.
    two = deviations.collect([{"policy_id": "p@1", "deviations": [hostile]},
                              {"policy_id": "p@2", "deviations": [hostile]}])
    page = deviations.section(two)
    assert len(two) == 2 and "<th>policy</th>" in page
    assert "The policies of these runs depart from" in page and "departs" not in page


def test_departures_on_a_board_of_several_policies_name_their_runs():
    v1 = {"run_id": "tool-a", "policy_id": "au-medicolegal@1.1.0",
          "deviations": [d.model_dump() for d in V1.deviations]}
    draft = {"run_id": "tool-b", "policy_id": "au-medicolegal@0.1.0-draft", "deviations": []}
    found = deviations.collect([v1, draft])
    assert [d["runs"] for d in found] == [["tool-a"], ["tool-a"]]
    page = deviations.section(found, {"au-medicolegal@1.1.0": ["tool-a"], "au-medicolegal@0.1.0-draft": ["tool-b"]})
    assert "scored under different policies" in page and "au-medicolegal@0.1.0-draft (tool-b)" in page
    assert "<th>runs</th>" in page and "<td>tool-a</td>" in page and "departs" not in page
    # Results scored before the policy version was recorded are named by their file.
    assert deviations.policy_label({"policy": "C:/x/policies/au-medicolegal-v0.yaml"}) == \
        "au-medicolegal-v0.yaml (version not recorded)"


# -- the ceiling: a mock that does everything the client's rules ask (D3) ------------------------


REFERENCE = NAMED["reference-replace"]


@pytest.mark.parametrize("variant", sorted(ALL_VARIANTS))
def test_reference_replace_is_clean_on_every_encoding(variant):
    ev = evaluate(variant, mocked(variant, REFERENCE))
    statuses = {m.mention_id: m.status for m in must(ev)}
    assert statuses and set(statuses.values()) == {MentionStatus.SAFE_REPLACED}, statuses
    s = ev.summary
    assert s["searchable_ok"] and s["collateral_ok"] and not s["protected_damaged"] and s["integrity_ok"], s
    assert s["clean_document"]


def test_reference_replace_makes_valid_fakes_of_every_kind():
    fx = fixture("digital")
    out = run_mock(fx.pdf, fx.gt, V1, REFERENCE)
    r = evaluate_pseudonyms(fx.gt, resolve(fx.gt, V1), out.pdf)
    c = r.checks
    assert c["replaced"] == c["name_components"] and c["collisions"] == 0 and c["collides_with_real_name"] == 0
    assert c["phone_valid_rate"] == c["email_valid_rate"] == c["email_from_new_name_rate"] == 1.0
    assert c["medicare_checksum_valid_rate"] == 1.0 and c["postcode_state_valid_rate"] == 1.0
    assert c["invalid_replacements"] == 0
    # Identifiers keep their format with every letter and digit changed.
    gt_mention = {f"d{i}": m for i, m in enumerate(out.targeted)}  # detections in GT order, none spurious
    claim = next(m.text for m in fx.gt.mentions if m.id == "clm0")
    fake = next(x.replacement for x in out.plan.regions if gt_mention[x.mentions[0]] == "clm0")
    assert len(fake) == len(claim) and fake[3] == claim[3] == "-"
    assert all(a != b for a, b in zip(fake, claim) if a.isalnum())
    assert fake[:3].isupper() and fake[4:].isdigit()
    # Every required kind gets a fake, the signature a drawn stand-in (a region with no text).
    regions = [x for x in out.plan.regions if x.action == "replace"]
    assert len(regions) == len(out.plan.regions)
    assert [x.replacement for x in regions if x.id.endswith("sig1")] == [None]


def test_entity_keyed_fakes_are_the_same_in_every_document_of_a_matter():
    fx = fixture("digital")
    reordered = fx.gt.model_copy(update={"mentions": list(reversed(fx.gt.mentions))})

    def fakes(gt, spec):
        out = run_mock(fx.pdf, gt, V1, spec)
        return {(e.component, e.replacement) for e in evaluate_pseudonyms(gt, resolve(gt, V1), out.pdf).replacements
                if e.entity == "E3"}

    # People appear in another order in the second document: per-document clusters give the husband
    # other fakes, fakes keyed on the GT entity stay the same.
    assert fakes(fx.gt, REFERENCE) == fakes(reordered, REFERENCE)
    clustered = MockSpec("by-cluster", replace_all=True)
    assert fakes(fx.gt, clustered) != fakes(reordered, clustered)


def _only(gt, doc_id: str, people: set[str]):
    """The document as another one of the matter that names only some of its people."""
    return gt.model_copy(update={"doc_id": doc_id, "mentions": [m for m in gt.mentions
                                                                 if m.entity is None or m.entity in people]})


def _planned_name_fakes(gt, matter_gts=None) -> dict[tuple[str, str], set[str]]:
    """(person, name component) -> the fakes the ceiling plans for them in one document."""
    plan, targeted, fakes = target_regions(gt, V1, REFERENCE, matter_gts)
    mentions = {m.id: m for m in gt.mentions}
    out: dict[tuple[str, str], set[str]] = {}
    for key, value in fakes.items():
        det, wid = key.split(":")
        m = mentions[targeted[int(det[1:])]]
        for comp, wids in m.components.items():
            if m.type is PiiType.PERSON_NAME and wid in wids:
                out.setdefault((m.entity, comp.value), set()).add(value.strip(".,;:()").casefold())
    return out


def test_every_person_keeps_one_fake_across_a_matter_whoever_else_a_document_names():
    gt = fixture("digital").gt
    # E1 (the claimant) and E5 (her GP) start from the same surname slot; a document that names only E2
    # and E5, or one that names E5 before E1, must not hand E5 the claimant's fake or change hers.
    docs = [gt.model_copy(update={"doc_id": "a"}),
            gt.model_copy(update={"doc_id": "b", "mentions": list(reversed(gt.mentions))}),
            _only(gt, "c", {"E2", "E5"}), _only(gt, "d", {"E1", "E5"})]
    by_doc = [_planned_name_fakes(d, docs) for d in docs]
    merged: dict[tuple[str, str], set[str]] = {}
    for fakes in by_doc:
        for key, values in fakes.items():
            merged.setdefault(key, set()).update(values)
    assert {e for e, _ in merged} == {"E1", "E2", "E3", "E5"} and ("E5", "surname") in by_doc[2]
    assert all(len(values) == 1 for values in merged.values()), merged  # one fake per person and part
    owners: dict[tuple[str, str], set[str]] = {}
    for (entity, comp), values in merged.items():
        for v in values:
            owners.setdefault((comp, v), set()).add(entity)
    assert all(len(people) == 1 for people in owners.values()), owners  # never one fake for two people
    # Without the rest of the matter the fakes still do not depend on the order of the mentions.
    assert _planned_name_fakes(docs[0]) == _planned_name_fakes(docs[1])


def test_the_ceiling_reads_any_spelling_of_gender_and_never_runs_out_of_fakes():
    from benchmark.mocks.mock import _GIVEN_KEYED, _assign
    from benchmark.schemas import AttributeValue

    gt = fixture("digital").gt
    # GT writes gender freely: "female" and "male", or a value the mock has no name pool for.
    spelled = {"E1": "female", "E2": "male", "E3": "X"}
    entities = [e.model_copy(update={"attributes": {**e.attributes, "gender": AttributeValue(value=spelled[e.id])}})
                if e.id in spelled else e for e in gt.entities]
    fakes = _planned_name_fakes(gt.model_copy(update={"entities": entities}))
    given = {e: next(iter(v)) for (e, c), v in fakes.items() if c == "given"}
    pools = {g: {n.casefold() for n in names} for g, names in _GIVEN_KEYED.items()}
    assert given["E1"] in pools["F"] and given["E2"] in pools["M"] and given["E3"] in pools["F"]
    # A matter with more people than a pool has names: each person still gets a fake no one else has.
    people = {(f"P{k:02d}", "given"): {f"Original{k}"} for k in range(40)}
    values = _assign(people, {}, frozenset(), salt="M1")
    assert len(_GIVEN_KEYED["F"]) < 40 and len({v.casefold() for v in values.values()}) == 40


def test_the_mock_adapter_keeps_each_persons_fake_across_the_documents_of_a_matter(tmp_path, monkeypatch):
    # As `rb run` calls it, one document at a time. Each document names only some of the matter's people,
    # in the order that hands E5 the claimant's surname when a document is planned on its own.
    from benchmark.gt.io import write_model
    from benchmark.mocks.adapter import main

    docs = {"c": ("digital", {"E2", "E5"}), "d": ("cropbox", {"E1", "E5"})}
    gts = {doc_id: _only(fixture(variant).gt, doc_id, people) for doc_id, (variant, people) in docs.items()}
    for doc_id, gt in gts.items():
        write_model(gt, tmp_path / "gt" / f"{doc_id}.gt.json")
    monkeypatch.setenv("RB_GT_DIRS", str(tmp_path / "gt"))
    fakes: dict[tuple[str, str], set[str]] = {}
    for doc_id, (variant, _) in docs.items():
        work = tmp_path / doc_id
        work.mkdir()
        (work / "input.pdf").write_bytes(fixture(variant).pdf)
        assert main(["--input", str(work / "input.pdf"), "--workdir", str(work / "out"), "--spec", "reference-replace",
                     "--policy", str(ROOT / "policies" / "au-medicolegal-v1.yaml")]) == 0
        whose = {wid: (m.entity, comp.value) for m in gts[doc_id].mentions if m.type is PiiType.PERSON_NAME
                 for comp, wids in m.components.items() for wid in wids}
        for entry in json.loads((work / "out" / "pseudonyms.json").read_text(encoding="utf-8"))["entries"]:
            if entry["original"] in whose:
                fakes.setdefault(whose[entry["original"]], set()).add(entry["replacement"].strip(".,;:()").casefold())
    assert {e for e, _ in fakes} == {"E1", "E2", "E5"}
    assert all(len(values) == 1 for values in fakes.values()), fakes  # one fake per person across the matter
    owners: dict[tuple[str, str], set[str]] = {}
    for (entity, comp), values in fakes.items():
        owners.setdefault((comp, next(iter(values))), set()).add(entity)
    assert all(len(people) == 1 for people in owners.values()), owners  # and no fake for two people


@pytest.mark.parametrize("words", [["(02)", "9876", "5432"], ["02", "9876", "5432"], ["0412", "555", "678"],
                                   ["04", "1234", "5678"], ["1300", "333", "767"], ["1800", "975", "707"],
                                   ["+61", "2", "9876", "5432"], ["+61", "412", "345", "678"],
                                   ["+61", "(0)412", "345", "678"], ["+61412345678"], ["131", "450"],
                                   ["Tel", "9876", "5432"]])
def test_a_fake_phone_keeps_its_kind_and_shares_no_word_with_the_original(words):
    from benchmark.evaluate.layers import text_leaks
    from benchmark.metrics.pseudonyms import phone_valid
    from benchmark.mocks.mock import _fake_phone

    fake = _fake_phone(words, 7)
    assert len(fake) == len(words) and fake == _fake_phone(words, 7)  # one value, one fake
    assert not any(text_leaks(w.strip(".,;:()"), f, CFG) for w, f in zip(words, fake)), fake
    if phone_valid(" ".join(words)):
        assert phone_valid(" ".join(fake)), fake
    assert all(t.count("(") == t.count(")") for t in fake), fake  # an area code in brackets stays whole
    if "".join(words).replace("+61", "0").replace("(0)", "").startswith("04"):
        # A mobile stays a mobile, except where its 04 has a word of its own: a fake cannot keep that word
        # unchanged without the evaluator reading it as the original.
        assert fake[0].startswith("04") or words[0] == "04", fake


@pytest.mark.parametrize("words", [["2123", "45670", "2"], ["2123", "45670", "1", "3"], ["2123456702"]])
def test_a_fake_medicare_number_is_valid_with_every_digit_changed(words):
    from benchmark.evaluate.layers import text_leaks
    from benchmark.metrics.pseudonyms import medicare_valid
    from benchmark.mocks.mock import _fake_medicare

    fake = _fake_medicare(words, 7)
    assert medicare_valid(" ".join(fake)) and len(fake) == len(words)
    assert not any(text_leaks(w, f, CFG) for w, f in zip(words, fake)), fake
    old, new = "".join(words), "".join(fake)
    assert all(a != b for k, (a, b) in enumerate(zip(old, new)) if k != 8)  # the check digit falls as it may


def _contacts_spec():
    """The digital fixture with a landline in (02) for the claimant, a Medicare issue number of 2, and
    the husband's own mobile and Medicare number."""
    from benchmark.synth.spec import L, MentionSpec, T, medico_legal_spec

    spec = medico_legal_spec()
    p0, p1 = spec.pages
    phone = next(i for i, line in enumerate(p0) if any("m:ph1" in t.tags for t in line.toks))
    p0[phone] = L(72, 180, 11, T("Phone:"), T("(02)", "m:ph1"), T("9876", "m:ph1"), T("5432", "m:ph1"),
                  T("Email:"), T("jane.smithers@example.com", "m:em1"))
    signed = next(i for i, line in enumerate(p0) if line.y == 360)  # above the signature, in reading order
    p0.insert(signed, L(72, 290, 11, T("His"), T("mobile:"), T("0433", "m:ph3"), T("765", "m:ph3"),
                        T("901", "m:ph3")))
    medicare = next(i for i, line in enumerate(p1) if any("m:gov1" in t.tags for t in line.toks))
    p1[medicare] = L(72, 140, 11, T("Medicare"), T("number"), T("2123", "m:gov1"), T("45670", "m:gov1"),
                     T("2", "m:gov1"))
    p1.insert(len(p1) - 1, L(72, 200, 11, T("His"), T("Medicare"), T("number"), T("3456", "m:gov3"),
                             T("78908", "m:gov3"), T("1", "m:gov3")))  # before the running footer
    spec.mentions["ph3"] = MentionSpec(PiiType.PHONE, "E3", subtype="mobile")
    spec.mentions["gov3"] = MentionSpec(PiiType.ID_GOVERNMENT, "E3", subtype="medicare")
    return spec


def test_the_ceiling_gives_each_person_their_own_phone_and_medicare_fake():
    from benchmark.metrics.pseudonyms import medicare_valid, phone_valid
    from benchmark.synth.fixtures import digital

    fx = digital(_contacts_spec())
    out = run_mock(fx.pdf, fx.gt, V1, REFERENCE)
    ev = evaluate_output(fx.pdf, out.pdf, fx.gt, V1, CFG)
    assert {m.status for m in must(ev)} == {MentionStatus.SAFE_REPLACED} and ev.summary["clean_document"], [
        (m.mention_id, m.status.value) for m in must(ev) if m.status is not MentionStatus.SAFE_REPLACED]
    gt_mention = {f"d{i}": m for i, m in enumerate(out.targeted)}
    fake = {}
    for r in out.plan.regions:
        if r.replacement:
            fake.setdefault(gt_mention[r.mentions[0]], []).append(r.replacement)
    assert fake["ph1"] != fake["ph3"] and fake["gov1"] != fake["gov3"]
    assert fake["ph1"][0] != "(02)" and phone_valid(" ".join(fake["ph1"])) and fake["ph3"][0].startswith("04")
    assert medicare_valid(" ".join(fake["gov1"])) and fake["gov1"][2] != "2"


def test_a_drawn_stand_in_with_true_redaction_counts_as_replaced():
    out = mocked("digital", MockSpec("true-stand-ins", stand_ins=True))
    assert _signature_status(out) is MentionStatus.SAFE_REPLACED


def test_true_redaction_with_fakes_at_the_original_size_fails_only_where_known():
    # The open calibration issue, pinned: at 150 DPI only the locality ("SA" over "QLD") is judged PARTIAL,
    # at 200 DPI the document is clean. A leak anywhere else, or a crash, fails here.
    fx = fixture("digital")
    out = mocked("digital", MockSpec("true-replace", replace_all=True, stand_ins=True))
    at_150 = evaluate("digital", out)
    assert {m.mention_id: m.status for m in must(at_150) if m.status is not MentionStatus.SAFE_REPLACED} == {
        "addr1-loc": MentionStatus.PARTIAL}
    at_200 = evaluate_output(fx.pdf, out, fx.gt, V1, EvalConfig(dpi=200))
    assert {m.status for m in must(at_200)} == {MentionStatus.SAFE_REPLACED} and at_200.summary["clean_document"]


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="open calibration issue (docs/metrics.md): correct fakes written in the original's font, "
                          "size and place can line up with its glyph bands and be judged PARTIAL; the ceiling draws "
                          "its fakes at 80% size to avoid this")
def test_true_redaction_with_fakes_at_the_original_size_is_clean():
    ev = evaluate("digital", mocked("digital", MockSpec("true-replace", replace_all=True, stand_ins=True)))
    assert next(m for m in must(ev) if m.mention_id == "addr1-loc").status is MentionStatus.SAFE_REPLACED


def test_reference_replace_fills_injected_oracle_regions_with_its_fakes():
    fx = fixture("digital")
    injected = {"text": oracle_text(fx.gt), "mentions": oracle_mentions(fx.gt, V1),
                "entities": oracle_entities(fx.gt, V1), "regions": oracle_regions(fx.gt, V1)}
    ev = evaluate("digital", run_mock(fx.pdf, fx.gt, V1, REFERENCE, injected=injected).pdf)
    assert {m.status for m in must(ev)} == {MentionStatus.SAFE_REPLACED} and ev.summary["clean_document"]
