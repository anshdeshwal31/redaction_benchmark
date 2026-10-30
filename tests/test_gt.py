import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.gt import Effective, load_policy, resolve, validate_gt
from benchmark.gt.address_split import PLACE_SUBTYPES
from benchmark.schemas import (
    EXPORTED,
    CandidateManifest,
    EntityKind,
    GTDocument,
    GTEntity,
    GTMention,
    GTPage,
    GTWord,
    MentionsArtifact,
    PiiType,
    PlanArtifact,
    Requirement,
    Role,
    Segment,
    WordSource,
)
from benchmark.synth import digital

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policies" / "au-medicolegal-v0.yaml"


@pytest.fixture(scope="module")
def fx():
    return digital()


@pytest.fixture(scope="module")
def policy():
    return load_policy(POLICY)


def test_policy_resolution_of_mentions(fx, policy):
    res = resolve(fx.gt, policy)
    m = res.mentions
    assert m["m1"].requirement is Requirement.MUST  # claimant's name
    assert m["m4"].requirement is Requirement.MUST  # family member
    assert m["hdr0"].requirement is Requirement.MUST  # running header form
    assert m["m3"].requirement is Requirement.MAY and m["m3"].pending == "Q1"  # examining expert
    assert m["sig1"].requirement is Requirement.MAY and m["sig1"].pending == "Q6"
    assert m["dob1"].requirement is Requirement.MAY and m["dob1"].pending == "Q3"
    assert m["date1"].requirement is Requirement.MUST_NOT  # dates are kept
    assert m["gov1"].requirement is Requirement.MUST
    assert m["org1"].requirement is Requirement.MAY and m["org1"].pending == "Q2"
    assert m["addr1"].requirement is Requirement.MUST


def test_policy_resolution_of_words(fx, policy):
    res = resolve(fx.gt, policy)
    words = {w.text: w.id for w in fx.gt.page(0).words}
    assert res.words[words["Ms"]] is Effective.MAY  # optional title
    assert res.words[words["strong"]] is Effective.MUST_NOT  # homograph, protected
    assert res.words[words["Sertraline"]] is Effective.MUST_NOT  # medication, protected
    assert res.words[words["Claimant:"]] is Effective.ORDINARY
    assert res.words[words["Elizabeth"]] is Effective.MUST


def test_must_beats_protected_when_spans_overlap(fx, policy):
    data = fx.gt.model_dump(mode="json", by_alias=True)
    strong = next(w for w in fx.gt.page(0).words if w.text == "strong")
    # Pretend an annotator also tagged the homograph as part of a private name.
    data["mentions"].append({"id": "mx", "type": "PERSON_NAME", "entity": "E1", "text": "strong",
                             "segments": [{"page": 0, "words": [strong.id]}]})
    res = resolve(GTDocument.model_validate(data), policy)
    assert res.words[strong.id] is Effective.MUST


def test_validator_flags_integrity_errors(fx):
    data = fx.gt.model_dump(mode="json", by_alias=True)
    data["mentions"][0]["text"] = "wrong text"
    data["mentions"][1]["segments"][0]["words"].append("no-such-word")
    data["pages"][0]["words"][1]["id"] = data["pages"][0]["words"][0]["id"]
    data["pages"][0]["words"][2]["poly"] = [[590, 10], [700, 10], [700, 20]]
    gt = GTDocument.model_validate(data)
    codes = {i.code for i in validate_gt(gt) if i.severity == "error"}
    assert {"text-mismatch", "unknown-word", "duplicate-id", "polygon-outside-page"} <= codes


def test_validator_warns_about_unannotated_repeats_and_homographs(fx):
    data = fx.gt.model_dump(mode="json", by_alias=True)
    data["classified"] = [c for c in data["classified"] if c["id"] != "hg1"]  # unclassify "strong"
    data["mentions"] = [m for m in data["mentions"] if m["id"] != "m8"]  # drop one "Smithers"
    codes = [i.code for i in validate_gt(GTDocument.model_validate(data))]
    assert "homograph-unclassified" in codes
    assert "unannotated-repeat" in codes or "homograph-unclassified" in codes


def test_schemas_reject_unknown_keys_and_missing_contracts():
    with pytest.raises(ValidationError):
        CandidateManifest.model_validate({"name": "x", "command": "c", "provides": ["plan"], "version": "1"})
    with pytest.raises(ValidationError):
        PlanArtifact.model_validate({"frame": "image_px", "regions": []})  # dpi missing
    with pytest.raises(ValidationError):
        MentionsArtifact.model_validate({"mentions": [{"id": "a", "label": "PER"}]})  # no anchor
    with pytest.raises(ValidationError):
        GTDocument.model_validate({"doc_id": "d", "source_sha256": "0" * 64, "gt_version": "1", "pages": [],
                                   "surprise": True})


def test_exported_json_schemas_are_current():
    for filename, model in EXPORTED.items():
        path = ROOT / "schemas" / filename
        assert path.exists(), f"run `rb schemas export` ({filename} missing)"
        assert json.loads(path.read_text(encoding="utf-8")) == model.model_json_schema(by_alias=True), filename


# -- addresses and places on their own (owner rulings 2 and 3 of 2026-10-22) ----------------------

PLACE_CODES = {"street-line-without-locality", "place-with-postcode-not-after-place-kind", "place-after-street-line"}


def _gt(lines: list[str], mentions: list[tuple]) -> GTDocument:
    """A fictional one-page GT: one line id per entry of ``lines``, 20 pt apart, and ``mentions`` as
    (id, type, subtype, entity, conditions, line index, first word, last word)."""
    words = []
    for li, line in enumerate(lines):
        x = 72.0
        for wi, text in enumerate(line.split()):
            w = 6.0 * len(text)
            words.append(GTWord(id=f"w{li}_{wi}", text=text, poly=[(x, 100 + 20 * li), (x + w, 100 + 20 * li),
                                                                     (x + w, 110 + 20 * li), (x, 110 + 20 * li)],
                                order=len(words), line=f"l{li}", source=WordSource.TEXT_LAYER))
            x += w + 4.0
    out = []
    for mid, kind, subtype, entity, conditions, li, a, b in mentions:
        wids = [f"w{li}_{k}" for k in range(a, b + 1)]
        out.append(GTMention(id=mid, type=kind, subtype=subtype, entity=entity, conditions=conditions,
                             text=" ".join(lines[li].split()[a:b + 1]), segments=[Segment(page=0, words=wids)]))
    return GTDocument(doc_id="d", source_sha256="0" * 64, gt_version="0.2", mentions=out,
                      pages=[GTPage(index=0, width_pt=595, height_pt=842, words=words)],
                      entities=[GTEntity(id="E1", kind=EntityKind.PERSON, role=Role.CLAIMANT)])


def _place_issues(gt: GTDocument) -> dict[str, str]:
    return {i.code: i.message for i in validate_gt(gt) if i.code in PLACE_CODES}


@pytest.mark.parametrize("lines, place, warns", [
    (["Place of birth: Penrith NSW 2750"], ("place_of_birth", 0, 3, 5), False),
    (["Penrith NSW 2750"], ("place_of_birth", 0, 0, 2), True),
    (["Date of accident: 12 June 2023, Penrith NSW 2750"], ("place_of_accident", 0, 6, 8), False),
    (["Injured 12 June, Penrith NSW 2750"], ("place_of_injury", 0, 3, 5), False),
    (["Place of incident:", "Kingswood NSW 2747"], ("place_of_incident", 1, 0, 2), False),
    (["She resides in Penrith NSW 2750."], ("suburb", 0, 3, 5), True),
    (["She resides in Penrith."], ("suburb", 0, 3, 3), False),  # no state or postcode
    (["Declared at Penrith NSW 2750 on 3 May 2024"], ("place_of_declaration", 0, 2, 4), False),
])
def test_a_place_with_a_postcode_needs_a_place_kind_label(lines, place, warns):
    subtype, li, a, b = place
    gt = _gt(lines, [("p1", PiiType.LOCATION, subtype, None, [], li, a, b)])
    issues = _place_issues(gt)
    assert ("place-with-postcode-not-after-place-kind" in issues) is warns
    if warns:
        assert "birth, injury, accident or incident, declaration" in issues["place-with-postcode-not-after-place-kind"]


def test_a_street_line_needs_its_locality():
    lines = ["12 Smith Street, Penrith NSW 2750"]
    street = ("a1", PiiType.ADDRESS, "residential", "E1", ["address_street_line"], 0, 0, 2)
    issues = _place_issues(_gt(lines, [street]))
    assert set(issues) == {"street-line-without-locality"}
    assert "birth, injury, accident or incident, declaration" in issues["street-line-without-locality"]
    loc = ("a1-loc", PiiType.LOCATION, "locality", "E1", ["address_locality"], 0, 3, 5)
    assert _place_issues(_gt(lines, [street, loc])) == {}
    # A DX line and a box line with no place stand alone.
    for text, last in (("DX 1234 Sydney", 2), ("Locked Bag 7", 2)):
        assert _place_issues(_gt([text], [("a1", PiiType.ADDRESS, "postal", "E1", ["address_street_line"],
                                           0, 0, last)])) == {}


def test_a_place_right_after_or_below_a_street_line_of_its_entity_is_flagged():
    street = ("a1", PiiType.ADDRESS, "residential", "E1", [], 0, 0, 2)
    for lines, place in ((["12 Smith Street, Penrith NSW 2750"], (0, 3, 5)),
                         (["12 Smith Street", "Penrith NSW 2750"], (1, 0, 2))):
        li, a, b = place
        issues = _place_issues(_gt(lines, [street, ("p1", PiiType.LOCATION, "suburb", "E1", [], li, a, b)]))
        assert "place-after-street-line" in issues
        assert "birth, injury, accident or incident, declaration" in issues["place-after-street-line"]
        issues = _place_issues(_gt(lines, [street, ("p1", PiiType.LOCATION, "suburb", None, [], li, a, b)]))
        assert "place-after-street-line" not in issues  # not the same entity
    # A labelled place on the next line is not right after the street line.
    lines = ["12 Smith Street, Penrith NSW 2750", "Place of birth: Parramatta NSW 2150"]
    gt = _gt(lines, [("a1", PiiType.ADDRESS, "residential", "E1", ["address_street_line"], 0, 0, 2),
                     ("a1-loc", PiiType.LOCATION, "locality", "E1", ["address_locality"], 0, 3, 5),
                     ("p1", PiiType.LOCATION, "place_of_birth", "E1", [], 1, 3, 5)])
    assert _place_issues(gt) == {}


def test_place_kind_and_street_subtypes_are_known_place_subtypes():
    assert {"place_of_birth", "place_of_injury", "place_of_accident", "place_of_incident", "street"} <= \
        set(PLACE_SUBTYPES)
    gt = _gt(["Place of birth: Penrith NSW 2750", "She walks along Smith Street"],
             [("p1", PiiType.LOCATION, "place_of_birth", None, [], 0, 3, 5),
              ("p2", PiiType.LOCATION, "street", None, [], 1, 3, 4)])
    assert [i for i in validate_gt(gt) if i.severity == "warning" and i.code != "orphan-entity"] == []
    labelled = gt.model_copy(update={"mentions": [gt.mentions[0].model_copy(update={"subtype": "locality"})]})
    message = next(i.message for i in validate_gt(labelled) if i.code == "locality-outside-address")
    assert "place_of_birth" in message and "street" in message
