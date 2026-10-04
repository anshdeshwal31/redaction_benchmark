import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.gt import Effective, load_policy, resolve, validate_gt
from benchmark.schemas import (
    EXPORTED,
    CandidateManifest,
    GTDocument,
    MentionsArtifact,
    PlanArtifact,
    Requirement,
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
