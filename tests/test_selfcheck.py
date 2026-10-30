import json
from pathlib import Path

import pytest

from benchmark.evaluate import EvalConfig, evaluate_output
from benchmark.gt import load_policy
from benchmark.mocks import MockSpec, run_mock
from benchmark.mocks.named import NAMED
from benchmark.selfcheck import _remove_only, _replace_all, selfcheck
from benchmark.synth import digital, mrc, rotated_content

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v0.yaml")
CLIENT = load_policy(Path(__file__).resolve().parents[1] / "policies" / "au-medicolegal-v1.yaml")
CFG = EvalConfig(dpi=150, renderers=("mupdf",))


@pytest.mark.parametrize("make", [digital, mrc, rotated_content])
def test_selfcheck_passes_on_fixtures(make):
    fx = make()
    results = selfcheck(fx.pdf, fx.gt, POLICY, EvalConfig(dpi=150, renderers=("mupdf",)))
    failed = [(r.name, r.observed) for r in results if r.passed is False]
    if make is rotated_content:
        # Measured limit (eval-0.4, S17): a line re-typeset at 1.02-1.035x on this rotated scan leaves its first
        # word ('June', in place, vertical) neither PRESERVED nor found MOVED: 1 of its 18 real words is
        # collateral, over the 1% mark. Pinned for the reviewers; the mark is not lowered.
        assert [name for name, _ in failed] == ["retypeset"] and "collateral 0.0556" in failed[0][1], failed
    else:
        assert failed == []
    names = {r.name for r in results}
    assert {"reference-true", "null", "overlay", "incremental", "dpi-scale"} <= names
    # eval-0.4: fakes drawn the way tools draw them; the address checks are not applicable without an address.
    assert set(EVAL04) <= names
    if make is not digital:  # no text layer: the pseudonym check needs the measurement OCR
        assert "pseudonyms" not in names


@pytest.mark.parametrize("make", [digital, rotated_content])
def test_client_rule_checks_follow_the_policy_passed_in(make):
    fx = make()
    under_client = {r.name: r for r in selfcheck(fx.pdf, fx.gt, CLIENT, CFG, only=["replace-all", "remove-only"])}
    assert under_client["replace-all"].passed is True and under_client["remove-only"].passed is True
    assert set(under_client["remove-only"].statuses) == {"NOT_REPLACED"}
    # The draft allows removal of every required mention: removal without a fake must be SAFE_REMOVED there.
    under_draft = {r.name: r for r in selfcheck(fx.pdf, fx.gt, POLICY, CFG, only=["replace-all", "remove-only"])}
    assert under_draft["replace-all"].passed is True and under_draft["remove-only"].passed is True
    assert set(under_draft["remove-only"].statuses) == {"SAFE_REMOVED"}


def test_client_rule_checks_catch_the_opposite_tool():
    fx = digital()
    required = {"replace_required": {m.id for m in fx.gt.mentions}}

    def evaluated(spec, policy=CLIENT):
        return evaluate_output(fx.pdf, run_mock(fx.pdf, fx.gt, policy, spec).pdf, fx.gt, policy, CFG)

    passed, observed = _replace_all(evaluated(MockSpec("remover")), required)
    assert passed is False and "not SAFE_REPLACED" in observed
    passed, observed = _remove_only(evaluated(NAMED["reference-replace"]), required)
    assert passed is False and "not as the policy implies" in observed
    # A leak is no removal at all: it fails too, whatever the policy (it used to pass as "0/0 hidden").
    leaking = evaluated(MockSpec("null", method="null"))
    assert _remove_only(leaking, required)[0] is False
    assert _remove_only(leaking, {"replace_required": set()})[0] is False
    # Removal where the policy allows it must be SAFE_REMOVED, not NOT_REPLACED.
    removed = evaluated(MockSpec("remover"), POLICY)
    assert _remove_only(removed, {"replace_required": set()})[0] is True
    assert _remove_only(removed, required)[0] is False


def test_every_check_passes_on_a_fixture_under_the_client_policy():
    # The drawn signature is required under the client policy: under an overlay box it is recoverable.
    fx = digital()
    results = selfcheck(fx.pdf, fx.gt, CLIENT, CFG)
    assert {r.name for r in results if r.passed is not True} == set(), [
        (r.name, r.observed) for r in results if r.passed is not True]


EVAL04 = ["replace-in-place", "retypeset", "retypeset-keep", "overprint", "overprint-veil", "shift-white",
          "text-next-line", "whole-address", "keeps-locality-moved", "erases-locality"]


def test_the_eval04_checks_follow_the_shared_roles_of_the_policy_passed_in():
    # Under the shared roles of policy 1.2.0 (owner, 2026-10-29) a fake mobile may keep its '04': the fakes
    # typeset in place are all replacements. With no shared roles (the shared structure of 1.1.1), or under
    # 1.2.0 without the phone prefix, the kept '04' shows the original's glyphs (PARTIAL), and the check
    # expects that. The address checks see a same-state fake (1.2.0) or another state's (no shared roles).
    from tests import calib

    fx = digital()
    for policy, mobile, only in ((CLIENT, "SAFE_REPLACED", EVAL04), (calib.V1_NO_SHARED, "PARTIAL", EVAL04),
                                 (calib.V1_PHONE_OFF, "PARTIAL", ["replace-in-place"])):
        results = {r.name: r for r in selfcheck(fx.pdf, fx.gt, policy, CFG, only=only)}
        assert {n: r.passed for n, r in results.items()} == dict.fromkeys(only, True), [
            (r.name, r.observed) for r in results.values() if r.passed is not True]
        ev = evaluate_output(fx.pdf, run_mock(fx.pdf, fx.gt, policy, NAMED["replace-in-place"]).pdf, fx.gt, policy,
                             CFG)
        assert ev.mention("ph1").status.value == mobile
        assert {m.status.value for m in ev.mentions if m.requirement == "MUST" and m.mention_id != "ph1"} == {
            "SAFE_REPLACED"}


def test_a_check_that_breaks_fails_as_an_error_and_the_other_checks_still_run(monkeypatch):
    import benchmark.selfcheck as sc

    real = sc.run_mock

    def breaks_on_null(source, gt, policy, spec, **kw):
        if spec.name == "null":
            raise KeyError("words of the document")
        return real(source, gt, policy, spec, **kw)

    monkeypatch.setattr(sc, "run_mock", breaks_on_null)
    fx = digital()
    results = {r.name: r for r in selfcheck(fx.pdf, fx.gt, POLICY, CFG, only=["null", "annotation"])}
    broken = results["null"]
    assert broken.passed is False and broken.observed == "ERROR: KeyError (could not run)"
    assert broken.extra == {"error": "KeyError"} and "words of the document" not in json.dumps(broken.to_dict())
    assert results["annotation"].passed is True  # the next check ran
