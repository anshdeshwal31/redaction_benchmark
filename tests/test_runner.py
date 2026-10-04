"""End-to-end: mocks run as external processes through the runner, then scoring, report,
compare, regression and candidate checks (PLAN.md phase 6 exit criterion)."""

import json
from pathlib import Path

import pytest

from benchmark.evaluate import EvalConfig
from benchmark.report.compare import compare
from benchmark.report.html import build_report
from benchmark.report.regression import check, promote
from benchmark.report.store import query
from benchmark.runner.artifacts import load_artifacts
from benchmark.runner.runner import load_candidate, run_candidate
from benchmark.runner.scoring import score_run
from benchmark.synth.register import register_synthetic

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policies" / "au-medicolegal-v0.yaml"
CFG = EvalConfig(dpi=150)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rb")
    mp = pytest.MonkeyPatch()
    for var, sub in (("RB_DATA_DIR", "data"), ("RB_DATASETS_DIR", "datasets"), ("RB_RUNS_DIR", "runs"),
                     ("RB_RESULTS_DIR", "results"), ("RB_GT_DIRS", "gt")):
        mp.setenv(var, str(tmp / sub))
    register_synthetic("0.0.1", ["digital"])
    yield tmp
    mp.undo()


def candidate(tmp: Path, spec: str, provides="output_pdf, text, mentions, entities, plan, run_info") -> Path:
    path = tmp / f"mock-{spec}.yaml"
    gt = (tmp / "gt").as_posix()
    path.write_text(f"""schema: rb.candidate/0.1
name: mock-{spec}
version: "0.1.0"
command: '{{python}} -m benchmark.mocks.adapter --input {{input}} --workdir {{workdir}} --mode {{mode}} --inject-dir {{inject_dir}} {{seed}} --spec {spec}'
provides: [{provides}]
accepts_injection: [text, mentions, entities, regions]
seed_flag: --seed
env:
  RB_GT_DIRS: "{gt}"
""", encoding="utf-8")
    return path


def run_and_score(tmp, spec, run_id, experiment="e2e", provides=None, oracle=None):
    cand = candidate(tmp, spec, provides) if provides else candidate(tmp, spec)
    run = run_candidate(cand, "synthetic@0.0.1", ROOT / "experiments" / f"{experiment}.yaml", POLICY, run_id=run_id)
    assert all(not i.error and not i.invalid_artifacts for i in run.items), run.items
    return score_run(run_id, POLICY, CFG, oracle_runs=oracle)


def must_rows(results, cols="status, attr_category, attr_confidence"):
    rows, _ = query(results, f"SELECT {cols} FROM mentions WHERE requirement='MUST' AND repeat=0")
    return rows


def test_inputs_are_neutral_and_gt_is_absent(env):
    res = run_and_score(env, "reference", "ref")
    run = json.loads((env / "runs" / "ref" / "run.json").read_text(encoding="utf-8"))
    work = env / "runs" / "ref" / run["items"][0]["workdir"]
    assert sorted(p.name for p in work.iterdir()) == ["input.pdf", "out", "stderr.log", "stdout.log"]
    assert run["candidate"]["version"] == "0.1.0" and run["environment"]["packages"]["pymupdf"]
    assert {r[0] for r in must_rows(res)} <= {"SAFE_REMOVED", "SAFE_REPLACED"}


def test_overlay_flow_report_and_query(env):
    res = run_and_score(env, "overlay", "overlay")
    assert {r for r in must_rows(res)} == {("LEAK_RECOVERABLE", "REMOVAL_INCOMPLETE", "confirmed")}
    report = build_report("overlay", res).read_text(encoding="utf-8")
    assert "REMOVAL_INCOMPLETE" in report and "data:image/png" in report
    shareable = build_report("overlay", res, shareable=True)
    assert shareable.name == "report-shareable.html"


def test_black_box_tier_ignores_undeclared_artifacts(env):
    res = run_and_score(env, "drops-headers", "bb", provides="output_pdf")
    rows, _ = query(res, "SELECT stages_exposed FROM documents WHERE must_mentions IS NOT NULL")
    assert rows == [("",)] or rows == [(None,)]
    assert {r[2] for r in must_rows(res) if r[1]} <= {"inferred"}


def test_oracle_regions_confirms_localization(env):
    oracle = run_and_score(env, "dpi-scale", "dpi-oracle", experiment="oracle-regions")
    assert {r[0] for r in must_rows(oracle)} <= {"SAFE_REMOVED", "SAFE_REPLACED"}
    e2e = run_and_score(env, "dpi-scale", "dpi-e2e", oracle={"oracle_regions": "dpi-oracle"})
    assert {r[1] for r in must_rows(e2e)} == {"LOCALIZATION_SYSTEMATIC"}


def test_compare_and_regression(env, tmp_path):
    a, b = env / "results" / "dpi-e2e", env / "results" / "ref"
    r = compare(a, b)
    assert r["counts"] == {"fixed": 13} and r["leak_rate_a"] == 1.0 and r["leak_rate_b"] == 0.0
    cases = tmp_path / "cases.yaml"
    case = promote(a, "dpi-e2e", "syn-digital", "m1", "dpi bug", path=cases)
    assert case["category"] == "LOCALIZATION_SYSTEMATIC"
    assert promote(a, "dpi-e2e", "syn-digital", "m1", path=cases)["fingerprint"] == case["fingerprint"]  # idempotent
    assert [c["result"] for c in check(b, path=cases)] == ["pass"]
    assert [c["result"] for c in check(a, path=cases)] == ["fail"]
    with pytest.raises(ValueError):
        promote(b, "ref", "syn-digital", "m1", path=cases)  # safe there: nothing to promote


def test_determinism_experiment(env):
    res = run_and_score(env, "nondeterministic", "nondet", experiment="determinism")
    rows, cols = query(res, "SELECT d3_identical, flip_rate FROM determinism")
    assert rows and rows[0][0] is False and rows[0][1] > 0
    tags, _ = query(res, "SELECT count(*) FROM mentions WHERE attr_tags LIKE '%NONDETERMINISTIC%'")
    assert tags[0][0] > 0


def test_candidate_check_rejects_broken_artifacts(env, tmp_path):
    manifest = load_candidate(candidate(env, "reference"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "output.pdf").write_bytes(b"%PDF-1.4 broken")
    (out / "plan.json").write_text('{"frame": "image_px", "regions": []}', encoding="utf-8")
    _, _, errors = load_artifacts(out, manifest)
    assert "output_pdf" in errors and "plan" in errors and "text" in errors  # text promised, missing


def test_experiment_not_applicable_when_injection_unsupported(env):
    cand = candidate(env, "reference")
    text = cand.read_text(encoding="utf-8").replace("accepts_injection: [text, mentions, entities, regions]",
                                                    "accepts_injection: []")
    cand.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not applicable"):
        run_candidate(cand, "synthetic@0.0.1", ROOT / "experiments" / "oracle-text.yaml", POLICY, run_id="na")
