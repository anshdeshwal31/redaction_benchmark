"""End-to-end: mocks run as external processes through the runner, then scoring, report,
compare, regression and candidate checks (PLAN.md phase 6 exit criterion)."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.dataset import document_pdf, ingest, load_manifest
from benchmark.evaluate import EvalConfig
from benchmark.report.compare import compare
from benchmark.report.html import build_report
from benchmark.report.regression import check, promote
from benchmark.report.store import query
from benchmark.runner.artifacts import load_artifacts
from benchmark.runner.runner import (
    CandidateEnvError,
    CommandValueError,
    build_command,
    load_candidate,
    resolve_version,
    run_candidate,
)
from benchmark.runner.scoring import score_run
from benchmark.schemas import CandidateManifest, DocumentEntry
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


def test_stage_swap_injects_another_candidates_text(env, tmp_path):
    # Tool A cannot read names; tool B reads perfectly. B fed with A's text inherits A's misses.
    run_and_score(env, "ocr-misses-names", "ocr-a")
    exp = tmp_path / "swap.yaml"
    exp.write_text('schema: rb.experiment/0.1\nname: swap-text\nmode: oracle_text\ninject_from: "run:ocr-a"\n',
                   encoding="utf-8")
    cand = candidate(env, "reference")
    run = run_candidate(cand, "synthetic@0.0.1", exp, POLICY, run_id="swap")
    assert all(not i.error for i in run.items) and run.items[0].injected
    res = score_run("swap", POLICY, CFG)
    leaks = [r for r in must_rows(res) if r[0] not in ("SAFE_REMOVED", "SAFE_REPLACED")]
    assert len(leaks) == 9  # the names A could not read


def test_failed_runs_count_in_the_strict_view(env, tmp_path):
    cand = candidate(env, "reference")
    text = cand.read_text(encoding="utf-8").replace("--spec reference", "--spec no-such-spec")
    broken = tmp_path / "broken.yaml"
    broken.write_text(text.replace("name: mock-reference", "name: mock-broken"), encoding="utf-8")
    run = run_candidate(broken, "synthetic@0.0.1", ROOT / "experiments" / "e2e.yaml", POLICY, run_id="broken")
    assert run.items[0].error
    res = score_run("broken", POLICY, CFG)
    rows = must_rows(res)
    assert rows and {r[0] for r in rows} == {"NO_OUTPUT"} and {r[1] for r in rows} == {"CANDIDATE_ERROR"}
    html = build_report("broken", res).read_text(encoding="utf-8")
    assert "leaking, completed runs only" in html and "from failed runs" in html


def test_board_scorecard_heatmaps_and_decomposition(env, tmp_path):
    from benchmark.report.board import build_board, decomposition, RunData

    out = build_board(["ref", "overlay", "dpi-oracle", "dpi-e2e"], tmp_path / "board.html")
    page = out.read_text(encoding="utf-8")
    assert "Scorecard" in page and "By PII type" in page and "Failure root causes" in page
    dec = decomposition([RunData("dpi-oracle"), RunData("dpi-e2e")])
    assert len(dec) == 1
    parts = dict(dec[0]["parts"])
    assert parts["application"] == 0.0 and abs(sum(parts.values()) - 1.0) < 1e-9


def test_ladder_dataset_and_curves(env, tmp_path):
    from benchmark.report.board import build_board, ladders, RunData
    from benchmark.synth.register import register_ladder

    m = register_ladder("dpi", "0.0.1", "synthetic@0.0.1")
    assert [d.derived_from.value for d in m.documents] == [300.0, 200.0, 150.0, 100.0]
    cand = candidate(env, "reference-raster")
    run_candidate(cand, "ladder-dpi@0.0.1", ROOT / "experiments" / "e2e.yaml", POLICY, run_id="lad")
    score_run("lad", POLICY, CFG)
    lad = ladders([RunData("lad")])
    assert sorted(v for v, _, _ in lad["dpi"]["mock-reference-raster"]) == [100.0, 150.0, 200.0, 300.0]
    page = build_board(["lad"], tmp_path / "b.html").read_text(encoding="utf-8")
    assert "Leak rate vs dpi" in page


# -- command placeholders: {policy}, {matter}, {env:NAME} ---------------------------------------------


def _manifest(command: str, version_command: str | None = None) -> CandidateManifest:
    return CandidateManifest(name="tool", command=command, provides=["output_pdf"],
                             **({"version_command": version_command} if version_command else {"version": "1"}))


def test_policy_matter_and_env_placeholders_reach_the_tool_intact(tmp_path, monkeypatch):
    # The path of this repository has spaces and an apostrophe ("Joel's projects"); so do these.
    odd = tmp_path / "Joel's tools" / "a b"
    odd.mkdir(parents=True)
    policy = odd / "policy v1.yaml"
    policy.write_text("x", encoding="utf-8")
    monkeypatch.setenv("RB_TEST_TOOL_PYTHON", sys.executable)
    monkeypatch.setenv("RB_TEST_TOOL_HOME", str(odd))
    manifest = _manifest('{env:RB_TEST_TOOL_PYTHON} -c "import sys; print(chr(10).join(sys.argv[1:]))" '
                         "{policy} {matter} {env:RB_TEST_TOOL_HOME} {matter}")
    cmd = build_command(manifest, tmp_path / "input.pdf", tmp_path / "out", "e2e", None, 0, policy=policy,
                        matter="Matter 7's")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [str(policy.resolve()), "Matter 7's", str(odd), "Matter 7's"]
    no_matter = build_command(manifest, tmp_path / "input.pdf", tmp_path / "out", "e2e", None, 0, policy=policy)
    assert subprocess.run(no_matter, shell=True, capture_output=True, text=True,
                          timeout=60).stdout.splitlines()[1] == ""  # "" when the document has no matter


def test_version_command_expands_python_repo_and_env(monkeypatch):
    monkeypatch.setenv("RB_TEST_TOOL_PYTHON", sys.executable)
    assert resolve_version(_manifest("x", '{env:RB_TEST_TOOL_PYTHON} -c "print(4.2)"')) == "4.2"
    assert resolve_version(_manifest("x", '{python} -c "import sys; print(sys.argv[1])" {repo}')) == str(ROOT)


def test_an_unset_tool_variable_fails_before_anything_runs(env, tmp_path, monkeypatch):
    monkeypatch.delenv("RB_NOT_SET_PYTHON", raising=False)
    cand = tmp_path / "tool.yaml"
    cand.write_text("schema: rb.candidate/0.1\nname: tool\nversion_command: '{env:RB_NOT_SET_PYTHON} -V'\n"
                    "command: '{env:RB_NOT_SET_PYTHON} -m tool {input} {workdir}'\nprovides: [output_pdf]\n",
                    encoding="utf-8")
    with pytest.raises(CandidateEnvError, match="RB_NOT_SET_PYTHON is not set; set it to the tool's interpreter"):
        run_candidate(cand, "synthetic@0.0.1", ROOT / "experiments" / "e2e.yaml", POLICY, run_id="unset")
    assert not (env / "runs" / "unset").exists()
    with pytest.raises(CandidateEnvError, match="invalid placeholder"):
        build_command(_manifest("{env:rb_lower} {input}"), tmp_path / "in.pdf", tmp_path, "e2e", None, 0)


# Commands run through cmd.exe on Windows, which acts on & | < > ^ ( ) outside quotes and expands %NAME%
# even inside them.
ARGV = '{python} -c "import sys; print(repr(sys.argv[1:]))" BEGIN {matter} END'
CMD_UNSAFE = ["x%OS%y", 'a"b & c', "a!b", "a\nb"]


def _argv(cmd: str) -> list[str]:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return ast.literal_eval(out.stdout.strip())


@pytest.mark.parametrize("value", ["M&A-07", "a^b", "a|b", "a<b>c", "(x) & y", "Smith & Jones", "trail\\"]
                         + ([] if os.name == "nt" else CMD_UNSAFE))
def test_shell_characters_in_a_value_reach_the_tool_unchanged(tmp_path, value):
    cmd = build_command(_manifest(ARGV), tmp_path / "in.pdf", tmp_path, "e2e", None, 0, matter=value)
    assert _argv(cmd) == ["BEGIN", value, "END"]


@pytest.mark.skipif(os.name != "nt", reason="only cmd.exe changes these inside quotes")
@pytest.mark.parametrize("value", CMD_UNSAFE)
def test_a_value_cmd_would_change_is_refused_and_named(tmp_path, value):
    with pytest.raises(CommandValueError, match=r"the matter id \(\{matter\}\) contains"):
        build_command(_manifest(ARGV), tmp_path / "in.pdf", tmp_path, "e2e", None, 0, matter=value)


def test_a_dataset_matter_is_a_plain_code(env):
    entry = load_manifest("synthetic", "0.0.1").documents[0].model_dump(mode="json")
    for ok in ("M1", "MX", "case-2024.07_b"):
        assert DocumentEntry.model_validate(dict(entry, matter=ok)).matter == ok
    for bad in ("M&A-07", "A^B", "x%OS%y", "Smith v Jones", "-M1"):
        with pytest.raises(ValidationError):
            DocumentEntry.model_validate(dict(entry, matter=bad))


def test_an_env_value_set_with_quotes_is_used_without_them(monkeypatch):
    monkeypatch.setenv("RB_TEST_TOOL_PYTHON", f'"{sys.executable}"')  # what cmd's `set X="..."` stores
    assert resolve_version(_manifest("x", '{env:RB_TEST_TOOL_PYTHON} -c "print(4.2)"')) == "4.2"


@pytest.mark.skipif(os.name != "nt", reason="only cmd.exe changes these inside quotes")
def test_an_env_value_cmd_would_change_stops_the_run_before_anything_runs(env, tmp_path, monkeypatch):
    monkeypatch.setenv("RB_TEST_PCT_PYTHON", "C:\\R%OS%D\\python.exe")
    cand = tmp_path / "pct.yaml"
    cand.write_text("schema: rb.candidate/0.1\nname: pct\nversion: '1'\n"
                    "command: '{env:RB_TEST_PCT_PYTHON} {input}'\nprovides: [output_pdf]\n", encoding="utf-8")
    with pytest.raises(CandidateEnvError, match="RB_TEST_PCT_PYTHON contains '%'"):
        run_candidate(cand, "synthetic@0.0.1", ROOT / "experiments" / "e2e.yaml", POLICY, run_id="pct")
    assert not (env / "runs" / "pct").exists()


# -- the matter the tool is given ---------------------------------------------------------------------


def test_a_matter_that_differs_between_dataset_and_gt_is_reported(env, tmp_path):
    src = tmp_path / "syn-digital.pdf"
    shutil.copyfile(document_pdf(load_manifest("synthetic", "0.0.1").document("syn-digital")), src)
    ingest("mattermix", "0.0.1", {"syn-digital": src}, split="synthetic", matters={"syn-digital": "M9"})
    run_candidate(candidate(env, "reference"), "mattermix@0.0.1", ROOT / "experiments" / "e2e.yaml", POLICY,
                  run_id="mattermix")
    res = score_run("mattermix", POLICY, CFG)
    scoring = json.loads((res / "scoring.json").read_text(encoding="utf-8"))
    assert scoring["matter_mismatch"] == [{"doc_id": "syn-digital", "dataset": "M9", "gt": "MX"}]
    assert "Matter ids differ between the dataset and the GT" in build_report("mattermix", res).read_text(
        encoding="utf-8")
    # Also when only one of them has a matter: the tool was told none, the GT groups the document with MX.
    ingest("matterless", "0.0.1", {"syn-digital": src}, split="synthetic")
    run_candidate(candidate(env, "reference"), "matterless@0.0.1", ROOT / "experiments" / "e2e.yaml", POLICY,
                  run_id="matterless")
    res = score_run("matterless", POLICY, CFG)
    scoring = json.loads((res / "scoring.json").read_text(encoding="utf-8"))
    assert scoring["matter_mismatch"] == [{"doc_id": "syn-digital", "dataset": None, "gt": "MX"}]
    assert "syn-digital (dataset none, GT MX)" in build_report("matterless", res).read_text(encoding="utf-8")
