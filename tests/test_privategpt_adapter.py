"""The runner for the client's PrivateGPT redaction (adapters/privategpt): his code graded as built, through the
text-level path. No test needs a model: a fake Ollama server (adapters/privategpt/fake_ollama.py) answers in his
format.

The fast tests need neither his environment nor docling: the manifest, the runner's helpers, the artifacts it
writes and its set-up failures. The tests that run his code need his environment (RB_PRIVATEGPT_PYTHON, built
as adapters/privategpt/README.md says) and his repository (RB_PRIVATEGPT_REPO, or the sibling folder
privategpt-redaction-testing); they are skipped without them. They run his code on fictional text only and
check that his repository is left as it was. On a machine without CUDA (his code asks docling for CUDA) they
also need RB_PRIVATEGPT_DOCLING_DEVICE=cpu, the runner's plumbing-check setting.
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.runner.runner import load_candidate
from benchmark.schemas import ArtifactKind, PiiType, RedactedTextArtifact
from benchmark.schemas.artifacts import RunInfo

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "privategpt"
RUNNER = ADAPTER / "run_privategpt.py"
MANIFEST = ROOT / "candidates" / "privategpt-text.yaml"
HIS_REPO = Path(os.environ.get("RB_PRIVATEGPT_REPO") or ROOT.parent / "privategpt-redaction-testing")
HIS_PYTHON = os.environ.get("RB_PRIVATEGPT_PYTHON", "")

needs_his_repo = pytest.mark.skipif(not (HIS_REPO / "backend" / "redaction").is_dir(),
                                    reason="the client's repository is not here (RB_PRIVATEGPT_REPO)")
needs_his_env = pytest.mark.skipif(not (HIS_PYTHON and Path(HIS_PYTHON).exists()
                                        and (HIS_REPO / "backend" / "redaction").is_dir()),
                                   reason="RB_PRIVATEGPT_PYTHON (his environment) or his repository is not set")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("rb_privategpt_runner", RUNNER)
fake_ollama = _load("rb_privategpt_fake_ollama", ADAPTER / "fake_ollama.py")


@pytest.fixture
def fake_server(tmp_path):
    servers = []

    def start(**kwargs):
        fake, server, url = fake_ollama.start(log_path=tmp_path / f"ollama-{len(servers)}.jsonl", **kwargs)
        servers.append(server)
        return fake, url

    yield start
    for server in servers:
        server.shutdown()


# -- manifest -----------------------------------------------------------------------------------------------------


def test_manifest_is_a_text_only_candidate():
    m = load_candidate(MANIFEST)
    assert m.primary_output is ArtifactKind.REDACTED_TEXT
    assert set(m.provides) == {ArtifactKind.REDACTED_TEXT, ArtifactKind.RUN_INFO}
    assert m.accepts_injection == []
    assert "{matter}" in m.command and "{policy}" not in m.command  # his tool follows its own rules
    assert "{env:RB_PRIVATEGPT_PYTHON}" in m.command and "{env:RB_PRIVATEGPT_REPO}" in m.command


@needs_his_repo
def test_label_map_covers_every_type_his_code_reports():
    """Every PII type his detector can report (its constants and the types his model is asked for) has a
    benchmark type, so his mappings are labelled. Read from his source, not imported."""
    src = (HIS_REPO / "backend" / "redaction" / "pii_detector.py").read_text(encoding="utf-8")
    constants = set(re.findall(r'^PII_TYPE_\w+ = "(\w+)"', src, flags=re.M))
    asked = re.search(r'- "type": one of (.+)\n', src).group(1)
    types = constants | set(re.findall(r'"(\w+)"', asked))
    assert len(types) >= 16
    label_map = load_candidate(MANIFEST).label_map
    assert types <= set(label_map), sorted(types - set(label_map))
    assert label_map["address"] is PiiType.ADDRESS and label_map["person_name"] is PiiType.PERSON_NAME


# -- runner helpers ---------------------------------------------------------------------------------------------


def test_one_project_per_matter_with_a_stable_session():
    a1, a2, b = runner.project_for("M1", "0" * 64), runner.project_for("M1", "1" * 64), runner.project_for("M2", "0" * 64)
    assert a1 == a2 and a1[0] == "matter-M1"  # every document of a matter: one project, one session
    assert b[1] != a1[1]
    lone1, lone2 = runner.project_for("", "a" * 64), runner.project_for("", "b" * 64)
    assert lone1[0].startswith("document-") and lone1[1] != lone2[1]  # no matter: a project per document


def test_store_is_shared_by_a_run_and_never_in_his_repository(tmp_path):
    out = tmp_path / "runs" / "priv-x" / "syn-digital" / "r0" / "out"
    out.mkdir(parents=True)
    assert runner.default_store_dir(out) == (tmp_path / "runs" / "priv-x" / "privategpt-store", "run")
    loose = tmp_path / "somewhere"
    loose.mkdir()
    assert runner.default_store_dir(loose) == (loose / "privategpt-store", "workdir")


def test_endpoint_and_model_come_from_the_environment():
    assert runner.settings({}) == {"ollama_url": "http://localhost:11434", "model": "qwen2.5:14b",
                                   "allow_remote": False}
    s = runner.settings({"RB_PRIVATEGPT_OLLAMA_URL": "http://10.0.0.5:11434/", "RB_PRIVATEGPT_MODEL": "m:1",
                         "RB_PRIVATEGPT_ALLOW_REMOTE": "1"})
    assert s == {"ollama_url": "http://10.0.0.5:11434", "model": "m:1", "allow_remote": True}


def test_a_remote_endpoint_needs_the_owners_go_ahead():
    for url in ("http://localhost:11434", "http://127.0.0.1:9", "http://[::1]:11434"):
        runner.check_endpoint_allowed(url, False)
    with pytest.raises(runner.SetupError, match="ALLOW_REMOTE"):
        runner.check_endpoint_allowed("http://54.0.0.1:11434", False)
    runner.check_endpoint_allowed("http://54.0.0.1:11434", True)
    with pytest.raises(runner.SetupError):
        runner.check_endpoint_allowed("file:///etc/passwd", True)


RESULT = {  # the shape of his redact_document() result, fictional values
    "original_text": "## Report\n\nMs Jane Doe saw Dr Ann Lee.\n",
    "redacted_text": "## Report\n\nMs Sarah Brown saw Dr Kate Hill.\n",
    "mappings": [{"pii_type": "person_name", "original": "Ms Jane Doe", "synthetic": "Ms Sarah Brown", "source": "llm"},
                 {"pii_type": "person_name", "original": "Dr Ann Lee", "synthetic": "Dr Kate Hill", "source": "llm"}],
    "verification": {"overall_confidence": 0.9, "checks_passed": 3, "checks_total": 4,
                     "issues": [{"severity": "critical", "type": "original_leak", "original_value": "Jane Doe",
                                 "location": "Jane Doe", "message": "Jane Doe"}]},
}


def test_his_result_becomes_a_valid_redacted_text_artifact_unchanged():
    art = RedactedTextArtifact.model_validate(runner.redacted_text_artifact(RESULT))
    assert art.original_text == RESULT["original_text"] and art.redacted_text == RESULT["redacted_text"]
    assert art.page_break == "\f"
    assert [(m.label, m.original, m.replacement) for m in art.mappings] == [
        ("person_name", "Ms Jane Doe", "Ms Sarah Brown"), ("person_name", "Dr Ann Lee", "Dr Kate Hill")]


def test_run_info_is_valid_and_quotes_no_values():
    summary = runner.verification_summary(RESULT["verification"])
    assert summary["issues_by_severity"] == {"critical": 1} and "Jane" not in json.dumps(summary)
    info = runner.run_info({"verification": summary}, "2026-10-30T00:00:00+00:00", "v", runner.settings({}))
    parsed = RunInfo.model_validate(info)
    assert parsed.release is None  # his tool holds nothing back: no release verdict
    assert parsed.models[0].name == "qwen2.5:14b"


class _Resp:
    def __init__(self, status):
        self.status_code = status


class _FakeRequests:
    class RequestException(Exception):
        pass

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.other = "passed through"

    def post(self, *args, **kwargs):
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_model_calls_are_counted_not_changed():
    class Detector:  # stands in for his pii_detector module
        class OllamaJSONError(RuntimeError):
            pass

        requests = _FakeRequests([_Resp(200), _Resp(500), _FakeRequests.RequestException("down")])

        @staticmethod
        def _ollama_chat_json(*args, **kwargs):
            if kwargs.get("fail"):
                raise Detector.OllamaJSONError("no json")
            return ["kept"]

    counter = runner.LlmCounter()
    runner.install_counters(Detector, counter)
    ok = _Resp(200)
    Detector.requests._real.outcomes.insert(0, ok)
    assert Detector.requests.post("u") is ok  # the very same answer
    Detector.requests.post("u")
    assert Detector.requests.post("u").status_code == 500
    with pytest.raises(_FakeRequests.RequestException):
        Detector.requests.post("u")
    assert Detector.requests.other == "passed through" and Detector.requests.RequestException is _FakeRequests.RequestException
    assert Detector._ollama_chat_json() == ["kept"]
    with pytest.raises(Detector.OllamaJSONError):
        Detector._ollama_chat_json(fail=True)
    assert counter.as_dict() == {"http_calls": 4, "http_failures": 2, "detector_calls": 2,
                                 "detector_calls_without_json": 1}


def test_the_laptop_device_setting_reaches_only_his_converters_device():
    """RB_PRIVATEGPT_DOCLING_DEVICE (plumbing checks without CUDA) changes the device his converters ask for and
    nothing else in their options."""
    from types import SimpleNamespace as NS

    def option(device):
        return NS(pipeline_options=NS(accelerator_options=NS(device=device), do_ocr=False))

    fast = NS(format_to_options={"pdf": option("cuda"), "docx": NS(pipeline_options=None)})
    ocr = NS(format_to_options={"pdf": option("cuda")})
    runner.set_docling_device(NS(_docling_converter=fast, _docling_converter_ocr=ocr), "cpu")
    assert fast.format_to_options["pdf"].pipeline_options.accelerator_options.device == "cpu"
    assert ocr.format_to_options["pdf"].pipeline_options.accelerator_options.device == "cpu"
    assert fast.format_to_options["pdf"].pipeline_options.do_ocr is False
    runner.set_docling_device(NS(_docling_converter=fast, _docling_converter_ocr=None), "cpu")  # no OCR converter


# -- the fake server --------------------------------------------------------------------------------------------


def test_fake_server_answers_his_prompts(fake_server):
    fake, url = fake_server(entities=[{"type": "organisation_name", "value": "Acme Widgets Pty Ltd"}])
    tags = runner._get_json(f"{url}/api/tags", 5)
    assert tags["models"][0]["name"] == "qwen2.5:14b"
    prompt = ("You are a Named Entity Recognition assistant for legal documents.\nText:\n---\n"
              "Ms Jane Doe works at Acme Widgets Pty Ltd.\n---\n\nJSON array:")
    reply = runner._post_json(f"{url}/api/chat", {"model": "qwen2.5:14b", "messages": [{"role": "user", "content": prompt}]}, 5)
    items = json.loads(reply["message"]["content"])
    assert {"type": "person_name", "value": "Ms Jane Doe"} in items
    assert {"type": "organisation_name", "value": "Acme Widgets Pty Ltd"} in items
    alias = runner._post_json(f"{url}/api/chat", {"messages": [{"role": "user", "content": "You are a name resolution assistant"}]}, 5)
    assert json.loads(alias["message"]["content"]) == {}
    assert all("content" not in r for r in fake.requests)  # the request log holds no prompt text


def test_runner_set_up_check_passes_on_the_fake_and_fails_without_a_model(fake_server):
    _, url = fake_server()
    runner.check_model(url, "qwen2.5:14b")
    with pytest.raises(runner.SetupError, match="ollama pull"):
        runner.check_model(url, "other-model:7b")
    with pytest.raises(runner.SetupError, match="does not answer"):
        runner.check_model("http://127.0.0.1:9", "qwen2.5:14b")


def test_a_plumbing_run_says_so_in_its_version(fake_server, tmp_path):
    """A run against the fake server, or with docling's device switched for a laptop, is not his tool as built: its
    version (which every board and demo row shows) carries the mark. A real endpoint, or none, adds nothing."""
    _, url = fake_server()
    no_git = tmp_path  # not a repository: his commit reads 'unknown'
    real = runner.version_line(no_git, {"RB_PRIVATEGPT_OLLAMA_URL": "http://127.0.0.1:9"})
    assert real.startswith("privategpt@unknown model=qwen2.5:14b runner=") and runner.PLUMBING_MARK not in real
    assert runner.version_line(no_git, {"RB_PRIVATEGPT_OLLAMA_URL": url}).endswith(
        " [plumbing check: fake model; not the tool as built]")
    both = runner.version_line(no_git, {"RB_PRIVATEGPT_OLLAMA_URL": url, "RB_PRIVATEGPT_DOCLING_DEVICE": "cpu"})
    assert both.endswith(" [plumbing check: fake model, docling on cpu; not the tool as built]")
    cpu = runner.version_line(no_git, {"RB_PRIVATEGPT_OLLAMA_URL": "http://127.0.0.1:9",
                                       "RB_PRIVATEGPT_DOCLING_DEVICE": "cpu"})
    assert cpu.endswith(" [plumbing check: docling on cpu; not the tool as built]")
    # An endpoint on another machine is not asked without the owner's go-ahead.
    assert not runner.is_fake_endpoint("http://203.0.113.7:11434", allow_remote=False)


def _run_runner(python, workdir, env_extra, *extra, input_pdf=None, matter="MT", timeout=1800):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **env_extra}
    cmd = [python, str(RUNNER), "--repo", str(HIS_REPO), "--input", str(input_pdf or workdir / "missing.pdf"),
           "--workdir", str(workdir), "--matter", matter, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


@pytest.mark.parametrize("env_extra, mode, expected", [
    ({"RB_PRIVATEGPT_OLLAMA_URL": "http://127.0.0.1:9"}, "e2e", "does not answer"),
    ({"RB_PRIVATEGPT_OLLAMA_URL": "http://203.0.113.7:11434"}, "e2e", "ALLOW_REMOTE"),
    ({}, "oracle_text", "not supported"),
])
def test_a_document_that_cannot_run_fails_loudly(tmp_path, env_extra, mode, expected):
    """Fails before his code runs: non-zero exit, error.json and run_info.json, no redacted text."""
    out = tmp_path / "out"
    dummy = tmp_path / "input.pdf"
    dummy.write_bytes(b"%PDF-1.4 not read\n")
    proc = _run_runner(sys.executable, out, env_extra, "--mode", mode, input_pdf=dummy, timeout=120)
    assert proc.returncode == (4 if mode == "e2e" else 2), proc.stdout + proc.stderr
    err = json.loads((out / "error.json").read_text(encoding="utf-8"))
    assert expected in err["error"]
    RunInfo.model_validate(json.loads((out / "run_info.json").read_text(encoding="utf-8")))
    assert not (out / "redacted_text.json").exists()


# -- his code, in his environment, against the fake server -----------------------------------------------------------


def _his_repo_state() -> str:
    """A fingerprint of his working tree, ignored files included (a written database or bytecode shows up). Only
    the hash is compared, so a failure never prints his file names. --no-optional-locks: git writes nothing."""
    status = subprocess.run(["git", "--no-optional-locks", "-C", str(HIS_REPO), "status", "--porcelain",
                             "--ignored"], capture_output=True, text=True, check=True).stdout
    return hashlib.sha256(status.encode("utf-8")).hexdigest()


@needs_his_env
def test_his_own_detector_parses_the_fake_servers_answers(fake_server):
    """His detect_llm_pii and resolve_name_aliases, unchanged, read the fake's answers."""
    fake, url = fake_server(entities=[{"type": "person_name", "value": "Harriet Quillfeather"}])
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(HIS_REPO / 'backend')!r})\n"
        "from redaction import pii_detector as p\n"
        "text = 'Ms Harriet Quillfeather was seen by Dr Bertram Oakhollow. Harriet Quillfeather agreed.'\n"
        f"found = p.detect_llm_pii(text, model='qwen2.5:14b', ollama_url={url!r})\n"
        f"aliases = p.resolve_name_aliases(['Ms Quillfeather', 'Harriet Quillfeather'], text, ollama_url={url!r})\n"
        "print('RESULT', json.dumps({'found': sorted(f['value'] for f in found), 'aliases': aliases}))\n")
    before = _his_repo_state()
    proc = subprocess.run([HIS_PYTHON, "-B", "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    assert "Harriet Quillfeather" in res["found"] and any("Oakhollow" in v for v in res["found"])
    assert res["aliases"] == {}
    chats = [r for r in fake.requests if r["path"] == "/api/chat"]
    assert {r["kind"] for r in chats} == {"ner", "alias"}
    assert all(r["model"] == "qwen2.5:14b" and r["options"]["seed"] == 42 for r in chats)  # his pinned options
    assert _his_repo_state() == before


def _fictional_pdf(path: Path, lines: list[str]) -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(path)
    doc.close()
    return path


DOC_A = ["Medical report", "Patient: Ms Harriet Quillfeather", "Seen by Dr Bertram Oakhollow on 04/05/2021.",
         "Ms Harriet Quillfeather reported back pain after lifting boxes at work.",
         "Email: harriet.q@example.org  Phone: 0412 555 019"]
DOC_B = ["Letter of referral", "Re: Ms Harriet Quillfeather",
         "Please review Ms Harriet Quillfeather for ongoing physiotherapy.", "Kind regards, Dr Mabel Thornquist"]


@needs_his_env
def test_his_tool_redacts_a_matter_as_one_project(tmp_path, fake_server):
    """Two fictional documents of one matter through his pipeline (docling, regex, the fake model, his fakes,
    his store): valid redacted text, his replacements kept one for one, the second document reuses the
    project's mappings, the store sits in the run folder and his repository is unchanged."""
    fake, url = fake_server(entities=[{"type": "person_name", "value": "Harriet Quillfeather"}])
    run = tmp_path / "runs" / "priv-test"
    env = {"RB_PRIVATEGPT_OLLAMA_URL": url}
    before = _his_repo_state()
    arts, infos = [], []
    for doc_id, lines in (("doc-a", DOC_A), ("doc-b", DOC_B)):
        out = run / doc_id / "r0" / "out"
        out.mkdir(parents=True)
        pdf = _fictional_pdf(run / doc_id / "r0" / "input.pdf", lines)
        proc = _run_runner(HIS_PYTHON, out, env, input_pdf=pdf)
        assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
        arts.append(RedactedTextArtifact.model_validate_json((out / "redacted_text.json").read_text(encoding="utf-8")))
        infos.append(json.loads((out / "run_info.json").read_text(encoding="utf-8")))
        assert not (out / "error.json").exists()
    a, b = arts
    assert "Quillfeather" in a.original_text and "Quillfeather" not in a.redacted_text
    assert "Oakhollow" not in a.redacted_text and "harriet.q@example.org" not in a.redacted_text
    fake_a = {m.original: m.replacement for m in a.mappings}
    fake_b = {m.original: m.replacement for m in b.mappings}
    shared = set(fake_a) & set(fake_b)
    assert shared and all(fake_a[k] == fake_b[k] for k in shared)  # one fake per person across the matter
    assert infos[0]["config"]["project_created"] and not infos[1]["config"]["project_created"]
    assert infos[1]["config"]["project_documents_before"] == 1 and infos[1]["config"]["project_mappings_before"] > 0
    assert infos[0]["config"]["session_id"] == infos[1]["config"]["session_id"]
    assert infos[1]["config"]["llm"]["http_failures"] == 0
    assert (run / "privategpt-store" / "redaction_data.db").is_file()
    assert _his_repo_state() == before


@needs_his_env
def test_a_model_failure_mid_document_withholds_the_output(tmp_path, fake_server):
    """The endpoint answers the set-up check, then fails his detector's calls: his code would carry on with
    regex only, so the runner withholds the output, exits 3 and says why."""
    _, url = fake_server(mode="http500", ok_calls=1)
    out = tmp_path / "runs" / "priv-test" / "doc-a" / "r0" / "out"
    out.mkdir(parents=True)
    pdf = _fictional_pdf(out.parent / "input.pdf", DOC_A)
    proc = _run_runner(HIS_PYTHON, out, {"RB_PRIVATEGPT_OLLAMA_URL": url}, input_pdf=pdf)
    assert proc.returncode == 3, (proc.stdout + proc.stderr)[-3000:]
    err = json.loads((out / "error.json").read_text(encoding="utf-8"))
    assert err["llm"]["http_failures"] > 0
    assert not (out / "redacted_text.json").exists()
