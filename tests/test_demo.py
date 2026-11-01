"""The demo web app (demo/): both tool columns fill from the adapter contract alone, jobs queue one at a
time, failures and timeouts read clearly, uploads are deleted, and the Benchmark page shows the
benchmark's own scorecard numbers."""

import io
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from benchmark.report import board as board_mod  # noqa: E402
from benchmark.report.store import write_tables  # noqa: E402
from benchmark.synth import ALL_VARIANTS  # noqa: E402
from demo import jobs as J  # noqa: E402
from demo.app import create_app  # noqa: E402
from demo.config import CandidateSlot, DemoConfig, load_config  # noqa: E402
from demo import results as R  # noqa: E402
from demo.results import FolderRun, summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "candidates" / "mock-reference-replace.yaml"
STANDIN = ROOT / "demo" / "candidates" / "standin-text.yaml"
UPLOAD_NAME = "Zebedee Quixley IME report.pdf"  # a file name that is personal data itself

# A PDF tool that holds its output back: copies the input, lists two replacements, reports a hold.
FAKE_PDF_TOOL = '''
import json, os, shutil, sys
src, out = sys.argv[1], sys.argv[2]
shutil.copyfile(src, out + "/output.pdf")
json.dump({"schema": "rb.pseudonyms/0.1", "entries": [
    {"cluster": "E1", "original": "Avery Example", "replacement": "Blake Sample"},
    {"cluster": "E2", "original": "12 Fictional St", "replacement": "48 Madeup Rd"}]}, open(out + "/pseudonyms.json", "w"))
json.dump({"schema": "rb.runinfo/0.1", "candidate": "fake-pdf", "candidate_version": os.environ.get("DEMO_TEST_VERSION", "9.9"),
           "notes": "held for review: output.pdf is the copy it would not release",
           "release": {"released": False, "checks_failed": ["re-read-failed"]}}, open(out + "/run_info.json", "w"))
'''


@pytest.fixture(scope="module")
def pdf() -> bytes:
    return ALL_VARIANTS["digital"]().pdf  # fictional; its GT is in gt/synthetic, so the mock can redact it


def _manifest(tmp: Path, name: str, command: str, provides: str, timeout_s: int = 300) -> Path:
    path = tmp / f"{name}.yaml"
    path.write_text(f"schema: rb.candidate/0.1\nname: {name}\nversion: \"0.1\"\ncommand: '{command}'\n"
                    f"provides: [{provides}]\ntimeout_s: {timeout_s}\n", encoding="utf-8")
    return path


def _standin(tmp: Path, name: str, extra: str, timeout_s: int = 300) -> Path:
    return _manifest(tmp, name, "{python} {repo}/demo/standin_text.py --input {input} --workdir {workdir} " + extra,
                     "redacted_text", timeout_s)


@pytest.fixture
def make_app(tmp_path):
    apps = []

    def make(candidates, **kw):
        # The work folder's path has a space and an apostrophe, like the repository's own.
        cfg = DemoConfig(candidates=[CandidateSlot(Path(m), label) for m, label in candidates],
                         work_dir=tmp_path / "Joel's demo" / "work dir", results_dirs=[tmp_path / "results"], **kw)
        app = create_app(cfg)
        app.config["TESTING"] = True
        apps.append(app)
        return app, app.extensions["demo_queue"], app.test_client(), app.jinja_env.globals["form_token"]

    yield make
    for app in apps:
        app.extensions["demo_queue"].shutdown()


def _upload(client, token, data: bytes, name: str = UPLOAD_NAME):
    return client.post("/upload", data={"token": token, "pdf": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def _job(queue, resp) -> J.Job:
    assert resp.status_code == 303, resp.get_data(as_text=True)[:500]
    return queue.get(resp.headers["Location"].rstrip("/").split("/")[-1])


def _wait(job: J.Job, timeout: float = 240) -> None:
    end = time.time() + timeout
    while job.state != J.FINISHED:
        assert time.time() < end, [(r.spec.name, r.state) for r in job.runs]
        time.sleep(0.2)


@pytest.mark.parametrize("template, found", [
    ("{python} -m tool --input {input} --workdir {workdir} {seed}", []),
    ("{env:RB_X_PYTHON} '{repo}/a b.py' \"--flag=x y\" {matter}", []),
    ("cd {repo} && {python} -m tool", ["&"]),
    ("{python} -m tool {input} | tee log", ["|"]),
    ("{python} -m tool $HOME", ["$"]),
    ("{python} -m tool \"$HOME\"", ["$"]),
    ("OLLAMA_HOST=x {python} -m tool", ["VAR=value before the command"]),
    ("~/venv/bin/python -m tool", ["~"]),
    ("{python} -c 'import sys; print(sys.argv)' {input}", []),
])
def test_shell_syntax_is_found_outside_quotes_only(template, found):
    assert J.shell_syntax(template) == found


def test_both_columns_fill_from_the_adapter_contract(make_app, pdf, tmp_path):
    app, queue, client, token = make_app([(MOCK, "Tool 2 (stand-in mock)"), (STANDIN, "Text tool (stand-in)")])
    page = client.get("/").get_data(as_text=True)
    assert page.count("badge ok") == 2 and "redacted text (not a PDF)" in page
    job = _job(queue, _upload(client, token, pdf))
    _wait(job)
    pdf_run, text_run = job.runs
    assert (pdf_run.state, text_run.state) == (J.DONE, J.DONE), [(r.state, r.message) for r in job.runs]
    assert (pdf_run.spec.kind, text_run.spec.kind) == ("pdf", "text")
    assert pdf_run.preview_pages == pdf_run.page_count >= 1 and job.original_pages == job.page_count
    # Each tool got its own copy of the upload under the neutral name, and its own output folder.
    for i in (0, 1):
        assert (job.dir / f"tool-{i}" / "input.pdf").read_bytes() == pdf
    assert J.replacements(job, pdf_run) and J.replacements(job, text_run)

    html = client.get(f"/jobs/{job.id}").get_data(as_text=True)
    assert "Tool 2 (stand-in mock)" in html and "Text tool (stand-in)" in html
    assert "This tool outputs text, not a PDF." in html and "http-equiv=\"refresh\"" not in html
    assert html.count("What was replaced") == 2
    assert f"/jobs/{job.id}/tool/0/1.png" in html and f"/jobs/{job.id}/original/1.png" in html
    # The original is one column wide: it shares the tool columns' grid, with an empty cell per other tool.
    original = html[html.index('<div class="grid original">'):html.index("<section")]
    assert "original/1.png" in original and original.count('class="spacer"') == len(job.runs) - 1
    got = client.get(f"/jobs/{job.id}/tool/0/output.pdf")
    assert got.status_code == 200 and got.data[:5] == b"%PDF-" and got.headers["Cache-Control"] == "no-store"
    assert client.get(f"/jobs/{job.id}/tool/0/1.png").data[:4] == b"\x89PNG"
    txt = client.get(f"/jobs/{job.id}/tool/1/redacted.txt")
    assert txt.status_code == 200 and txt.data.strip()
    # The .json download is the redacted text only: the tool's own file also holds the original text and every
    # original value, which a file named "redacted" must never carry.
    got = client.get(f"/jobs/{job.id}/tool/1/redacted.json")
    assert got.status_code == 200 and got.headers["Content-Disposition"].endswith(".json")
    own = json.loads((job.dir / "tool-1" / "out" / "redacted_text.json").read_text(encoding="utf-8"))
    served = got.get_json()
    assert set(served) == {"redacted_text", "page_break"} and served["redacted_text"] == own["redacted_text"]
    originals = [m["original"] for m in own["mappings"] if m["original"] not in own["redacted_text"]]
    assert originals and not any(o in got.get_data(as_text=True) for o in originals)
    assert own["original_text"] not in got.get_data(as_text=True)
    assert client.get(f"/jobs/{job.id}/tool/1/output.pdf").status_code == 404  # a text tool has no PDF
    status = client.get(f"/jobs/{job.id}/status.json").get_json()
    assert [t["state"] for t in status["tools"]] == ["done", "done"]
    # The uploaded file's name is kept nowhere: not on disk, not on the pages.
    names = [p.name for p in job.dir.rglob("*")]
    assert not any("Zebedee" in n or "Quixley" in n for n in names)
    assert "Quixley" not in html and "Quixley" not in client.get("/").get_data(as_text=True)


def test_a_held_back_pdf_shows_its_reasons_and_replacements(make_app, pdf, tmp_path, monkeypatch):
    script = tmp_path / "fake pdf tool.py"
    script.write_text(FAKE_PDF_TOOL, encoding="utf-8")
    monkeypatch.setenv("DEMO_TEST_TOOL", str(script))
    cand = _manifest(tmp_path, "fake-pdf", "{python} {env:DEMO_TEST_TOOL} {input} {workdir}", "output_pdf, pseudonyms")
    app, queue, client, token = make_app([(cand, "PDF tool")])
    job = _job(queue, _upload(client, token, pdf))
    _wait(job)
    run = job.runs[0]
    assert run.state == J.DONE and run.version == "9.9"
    assert run.release == {"released": False, "checks_failed": ["re-read-failed"]}
    html = client.get(f"/jobs/{job.id}").get_data(as_text=True)
    assert "Held back." in html and "re-read-failed" in html
    assert "Avery Example" in html and "48 Madeup Rd" in html
    # The held-back copy is offered for review, never as "the redacted PDF", after a warning and the tool's note.
    assert "Download the redacted PDF" not in html
    assert "Download the held-back output (not safe to share)" in html and "do not share it" in html
    assert "output.pdf is the copy it would not release" in html
    assert "Not the tool as built" not in html


def test_a_plumbing_run_is_marked_as_not_the_tool_as_built(make_app, pdf, tmp_path, monkeypatch):
    script = tmp_path / "fake pdf tool.py"
    script.write_text(FAKE_PDF_TOOL, encoding="utf-8")
    monkeypatch.setenv("DEMO_TEST_TOOL", str(script))
    monkeypatch.setenv("DEMO_TEST_VERSION", "privategpt@abc1234 model=qwen2.5:14b runner=0123456789ab "
                                            "[plumbing check: fake model, docling on cpu; not the tool as built]")
    cand = _manifest(tmp_path, "fake-pdf", "{python} {env:DEMO_TEST_TOOL} {input} {workdir}", "output_pdf, pseudonyms")
    app, queue, client, token = make_app([(cand, "PDF tool")])
    job = _job(queue, _upload(client, token, pdf))
    _wait(job)
    assert job.runs[0].plumbing == "plumbing check: fake model, docling on cpu; not the tool as built"
    assert "Not the tool as built." in client.get(f"/jobs/{job.id}").get_data(as_text=True)
    # A run that records the laptop's docling setting, from before the version carried the mark, is caught too.
    assert J.plumbing_reason("privategpt@abc model=m runner=r", {"docling_device_override": "cpu"}).startswith(
        "plumbing check: docling on cpu")
    assert J.plumbing_reason("0.1.0+g46c4498", {}) == "" and J.plumbing_reason(None) == ""


def test_failures_timeouts_and_tools_that_cannot_run_read_clearly(make_app, pdf, tmp_path, monkeypatch):
    monkeypatch.delenv("RB_DEMO_UNSET_PYTHON", raising=False)
    failing = _standin(tmp_path, "fails", "--fail")
    slow = _standin(tmp_path, "slow", "--sleep 60")
    unset = _manifest(tmp_path, "unset", "{env:RB_DEMO_UNSET_PYTHON} -m tool {input} {workdir}", "output_pdf")
    shell = _manifest(tmp_path, "shell", "cd {repo} && {python} -m tool {input}", "output_pdf")
    says = _manifest(tmp_path, "says", '{python} -c "print(424242, 40 + 2); raise SystemExit(4)"', "redacted_text")
    app, queue, client, token = make_app([(failing, "Fails"), (slow, "Slow"), (unset, "Unset"), (shell, "Shell"),
                                          (tmp_path / "missing.yaml", "Missing"), (says, "Says why")], job_timeout_s=2)
    page = client.get("/").get_data(as_text=True)
    assert page.count("not ready") == 3 and "RB_DEMO_UNSET_PYTHON is not set" in page
    assert "shell syntax (&amp;)" in page and "candidate manifest not found" in page
    job = _job(queue, _upload(client, token, pdf))
    _wait(job, timeout=60)
    states = {r.spec.label: (r.state, r.message) for r in job.runs}
    assert states["Fails"][0] == J.FAILED and "exited with code 3" in states["Fails"][1]
    assert states["Slow"][0] == J.TIMED_OUT and "did not finish in time" in states["Slow"][1]
    assert job.runs[1].elapsed < 20  # stopped, with what it started, not left to run
    for label in ("Unset", "Shell", "Missing"):
        assert states[label][0] == J.NOT_RUN and states[label][1].startswith("Not run: ")
    html = client.get(f"/jobs/{job.id}").get_data(as_text=True)
    assert "stand-in: failing on request" in html  # the failed tool's own error output, on its card
    assert states["Says why"][0] == J.FAILED and "424242 42" in html  # its standard output, when no errors


def test_jobs_run_one_at_a_time_in_upload_order(make_app, pdf, tmp_path):
    slow = _standin(tmp_path, "slow", "--sleep 1.5")
    app, queue, client, token = make_app([(slow, "A"), (slow, "B")])
    first = _job(queue, _upload(client, token, pdf))
    second = _job(queue, _upload(client, token, pdf))
    end = time.time() + 30
    while first.state != J.RUNNING and time.time() < end:
        time.sleep(0.05)
    assert second.state == J.QUEUED and queue.ahead(second) == 1
    assert "1 job ahead" in client.get(f"/jobs/{second.id}").get_data(as_text=True)
    _wait(second, timeout=60)
    spans = sorted((r.started, r.finished) for j in (first, second) for r in j.runs)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # never two tools at once
    assert first.runs[-1].finished <= second.runs[0].started


def test_uploads_are_deleted_on_request_after_the_retention_time_and_on_restart(make_app, pdf, tmp_path):
    app, queue, client, token = make_app([(STANDIN, "Text")], keep_minutes=5)
    job = _job(queue, _upload(client, token, pdf))
    _wait(job)
    assert client.post(f"/jobs/{job.id}/delete", data={}).status_code == 400  # no form token
    assert job.dir.exists()
    assert client.post(f"/jobs/{job.id}/delete", data={"token": token}).status_code == 303
    assert not job.dir.exists() and client.get(f"/jobs/{job.id}").status_code == 404

    later = _job(queue, _upload(client, token, pdf))
    _wait(later)
    assert queue.expire(now=later.finished + 60) == [] and later.dir.exists()
    assert queue.expire(now=later.finished + 5 * 60) == [later.id] and not later.dir.exists()

    # Stopping the app stops a running tool, with everything it started, and deletes every job folder.
    slow = _standin(tmp_path, "slow", "--sleep 120")
    app2, queue2, client2, token2 = make_app([(slow, "Slow")])
    running = _job(queue2, _upload(client2, token2, pdf))
    end = time.time() + 30
    while not _processes_mentioning(running.dir) and time.time() < end:
        time.sleep(0.1)
    assert _processes_mentioning(running.dir) and running.dir.exists()
    queue2.shutdown()
    assert not running.dir.exists() and not _processes_mentioning(running.dir) and not queue2.jobs
    queue2.shutdown()  # a second call (atexit after a signal) does nothing

    # A new start deletes the job folders an earlier start left, and nothing else.
    work = queue.cfg.work_dir
    left = work / ("a" * 32)
    left.mkdir()
    (left / J.MARKER).write_text("x", encoding="utf-8")
    (work / ("b" * 32)).mkdir()  # not marked as the demo's
    (work / "notes.txt").write_text("keep", encoding="utf-8")
    assert J.purge_leftovers(work) == 1
    assert not left.exists() and (work / ("b" * 32)).exists() and (work / "notes.txt").exists()


def _processes_mentioning(path: Path) -> list[int]:
    """Processes whose command line names ``path`` (a tool started for a job names its folder). Linux only."""
    found = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if str(path).encode() in (d / "cmdline").read_bytes():
                    found.append(int(d.name))
            except OSError:
                continue
    return found


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads /proc (Linux)")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_stopping_the_app_stops_the_running_tool_and_deletes_the_uploads(tmp_path, pdf, sig):
    """The real app (python -m demo), stopped while a tool runs: before, SIGTERM ended it at once and Ctrl-C left the
    tool running in its own process group, still writing into a job folder that stayed on disk."""
    pytest.importorskip("waitress")
    slow = _standin(tmp_path, "slow", "--sleep 120")
    work, port = tmp_path / "Joel's demo" / "work", _free_port()
    config = tmp_path / "demo.yaml"
    config.write_text(json.dumps({"candidates": [{"manifest": str(slow), "label": "Slow"}], "work_dir": str(work),
                                  "results_dirs": [str(tmp_path / "results")]}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    app = subprocess.Popen([sys.executable, "-m", "demo", "--config", str(config), "--port", str(port)], cwd=ROOT,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        end = time.time() + 60
        while True:
            try:
                page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
                break
            except OSError:
                assert time.time() < end and app.poll() is None, "the app did not start"
                time.sleep(0.2)
        token = re.search(r'name="token" value="([^"]+)"', page).group(1)
        boundary = "rbdemo" + secrets.token_hex(8)
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="token"\r\n\r\n{token}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="pdf"; filename="x.pdf"\r\n'
                f"Content-Type: application/pdf\r\n\r\n").encode() + pdf + f"\r\n--{boundary}--\r\n".encode()
        urllib.request.urlopen(urllib.request.Request(
            base + "/upload", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}),
            timeout=30).read()
        end = time.time() + 60
        while not _processes_mentioning(work) and time.time() < end:
            time.sleep(0.2)
        assert _processes_mentioning(work) and [p for p in work.iterdir() if (p / J.MARKER).exists()]
        app.send_signal(sig)
        assert app.wait(timeout=60) == 0
    finally:
        if app.poll() is None:
            app.kill()
    assert not _processes_mentioning(work)
    assert not [p for p in work.iterdir() if (p / J.MARKER).exists()]


def test_bad_uploads_are_refused_with_a_reason(make_app, pdf):
    app, queue, client, token = make_app([(STANDIN, "Text")], max_upload_mb=0.5)
    for data, reason in ((b"", "empty"), (b"hello, not a pdf", "not a PDF"), (b"%PDF-1.7\n garbage", "PDF")):
        resp = _upload(client, token, data)
        assert resp.status_code == 400 and reason in resp.get_data(as_text=True)
    assert _upload(client, token, b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024)).status_code == 413
    assert client.post("/upload", data={"pdf": (io.BytesIO(pdf), "x.pdf")},
                       content_type="multipart/form-data").status_code == 400  # no form token
    assert not queue.jobs


def _tables():
    """A small scored run: two documents, one graded on its text with a word its tool did not extract."""
    base = {"candidate": "tool-x", "candidate_version": "1.0", "experiment": "e2e", "mode": "e2e",
            "dataset": "synthetic@0.0.1", "repeat": 0}
    docs = [dict(base, doc_id="d1", evidence="pdf", clean_document=True, searchable_rate=0.9,
                 **{"self_check.verdict": "correct_release"}),
            dict(base, doc_id="d2", evidence="text", clean_document=False, searchable_rate=None,
                 **{"self_check.verdict": "unsafe_release"})]
    statuses = [("d1", "SAFE_REPLACED"), ("d1", "NOT_REPLACED"), ("d1", "NOT_EVALUATED"), ("d2", "LEAK_VISIBLE"),
                ("d2", "SAFE_REPLACED"), ("d2", "NOT_EVALUATED"), ("d2", "SAFE_REMOVED")]
    mentions = [dict(base, doc_id=d, mention_id=f"m{i}", requirement="MUST", status=s, type="PERSON_NAME",
                     shared_words="", shared_chars=None, shared_pairs=None) for i, (d, s) in enumerate(statuses)]
    words = [dict(base, doc_id=d, word_id=f"w{i}", requirement=r, status=s, visible_fraction=v)
             for i, (d, r, s, v) in enumerate([("d1", "ORDINARY", "PRESERVED", 1.0), ("d1", "ORDINARY", "DAMAGED", 0.2),
                                               ("d1", "MUST", "REMOVED", 0.0), ("d2", "ORDINARY", "MOVED", 1.0),
                                               ("d2", "MUST_NOT", "PRESERVED", 1.0)])]
    return {"documents": docs, "mentions": mentions, "words": words}


def test_scored_runs_and_boards_show_the_benchmarks_own_numbers(make_app, tmp_path, monkeypatch):
    results = tmp_path / "results"
    run = results / "x-e2e"
    write_tables(run / "tables", _tables())
    (run / "scoring.json").write_text(json.dumps({"policy_id": "au-medicolegal@1.2.0",
                                                  "eval_config": {"version": "eval-0.4"}}), encoding="utf-8")
    # The board's own RunData (which also reads the run's run.json) gives the same scorecard row.
    monkeypatch.setenv("RB_RESULTS_DIR", str(results))
    monkeypatch.setattr(board_mod, "load_run", lambda run_id: SimpleNamespace(
        candidate={"name": "tool-x", "version": "1.0"}, dataset={"name": "synthetic", "version": "0.0.1"},
        experiment=SimpleNamespace(mode="e2e", name="e2e")))
    row = board_mod.scorecard([FolderRun(run)])[0]
    assert row == board_mod.scorecard([board_mod.RunData("x-e2e")])[0]
    assert row["n"] == 6 and row["evidence"] == "pdf+text" and row["clean"] == "1/2"  # d1's NOT_EVALUATED left out
    x = summary(row)  # leaked: d2's visible name and its word the tool's text lacks; d1's damaged word
    assert (x["leak"], x["not_replaced"], x["over"], x["clean"], x["graded_on"]) == (
        "33.3%", "16.7%", "33.3%", "1/2", "pdf + text")
    # The word d2's tool never extracted counts as leaked in the strict rate; the page says how many there were, and
    # the rate on the items the tool did handle.
    assert (row["not_seen"], row["n_completed"], x["not_seen"], x["leak_completed"], x["handled"]) == (1, 5, 1, "20.0%", 5)
    assert not x["plumbing"] and summary(dict(row, version="1.0 [plumbing check: fake model; not the tool as built]"))[
        "plumbing"]
    # A tool graded on its text alone: the headline is the rate on the items it handled (owner, 2026-10-31), with the
    # strict rate and the count of items its reader never extracted under it.
    t = summary(dict(row, evidence="text"))
    assert (t["leak"], t["leak_strict"], t["not_seen"], t["leak_completed"], t["leak_ci"]) == ("20.0%", "33.3%", 1, "", "")
    assert x["leak_strict"] == ""

    (results / "board-demo.html").write_text("<html><body><h1>Demo &amp; board</h1></body></html>", encoding="utf-8")
    (results / "board-demo.json").write_text(json.dumps({"runs": ["x-e2e"], "scorecard": [row]}), encoding="utf-8")
    (results / "board-old.html").write_text("<h1>Old board</h1>", encoding="utf-8")  # no scorecard file
    app, queue, client, token = make_app([])
    page = client.get("/benchmark").get_data(as_text=True)
    assert "Demo &amp; board" in page and "Old board" in page and "no scorecard file" in page
    assert page.count("33.3%") >= 4 and page.count("16.7%") >= 2 and "released, but not clean 1" in page and "What the numbers mean" in page
    assert page.count("1 of 6 never reached its text") == 2 and page.count("20.0% on the 5 items it handled") == 2
    assert "<th>Required items leaked</th>" in page and "never in the text the tool's own reader" in page
    # A text-graded row: not every item such a tool never saw is a header or footer (a sideways page, a scan its
    # OCR missed), so the page does not say they all are.
    (results / "board-text.html").write_text("<h1>Text board</h1>", encoding="utf-8")
    (results / "board-text.json").write_text(json.dumps({"runs": ["x-e2e"], "scorecard": [dict(row, evidence="text")]}),
                                             encoding="utf-8")
    text_page = " ".join(client.get("/benchmark").get_data(as_text=True).split())
    assert "1 of 6 never reached its output (text its reader never extracted, such as page headers and footers)" in text_page
    assert "headers and footers its reader dropped" not in text_page
    assert "Check Now" in page and "plumbing check: not the tool as built" not in page
    assert str(tmp_path) not in page  # no results-folder path on the page the client sees
    # Only the runs named for the demo are listed (a folder may also hold plumbing checks); boards stay whole.
    app2, *_ = make_app([], runs=["another-run"])
    page = app2.test_client().get("/benchmark").get_data(as_text=True)
    assert "Only the runs chosen for this demo: another-run." in page and "No scored runs found" in page
    assert page.count("1 of 6 never reached its text") == 1  # the board's row
    resp = client.get("/benchmark/board/0/board-demo.html")
    assert resp.status_code == 200 and "script-src" not in resp.headers["Content-Security-Policy"]
    for bad in ("/benchmark/board/0/..%2Fx-e2e%2Fscoring.json", "/benchmark/board/1/board-demo.html",
                "/benchmark/board/0/other.html"):
        assert client.get(bad).status_code == 404


def _repeated_run(results: Path) -> None:
    """``x-e2e`` (one repeat, ``_tables``) and ``x-det``, the same tool's repeated run: 3 identical repeats. d2's output
    differs once, and d1's first item leaks in the last repeat, so 1 of its 6 counted items changes verdict."""
    write_tables(results / "x-e2e" / "tables", _tables())
    reps = {"documents": [], "mentions": [], "words": []}
    for r in range(3):
        base = _tables()
        for d in base["documents"]:
            d.update(repeat=r, experiment="determinism",
                     output_sha256="a" if d["doc_id"] == "d1" else ("c" if r == 1 else "b"))
        for m in base["mentions"]:
            m.update(repeat=r, experiment="determinism")
            if r == 2 and m["mention_id"] == "m0":
                m["status"] = "LEAK_VISIBLE"
        for w in base["words"]:
            w.update(repeat=r, experiment="determinism")
        for k in reps:
            reps[k] += base[k]
    write_tables(results / "x-det" / "tables", reps)
    for run in ("x-e2e", "x-det"):
        (results / run / "scoring.json").write_text(json.dumps({"policy_id": "au-medicolegal@1.2.0"}), encoding="utf-8")


def test_repeated_runs_report_how_often_output_and_verdicts_agree(tmp_path, monkeypatch):
    # Read from the scored tables, so it works for a text-graded tool too (the determinism table compares PDFs only).
    results = tmp_path / "results"
    _repeated_run(results)
    e2e, det = (board_mod.scorecard([FolderRun(results / r)])[0] for r in ("x-e2e", "x-det"))
    assert e2e["stability"] is None
    assert det["stability"] == {"repeats": 3, "same_output": (1, 2), "unstable": (1, 6)}
    assert board_mod.repeat_stability(*(_tables()[k] for k in ("documents", "mentions"))) is None
    monkeypatch.setenv("RB_RESULTS_DIR", str(results))
    monkeypatch.setattr(board_mod, "load_run", lambda run_id: SimpleNamespace(
        candidate={"name": "tool-x", "version": "1.0"}, dataset={"name": "synthetic", "version": "0.0.1"},
        experiment=SimpleNamespace(mode="e2e", name="determinism")))
    assert board_mod.scorecard([board_mod.RunData("x-det")])[0]["stability"] == det["stability"]


def test_the_overall_score_weighs_quality_accuracy_and_determinism(make_app, tmp_path, monkeypatch):
    results = tmp_path / "results"
    _repeated_run(results)
    e2e, det = (board_mod.scorecard([FolderRun(results / r)])[0] for r in ("x-e2e", "x-det"))
    (o,) = R.overall([e2e, det])
    # 50% x (1 - 33.3% leaked) + 30% x (1 - 33.3% over-redaction) + 20% x (1 - 1/6 unstable) = 70.0
    assert o["score"] == 70.0 and o["parts"] == {"quality": "66.7%", "accuracy": "66.7%", "determinism": "83.3%"}
    assert o["runs"] == "x-e2e, x-det"
    assert o["notes"]["determinism"] == ("1 of 6 required items changed verdict across 3 identical runs; "
                                         "identical output every run on 1 of 2 documents")
    # A text-graded tool's quality is on the items it handled, as in its headline.
    (t,) = R.overall([dict(e2e, evidence="text"), det])
    assert t["parts"]["quality"] == "80.0%" and t["notes"]["quality"].endswith("(of the items it handled)")
    # Without a repeated run there is no overall score, and the page says why; plumbing checks are left out.
    (n,) = R.overall([e2e, dict(det, candidate="tool-y", version="1 [plumbing check: fake model]")])
    assert n["score"] is None and n["notes"]["determinism"] == "not measured: no repeated run"

    monkeypatch.setenv("RB_RESULTS_DIR", str(results))
    app, queue, client, token = make_app([])
    page = " ".join(client.get("/benchmark").get_data(as_text=True).split())
    assert "Overall score" in page and ">70.0<" in page.replace(" <", "<")
    assert "Overall = 50% &times; redaction quality + 30% &times; accuracy + 20% &times; determinism" in page
    assert "1 of 6 required items changed verdict across 3 identical runs" in page


def test_config_file_and_environment(tmp_path):
    f = tmp_path / "demo.yaml"
    f.write_text("candidates:\n  - manifest: candidates/mock-reference.yaml\n    label: Mock\n    note: >-\n"
                 "      Only its first\n      pass.\nresults_dirs: [/data/results]\nruns: [run-a]\nkeep_minutes: 5\n"
                 "port: 9000\n", encoding="utf-8")
    cfg = load_config(f, environ={})
    assert [(c.manifest, c.label, c.note) for c in cfg.candidates] == [
        (ROOT / "candidates" / "mock-reference.yaml", "Mock", "Only its first pass.")]
    assert cfg.runs == ["run-a"]
    assert load_config(f, environ={"RB_DEMO_RUNS": "run-b, run-c"}).runs == ["run-b", "run-c"]
    listed = load_config(f, environ={"RB_DEMO_CANDIDATES": "candidates/mock-reference.yaml"}).candidates
    assert [(c.label, c.note) for c in listed] == [("Mock", "Only its first pass.")]  # the file's label and note
    assert cfg.results_dirs == [Path("/data/results")] and cfg.keep_minutes == 5 and cfg.port == 9000
    assert cfg.host == "127.0.0.1"
    env = {"RB_DEMO_CANDIDATES": os.pathsep.join(["/a/x.yaml", "candidates/y.yaml"]), "RB_DEMO_PORT": "9100",
           "RB_DEMO_WORK_DIR": "/tmp/w", "RB_DEMO_RESULTS_DIRS": "/r1" + os.pathsep + "/r2"}
    cfg = load_config(f, environ=env)
    assert [c.label for c in cfg.candidates] == ["x", "y"] and cfg.port == 9100
    assert cfg.work_dir == Path("/tmp/w") and cfg.results_dirs == [Path("/r1"), Path("/r2")]
    default = load_config(environ={})
    assert [c.manifest.name for c in default.candidates] == ["mlredact-client.yaml", "privategpt-text.yaml"]
    assert "Check Now" in default.candidates[1].note and default.runs == []
    assert default.host == "127.0.0.1"
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml", environ={})
    assert sys.executable  # the stand-ins run on the benchmark's own interpreter ({python})
