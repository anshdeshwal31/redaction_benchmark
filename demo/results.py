"""The Benchmark page's data: comparison boards (``rb board``) and scored runs (``rb score``) found in
the configured results folders.

Every number comes from the benchmark's own scorecard (``benchmark.report.board.scorecard``): a
board's rows are read from the JSON file ``rb board`` writes next to its HTML, and a scored run's
rows are computed from its tables the same way, so the page and the board never disagree.
"""

from __future__ import annotations

import html
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from benchmark.report import deviations
from benchmark.report.board import RunData, repeat_stability, scorecard, time_per_document
from benchmark.report.store import read_rows
from benchmark.runner.runner import runs_root

_TITLE = re.compile(r"<h1>(.*?)</h1>", re.S)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.html$")


class FolderRun(RunData):
    """A scored run read from its results folder. ``RunData`` also reads the run's ``run.json`` from the runs
    folder, for the names its tables already hold; the rows kept are the same. The time each document took is only
    in ``run.json``: it is read from the runs folder when the run is there, and left out otherwise."""

    def __init__(self, folder: Path):  # noqa: D107 - mirrors RunData.__init__ without run.json
        self.run_id = folder.name
        documents = read_rows(folder, "documents")
        first = documents[0] if documents else {}
        self.candidate = first.get("candidate") or "?"
        self.version = first.get("candidate_version") or "?"
        self.mode = first.get("mode") or "?"
        self.dataset = first.get("dataset") or "?"
        self.run = SimpleNamespace(experiment=SimpleNamespace(name=first.get("experiment") or self.mode))
        self.documents = [d for d in documents if d.get("repeat") == 0]
        self.evidence = "+".join(sorted({d.get("evidence") or "pdf" for d in self.documents})) or "pdf"
        text_docs = {d["doc_id"] for d in self.documents if d.get("evidence") == "text"}
        self.mentions = [m for m in read_rows(folder, "mentions") if m["requirement"] == "MUST" and m["repeat"] == 0
                         and (m["status"] != "NOT_EVALUATED" or m["doc_id"] in text_docs)]
        self.words = [w for w in read_rows(folder, "words") if w["repeat"] == 0 and w["status"] != "NOT_EVALUATED"]
        self.determinism = read_rows(folder, "determinism")
        self.stability = repeat_stability(documents, read_rows(folder, "mentions"))
        run_json = runs_root() / self.run_id / "run.json"
        self.time = (time_per_document(json.loads(run_json.read_text(encoding="utf-8")).get("items") or [])
                     if run_json.is_file() else None)
        self.matters = read_rows(folder, "matters")
        self.label = f"{self.candidate} · {self.run.experiment.name}"
        scoring = folder / "scoring.json"
        self.scoring = json.loads(scoring.read_text(encoding="utf-8")) if scoring.exists() else {}
        self.scoring.setdefault("run_id", self.run_id)
        self.policy = deviations.policy_label(self.scoring)
        self.eval_version = (self.scoring.get("eval_config") or {}).get("version")
        self.eval_digest = self.scoring.get("eval_digest")


@dataclass
class Board:
    folder: int  # index into the configured results folders
    html: str | None  # file name of the board page, when there is one
    title: str
    rows: list[dict] = field(default_factory=list)
    error: str = ""


@dataclass
class ScoredRun:
    folder: int
    run_id: str
    row: dict | None = None
    error: str = ""


def find_boards(dirs: list[Path]) -> list[Board]:
    """Boards in the top level of each folder: an HTML page with the scorecard JSON ``rb board``
    writes next to it (named board-*.html by default)."""
    boards = []
    for i, d in enumerate(dirs):
        if not d.is_dir():
            continue
        for page in sorted(d.glob("*.html")):
            data = page.with_suffix(".json")
            if not SAFE_NAME.match(page.name) or not (page.name.startswith("board") or data.exists()):
                continue
            title = page.stem
            m = _TITLE.search(page.read_text(encoding="utf-8", errors="replace")[:200_000])
            if m:
                title = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() or title
            board = Board(i, page.name, title)
            try:
                board.rows = json.loads(data.read_text(encoding="utf-8"))["scorecard"]
            except (OSError, ValueError, KeyError, TypeError):
                board.error = "no scorecard file next to this board (rb board writes one); open the board to see it"
            boards.append(board)
    return boards


_cache: dict[tuple, dict] = {}
_cache_lock = threading.Lock()


def _stamp(folder: Path) -> tuple:
    files = [folder / "scoring.json", *sorted((folder / "tables").glob("*.parquet"))]
    return tuple((f.name, f.stat().st_mtime_ns) for f in files if f.exists())


def find_scored_runs(dirs: list[Path], only: list[str] | None = None) -> list[ScoredRun]:
    """Scored runs: sub-folders holding scoring.json and the tables, with their scorecard row. ``only``: the run ids
    to list (the demo's ``runs`` setting), so a folder can also hold runs the page must not show, such as plumbing
    checks; empty or None lists every one."""
    runs = []
    for i, d in enumerate(dirs):
        if not d.is_dir():
            continue
        for folder in sorted(p for p in d.iterdir() if p.is_dir()):
            if only and folder.name not in only:
                continue
            if not (folder / "scoring.json").exists() or not (folder / "tables" / "documents.parquet").exists():
                continue
            run = ScoredRun(i, folder.name)
            key = (str(folder), _stamp(folder))
            with _cache_lock:
                row = _cache.get(key)
            if row is None:
                try:
                    row = scorecard([FolderRun(folder)])[0]
                except Exception as exc:  # noqa: BLE001 - one unreadable run must not hide the others
                    run.error = f"could not be read ({type(exc).__name__})"
                else:
                    with _cache_lock:
                        _cache[key] = row
            run.row = row
            runs.append(run)
    return runs


# -- presentation --------------------------------------------------------------------------------

MODE_NAMES = {"e2e": "end to end", "oracle_text": "given perfect text", "oracle_mentions": "given perfect detection",
              "oracle_entities": "given perfect detection and grouping", "oracle_regions": "given perfect regions"}
VERDICTS = (("correct_release", "released, clean"), ("correct_hold", "held back, not clean"),
            ("false_alarm", "held back, but clean"), ("unsafe_release", "released, but not clean"))


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def summary(row: dict) -> dict:
    """One scorecard row in the words of the Benchmark page."""
    ci = row.get("ci")
    evidence = row.get("evidence") or "pdf"
    verdicts = row.get("self_check") or {}
    version = row.get("version") or ""
    # Required items a text-graded tool's own reader never extracted (NOT_EVALUATED). The owner's choice
    # (2026-10-31): the headline is the leak rate on the items the tool handled, with the count of items that
    # never reached its output and the strict rate (which counts them as leaked) shown under it.
    not_seen = row.get("not_seen")
    completed = row.get("leak_completed")
    handled_headline = evidence == "text" and bool(not_seen) and completed is not None
    return {
        "candidate": row.get("candidate"), "version": row.get("version"), "run": row.get("run"),
        "mode": MODE_NAMES.get(row.get("mode"), row.get("mode")), "headline": row.get("mode") == "e2e",
        "dataset": row.get("dataset"), "policy": row.get("policy"), "eval": row.get("eval"),
        "graded_on": {"pdf": "PDF", "text": "text only"}.get(evidence, evidence.replace("+", " + ")),
        "docs": row.get("docs"), "required": row.get("n"),
        "leak": _pct(completed if handled_headline else row.get("leak")),
        "leak_ci": "" if handled_headline or not ci else f"{ci[0] * 100:.1f}–{ci[1] * 100:.1f}%",
        "leak_strict": _pct(row.get("leak")) if handled_headline else "",
        "not_replaced": _pct(row.get("not_replaced")), "failure": _pct(row.get("fail")),
        "over": _pct(row.get("collateral")), "clean": row.get("clean"),
        "searchable": "not checked" if evidence == "text" else _pct(row.get("searchable")),
        "self_check": ", ".join(f"{name} {verdicts[k]}" for k, name in VERDICTS if verdicts.get(k))
        or "not reported",
        "warn": bool(row.get("given_policy")),
        "not_seen": not_seen or 0,
        "leak_completed": _pct(completed) if not handled_headline and completed != row.get("leak") else "",
        "handled": row.get("n_completed"),
        "plumbing": "plumbing check" in version,
    }


# -- overall score ---------------------------------------------------------------------------------

# The owner's priorities (2026-11-01): redaction quality first (70%), then accuracy, then determinism. The weights
# are the demo's choice, not the benchmark's; every part is read from the benchmark's own scorecard.
WEIGHTS = (("quality", 0.7), ("accuracy", 0.2), ("determinism", 0.1))


def _headline_leak(row: dict) -> tuple[float | None, bool]:
    """The leak rate the page shows as the tool's headline (see ``summary``), and whether it is the rate on the items
    the tool handled (a text-graded tool whose reader never extracted some of them)."""
    if (row.get("evidence") or "pdf") == "text" and row.get("not_seen") and row.get("leak_completed") is not None:
        return row["leak_completed"], True
    return row.get("leak"), False


def overall(rows: list[dict], labels: dict[str, str] | None = None) -> list[dict]:
    """One overall score per tool, out of 100: 70% redaction quality, 20% accuracy, 10% determinism (``WEIGHTS``).

    - quality: 1 - the headline leak rate of the tool's end-to-end run;
    - accuracy: 1 - its over-redaction (ordinary words damaged or removed);
    - determinism: 1 - the share of required items whose verdict changed between the identical repeats of the
      tool's repeated run (``experiments/determinism.yaml``).

    The end-to-end run is the tool's run with a single repeat in end-to-end mode; the repeated run is its run with
    the most repeats. A part that is missing leaves the overall score out, and says why. ``labels``: names to show
    for the tools, by candidate name."""
    tools: dict[str, dict] = {}
    for row in rows:
        if row.get("mode") != "e2e" or "plumbing check" in (row.get("version") or ""):
            continue
        t = tools.setdefault(row.get("candidate") or "?", {"e2e": None, "repeated": None})
        stability = row.get("stability")
        if stability:
            if t["repeated"] is None or stability["repeats"] > t["repeated"]["stability"]["repeats"]:
                t["repeated"] = row
        elif t["e2e"] is None:
            t["e2e"] = row
    out = []
    for name, t in tools.items():
        e2e, rep = t["e2e"], t["repeated"]
        parts: dict[str, float | None] = {"quality": None, "accuracy": None, "determinism": None}
        notes: dict[str, str] = {}
        if e2e is not None:
            leak, handled_only = _headline_leak(e2e)
            parts["quality"] = None if leak is None else 1 - leak
            notes["quality"] = f"{_pct(leak)} of required items leaked" + (" (of the items it handled)" if handled_only else "")
            over = e2e.get("collateral")
            parts["accuracy"] = None if over is None else 1 - over
            notes["accuracy"] = f"{_pct(over)} of ordinary words damaged"
        else:
            notes["quality"] = notes["accuracy"] = "no end-to-end run"
        if rep is not None:
            s = rep["stability"]
            unstable, n = s["unstable"]
            parts["determinism"] = None if not n else 1 - unstable / n
            same, docs = s["same_output"]
            notes["determinism"] = (f"{unstable} of {n} required items changed verdict across {s['repeats']} identical "
                                    f"runs; identical output every run on {same} of {docs} documents")
        else:
            notes["determinism"] = "not measured: no repeated run"
        score = (None if any(v is None for v in parts.values())
                 else round(100 * sum(parts[k] * w for k, w in WEIGHTS), 1))
        out.append({
            "candidate": name, "label": (labels or {}).get(name, name), "score": score,
            "parts": {k: _pct(v) for k, v in parts.items()}, "notes": notes,
            "runs": ", ".join(r["run"] for r in (e2e, rep) if r is not None),
            "versions": ", ".join(sorted({r.get("version") or "?" for r in (e2e, rep) if r is not None})),
        })
    return sorted(out, key=lambda x: (x["score"] is None, -(x["score"] or 0)))
