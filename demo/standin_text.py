"""Stand-in text tool for the demo's tests and smoke runs: NOT the client's tool.

It speaks the adapter contract of a text-only tool (docs/adapter_guide.md, section 8): it extracts
each page's text, replaces e-mail addresses, numbers of three or more digits and pairs of capitalised
words with made-up values, and writes ``redacted_text.json`` with its mappings. It is crude on
purpose; it only gives the demo's text column something to show without a language model.

    python demo/standin_text.py --input input.pdf --workdir out [--sleep S] [--fail]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pymupdf

FIRST = ["Alex", "Jordan", "Casey", "Riley", "Morgan", "Taylor", "Jamie", "Quinn"]
LAST = ["Hartley", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Abernathy", "Castell", "Fenwick"]
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
DIGITS = re.compile(r"\b\d{3,}\b")
NAME = re.compile(r"\b([A-Z][a-z]{2,}) ([A-Z][a-z]{2,})\b")


def redact(text: str, rng: random.Random, seen: dict) -> str:
    def fake(label: str, original: str, make) -> str:
        if original not in seen:
            seen[original] = (label, make())
        return seen[original][1]

    text = EMAIL.sub(lambda m: fake("EMAIL", m.group(0), lambda: f"user{rng.randint(100, 999)}@example.org"), text)
    text = DIGITS.sub(lambda m: fake("NUMBER", m.group(0),
                                     lambda: "".join(rng.choice("0123456789") for _ in m.group(0))), text)
    return NAME.sub(lambda m: fake("NAME", m.group(0), lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"), text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--sleep", type=float, default=0.0, help="wait this long first (to test progress and timeouts)")
    ap.add_argument("--fail", action="store_true", help="write nothing and exit with code 3")
    a = ap.parse_args(argv)
    time.sleep(a.sleep)
    if a.fail:
        print("stand-in: failing on request", file=sys.stderr)
        return 3
    with pymupdf.open(a.input) as doc:
        pages = [page.get_text() for page in doc]
    rng, seen = random.Random(0), {}
    redacted = [redact(p, rng, seen) for p in pages]
    out = Path(a.workdir)
    (out / "redacted_text.json").write_text(json.dumps({
        "schema": "rb.redacted_text/0.1", "original_text": "\f".join(pages), "redacted_text": "\f".join(redacted),
        "page_break": "\f",
        "mappings": [{"label": label, "original": o, "replacement": r} for o, (label, r) in seen.items()],
    }, ensure_ascii=False), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({
        "schema": "rb.runinfo/0.1", "candidate": "standin-text", "candidate_version": "0.1.0",
        "notes": "demo stand-in, not the client's tool"}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
