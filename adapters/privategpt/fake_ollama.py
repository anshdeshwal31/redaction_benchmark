"""A fake Ollama server for plumbing checks of the PrivateGPT runner, on a machine without Ollama.

It answers the calls the client's redaction makes, in the format his code expects:

- ``GET /api/tags``: the model list (the runner's set-up check);
- ``POST /api/chat``: for his detection prompt, a JSON array of ``{"type", "value"}`` items found in the
  prompt's text chunk; for his name-alias prompt, a JSON object (empty); anything else, ``[]``;
- ``POST /api/generate`` (his verifier, only without spaCy): ``[]``.

The "model" is a few fixed rules, not a language model: names after a title (Mr, Ms, Dr, ...), plus any
values listed in an entities file that occur in the chunk. Results made with it are plumbing checks only,
never the client's scores.

Failure modes for tests: ``ok``; ``http500`` (every chat call after the first ``--ok-calls`` answers 500);
``nojson`` (chat answers are not JSON). Each request is logged as one JSON line (path, model, options,
prompt kind and length; never the prompt text).

Usage: python fake_ollama.py [--port 0] [--port-file F] [--entities E.json] [--mode ok] [--log L.jsonl]
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_MODEL = "qwen2.5:14b"
TITLED_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.? [A-Z][a-z]+(?: [A-Z][a-z]+)?\b")
NER_MARK = "Named Entity Recognition assistant"
ALIAS_MARK = "name resolution assistant"


def ner_chunk(prompt: str) -> str:
    """The text chunk his detection prompt wraps between 'Text:\\n---\\n' and '\\n---\\n\\nJSON array:'."""
    start = prompt.find("Text:\n---\n")
    end = prompt.rfind("\n---\n\nJSON array:")
    if start == -1 or end == -1 or end < start:
        return ""
    return prompt[start + len("Text:\n---\n"):end]


def find_entities(chunk: str, entities: list[dict]) -> list[dict]:
    found, seen = [], set()
    for e in entities:
        if e.get("value") and e["value"] in chunk and (e["type"], e["value"]) not in seen:
            seen.add((e["type"], e["value"]))
            found.append({"type": e["type"], "value": e["value"]})
    for m in TITLED_NAME.finditer(chunk):
        if ("person_name", m.group(0)) not in seen:
            seen.add(("person_name", m.group(0)))
            found.append({"type": "person_name", "value": m.group(0)})
    return found


class FakeOllama:
    def __init__(self, model: str = DEFAULT_MODEL, entities: list[dict] | None = None, mode: str = "ok",
                 ok_calls: int = 1, log_path: str | Path | None = None):
        self.model = model
        self.entities = entities or []
        self.mode = mode
        self.ok_calls = ok_calls
        self.log_path = Path(log_path) if log_path else None
        self.requests: list[dict] = []
        self.chat_calls = 0
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        with self._lock:
            self.requests.append(entry)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

    def chat(self, body: dict) -> tuple[int, dict]:
        with self._lock:
            self.chat_calls += 1
            n = self.chat_calls
        messages = body.get("messages") or []
        prompt = messages[0].get("content", "") if messages else ""
        kind = "ner" if NER_MARK in prompt else "alias" if ALIAS_MARK in prompt else "other"
        self.record({"path": "/api/chat", "model": body.get("model"), "options": body.get("options"),
                     "stream": body.get("stream"), "kind": kind, "prompt_chars": len(prompt),
                     "messages": len(messages)})
        if self.mode == "http500" and n > self.ok_calls:
            return 500, {"error": "fake failure"}
        if self.mode == "nojson":
            content = "I could not find anything."
        elif kind == "ner":
            content = json.dumps(find_entities(ner_chunk(prompt), self.entities))
        elif kind == "alias":
            content = "{}"
        else:
            content = "[]"
        return 200, {"model": body.get("model"), "message": {"role": "assistant", "content": content},
                     "done": True}

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _send(self, status: int, data: dict) -> None:
                raw = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/api/tags":
                    fake.record({"path": "/api/tags"})
                    # The runner marks the version of a run against this server as a plumbing check.
                    self._send(200, {"models": [{"name": fake.model, "model": fake.model}], "rb_fake_ollama": True})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    self._send(400, {"error": "bad json"})
                    return
                if self.path == "/api/chat":
                    self._send(*fake.chat(body))
                elif self.path == "/api/generate":
                    fake.record({"path": "/api/generate", "model": body.get("model")})
                    self._send(200, {"model": body.get("model"), "response": "[]", "done": True})
                else:
                    self._send(404, {"error": "not found"})

        return Handler


def start(port: int = 0, **kwargs) -> tuple[FakeOllama, ThreadingHTTPServer, str]:
    """Start a fake server on a background thread; returns (fake, server, base URL). Stop with
    server.shutdown()."""
    fake = FakeOllama(**kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return fake, server, f"http://127.0.0.1:{server.server_address[1]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fake Ollama server for plumbing checks")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--port-file", help="write the chosen port here once listening")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--entities", help='JSON list of {"type", "value"} the fake "detects" when they occur')
    ap.add_argument("--mode", choices=["ok", "http500", "nojson"], default="ok")
    ap.add_argument("--ok-calls", type=int, default=1, help="http500 mode: chat calls answered before failing")
    ap.add_argument("--log", help="request log (JSON lines, no prompt text)")
    args = ap.parse_args(argv)
    entities = json.loads(Path(args.entities).read_text(encoding="utf-8")) if args.entities else []
    fake = FakeOllama(model=args.model, entities=entities, mode=args.mode, ok_calls=args.ok_calls,
                      log_path=args.log)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), fake.handler())
    if args.port_file:
        Path(args.port_file).write_text(str(server.server_address[1]), encoding="utf-8")
    print(f"fake ollama on http://127.0.0.1:{server.server_address[1]} mode={args.mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
