"""A front for the demo when the machine it runs on is switched off part of the time.

``python -m demo.gate [--port 8790] [--upstream 127.0.0.1:8792]`` passes every request on to the demo at
``--upstream`` (for example the local end of an SSH tunnel to the demo's server) and sends its answer back unchanged.
When the demo cannot be reached, it answers with a short page saying that the demo is not available right now,
instead of a proxy's error page. It never says why, and has no controls. Standard library only.
"""

from __future__ import annotations

import argparse
import http.client
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("demo.gate")

# Headers that belong to one connection and are not passed on (RFC 9110 section 7.6.1).
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
              "trailer", "transfer-encoding", "upgrade"}
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 300
CHUNK = 64 * 1024

UNAVAILABLE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Redaction demo</title>
<style>
:root { --bg: #f7f7f5; --panel: #fff; --ink: #1d1d1b; --muted: #66665f; --line: #deded8; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --panel: #1f1f1d; --ink: #ececea; --muted: #a3a39c;
  --line: #3a3a36; } }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI",
  sans-serif; }
header { background: var(--panel); border-bottom: 1px solid var(--line); padding: 10px 16px; font-weight: 600; }
main { max-width: 640px; margin: 0 auto; padding: 16px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
  margin: 12px 0; }
p { margin: 6px 0; }
.mut { color: var(--muted); font-size: 13px; }
</style></head>
<body><header>Redaction demo</header>
<main><div class="card"><p><b>The demo is not available right now.</b></p>
<p>Ask Ansh to switch it on.</p></div></main></body></html>
""".encode("utf-8")


class Gate(BaseHTTPRequestHandler):
    upstream: tuple[str, int] = ("127.0.0.1", 8792)
    protocol_version = "HTTP/1.1"
    server_version = "demo"
    sys_version = ""

    def _relay(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
        conn = http.client.HTTPConnection(*self.upstream, timeout=CONNECT_TIMEOUT_S)
        try:
            conn.connect()
            conn.sock.settimeout(READ_TIMEOUT_S)
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            log.info("demo not reachable (%s): %s %s", type(exc).__name__, self.command, self.path.split("?")[0])
            self._unavailable()
            return
        try:
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            data = resp.read()  # the demo's answers are pages, page images and files of up to tens of MB
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except (OSError, http.client.HTTPException) as exc:
            log.info("demo stopped answering (%s)", type(exc).__name__)
            self.close_connection = True
        finally:
            conn.close()

    def _unavailable(self) -> None:
        self.send_response(503)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(UNAVAILABLE)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Retry-After", "300")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(UNAVAILABLE)

    do_GET = do_POST = do_HEAD = _relay

    def log_message(self, fmt, *args):  # noqa: D102 - quiet: paths of job pages are not logged
        pass


def serve(port: int, upstream: tuple[str, int], host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("BoundGate", (Gate,), {"upstream": upstream})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m demo.gate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--upstream", default="127.0.0.1:8792", help="host:port of the demo")
    args = ap.parse_args(argv)
    host, _, port = args.upstream.rpartition(":")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = serve(args.port, (host or "127.0.0.1", int(port)), args.host)
    log.info("front on %s:%d for the demo at %s", args.host, args.port, args.upstream)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

