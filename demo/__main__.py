"""Start the demo: ``python -m demo [--host H] [--port P] [--config demo.yaml]`` from the benchmark root.

It listens on 127.0.0.1 unless told otherwise: reach it through an SSH tunnel (demo/README.md).
"""

from __future__ import annotations

import argparse
import atexit
import ipaddress
import logging
import signal
import sys

from .app import create_app
from .config import load_config


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _stop_cleanly_on_signals() -> None:
    """SIGTERM (and SIGHUP, when the terminal closes) end the app the way Ctrl-C does, so the code after the server
    returns runs: it stops a running tool and deletes the job folders. Without this, SIGTERM ends the process at once,
    the tool (in its own process group) keeps running, and the uploads stay on disk until the next start."""

    def stop(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demo", description=__doc__)
    ap.add_argument("--config", help="settings file (default: $RB_DEMO_CONFIG or demo/demo.yaml)")
    ap.add_argument("--host", help="address to listen on (default 127.0.0.1)")
    ap.add_argument("--port", type=int, help="port (default 8765)")
    ap.add_argument("--threads", type=int, default=8, help="request threads (the tools still run one at a time)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    cfg.host = a.host or cfg.host
    cfg.port = a.port or cfg.port
    log = logging.getLogger("demo")
    if not _loopback(cfg.host):
        log.warning("listening on %s: anyone who can reach this address can upload documents and see the "
                    "results; the demo has no login. Prefer 127.0.0.1 and an SSH tunnel.", cfg.host)
    app = create_app(cfg)
    for spec in app.extensions["demo_queue"].specs():
        log.info("tool %s (%s output): %s", spec.name, spec.kind, "; ".join(spec.problems) or "ready")
    log.info("results folders: %s", ", ".join(str(d) for d in cfg.results_dirs))
    log.info("open http://%s:%d/", cfg.host, cfg.port)
    queue = app.extensions["demo_queue"]
    atexit.register(queue.shutdown)  # a second call does nothing
    _stop_cleanly_on_signals()
    try:
        from waitress import serve
    except ImportError:
        log.warning("waitress is not installed (pip install -e '.[demo]'); using Flask's development server")
        serve = None
    try:
        if serve is None:
            app.run(host=cfg.host, port=cfg.port, threaded=True)
        else:
            serve(app, host=cfg.host, port=cfg.port, threads=a.threads, ident="", _quiet=True)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("stopping: a running tool is stopped and every job folder deleted")
        queue.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
