"""The demo's web pages (Flask, server-rendered; no JavaScript, a running job's page reloads itself)."""

from __future__ import annotations

import io
import secrets
import time

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for

from . import jobs as J
from . import results as R
from .config import DemoConfig, load_config

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    # Pages with documents in them are never kept in the browser's cache.
    "Cache-Control": "no-store",
}
PAGE_CSP = "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
# A board is a static page the benchmark wrote: inline styles and SVG, no scripts.
BOARD_CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; frame-ancestors 'none'"


def _ago(t: float | None) -> str:
    if t is None:
        return ""
    s = int(time.time() - t)
    return f"{s}s ago" if s < 60 else f"{s // 60} min ago" if s < 3600 else f"{s // 3600} h ago"


def create_app(cfg: DemoConfig | None = None) -> Flask:
    cfg = cfg or load_config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.max_upload_mb * 1024 * 1024) + 64 * 1024
    queue = J.JobQueue(cfg)
    app.extensions["demo_queue"] = queue
    app.extensions["demo_config"] = cfg
    # Every form carries this value, so another site open in the same browser cannot post to the app.
    form_token = secrets.token_urlsafe(24)
    app.jinja_env.globals.update(form_token=form_token, duration=J.duration, ago=_ago, S=J)

    def check_token():
        if not secrets.compare_digest(request.form.get("token", ""), form_token):
            abort(400, "This form has expired: reload the page and try again.")

    def job_or_404(job_id: str) -> J.Job:
        job = queue.get(job_id)
        if job is None:
            abort(404, "No such job: it finished more than the retention time ago, or was deleted.")
        return job

    def run_or_404(job: J.Job, index: int) -> J.ToolRun:
        if not 0 <= index < len(job.runs):
            abort(404)
        return job.runs[index]

    @app.after_request
    def headers(resp):
        for k, v in SECURITY_HEADERS.items():
            resp.headers[k] = v
        resp.headers.setdefault("Content-Security-Policy", PAGE_CSP)
        return resp

    @app.get("/")
    def index():
        with_jobs = sorted(queue.jobs.values(), key=lambda j: j.created, reverse=True)
        return render_template("index.html", specs=queue.specs(), jobs=with_jobs, cfg=cfg, error=None)

    @app.post("/upload")
    def upload():
        check_token()
        f = request.files.get("pdf")  # the file's own name is never read
        try:
            if f is None:
                raise J.UploadError("Choose a PDF file first.")
            job = queue.submit(f.read())
        except J.UploadError as exc:
            with_jobs = sorted(queue.jobs.values(), key=lambda j: j.created, reverse=True)
            return render_template("index.html", specs=queue.specs(), jobs=with_jobs, cfg=cfg, error=str(exc)), 400
        return redirect(url_for("job_page", job_id=job.id), code=303)

    @app.get("/jobs/<job_id>")
    def job_page(job_id):
        job = job_or_404(job_id)
        views = []
        for run in job.runs:
            view = {"run": run, "replacements": [], "pages": [], "output": ""}
            if run.state == J.DONE:
                view["replacements"] = J.replacements(job, run)
                if run.spec.kind == "text":
                    art = J.redacted_text(job, run)
                    view["pages"] = J.text_pages(art) if art else []
            elif run.state == J.FAILED:
                view["output"] = J.output_tail(job, run)
            views.append(view)
        return render_template("job.html", job=job, views=views, ahead=queue.ahead(job), cfg=cfg)

    @app.get("/jobs/<job_id>/status.json")
    def job_status(job_id):
        job = job_or_404(job_id)
        return jsonify({"id": job.id, "state": job.state, "ahead": queue.ahead(job), "pages": job.page_count,
                        "tools": [{"name": r.spec.name, "label": r.spec.label, "output": r.spec.kind,
                                   "state": r.state, "message": r.message, "elapsed_s": r.elapsed,
                                   "release": r.release} for r in job.runs]})

    @app.post("/jobs/<job_id>/delete")
    def job_delete(job_id):
        check_token()
        queue.delete(job_id)
        return redirect(url_for("index"), code=303)

    @app.get("/jobs/<job_id>/original/<int:page>.png")
    def original_page(job_id, page):
        job = job_or_404(job_id)
        return _png(job.dir / "original", page)

    @app.get("/jobs/<job_id>/tool/<int:index>/<int:page>.png")
    def tool_page(job_id, index, page):
        job = job_or_404(job_id)
        run_or_404(job, index)
        return _png(job.dir / f"tool-{index}" / "pages", page)

    @app.get("/jobs/<job_id>/tool/<int:index>/output.pdf")
    def tool_pdf(job_id, index):
        job = job_or_404(job_id)
        run = run_or_404(job, index)
        path = job.dir / f"tool-{index}" / "out" / "output.pdf"
        if run.state != J.DONE or not path.exists():
            abort(404)
        return send_file(path, mimetype="application/pdf", as_attachment=True,
                         download_name=f"redacted-{run.spec.name}-{job.id[:8]}.pdf")

    @app.get("/jobs/<job_id>/tool/<int:index>/redacted.<ext>")
    def tool_text(job_id, index, ext):
        job = job_or_404(job_id)
        run = run_or_404(job, index)
        art = J.redacted_text(job, run) if run.state == J.DONE else None
        if art is None or ext not in ("txt", "json"):
            abort(404)
        name = f"redacted-{run.spec.name}-{job.id[:8]}.{ext}"
        if ext == "json":  # built here: the tool's own file also holds the original text
            return send_file(io.BytesIO(J.redacted_json(art)), mimetype="application/json", as_attachment=True,
                             download_name=name)
        return send_file(io.BytesIO(J.plain_text(art).encode("utf-8")), mimetype="text/plain",
                         as_attachment=True, download_name=name)

    @app.get("/benchmark")
    def benchmark():
        boards = R.find_boards(cfg.results_dirs)
        scored = R.find_scored_runs(cfg.results_dirs, cfg.runs)
        # Folder names only: the page is shown to the client, and a full path names the machine's user and layout.
        # The tools' names as the job pages show them (the demo's labels), and a plain name for the reference.
        labels = {"mock-reference-replace": "Perfect reference (a stand-in)"}
        labels.update({s.name: s.label for s in queue.specs()})
        overall = R.overall([s.row for s in scored if s.row], labels)
        return render_template("benchmark.html", boards=boards, scored=scored, dirs=[d.name for d in cfg.results_dirs],
                               summary=R.summary, only_runs=cfg.runs, overall=overall)

    @app.get("/benchmark/board/<int:folder>/<name>")
    def board(folder, name):
        if not 0 <= folder < len(cfg.results_dirs) or not R.SAFE_NAME.match(name):
            abort(404)
        if name not in {b.html for b in R.find_boards(cfg.results_dirs) if b.folder == folder}:
            abort(404)
        resp = send_from_directory(cfg.results_dirs[folder], name, mimetype="text/html")
        resp.headers["Content-Security-Policy"] = BOARD_CSP
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return render_template("message.html", title="File too large",
                               text=f"The upload limit is {cfg.max_upload_mb:g} MB."), 413

    @app.errorhandler(400)
    @app.errorhandler(404)
    def not_found(err):
        return render_template("message.html", title=err.name, text=err.description), err.code

    return app


def _png(folder, page: int):
    path = folder / f"p{page:03d}.png"
    if page < 1 or not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")
