"""PP-OCRv6 worker for ``benchmark.ocr`` (PLAN.md §5.9).

Runs inside the separate OCR environment (``.venv-ocr``, pinned by
``requirements-ocr.txt``): PaddleOCR pins numpy, PyYAML and OpenCV to versions that
conflict with the benchmark's own pins, so it cannot share the benchmark environment.
This file imports nothing from the benchmark package and only reports raw engine output;
all geometry (frames, word polygons, angles) is done by ``benchmark.ocr``.

Protocol: one JSON request per stdin line, one JSON reply per stdout line.

    {"op": "version"}                -> {"ok": true, "version": {...}}
    {"op": "ocr", "image": "<path>"} -> {"ok": true, "angle": 0|90|180|270, "lines": [...]}

Library logging is redirected to stderr so that stdout carries only replies.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

# Pinned models (names; the weights' hashes are reported by the "version" op).
MODELS = {
    "text_detection_model_name": "PP-OCRv6_medium_det",
    "text_recognition_model_name": "PP-OCRv6_medium_rec",
    "doc_orientation_classify_model_name": "PP-LCNet_x1_0_doc_ori",
}
PACKAGES = ("paddleocr", "paddlex", "onnxruntime", "numpy", "opencv-contrib-python")


def _model_hash(name: str) -> str | None:
    root = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models" / f"{name}_onnx"
    files = sorted(p for p in root.glob("*") if p.is_file()) if root.is_dir() else []
    if not files:
        return None
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _version() -> dict:
    return {"engine": "PP-OCRv6", "runtime": "onnxruntime-cpu",
            "packages": {p: pkg_version(p) for p in PACKAGES},
            "models": {name: _model_hash(name) for name in MODELS.values()}}


def _ocr(engine, image: str) -> dict:
    res = engine.predict(image)[0].json["res"]
    angle = int((res.get("doc_preprocessor_res") or {}).get("angle") or 0)
    lines = [{"text": t, "score": float(s), "poly": [[float(x), float(y)] for x, y in poly],
              "tokens": list(toks), "token_boxes": [[float(v) for v in b] for b in boxes]}
             for t, s, poly, toks, boxes in zip(res.get("rec_texts", []), res.get("rec_scores", []),
                                                res.get("rec_polys", []), res.get("text_word", []),
                                                res.get("text_word_boxes", []))]
    return {"angle": angle if angle in (90, 180, 270) else 0, "lines": lines}


def main() -> None:
    reply = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # library prints (Python or native) go to stderr
    sys.stdout = sys.stderr
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)
    from paddleocr import PaddleOCR

    engine = PaddleOCR(**MODELS, engine="onnxruntime", device="cpu", return_word_box=True,
                       use_doc_orientation_classify=True, use_doc_unwarping=False,
                       use_textline_orientation=False)  # flips some upright lines 180° on scans
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
            if req.get("op") == "quit":
                break
            if req.get("op") == "version":
                out = {"ok": True, "version": _version()}
            elif req.get("op") == "ocr":
                out = {"ok": True, **_ocr(engine, req["image"])}
            else:
                out = {"ok": False, "error": f"unknown op {req.get('op')!r}"}
        except Exception as e:  # report to the caller; the worker stays alive
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        reply.write(json.dumps(out) + "\n")
        reply.flush()


if __name__ == "__main__":
    main()
