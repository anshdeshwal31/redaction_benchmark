"""Dataset registry and automatic condition measurement."""

from .conditions import classify_encoding, measure_skew, page_conditions
from .registry import (
    data_dir,
    describe_document,
    document_pdf,
    ingest,
    load_manifest,
    manifest_path,
    store_path,
    validate_manifest,
)

__all__ = [
    "classify_encoding",
    "data_dir",
    "describe_document",
    "document_pdf",
    "ingest",
    "load_manifest",
    "manifest_path",
    "measure_skew",
    "page_conditions",
    "store_path",
    "validate_manifest",
]
