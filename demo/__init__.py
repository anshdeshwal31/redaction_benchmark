"""Demo web app: upload a PDF, run every configured redaction tool on it side by side, and show the
benchmark's comparison boards (demo/README.md).

The app knows the tools only through their candidate manifests (docs/adapter_guide.md): it builds each
tool's command with the benchmark's own runner and reads the files the adapter contract defines.
"""
