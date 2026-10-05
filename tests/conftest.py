import pytest


@pytest.fixture(autouse=True)
def _no_measurement_ocr(monkeypatch, tmp_path):
    """Tests never launch a real OCR engine (slow, machine-dependent): the measurement OCR
    reads as not installed unless a test patches ``benchmark.ocr`` itself."""
    monkeypatch.setenv("RB_OCR_ENGINE", "ppocrv6")
    monkeypatch.setenv("RB_PPOCR_PYTHON", str(tmp_path / "no-ocr-env" / "python.exe"))
