import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-calibration", action="store_true",
        help="skip the slow eval-0.4 calibration tests (marker 'calibration'); also RB_SKIP_CALIBRATION=1. "
             "Run them before every commit to benchmark/evaluate, the policies, the config or the mocks, "
             "and before any scoring that is reported.")


def pytest_collection_modifyitems(config, items):
    """The calibration tests run by default. A quick loop can leave them out with --skip-calibration
    or RB_SKIP_CALIBRATION=1; the skip is reported, so a quick run never reads as a full one."""
    if not (config.getoption("--skip-calibration") or os.environ.get("RB_SKIP_CALIBRATION") == "1"):
        return
    skip = pytest.mark.skip(reason="calibration test skipped (--skip-calibration or RB_SKIP_CALIBRATION=1)")
    for item in items:
        if "calibration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_measurement_ocr(monkeypatch, tmp_path):
    """Tests never launch a real OCR engine (slow, machine-dependent): the measurement OCR
    reads as not installed unless a test patches ``benchmark.ocr`` itself."""
    monkeypatch.setenv("RB_OCR_ENGINE", "ppocrv6")
    monkeypatch.setenv("RB_PPOCR_PYTHON", str(tmp_path / "no-ocr-env" / "python.exe"))
