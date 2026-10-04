from benchmark.gt import validate_gt
from benchmark.gt.draft import draft_gt, review_sheets
from benchmark.synth import digital, raster


def test_draft_from_text_layer_prelabels_contact_details_and_titled_names(tmp_path):
    fx = digital()
    gt, notes = draft_gt(fx.pdf, "d")
    assert notes == [] and sum(len(p.words) for p in gt.pages) == sum(len(p.words) for p in fx.gt.pages)
    found = {(m.type.value, m.text) for m in gt.mentions}
    assert ("EMAIL", "jane.smithers@example.com") in found
    assert ("PHONE", "0400 123 456") in found
    assert ("PERSON_NAME", "Jane Elizabeth Smithers") in found
    assert all("prelabel" in m.conditions and m.entity is None for m in gt.mentions)
    assert [i for i in validate_gt(gt) if i.severity == "error"] == []
    assert len(review_sheets(gt, fx.pdf, tmp_path)) == len(gt.pages)


def test_draft_without_text_or_ocr_says_so(monkeypatch):
    monkeypatch.setenv("RB_TESSERACT", "")
    monkeypatch.setattr("benchmark.ocr.tesseract_path", lambda: None)
    gt, notes = draft_gt(raster().pdf, "r")
    assert all(not p.words for p in gt.pages) and len(notes) == len(gt.pages)
