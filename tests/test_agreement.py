import pytest

from benchmark.gt.agreement import agreement
from benchmark.schemas import GTDocument
from benchmark.synth import digital


@pytest.fixture(scope="module")
def gt():
    return digital().gt


def second_annotator(gt, edits):
    """B's copy: renamed word ids, slightly different boxes, then the given edits."""
    d = gt.model_dump(mode="json", by_alias=True)
    rename = {}
    for p in d["pages"]:
        for w in p["words"]:
            rename[w["id"]] = "b_" + w["id"]
            w["id"] = rename[w["id"]]
            w["poly"] = [[x + 0.4, y - 0.3] for x, y in w["poly"]]
            w["char_polys"] = None
    for m in d["mentions"]:
        for s in m["segments"]:
            s["words"] = [rename[x] for x in s["words"]]
        m["optional_words"] = [rename[x] for x in m["optional_words"]]
        m["components"] = {k: [rename[x] for x in v] for k, v in m["components"].items()}
    for c in d["classified"]:
        for s in c["segments"]:
            s["words"] = [rename[x] for x in s["words"]]
    edits(d, rename)
    return GTDocument.model_validate(d)


def test_identical_annotation_agrees_perfectly(gt):
    r = agreement(gt, second_annotator(gt, lambda d, r: None))
    assert r.words_matched == r.words_a and r.word_text_agreement == 1.0
    assert r.strict_f1 == r.tolerant_f1 == 1.0 and r.token_kappa == 1.0 and r.entity_link_b3["f1"] == 1.0
    assert r.passed


def test_title_inclusion_is_tolerated_but_not_strict(gt):
    def edit(d, rename):
        m1 = next(m for m in d["mentions"] if m["id"] == "m1")
        title = m1["optional_words"].pop()
        m1["segments"][0]["words"].insert(0, title)
        m1["text"] = "Ms " + m1["text"]

    r = agreement(gt, second_annotator(gt, edit))
    assert r.tolerant_f1 == 1.0 and r.strict_f1 < 1.0


def test_missed_mention_type_change_and_entity_split(gt):
    def edit(d, rename):
        d["mentions"] = [m for m in d["mentions"] if m["id"] != "m4"]  # B misses the husband
        next(m for m in d["mentions"] if m["id"] == "org1")["type"] = "LOCATION"
        next(m for m in d["mentions"] if m["id"] == "m5")["entity"] = "E3"  # nickname linked to the wrong person

    r = agreement(gt, second_annotator(gt, edit))
    assert r.disagreements["only_in_a"] == ["m4"] and r.disagreements["only_in_b"] == []
    assert any(x.startswith("org1:") for x in r.disagreements["type_differs"])
    assert r.tolerant_f1 < 1.0 and r.entity_link_b3["f1"] < 1.0 and r.token_kappa < 1.0


def test_different_sources_are_rejected(gt):
    other = gt.model_copy(update={"source_sha256": "0" * 64})
    with pytest.raises(ValueError):
        agreement(gt, other)
