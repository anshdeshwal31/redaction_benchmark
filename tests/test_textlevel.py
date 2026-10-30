"""The text-level scoring path (eval-0.4 S19; benchmark/evaluate/textlevel.py): a tool that emits redacted text
is graded on that text, with the PDF path's statuses, cut-offs and summary keys.

GT: gt/synthetic/syn-digital.gt.json (fictional). Simulated text tools write the GT words line by line, pages
split by a form feed, and change words of their choice. Policy: the client's, 1.2.0, with its shared roles.
"""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pytest

from benchmark.evaluate import EvalConfig, MentionStatus, WordStatus, evaluate_output
from benchmark.evaluate.textlevel import is_placeholder, score_text_document
from benchmark.gt import load_policy
from benchmark.gt.io import load_gt
from benchmark.schemas import RedactedTextArtifact
from benchmark.score import score_text

ROOT = Path(__file__).resolve().parents[1]
V1 = load_policy(ROOT / "policies" / "au-medicolegal-v1.yaml")
CFG = EvalConfig()

# Realistic fakes for every required word of the fictional claimant's documents (word text -> fake). The
# claim number keeps its 'CLM-' prefix, the address its street type and same-state 'QLD' with the postcode's
# '4', the mobile its '04', the email its domain and the organisation its legal form (policy 1.2.0).
FAKES = {
    "SMITHERS,": "DUNBAR,", "SMITHERS": "DUNBAR", "Jane": "Ruth", "Elizabeth": "Caroline", "Smithers": "Dunbar",
    "Lizzie.": "Rosa.", "Peter": "Henry", "Strong": "Marsh", "Tom": "Ivan", "Mia": "Zoe", "Hart,": "Quinn,",
    "CLM-48213": "CLM-70596", "14": "27", "Example": "Banksia", "Faketown": "Marsden", "4000": "4132",
    "0400": "0412", "123": "987", "456": "781", "jane.smithers@example.com": "ruth.dunbar@example.com",
    "Acme": "Brindle", "Widgets": "Holdings", "2123": "3456", "45670": "78912", "1": "2",
}


@lru_cache(maxsize=None)
def gt():
    return load_gt(ROOT / "gt" / "synthetic" / "syn-digital.gt.json")


def required_words() -> dict[str, str]:
    """Word id -> mention id of every word of a required mention (titles are optional words, not in it)."""
    from benchmark.gt.policy import resolve

    res = resolve(gt(), V1)
    return {wid: m.id for m in gt().mentions if res.mentions[m.id].requirement.value == "MUST"
            for seg in m.segments for wid in seg.words}


def texts(change=None, drop=()) -> RedactedTextArtifact:
    """The GT words line by line (pages split by a form feed) as the original text, and the same with ``change``
    (word id -> new text; '' removes the word) as the redacted text. Words in ``drop`` are in neither."""
    change = change or {}
    pages_o, pages_r = [], []
    for page in gt().pages:
        lines = defaultdict(list)
        for w in sorted(page.words, key=lambda w: w.order):
            if w.id not in drop:
                lines[w.line].append(w)
        pages_o.append("\n".join(" ".join(w.text for w in ws) for ws in lines.values()))
        pages_r.append("\n".join(" ".join(t for t in (change.get(w.id, w.text) for w in ws) if t)
                                 for ws in lines.values()))
    return RedactedTextArtifact(original_text="\f".join(pages_o), redacted_text="\f".join(pages_r))


def faked(**extra) -> dict[str, str]:
    """Every required word replaced by its fake (shared words left as they are)."""
    words = gt().word_index()
    out = {wid: FAKES[words[wid][1].text] for wid in required_words() if words[wid][1].text in FAKES}
    out.update(extra)
    return out


def every_mention_as(text: str) -> dict[str, str]:
    """Each required mention's first word becomes ``text``, its other words are removed."""
    change = {}
    must_ids = set(required_words().values())
    for m in gt().mentions:
        wids = [wid for seg in m.segments for wid in seg.words]
        if m.id in must_ids and wids:
            change.update(dict.fromkeys(wids[1:], ""))
            change[wids[0]] = text
    return change


def evaluate(art, policy=V1):
    return score_text_document(gt(), policy, art, CFG)


def must(ev) -> dict[str, str]:
    return {m.mention_id: m.status.value for m in ev.mentions if m.requirement == "MUST"}


def test_a_perfect_replacement_is_clean():
    ev = evaluate(texts(faked()))
    assert ev.evidence == "text" and ev.render_configs == ["text"]
    assert set(must(ev).values()) == {"SAFE_REPLACED"}, must(ev)
    # The signature has no text: the text path leaves it out.
    assert "sig1" not in {m.mention_id for m in ev.mentions}
    s = ev.summary
    assert s["collateral_words"] == 0 and s["protected_damaged"] == [] and s["searchable_rate"] is None
    assert s["clean_document"] and s["moved_words"] == 0
    assert {w.status for w in ev.words} <= {WordStatus.PRESERVED, WordStatus.DAMAGED, WordStatus.REMOVED}
    # The shared structure is reported like the PDF path's (the pair held: 'QLD' with the postcode's '4').
    loc = ev.mention("addr1-loc")
    assert loc.shared_pairs and all(ok for _, ok in loc.shared_pairs) and loc.locality_shape == "suburb+state+postcode"


def test_keeping_the_locality_is_a_visible_leak():
    words = gt().word_index()
    loc = {wid for wid, mid in required_words().items() if mid == "addr1-loc"}
    ev = evaluate(texts({wid: t for wid, t in faked().items() if wid not in loc}))
    assert ev.mention("addr1-loc").status is MentionStatus.LEAK_VISIBLE
    assert ev.mention("addr1").status is MentionStatus.SAFE_REPLACED
    assert {words[w][1].text for w in loc} == {"Faketown", "QLD", "4000"}


def test_placeholders_are_not_replacements():
    ev = evaluate(texts(every_mention_as("[REDACTED]")))
    assert set(must(ev).values()) == {"NOT_REPLACED"}, must(ev)
    assert not ev.summary["clean_document"]


def test_placeholder_forms():
    for t in ("[REDACTED]", "<PERSON>", "{NAME_1}", "XXXX", "****", "█████", "PERSON_2", "REDACTED", "ADDRESS"):
        assert is_placeholder(t), t
    for t in ("Ruth", "DUNBAR", "(02)", "CLM-70596", "0412", "Marsden"):
        assert not is_placeholder(t), t


def test_one_text_for_every_mention_is_a_placeholder():
    # 'Removed' is no tag of its own, but the same text standing for every different original is one.
    assert set(must(evaluate(texts(every_mention_as("Removed")))).values()) == {"NOT_REPLACED"}


def test_an_original_surname_moved_to_another_sentence_is_a_visible_leak():
    words = gt().word_index()
    daily = next(w.id for w in gt().page(0).words if w.text == "daily")
    ev = evaluate(texts(faked(**{daily: "daily Smithers"})))
    m2 = ev.mention("m2")  # 'Ms Smithers takes ...': the surname alone
    assert m2.status is MentionStatus.LEAK_VISIBLE and m2.displaced_residual == 1.0
    # Every mention holding the surname shows it; the others stay safe.
    holding = {m.id for m in gt().mentions for seg in m.segments for wid in seg.words
               if words[wid][1].text.strip(",").casefold() == "smithers"}
    assert {mid for mid, st in must(ev).items() if st != "SAFE_REPLACED"} == holding


def test_a_whole_address_fake_is_exact_for_both_parts():
    sc = score_text(gt(), V1, texts(faked()), config=CFG)
    assert sc.evaluation.mention("addr1").status is MentionStatus.SAFE_REPLACED
    assert sc.evaluation.mention("addr1-loc").status is MentionStatus.SAFE_REPLACED
    match = {x.mention_id: x for x in sc.detection.matches}
    assert match["addr1"].boundary == match["addr1-loc"].boundary == "exact"
    assert match["addr1"].best_pred == match["addr1-loc"].best_pred
    # The fakes are read per word and per address unit for the pseudonym checks.
    p = sc.pseudonyms
    assert p.evaluable and p.checks["survived"] == 0 and p.checks["postcode_state_valid_rate"] == 1.0
    assert p.checks["value_consistency_rate"] == 1.0
    other = {o.mention_id: o.replacement for o in p.other}
    assert other["addr1-loc"] == "Marsden QLD 4132" and other["ph1"] == "0412 987 781"


def test_words_missing_from_the_original_text_are_not_evaluated():
    tom = next(w.id for w in gt().page(0).words if w.text == "Tom")
    ev = evaluate(texts(faked(), drop={tom}))
    assert ev.mention("m4").status is MentionStatus.NOT_EVALUATED
    assert next(w for w in ev.words if w.word_id == tom).status is WordStatus.NOT_EVALUATED
    s = ev.summary
    assert s["must_evaluated"] == s["must_mentions"] - 1 and not s["clean_document"]


def test_summary_keys_equal_the_pdf_paths():
    from benchmark.synth import digital

    fx = digital()
    pdf = evaluate_output(fx.pdf, fx.pdf, fx.gt, V1, EvalConfig(dpi=150, renderers=("mupdf",)))
    assert set(evaluate(texts(faked())).summary) == set(pdf.summary)
    assert pdf.evidence == "pdf"


def test_a_page_break_count_that_does_not_match_is_aligned_as_one_text():
    art = texts(faked())
    one = RedactedTextArtifact(original_text=art.original_text.replace("\f", "\n"),
                               redacted_text=art.redacted_text.replace("\f", "\n"))
    ev = evaluate(one)
    assert set(must(ev).values()) == {"SAFE_REPLACED"} and ev.integrity.ok
    flat = RedactedTextArtifact(original_text=art.original_text, redacted_text=art.redacted_text.replace("\f", "\n"))
    ev = evaluate(flat)
    assert set(must(ev).values()) == {"SAFE_REPLACED"} and ev.integrity.validator_warnings


@pytest.mark.parametrize("bad", [dict(original_text="a"), dict(redacted_text="a"),
                                 dict(original_text="a", redacted_text="b", schema="rb.redacted_text/9")])
def test_the_artifact_is_validated(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RedactedTextArtifact.model_validate(bad)
