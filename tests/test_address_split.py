"""Address split for GT 0.2: street line stays ADDRESS, suburb/state/postcode become LOCATION/locality."""

from __future__ import annotations

import pytest

from benchmark.gt.address_split import split_addresses
from benchmark.gt.validate import validate_gt
from benchmark.schemas import (EntityKind, GTDocument, GTEntity, GTMention, GTPage, GTWord, PiiType, Role, Segment,
                               WordSource)


def _doc(lines: list[list[str]], boxes: bool = False, char_span: tuple[int, int, int] | None = None) -> GTDocument:
    """One ADDRESS mention over ``lines`` of words; ``boxes`` puts every word in its own segment."""
    words, segs = [], []
    order = 0
    for li, line in enumerate(lines):
        ids = []
        for k, text in enumerate(line):
            x = 72 + 60 * k
            y = 100 + 20 * li
            wid = f"w{order}"
            words.append(GTWord(id=wid, text=text, poly=[(x, y), (x + 50, y), (x + 50, y + 10), (x, y + 10)],
                                order=order, line=f"l{li}", source=WordSource.TEXT_LAYER))
            ids.append(wid)
            order += 1
        if boxes:
            segs.extend(Segment(page=0, words=[w]) for w in ids)
        else:
            segs.append(Segment(page=0, words=ids))
    if char_span is not None:
        si, start, end = char_span
        segs[si] = Segment(page=0, words=segs[si].words, char_start=start, char_end=end)
    text = " ".join(
        " ".join((next(w for w in words if w.id == wid).text[s.char_start:s.char_end] if s.char_start is not None
                  else next(w for w in words if w.id == wid).text) for wid in s.words) for s in segs)
    m = GTMention(id="a1", type=PiiType.ADDRESS, subtype="residential", entity="E1", text=text, segments=segs)
    return GTDocument(doc_id="d", source_sha256="0" * 64, gt_version="0.1", pages=[
        GTPage(index=0, width_pt=595, height_pt=842, words=words)], mentions=[m],
        entities=[GTEntity(id="E1", kind=EntityKind.PERSON, role=Role.CLAIMANT)])


def _parts(gt: GTDocument) -> tuple[str | None, str | None]:
    street = next((m.text for m in gt.mentions if m.type is PiiType.ADDRESS), None)
    loc = next((m.text for m in gt.mentions if m.type is PiiType.LOCATION), None)
    return street, loc


@pytest.mark.parametrize("lines, street, loc", [
    ([["14", "Example", "Street,", "Faketown", "QLD", "4000"]], "14 Example Street,", "Faketown QLD 4000"),
    ([["3", "Rose", "Rd", "Kelvin", "Grove", "QLD", "4059"]], "3 Rose Rd", "Kelvin Grove QLD 4059"),
    ([["Unit", "2,", "9", "High", "St,", "Brisbane,", "QLD", "4000"]], "Unit 2, 9 High St,", "Brisbane, QLD 4000"),
    ([["PO", "Box", "123", "Toowong", "QLD", "4066"]], "PO Box 123", "Toowong QLD 4066"),
    ([["Level", "4", "100", "Queen"], ["Spring", "Hill", "QLD", "4000", "Australia"]], "Level 4 100 Queen",
     "Spring Hill QLD 4000 Australia"),
    ([["12", "Ocean", "Pde", "Coogee", "New", "South", "Wales", "2034"]], "12 Ocean Pde", "Coogee New South Wales 2034"),
    ([["1", "Victoria", "Street", "Ipswich", "QLD", "4305"]], "1 Victoria Street", "Ipswich QLD 4305"),
])
def test_split(lines, street, loc):
    new, notes = split_addresses(_doc(lines), version="0.2-test")
    assert _parts(new) == (street, loc)
    assert notes[0].outcome == "split" and notes[0].confident
    assert new.gt_version == "0.2-test"
    assert not [i for i in validate_gt(new) if i.severity == "error"]
    a, b = new.mentions
    assert a.id == "a1" and b.id == "a1-loc" and b.entity == a.entity == "E1"
    assert b.subtype == "locality" and "address_locality" in b.conditions and "address_street_line" in a.conditions


def test_form_boxes_split_at_box_boundary():
    # A form with one box per field: street, suburb, state, postcode. No street-type word.
    new, notes = split_addresses(_doc([["5", "Gum", "Tree"], ["Camp", "Hill"], ["QLD"], ["4152"]], boxes=False))
    assert _parts(new) == ("5 Gum Tree", "Camp Hill QLD 4152")
    assert notes[0].confident


def test_locality_only_becomes_location():
    new, notes = split_addresses(_doc([["Ipswich", "QLD", "4305"]]))
    assert [m.type for m in new.mentions] == [PiiType.LOCATION]
    assert new.mentions[0].id == "a1" and notes[0].outcome == "locality_only" and notes[0].confident


def test_no_state_or_postcode_is_left_alone_and_flagged():
    gt = _doc([["14", "Example", "Street"]])
    new, notes = split_addresses(gt)
    assert new.mentions == gt.mentions
    assert notes[0].outcome == "unsplit" and not notes[0].confident


def test_no_boundary_is_flagged_uncertain():
    new, notes = split_addresses(_doc([["14", "Example", "Faketown", "QLD", "4000"]]))
    assert _parts(new) == ("14 Example", "Faketown QLD 4000")
    assert not notes[0].confident


def test_sub_word_segment_is_never_cut():
    # The postcode box holds "4000." with the full stop outside the mention.
    new, _ = split_addresses(_doc([["9", "Hay", "St"], ["Perth", "WA", "4000."]], boxes=True, char_span=(5, 0, 4)))
    loc = next(m for m in new.mentions if m.type is PiiType.LOCATION)
    assert loc.text == "Perth WA 4000"
    assert loc.segments[-1].char_start == 0 and loc.segments[-1].char_end == 4
    assert not [i for i in validate_gt(new) if i.severity == "error"]


def test_nested_mentions_follow_their_part():
    gt = _doc([["14", "Example", "Street,", "Faketown", "QLD", "4000"]])
    child = GTMention(id="c1", type=PiiType.LOCATION, subtype="suburb", entity=None, text="Faketown",
                      segments=[Segment(page=0, words=["w3"])], parent="a1")
    new, _ = split_addresses(gt.model_copy(update={"mentions": [*gt.mentions, child]}))
    assert next(m for m in new.mentions if m.id == "c1").parent == "a1-loc"
    assert not [i for i in validate_gt(new) if i.severity == "error"]
