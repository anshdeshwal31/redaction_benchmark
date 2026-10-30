"""Address split for GT 0.2: street line stays ADDRESS, suburb/state/postcode become LOCATION/locality."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.gt import load_gt
from benchmark.gt.address_split import (ROLE_NAMES, STREET_TYPE_NAMES, STREET_TYPES, canonical_street_type,
                                        mention_address_roles, read_end, split_addresses)
from benchmark.gt.validate import validate_gt
from benchmark.schemas import (EntityKind, GTDocument, GTEntity, GTMention, GTPage, GTWord, PiiType, Role, Segment,
                               WordSource)
from tests.address_cases import build_gt, build_split_gt, expected_outcome, table, unit_key, units, word_id

ROOT = Path(__file__).resolve().parents[1]


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


# -- the case table (docs/address_cases.json) ----------------------------------------------------

# Units the splitter cannot reach yet; each must be flagged for the GT review. Overseas addresses
# (no foreign country or postcode is read), 'IN YOUR CAPITAL CITY' and the street lines followed by a
# suburb alone (no state or postcode to anchor on), an OCR letter in the postcode, and a directional
# after the street type with no comma, which only the postcode's owner decides (no gazetteer here).
SPLIT_ALLOWED = {
    "overseas-one-line/A1", "overseas-block/A1", "overseas-in-sentence/A1", "overseas-uk-postcode/A1",
    "gpo-box-in-your-capital-city/A1",
    "no-state-no-postcode/A1", "no-state-no-postcode-block/A1", "sentence-suburb-tail/A1",
    "sentence-suburb-tail-kept-place/A1", "sentence-suburb-tail-postal-name/A1", "corner-suburb-only-letterhead/A1",
    "footer-repeat-without-state-or-postcode/A2",
    "ocr-letter-o-in-postcode/A1", "street-suffix-no-comma-postcode-decides/A1",
}


def test_splitter_agrees_with_the_case_table():
    by_case: dict[str, tuple[dict, list[dict]]] = {}
    for case, unit in units():
        by_case.setdefault(case["id"], (case, []))[1].append(unit)
    keys, agree, wrong, unflagged = set(), [], [], []
    for case, case_units in by_case.values():
        new, notes = split_addresses(build_gt(case))
        note = {n.mention_id: n for n in notes}
        street_of = {m.id: {w for seg in m.segments for w in seg.words} for m in new.mentions}
        for unit in case_units:
            key = unit_key(case, unit)
            keys.add(key)
            n = note[unit["id"]]
            outcome = expected_outcome(unit)
            same = n.outcome == outcome and (outcome != "split"
                                             or street_of[unit["id"]] == {word_id(p) for p in unit["street_line"]})
            if same:
                agree.append(key)
            elif key not in SPLIT_ALLOWED:
                wrong.append(f"{key}: {n.outcome} ({n.reason})")
            if key in SPLIT_ALLOWED and n.confident:
                unflagged.append(key)
    assert SPLIT_ALLOWED <= keys
    assert not wrong and not unflagged, (f"{len(agree)} of {len(keys)} units agree; wrong: {wrong}; "
                                         f"allowed but not flagged: {unflagged}")
    assert len(agree) == len(keys) - len(SPLIT_ALLOWED), f"{len(agree)} of {len(keys)} units agree"


@pytest.mark.parametrize("lines, street, loc", [
    ([["7", "King", "Street,", "St", "Marys", "NSW", "2760"]], "7 King Street,", "St Marys NSW 2760"),
    ([["12", "Smith", "Street", "St", "Ives", "NSW", "2075"]], "12 Smith Street", "St Ives NSW 2075"),
    ([["4", "Rose", "Street", "Lane", "Cove", "NSW", "2066"]], "4 Rose Street", "Lane Cove NSW 2066"),
    ([["CMB", "12", "Wagga", "Wagga", "NSW", "2650"]], "CMB 12", "Wagga Wagga NSW 2650"),
    ([["PO", "Box", "Q123", "Sydney", "South", "NSW", "1235"]], "PO Box Q123", "Sydney South NSW 1235"),
    ([["12", "Jetty", "Road,", "Glenelg,", "South", "Australia"]], "12 Jetty Road,", "Glenelg, South Australia"),
    ([["Glenelg,", "South", "Australia"]], None, "Glenelg, South Australia"),
    ([["12", "Hay", "St,", "Perth,", "Western", "Australia,", "Australia"]], "12 Hay St,",
     "Perth, Western Australia, Australia"),
    ([["12", "Smith", "Street", "Penrith", "NSW2750"]], "12 Smith Street", "Penrith NSW2750"),
    ([["PENRITH", "NSW2750"]], None, "PENRITH NSW2750"),
    ([["12", "Smith", "Street,", "Penrith", "N.S.W.", "2750"]], "12 Smith Street,", "Penrith N.S.W. 2750"),
    ([["TWEED", "HEADS", "SOUTH", "NSW", "2486"]], None, "TWEED HEADS SOUTH NSW 2486"),
    ([["Box", "Hill", "North", "VIC", "3129"]], None, "Box Hill North VIC 3129"),
    ([["12", "Ocean", "View", "St", "Penrith", "NSW", "2750"]], "12 Ocean View St", "Penrith NSW 2750"),
    ([["Cnr", "Smith", "&", "Jones", "Streets", "Penrith", "NSW", "2750"]], "Cnr Smith & Jones Streets",
     "Penrith NSW 2750"),
    ([["12", "Smith", "Street,", "Sydney", "NSW", "NSW", "2000"]], "12 Smith Street,", "Sydney NSW NSW 2000"),
])
def test_named_cases_split_as_the_case_table_says(lines, street, loc):
    new, notes = split_addresses(_doc(lines))
    assert _parts(new) == (street, loc)
    assert not [i for i in validate_gt(new) if i.severity == "error"]


def test_a_directional_after_the_street_type_is_flagged():
    # 2112 is Ryde's postcode and 2113 North Ryde's: only a gazetteer could place 'North'.
    new, notes = split_addresses(_doc([["12", "Smith", "Street", "North", "Ryde", "NSW", "2112"]]))
    assert _parts(new) == ("12 Smith Street", "North Ryde NSW 2112") and not notes[0].confident
    new, notes = split_addresses(_doc([["12", "Smith", "Street", "North,", "Ryde", "NSW", "2112"]]))
    assert _parts(new) == ("12 Smith Street North,", "Ryde NSW 2112") and notes[0].confident


def test_a_dx_line_stays_whole():
    gt = _doc([["DX", "1234", "Sydney", "NSW"]])
    new, notes = split_addresses(gt)
    assert new.mentions == gt.mentions
    assert notes[0].outcome == "unsplit" and notes[0].confident


@pytest.mark.parametrize("text, state, postcode, country", [
    ("Mount Victoria 2786", [], ["2786"], []),  # 2786 is a NSW postcode: no state
    ("4 Rose Lane, Mount Victoria", [], [], []),
    ("Port Victoria 5573", [], ["5573"], []),
    ("Ballarat Victoria 3350", ["Victoria"], ["3350"], []),
    ("Victoria Point QLD 4165", ["QLD"], ["4165"], []),
    ("Mount Victoria NSW 2786", ["NSW"], ["2786"], []),
    ("Tharwa ACT 2620", ["ACT"], ["2620"], []),  # a cross-border postcode: codes are not range-checked
    ("Penrith New South Wales 2750", ["New", "South", "Wales"], ["2750"], []),
    ("Penrith N.S.W. 2750", ["N.S.W."], ["2750"], []),
    ("Sydney NSW NSW 2000", ["NSW", "NSW"], ["2000"], []),
    ("Penrith NSW2750", ["NSW2750"], ["NSW2750"], []),
    ("Penrith N.S.W.2750", ["N.S.W.2750"], ["N.S.W.2750"], []),
    ("Glenelg, South Australia", ["South", "Australia"], [], []),
    ("Perth, Western Australia, Australia", ["Western", "Australia,"], [], ["Australia"]),
    ("SPRINGVALE VIC 3171 AUSTRALIA", ["VIC"], ["3171"], ["AUSTRALIA"]),
])
def test_end_reading(text, state, postcode, country):
    words = text.split()
    r = read_end(words)
    assert ([words[i] for i in r.state], [words[i] for i in r.postcode], [words[i] for i in r.country]) == \
        (state, postcode, country)


def _join_split_addresses(gt: GTDocument) -> GTDocument:
    """Each street line joined back with its '-loc' part into the one ADDRESS mention it came from."""
    by_id = {m.id: m for m in gt.mentions}
    out = []
    for m in gt.mentions:
        loc = by_id.get(f"{m.id}-loc")
        if m.type is PiiType.LOCATION and m.id.endswith("-loc") and m.id[:-4] in by_id:
            continue
        if m.type is PiiType.ADDRESS and loc is not None:
            segs = [*m.segments, *loc.segments]
            a, b = m.segments[-1], loc.segments[0]
            if a.page == b.page and a.char_start is None and b.char_start is None:
                segs = [*m.segments[:-1], Segment(page=a.page, words=[*a.words, *b.words]), *loc.segments[1:]]
            m = m.model_copy(update={"segments": segs, "text": f"{m.text} {loc.text}",
                                     "conditions": sorted(set(m.conditions) - {"address_street_line"})})
        out.append(m)
    return gt.model_copy(update={"mentions": out})


@pytest.mark.parametrize("path", sorted((ROOT / "gt" / "synthetic").glob("*.gt.json")), ids=lambda p: p.name)
def test_resplitting_the_synthetic_gt_gives_the_same_mentions(path):
    gt = load_gt(path)
    new, _ = split_addresses(_join_split_addresses(gt))
    assert new.mentions == gt.mentions


# -- roles of an address unit's words (address_roles) --------------------------------------------

# Words whose table role the reader may miss: OCR forms of a state or postcode ('N5W', '275O'), which
# GT words hold only where the page itself prints them, and the town and postcode of an overseas unit
# (no foreign gazetteer); an overseas unit's country must still read 'country'.
ROLE_ALLOWED = {"ocr-letter-o-in-postcode/A1": {"suburb", "state", "postcode"},
                "ocr-state-n5w/A1": {"suburb", "state", "postcode"}}
OVERSEAS_ALLOWED = {"suburb", "postcode"}
STRUCTURE_PARTS = {"state", "designator"}


def test_address_roles_match_the_case_table():
    assert set(ROLE_NAMES) == set(table()["roles"])
    by_case: dict[str, tuple[dict, list[dict]]] = {}
    for case, unit in units():
        by_case.setdefault(case["id"], (case, []))[1].append(unit)
    total, agree, wrong = 0, 0, []
    for case, case_units in by_case.values():
        roles = {r.wid: r for rs in mention_address_roles(build_split_gt(case)).values() for r in rs}
        for unit in case_units:
            key = unit_key(case, unit)
            total += 1
            allowed = ROLE_ALLOWED.get(key, OVERSEAS_ALLOWED if unit.get("overseas") else set())
            diffs = [(text, role, roles[word_id((li, wi))].role if word_id((li, wi)) in roles else None)
                     for li, wi, text, role in unit["words"]
                     if word_id((li, wi)) not in roles or roles[word_id((li, wi))].role != role]
            agree += not diffs
            wrong += [f"{key}: {t} {want} -> {got}" for t, want, got in diffs if want not in allowed]
            for ps in unit.get("partial_structure", []):
                text = case["lines"][ps["at"][0]]["text"].split()[ps["at"][1]]
                parts = roles[word_id(ps["at"])].parts
                keep = "".join(text[a:b] for a, b, r in parts if r in STRUCTURE_PARTS)
                change = "".join(text[a:b] for a, b, r in parts if r not in STRUCTURE_PARTS)
                if (keep, change) != (ps["keep"], ps["change"]):
                    wrong.append(f"{key}: {text} parts {parts}")
    assert not wrong, f"{agree} of {total} units agree word for word; wrong: {wrong}"
    assert agree >= total - len(ROLE_ALLOWED), f"{agree} of {total} units agree word for word"


@pytest.mark.parametrize("lines, word, role", [
    ([["12", "Victoria", "Road,", "Rydalmere", "NSW", "2116"]], "Victoria", "street_name"),
    ([["4", "Rose", "Lane,", "Mount", "Victoria"]], "Victoria", "suburb"),
    ([["Mount", "Victoria", "2786"]], "Victoria", "suburb"),
    ([["3", "Queensland", "Avenue,", "Victoria", "Point", "QLD", "4165"]], "Queensland", "street_name"),
    ([["3", "Queensland", "Avenue,", "Victoria", "Point", "QLD", "4165"]], "Victoria", "suburb"),
    ([["TWEED", "HEADS", "SOUTH", "NSW", "2486"]], "SOUTH", "suburb"),
    ([["8", "Station", "Street,", "Box", "Hill", "Vic", "3128"]], "Box", "suburb"),
    ([["Lot", "3", "Old", "Bells", "Line", "of", "Road,", "Kurrajong", "NSW", "2758"]], "of", "street_name"),
    ([["Level", "4,", "Australia", "Square,", "264", "George", "Street,", "Sydney", "NSW", "2000"]], "Australia",
     "building"),
    ([["Shop", "4", "Westfield", "Plaza,", "Penrith", "NSW", "2750"]], "Plaza,", "building"),
    ([["12", "Smith", "St,", "Penrith", "New", "South", "Wales", "2750"]], "Wales", "state"),
    ([["Unit", "2,", "9", "High", "St,", "Brisbane,", "QLD", "4000"]], "St,", "street_type"),
])
def test_address_roles_never_take_another_role_for_structure(lines, word, role):
    new, _ = split_addresses(_doc(lines))
    roles = {r.wid: r.role for rs in mention_address_roles(new).values() for r in rs}
    assert {roles[w.id] for w in new.pages[0].words if w.text == word} == {role}


@pytest.mark.parametrize("lines, roles", [
    ([["12", "Queen", "Street,", "Auckland", "1010,", "New", "Zealand."]],
     ["number", "street_name", "street_type", "suburb", "postcode", "country", "country"]),
    ([["10", "High", "Street,", "Oxford", "OX1", "4AA,", "United", "Kingdom"]],
     ["number", "street_name", "street_type", "suburb", "postcode", "postcode", "country", "country"]),
    ([["Cardiff", "CF10", "1AA,", "Wales"]], ["suburb", "postcode", "postcode", "country"]),
    ([["GPO", "Box", "9999"], ["IN", "YOUR", "CAPITAL", "CITY"]],
     ["box_kind", "box_kind", "box_number", *["capital_city_phrase"] * 4]),
    ([["DX", "1234", "Sydney", "NSW"]], ["dx", "dx_number", "dx_exchange", "dx_exchange"]),
    ([["123", "George", "Street", "Sydney"]], ["number", "street_name", "street_type", "suburb"]),
    ([["Locked", "Bag", "7"]], ["box_kind", "box_kind", "box_number"]),
])
def test_address_roles_of_an_unsplit_address(lines, roles):
    assert [r.role for r in mention_address_roles(_doc(lines))["a1"]] == roles


def test_sub_word_roles():
    new, _ = split_addresses(_doc([["12", "Smith", "Street", "Penrith", "NSW2750,"]]))
    glued = next(r for r in mention_address_roles(new)["a1-loc"] if r.wid == "w4")
    assert glued.role == "state+postcode" and glued.parts == ((0, 3, "state"), (3, 7, "postcode"))
    # A sub-word segment: offsets stay in the GT word's text.
    gt = _doc([["12", "Smith", "Street"], ["Penrith", "(NSW2750)"]], boxes=True, char_span=(4, 1, 8))
    glued = next(r for r in mention_address_roles(split_addresses(gt)[0])["a1-loc"] if r.wid == "w4")
    assert glued.parts == ((1, 4, "state"), (4, 8, "postcode"))
    new, _ = split_addresses(_doc([["B02/53", "Smith", "Street,", "Newtown", "NSW", "2042"]]))
    hotel = mention_address_roles(new)["a1"][0]
    assert hotel.role == "unit_number+number" and hotel.parts == ((0, 1, "designator"), (1, 6, "unit_number+number"))


def test_canonical_street_types():
    assert STREET_TYPES <= set(STREET_TYPE_NAMES)
    assert [canonical_street_type(t) for t in ("St.", "Streets", "Cres,", "Cr", "Smith")] == \
        ["street", "street", "crescent", "crescent", None]
