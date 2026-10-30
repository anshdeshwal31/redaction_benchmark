"""Shared structure read from GT (eval-0.4; benchmark/gt/shared_roles.py): what a realistic fake may keep
and nothing else. Fictional strings and the address case table (docs/address_cases.json)."""

import pytest

from benchmark.gt.address_split import LOCALITY_SUBTYPE
from benchmark.gt.shared_roles import ROLE_NAMES, document_shared_units
from benchmark.schemas import GTDocument, GTMention, GTPage, GTWord, PiiType, Segment, SharedRoles, WordSource
from tests import address_cases, calib

V1 = calib.V1
V1_PHONE_OFF = calib.V1_PHONE_OFF
V0 = calib.V0


def _doc(kind: str, text: str, split_at: int | None = None) -> tuple[GTDocument, list[GTWord]]:
    """One line of words and one mention of them: an ADDRESS (unsplit), or with ``split_at`` a street line
    and its '-loc' locality; 'LOCATION' is a locality on its own; any other type a mention of its own."""
    ws, x = [], 72.0
    for k, t in enumerate(text.split(" ")):
        w = 5.5 * len(t) + 1.0
        ws.append(GTWord(id=f"w{k}", text=t, poly=[(x, 40), (x + w, 40), (x + w, 50), (x, 50)], order=k, line="l0",
                         block="b0", source=WordSource.TEXT_LAYER))
        x += w + 3.0
    ids = [w.id for w in ws]
    if kind == "ADDRESS" and split_at is not None:
        mentions = [GTMention(id="a", type=PiiType.ADDRESS, entity="E", text=text, conditions=["address_street_line"],
                              segments=[Segment(page=0, words=ids[:split_at])]),
                    GTMention(id="a-loc", type=PiiType.LOCATION, subtype=LOCALITY_SUBTYPE, entity="E", text=text,
                              conditions=["address_locality"], segments=[Segment(page=0, words=ids[split_at:])])]
    elif kind == "LOCATION":
        mentions = [GTMention(id="a", type=PiiType.LOCATION, subtype=LOCALITY_SUBTYPE, entity="E", text=text,
                              segments=[Segment(page=0, words=ids)])]
    else:
        mentions = [GTMention(id="a", type=PiiType(kind), entity="E", text=text, segments=[Segment(page=0, words=ids)])]
    gt = GTDocument(doc_id="d", source_sha256="0" * 64, gt_version="0.1", mentions=mentions,
                    pages=[GTPage(index=0, width_pt=595.0, height_pt=842.0, words=ws)])
    return gt, ws


def _shared(kind: str, text: str, policy=V1, split_at=None) -> tuple[list[str], list, dict]:
    """(shared parts as text, pairs as (state, postcode, prefix), identifying slices) of every mention."""
    gt, ws = _doc(kind, text, split_at)
    t = {w.id: w.text for w in ws}
    got, pairs, cut = [], [], {}
    for su in document_shared_units(gt, policy).values():
        got += [t[w] for w in su.full]
        for w in sorted({w for w, _ in su.chars}):
            ks = sorted(k for ww, k in su.chars if ww == w)
            got.append(t[w][ks[0]:ks[-1] + 1] if ks == list(range(ks[0], ks[-1] + 1)) else f"{t[w]}{ks}")
        pairs += [(t[s], t[p], "".join(t[p][k] for k in ks)) for s, p, ks in su.pairs]
        cut.update({t[w]: t[w][a:b] for w, (a, b) in su.needle_cut.items()})
    return sorted(got), pairs, cut


# The revision's 32 reader strings (scratch/revise/check_roles.py) with its expected output, read here
# with the address role reader: 'Plaza' is a building word, so 'Shop 4 Westfield Plaza' shares 'Shop'
# only; a postcode with no state word shares the prefix of the state its range gives as plain characters.
READER = [
    ("ADDRESS", "14 Example Street,", ["Street,"]),
    ("ADDRESS", "3 Court Street", ["Street"]),
    ("ADDRESS", "PO Box 77,", ["Box", "PO"]),
    ("ADDRESS", "12 Box Street", ["Street"]),
    ("ADDRESS", "Unit 5, 12 Smith St", ["St", "Unit"]),
    ("ADDRESS", "1 The Esplanade", []),
    ("ADDRESS", "22 The Avenue", []),
    ("ADDRESS", "22 Esplanade", []),
    ("ADDRESS", "Shop 4 Westfield Plaza", ["Shop"]),
    ("LOCATION", "Faketown QLD 4000", ["4", "QLD"]),
    ("LOCATION", "Darwin NT 0812", ["08", "NT"]),
    ("LOCATION", "Victoria Point QLD 4165", ["4", "QLD"]),
    ("LOCATION", "Mount Victoria 2786", ["2"]),
    ("LOCATION", "Mount Victoria NSW 2786", ["2", "NSW"]),
    ("LOCATION", "QLD", []),
    ("ORGANISATION", "Acme Widgets Pty Ltd", ["Ltd", "Pty"]),
    ("ORGANISATION", "Pty Ltd", []),
    ("EMAIL", "jane.smithers@example.com", ["@example.com"]),
    ("EMAIL", "@example.com", []),
    ("ID_CLAIM", "CLM-48213", ["CLM-"]),
    ("ID_CLAIM", "CLM 48213", ["CLM"]),
    ("ID_CLAIM", "SMITHERS-0012", []),
    ("ID_CLAIM", "48213", []),
    ("ID_CLAIM", "CLM-", []),
    ("ID_GOVERNMENT", "2123 45670 1", []),
    ("PHONE", "0400 123 456", []),  # with the phone role off
]
PHONE = [("PHONE", "0400 123 456", ["04"]), ("PHONE", "+61 412 345 678", ["+61"]), ("PHONE", "(02) 9876 5432", ["(02)"]),
         ("PHONE", "1300 333 767", ["1300"]), ("PHONE", "13 22 11", ["13"]), ("PHONE", "9876 5432", [])]


@pytest.mark.parametrize("kind, text, want", READER + PHONE, ids=[f"{k}:{t}" for k, t, _ in READER + PHONE])
def test_shared_roles_reader(kind, text, want):
    policy = V1_PHONE_OFF if (kind, text, want) in READER else V1
    got, _, _ = _shared(kind, text, policy)
    assert got == sorted(want)


def test_shared_roles_reader_details():
    # pairs and identifying slices
    assert _shared("LOCATION", "Darwin NT 0812")[1] == [("NT", "0812", "08")]
    assert _shared("LOCATION", "Faketown QLD 4000")[1] == [("QLD", "4000", "4")]
    assert _shared("LOCATION", "Mount Victoria 2786")[1] == []  # no state word: no pair
    assert _shared("EMAIL", "jane.smithers@example.com")[2] == {"jane.smithers@example.com": "jane.smithers"}
    assert _shared("ID_CLAIM", "CLM-48213")[2] == {"CLM-48213": "48213"}
    assert _shared("PHONE", "0400 123 456")[2] == {"0400": "00"}
    # spelled-out, glued, doubled states and the prefix rule
    assert _shared("LOCATION", "Faketown Queensland 4000")[:2] == (["4", "Queensland"], [("Queensland", "4000", "4")])
    assert _shared("LOCATION", "Penrith NSW2750")[:2] == (["NSW2"], [("NSW2750", "NSW2750", "2")])
    assert _shared("LOCATION", "Darlinghurst NSW 1300")[:2] == (["NSW"], [("NSW", "1300", "")])
    assert _shared("LOCATION", "Darwin NT 0900")[:2] == (["0", "NT"], [("NT", "0900", "0")])
    assert _shared("LOCATION", "Canberra ACT 0200")[:2] == (["ACT"], [("ACT", "0200", "")])
    assert _shared("LOCATION", "Sydney NSW NSW 2000")[0] == ["2", "NSW", "NSW"]
    # street lines
    assert _shared("ADDRESS", "Corner of Smith and King Streets")[0] == ["Corner", "and", "of"]
    assert _shared("ADDRESS", "Flat 2, 5 Smith St")[0] == ["Flat", "St"]
    assert _shared("ADDRESS", "12 Queen Street, Auckland 1010, New Zealand")[0] == ["New", "Street,", "Zealand"]
    assert _shared("ADDRESS", "14 Example Street, Faketown QLD 4000", split_at=3)[:2] == (
        ["4", "QLD", "Street,"], [("QLD", "4000", "4")])
    # the never-list: a shared-looking word in another role
    for text in ("12 Victoria Road", "Mount Victoria NSW 2786", "Victoria Point QLD 4165", "3 Queensland Avenue",
                 "Box Hill VIC 3128", "Tweed Heads South NSW 2486"):
        kind = "LOCATION" if text[0].isalpha() else "ADDRESS"
        got = _shared(kind, text)[0]
        assert not {"Victoria", "Queensland", "South", "Box"} & set(got), (text, got)


def test_policy_v0_shares_nothing():
    for kind, text, _ in READER + PHONE:
        assert _shared(kind, text, V0)[0] == []


def test_shared_roles_are_validated():
    with pytest.raises(ValueError, match="not roles of"):
        SharedRoles(types=[PiiType.EMAIL], roles=["legal_form"])
    with pytest.raises(ValueError, match="together"):
        SharedRoles(types=[PiiType.LOCATION], roles=["state"])
    assert set(ROLE_NAMES) == {"ADDRESS", "LOCATION", "ORGANISATION", "EMAIL", "ID", "PHONE"}


NEVER = {"number", "unit_number", "level_number", "unit_number+number", "box_number", "dx_number", "suburb",
         "street_name", "building", "box_extra", "dx_exchange", "street_suffix"}
OCR_UNITS = {"ocr-letter-o-in-postcode", "ocr-state-n5w"}  # the role reader's two allow-listed units (S06b)


def test_shared_roles_never_excuse_identifying_words():
    # Over every core unit of the case table: the shared words are the table's structure words or the
    # street type, never a number, a suburb, a street name, a building word, a word after a box number, a
    # DX exchange, nor a postcode digit after the state's prefix.
    checked = 0
    for case in address_cases.table()["cases"]:
        if case["tier"] != "core" or case["id"] in OCR_UNITS:
            continue
        gt = address_cases.build_split_gt(case)
        shared = document_shared_units(gt, V1)
        for u in case["expect"]["addresses"]:
            role = {address_cases.word_id(w[:2]): w[3] for w in u["words"]}
            text = {address_cases.word_id(w[:2]): w[2] for w in u["words"]}
            structure = {address_cases.word_id(p) for p in u["structure"]}
            for mid in (u["id"], f"{u['id']}-loc"):
                su = shared.get(mid)
                if su is None:
                    continue
                for wid in su.full:
                    assert wid in structure or role[wid] == "street_type", (case["id"], text[wid], role[wid])
                    assert role[wid] not in NEVER, (case["id"], text[wid], role[wid])
                for wid, k in su.chars:
                    assert role[wid] in ("postcode", "state+postcode", "unit_number+number", "designator"), (
                        case["id"], text[wid], role[wid])
                    if role[wid] == "postcode":
                        assert k < 2 and text[wid][k].isdigit(), (case["id"], text[wid], k)
                checked += 1
    assert checked > 150, checked
