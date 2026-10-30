"""The address case table (docs/address_cases.json) for tests: loader and GT builders.

Every case is fictional. ``build_gt`` follows the table's own splitter check (check_splitter.build_gt):
one ADDRESS mention per expected unit over the unit's words, one segment per run of adjacent words on
a line, word ids ``w{line}_{word}`` and one line and block id per case line, so the splitter sees the
case's line breaks. ``build_split_gt`` builds the GT as the table splits it.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from benchmark.gt.address_split import ADDRESS_LOCALITY, LOCALITY_SUBTYPE
from benchmark.schemas import GTDocument, GTMention, GTPage, GTWord, PiiType, Segment, WordSource

TABLE = Path(__file__).resolve().parents[1] / "docs" / "address_cases.json"


@cache
def table() -> dict:
    return json.loads(TABLE.read_text("utf-8"))


def units():
    """(case, unit) for every expected address unit, in table order."""
    for case in table()["cases"]:
        for unit in case["expect"]["addresses"]:
            yield case, unit


def unit_key(case: dict, unit: dict) -> str:
    return f"{case['id']}/{unit['id']}"


def expected_outcome(unit: dict) -> str:
    if unit["street_line"] and unit["locality"]:
        return "split"
    return "locality_only" if unit["locality"] else "unsplit"


def word_id(pos) -> str:
    line, word = pos
    return f"w{line}_{word}"


def _pages(case: dict) -> list[GTPage]:
    pages: dict[int, list[GTWord]] = {}
    order: dict[int, int] = {}
    for li, line in enumerate(case["lines"]):
        p = line["page"]
        x = float(line["x0"])
        y = 40.0 + 14.0 * line["row"]
        for wi, text in enumerate(line["text"].split()):
            w = 5.5 * len(text) + 1.0
            o = order.get(p, 0)
            order[p] = o + 1
            pages.setdefault(p, []).append(GTWord(
                id=f"w{li}_{wi}", text=text, poly=[(x, y), (x + w, y), (x + w, y + 10.0), (x, y + 10.0)], order=o,
                line=f"l{li}", block=f"b{li}", source=WordSource.TEXT_LAYER))
            x += w + 3.0
    return [GTPage(index=p, width_pt=595.0, height_pt=842.0, words=ws) for p, ws in sorted(pages.items())]


def _segments(case: dict, positions) -> list[Segment]:
    """One segment per run of adjacent words on one line."""
    runs: list[list] = []
    for li, wi in positions:
        if runs and runs[-1][0] == li and runs[-1][2] == wi - 1:
            runs[-1][1].append(f"w{li}_{wi}")
            runs[-1][2] = wi
        else:
            runs.append([li, [f"w{li}_{wi}"], wi, case["lines"][li]["page"]])
    return [Segment(page=r[3], words=r[1]) for r in runs]


def _text(case: dict, positions) -> str:
    return " ".join(case["lines"][li]["text"].split()[wi] for li, wi in positions)


def _doc(case: dict, mentions: list[GTMention]) -> GTDocument:
    return GTDocument(doc_id=case["id"], source_sha256="0" * 64, gt_version="0.1", pages=_pages(case),
                      mentions=mentions)


def build_gt(case: dict) -> GTDocument:
    """One unsplit ADDRESS mention per expected unit (entity ``E-<unit id>``), for the splitter."""
    mentions = []
    for u in case["expect"]["addresses"]:
        pos = u["street_line"] + u["locality"]
        mentions.append(GTMention(id=u["id"], type=PiiType.ADDRESS, entity=f"E-{u['id']}", text=_text(case, pos),
                                  segments=_segments(case, pos)))
    return _doc(case, mentions)


def build_split_gt(case: dict) -> GTDocument:
    """The GT as the table splits each unit: a street line (ADDRESS, ``address_street_line``) and its
    ``<id>-loc`` locality (LOCATION/locality, ``address_locality``), a locality-only LOCATION, or an
    unsplit ADDRESS (a DX line, a street line with no place)."""
    mentions = []
    for u in case["expect"]["addresses"]:
        entity = f"E-{u['id']}"
        street, loc = u["street_line"], u["locality"]
        if street:
            mentions.append(GTMention(id=u["id"], type=PiiType.ADDRESS, entity=entity, text=_text(case, street),
                                      segments=_segments(case, street),
                                      conditions=["address_street_line"] if loc else []))
        if loc:
            mentions.append(GTMention(id=f"{u['id']}-loc" if street else u["id"], type=PiiType.LOCATION,
                                      subtype=LOCALITY_SUBTYPE, entity=entity, text=_text(case, loc),
                                      segments=_segments(case, loc), conditions=[ADDRESS_LOCALITY]))
    return _doc(case, mentions)
