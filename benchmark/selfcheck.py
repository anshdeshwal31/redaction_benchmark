"""Evaluator self-check on any document with GT (PLAN.md Verification).

Runs the fault-injecting mocks on a document and checks the evaluator reaches the
expected verdict for each injected fault. Use it on every newly annotated real document:
the thresholds were calibrated on synthetic fixtures, and this is how we find out whether
they hold on real layouts, scans and fonts.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

from .evaluate import SAFE, DocumentEvaluation, EvalConfig, MentionStatus, evaluate_output
from .gt.policy import resolve
from .metrics import InkIndex, evaluate_plan
from .mocks import MockSpec, run_mock
from .schemas import GTDocument, PiiType, PolicyProfile


@dataclass
class Check:
    name: str
    spec: MockSpec
    expect: str
    test: Callable[[DocumentEvaluation, dict], tuple[bool, str]]
    needs_text_layer: bool = False


@dataclass
class CheckResult:
    name: str
    expect: str
    passed: bool | None  # None = not applicable to this document
    observed: str
    statuses: dict
    seconds: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _must(ev):
    return [m for m in ev.mentions if m.requirement == "MUST" and m.status is not MentionStatus.NOT_EVALUATED]


def _all(statuses: set[MentionStatus]):
    def t(ev, _):
        bad = [m.mention_id for m in _must(ev) if m.status not in statuses]
        return not bad, f"{len(bad)} mention(s) outside {sorted(s.value for s in statuses)}: {bad[:8]}"
    return t


def _share_not_safe(share: float):
    def t(ev, _):
        ms = _must(ev)
        n = sum(1 for m in ms if m.status not in SAFE)
        return (bool(ms) and n >= share * len(ms)), f"{n}/{len(ms)} not safe (need >= {share:.0%})"
    return t


def _none_safe(ev, _):
    ms = _must(ev)
    safe = [m.mention_id for m in ms if m.status in SAFE]
    st = Counter(m.status.value for m in ms)
    return not safe and bool(ms), f"{len(safe)} judged safe: {safe[:8]}; statuses {dict(st)}"


def _reference(ev, _):
    ok, msg = _all(SAFE)(ev, _)
    s = ev.summary
    coll = s.get("collateral_rate") or 0.0
    prot = s.get("protected_damaged") or []
    good = ok and coll <= 0.02 and not prot
    return good, f"{msg}; collateral {coll:.3f}; protected damaged {len(prot)}"


def _overlay(ev, _):
    bad = [m.mention_id for m in _must(ev)
           if m.status is not MentionStatus.LEAK_RECOVERABLE or m.visible_residual > 0.02]
    return not bad, f"{len(bad)} mention(s) not 'hidden but recoverable': {bad[:8]}"


def _incremental(ev, _):
    bad = [m.mention_id for m in _must(ev) if "earlier_revision" not in m.l2_sources]
    return not bad and ev.revisions_checked >= 1, f"{len(bad)} without earlier_revision evidence"


def _scale(ev, extra):
    ok, msg = _share_not_safe(1.0)(ev, extra)
    diag = extra.get("diagnosis")
    return ok and diag == "systematic: scale 72/300", f"{msg}; diagnosis {diag!r}"


def _pseudonyms(ev, _):
    names = [m for m in _must(ev) if m.type == PiiType.PERSON_NAME.value and m.mention_id in _WITH_COMPONENTS]
    if not names:
        return None, "no required person names with name components (e.g. machine drafts): not applicable"
    bad = [m.mention_id for m in names if m.status is not MentionStatus.SAFE_REPLACED]
    return not bad, f"{len(bad)}/{len(names)} names not SAFE_REPLACED: {bad[:8]}"


_WITH_COMPONENTS: set[str] = set()  # set per document by selfcheck(); only these can be pseudonymised

CHECKS = [
    Check("reference-true", MockSpec("reference"), "all SAFE, collateral <= 2%, no protected damage", _reference),
    Check("reference-raster", MockSpec("reference-raster", method="rasterize"), "all SAFE, collateral <= 2%",
          _reference),
    Check("null", MockSpec("null", method="null"), "all LEAK_VISIBLE", _all({MentionStatus.LEAK_VISIBLE})),
    Check("overlay", MockSpec("overlay", method="overlay"), "all LEAK_RECOVERABLE, nothing visible", _overlay),
    Check("annotation", MockSpec("annotation", method="annotation"), "all LEAK_VISIBLE",
          _all({MentionStatus.LEAK_VISIBLE})),
    Check("blur", MockSpec("blur", method="blur"), "none judged safe (blur is not redaction)", _none_safe),
    Check("incremental", MockSpec("incremental", incremental=True), "all recoverable from an earlier revision",
          _incremental),
    Check("partial-half", MockSpec("partial", partial=0.5), ">= 80% not safe", _share_not_safe(0.8)),
    Check("shift-half-line", MockSpec("shift", method="rasterize", shift_pt=(0, 6)), ">= 50% not safe",
          _share_not_safe(0.5)),
    Check("dpi-scale", MockSpec("scale", scale=0.24), "all not safe; diagnosed as scale 72/300", _scale),
    Check("pseudonyms", MockSpec("pseud", pseudonyms="consistent"), "names SAFE_REPLACED", _pseudonyms,
          needs_text_layer=True),
]


def half_line_shift(gt: GTDocument) -> tuple[float, float]:
    """Half the median text-line height, perpendicular to the dominant reading direction."""
    import math

    import numpy as np

    from .geometry import as_poly

    heights, angles = [], []
    for p in gt.pages:
        for w in p.words:
            t = math.radians(w.angle)
            v = np.array([-math.sin(t), math.cos(t)])
            proj = as_poly(w.poly) @ v
            heights.append(float(proj.max() - proj.min()))
            angles.append(w.angle)
    if not heights:
        return (0.0, 6.0)
    half = 0.5 * float(np.median(heights))
    t = math.radians(Counter(round(a / 90) * 90 % 360 for a in angles).most_common(1)[0][0])
    return (round(-math.sin(t) * half, 2), round(math.cos(t) * half, 2))


def selfcheck(source: bytes, gt: GTDocument, policy: PolicyProfile, config: EvalConfig | None = None,
              only: list[str] | None = None) -> list[CheckResult]:
    import pymupdf

    cfg = config or EvalConfig()
    res = resolve(gt, policy)
    _WITH_COMPONENTS.clear()
    _WITH_COMPONENTS.update(m.id for m in gt.mentions if m.components)
    has_text = any(p.get_text().strip() for p in pymupdf.open("pdf", source))
    index = None
    out: list[CheckResult] = []
    for c in CHECKS:
        if only and c.name not in only:
            continue
        if c.name == "shift-half-line":
            c = Check(c.name, MockSpec("shift", method="rasterize", shift_pt=half_line_shift(gt)), c.expect, c.test)
        if c.needs_text_layer and not has_text:
            continue
        t0 = time.time()
        mock = run_mock(source, gt, policy, c.spec)
        ev = evaluate_output(source, mock.pdf, gt, policy, cfg)
        extra = {}
        if c.name == "dpi-scale":
            index = index or InkIndex.build(gt, source, 100)
            extra["diagnosis"] = evaluate_plan(mock.plan, gt, res, index).document_diagnosis
        passed, observed = c.test(ev, extra)
        statuses = dict(Counter(m.status.value for m in ev.mentions if m.requirement == "MUST"))
        out.append(CheckResult(c.name, c.expect, passed, observed, statuses, round(time.time() - t0, 1), extra))
    return out
