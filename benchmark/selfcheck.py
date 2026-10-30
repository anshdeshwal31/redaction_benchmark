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

from .evaluate import HIDDEN, DocumentEvaluation, EvalConfig, MentionStatus, evaluate_output

# The self-check tests leak detection, so "safe" here means hidden: a removal under a policy that
# requires replacement is hidden (NOT_REPLACED), not leaked.
SAFE = HIDDEN
from .gt.policy import resolve
from .metrics import InkIndex, evaluate_plan
from .mocks import MockSpec, run_mock
from .mocks.named import NAMED
from .schemas import Action, GTDocument, PiiType, PolicyProfile, Requirement


@dataclass
class Check:
    name: str
    spec: MockSpec
    expect: str
    test: Callable[[DocumentEvaluation, dict], tuple[bool, str]]
    needs_text_layer: bool = False
    # More mock specs the same expectation must hold for (each run on its own; the check passes when all do).
    variants: tuple[MockSpec, ...] = ()


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


def _replace_all(ev, _):
    """Everything the client's rules ask for: every required mention replaced, text searchable."""
    ms = _must(ev)
    if not ms:
        return None, "no required mentions evaluated: not applicable"
    bad = [m.mention_id for m in ms if m.status is not MentionStatus.SAFE_REPLACED]
    searchable = ev.summary.get("searchable_ok")
    return not bad and bool(searchable), (f"{len(bad)}/{len(ms)} not SAFE_REPLACED: {bad[:8]}; searchable "
                                          f"{ev.summary.get('searchable_rate')} ({'ok' if searchable else 'FAIL'})")


def _replace_in_place(ev, context):
    """Fakes typeset in place in the original's font and size, after true redaction: every required mention
    replaced. A fake mobile keeping its '04' at the same place shows the original's glyphs there: PARTIAL,
    unless the policy shares the phone prefix (owner, 2026-10-29)."""
    ms = _must(ev)
    if not ms:
        return None, "no required mentions evaluated: not applicable"
    kept = context.get("mobile_prefix_kept", set())
    shared = context.get("phone_prefix", False)
    bad = [m.mention_id for m in ms if m.status is not (
        MentionStatus.PARTIAL if (m.mention_id in kept and not shared) else MentionStatus.SAFE_REPLACED)]
    coll = ev.summary.get("collateral_rate") or 0.0
    return not bad and coll <= 0.02, (f"{len(bad)}/{len(ms)} not as expected (SAFE_REPLACED; PARTIAL for "
                                      f"{0 if shared else len(kept)} mobile(s) keeping '04' unshared): {bad[:8]}; "
                                      f"collateral {coll:.3f}")


def _retypeset(ev, _):
    """Lines holding a fake re-typeset like tool 2's: the kept words moved, not damaged, and searchable."""
    ms = _must(ev)
    if not ms:
        return None, "no required mentions evaluated: not applicable"
    bad = [m.mention_id for m in ms if m.status is not MentionStatus.SAFE_REPLACED]
    s = ev.summary
    ok = not bad and bool(s.get("collateral_ok")) and bool(s.get("searchable_ok"))
    return ok, (f"{len(bad)}/{len(ms)} not SAFE_REPLACED: {bad[:8]}; collateral {s.get('collateral_real_rate')}; "
                f"searchable {s.get('searchable_rate')}; moved {s.get('moved_words')}")


def _text_off_a_line(ev, _):
    s = ev.summary
    if not s.get("searchable_words"):
        return None, "no kept words to search: not applicable"
    return s.get("searchable_ok") is False, f"searchable {s.get('searchable_rate')} (must fail)"


def _address_parts(ev, kind: str | None = None):
    """Required address parts: street lines and localities (``kind`` 'locality' for localities only)."""
    return [m for m in _must(ev) if m.address_unit and (kind is None or m.type == PiiType.LOCATION.value)]


def _whole_address(ev, _):
    parts = _address_parts(ev)
    if not parts:
        return None, "no required address parts: not applicable"
    bad = [m.mention_id for m in parts if m.status is not MentionStatus.SAFE_REPLACED]
    return not bad, f"{len(bad)}/{len(parts)} address parts not SAFE_REPLACED: {bad[:8]}"


def _locality_kept(ev, _):
    parts = _address_parts(ev, "locality")
    if not parts:
        return None, "no required localities: not applicable"
    safe = [m.mention_id for m in parts if m.status in SAFE]
    return not safe, f"{len(safe)}/{len(parts)} kept localities judged safe: {safe[:8]}"


def _locality_erased(ev, context):
    required = context.get("replace_required", set())
    parts = [m for m in _address_parts(ev, "locality") if m.mention_id in required]
    if not parts:
        return None, "no locality the policy requires a fake for: not applicable"
    bad = [m.mention_id for m in parts if m.status is not MentionStatus.NOT_REPLACED]
    return not bad, f"{len(bad)}/{len(parts)} erased localities not NOT_REPLACED: {bad[:8]}"


def _mobiles_keeping_prefix(gt: GTDocument, mock) -> set[str]:
    """Phone mentions whose original and fake both start with a mobile's '04' (the mock's fakes)."""
    words = gt.word_index()
    fake = {r.mention.split(":", 1)[1]: r.replacement for r in (mock.pseudonyms.mentions if mock.pseudonyms else [])}
    out = set()
    for m in gt.mentions:
        if m.type is not PiiType.PHONE or not m.segments or not m.segments[0].words:
            continue
        first = m.segments[0].words[0]
        if words[first][1].text.startswith("04") and (fake.get(first) or "").startswith("04"):
            out.add(m.id)
    return out


def _remove_only(ev, context):
    """True removal without fakes gets the verdict the policy implies for every required mention:
    NOT_REPLACED where a fake is required, SAFE_REMOVED where removal is allowed. A leak fails."""
    required = context.get("replace_required", set())
    ms = _must(ev)
    if not ms:
        return None, "no required mentions evaluated: not applicable"
    fake = sum(1 for m in ms if m.mention_id in required)
    bad = [m.mention_id for m in ms if m.status is not (
        MentionStatus.NOT_REPLACED if m.mention_id in required else MentionStatus.SAFE_REMOVED)]
    return not bad, (f"{len(bad)}/{len(ms)} not as the policy implies (NOT_REPLACED for {fake} that need a fake, "
                     f"SAFE_REMOVED for {len(ms) - fake} that may be removed): {bad[:8]}")


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
    # The client's rules (Q6, Q7, Q11): the perfect replacer passes, a perfect remover does not.
    Check("replace-all", NAMED["reference-replace"], "all SAFE_REPLACED and searchable", _replace_all),
    Check("remove-only", MockSpec("remove-only"), "NOT_REPLACED where the policy requires a fake, SAFE_REMOVED "
          "where it allows removal", _remove_only),
    # eval-0.4: fakes drawn the way tools draw them (docs/metrics.md, "Character identity", "Moved words",
    # "Address units"). Correct ones pass; an original left under, beside or behind its fake does not.
    Check("replace-in-place", NAMED["replace-in-place"], "all SAFE_REPLACED (a mobile keeping '04' PARTIAL unless "
          "the policy shares the phone prefix), collateral <= 2%", _replace_in_place),
    Check("retypeset", NAMED["retypeset"], "all SAFE_REPLACED, kept words moved not damaged, searchable",
          _retypeset),
    Check("retypeset-keep", NAMED["retypeset-keep"], "none judged safe (originals re-typeset)", _none_safe),
    Check("overprint", NAMED["overprint"], ">= 80% not safe", _share_not_safe(0.8)),
    Check("overprint-veil", NAMED["overprint-veil"], ">= 80% not safe under 30% and 60% white boxes",
          _share_not_safe(0.8), variants=(MockSpec("overprint-veil-60", replace_all=True, stand_ins=True, veil=0.6),)),
    Check("shift-white", NAMED["shift-white"], ">= 80% not safe with the white box half a line down or up",
          _share_not_safe(0.8), variants=(MockSpec("shift-white-up", replace_all=True, stand_ins=True,
                                                   shift_white=-0.5),)),
    Check("text-next-line", NAMED["text-next-line"], "not searchable (text layer one line off)", _text_off_a_line),
    Check("whole-address", NAMED["whole-address-replace"], "address parts SAFE_REPLACED", _whole_address),
    Check("keeps-locality-moved", NAMED["keeps-locality-moved"], "the kept locality not safe, moved along its "
          "line or to the next line", _locality_kept,
          variants=(MockSpec("keeps-locality-next-line", replace_all=True, stand_ins=True, fakes_in_place=True,
                             keeps_locality_moved="next_line"),)),
    Check("erases-locality", NAMED["erases-locality"], "the erased locality NOT_REPLACED where the policy requires "
          "a fake", _locality_erased),
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
    replace_required = {mid for mid, r in res.mentions.items()
                        if r.requirement is Requirement.MUST and r.action is Action.REPLACE}
    phone_prefix = any(PiiType.PHONE in e.types and "phone_prefix" in e.roles for e in policy.shared_roles)
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
        try:
            results, statuses, extra = [], Counter(), {}
            for spec in (c.spec, *c.variants):
                mock = run_mock(source, gt, policy, spec)
                ev = evaluate_output(source, mock.pdf, gt, policy, cfg)
                if c.name == "dpi-scale":
                    index = index or InkIndex.build(gt, source, 100)
                    extra["diagnosis"] = evaluate_plan(mock.plan, gt, res, index).document_diagnosis
                results.append(c.test(ev, dict(extra, replace_required=replace_required, phone_prefix=phone_prefix,
                                               mobile_prefix_kept=_mobiles_keeping_prefix(gt, mock))))
                statuses.update(m.status.value for m in ev.mentions if m.requirement == "MUST")
        except Exception as exc:  # noqa: BLE001 - a check that breaks fails; the other checks still run
            out.append(error_result(c.name, c.expect, exc, round(time.time() - t0, 1)))
            continue
        verdicts = [p for p, _ in results]
        passed = None if all(p is None for p in verdicts) else all(p is not False for p in verdicts)
        observed = " | ".join(o for _, o in results) if len(results) > 1 else results[0][1]
        out.append(CheckResult(c.name, c.expect, passed, observed, dict(statuses), round(time.time() - t0, 1), extra))
    return out


def error_result(name: str, expect: str, exc: BaseException, seconds: float = 0.0) -> CheckResult:
    """A check (or a whole document) that raised: failed, named by the exception's class only, since
    its message can quote document text."""
    kind = type(exc).__name__
    return CheckResult(name, expect, False, f"ERROR: {kind} (could not run)", {}, seconds, {"error": kind})
