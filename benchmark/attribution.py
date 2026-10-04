"""Failure attribution (PLAN.md §11): for every failing required mention, the first stage
in causal order that failed given correct upstream input.

Each attribution is labelled ``confirmed`` (direct artifact or counterfactual evidence)
or ``inferred`` (black-box heuristic) and the two are never mixed. Over-redaction of
protected and ordinary words is attributed separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from .evaluate import SAFE, DocumentEvaluation, MentionStatus, WordStatus
from .geometry import as_poly, bbox
from .gt.policy import Effective, Resolution, resolve_fact
from .metrics.detection import DetectionResult
from .metrics.localization import FidelityResult, LocalizationResult
from .metrics.pseudonyms import PseudonymResult
from .metrics.resolution import ResolutionResult
from .metrics.text_acq import TextAcqResult
from .schemas import GTDocument, PiiType, PolicyProfile, Requirement


@dataclass
class Attribution:
    mention_id: str
    status: str
    stage: str
    category: str
    confidence: str  # confirmed | inferred
    detail: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class WordAttribution:
    word_id: str
    requirement: str
    category: str
    confidence: str


@dataclass
class AttributionResult:
    mentions: list[Attribution]
    words: list[WordAttribution]
    funnel: dict
    counts: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _near(a, b, gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def attribute(
    gt: GTDocument,
    policy: PolicyProfile,
    res: Resolution,
    ev: DocumentEvaluation,
    text: TextAcqResult | None = None,
    det: DetectionResult | None = None,
    loc: LocalizationResult | None = None,
    fid: FidelityResult | None = None,
    pseud: PseudonymResult | None = None,
    resolution: ResolutionResult | None = None,
    oracle: dict[str, dict[str, str]] | None = None,
    unstable: set[str] | None = None,
    plan_regions_by_mention: dict[str, list[str]] | None = None,
) -> AttributionResult:
    words = gt.word_index()
    mentions = {m.id: m for m in gt.mentions}
    matches = {x.mention_id: x for x in det.matches} if det else {}
    coverage = {c.mention_id: c.coverage for c in loc.mentions} if loc else {}
    diag = {d.page: d for d in loc.diagnoses} if loc else {}
    unrecognisable = set(text.mention_recognisability["unrecognisable"]) if text else set()
    rec_words = {w.word_id: w for w in text.words} if text else {}
    not_applied = {r.region_id for r in fid.regions if r.status == "not_applied"} if fid else set()
    oracle_status = oracle or {}  # mode -> {mention id: status value} from oracle runs
    safe_values = {s.value for s in SAFE}
    safe_by_entity: dict[str, int] = Counter()
    for m in ev.mentions:
        if m.requirement == Requirement.MUST.value and m.status in SAFE and m.entity:
            safe_by_entity[m.entity] += 1
    detected_entities = {x.entity for x in matches.values() if x.entity and x.detected}

    pseud_tags: dict[str, list[str]] = {}
    if pseud is not None and pseud.evaluable:
        by_word = {}
        for r in pseud.replacements:
            by_word.setdefault(r.word_id, []).append(r.mention_id)
        for wid in pseud.issues.get("survived", []):
            for mid in by_word.get(wid, []):
                pseud_tags.setdefault(mid, []).append("PSEUDONYM_COMPONENT_SURVIVED")
        for mid in pseud.issues.get("honorific_gender_mismatch", []):
            pseud_tags.setdefault(mid, []).append("PSEUDONYM_ATTRIBUTE")
        inconsistent_entities = {k.split(":")[0] for k in pseud.issues.get("entity_inconsistent", {})}
        for m in gt.mentions:
            if m.entity in inconsistent_entities and m.type is PiiType.PERSON_NAME:
                pseud_tags.setdefault(m.id, []).append("PSEUDONYM_INCONSISTENT")
        colliding = {tuple(k.split(":", 1)) for k in pseud.issues.get("collisions", {})}
        for r in pseud.replacements:
            if r.replacement and (r.component, r.replacement.strip(".,;:()").casefold()) in colliding:
                if "PSEUDONYM_COLLISION" not in pseud_tags.get(r.mention_id, []):
                    pseud_tags.setdefault(r.mention_id, []).append("PSEUDONYM_COLLISION")
        for mid in pseud.issues.get("invalid", {}):
            pseud_tags.setdefault(mid, []).append("PSEUDONYM_INVALID")
    # A pseudonym inconsistency or collision is a resolution failure when the tool's own
    # clusters (not ones inferred from its pseudonyms, which would be circular) split or
    # merged the entity.
    if resolution is not None and resolution.source == "entities_artifact":
        split = {x["entity"] for x in resolution.false_splits}
        merged = {e for x in resolution.false_merges for e in x["entities"]}
        for m in gt.mentions:
            tags = pseud_tags.get(m.id, [])
            if m.entity in split and "PSEUDONYM_INCONSISTENT" in tags:
                tags[tags.index("PSEUDONYM_INCONSISTENT")] = "RESOLUTION_SPLIT"
            if m.entity in merged and "PSEUDONYM_COLLISION" in tags:
                tags[tags.index("PSEUDONYM_COLLISION")] = "RESOLUTION_MERGE"

    out: list[Attribution] = []
    for mo in ev.mentions:
        if mo.requirement != Requirement.MUST.value:
            continue
        tags = list(pseud_tags.get(mo.mention_id, []))
        if unstable and mo.mention_id in unstable:
            tags.append("NONDETERMINISTIC")
        if mo.status in SAFE:
            if tags:
                out.append(Attribution(mo.mention_id, mo.status.value, "NONE", "SAFE", "confirmed", tags=tags))
            continue
        m = mentions[mo.mention_id]
        conf = "confirmed"

        def done(stage, category, detail="", confidence=conf):
            out.append(Attribution(mo.mention_id, mo.status.value, stage, category, confidence, detail, tags))

        if mo.status is MentionStatus.NOT_EVALUATED:
            done("UNKNOWN", "NOT_EVALUATED", "integrity or geometry problem")
            continue
        # counterfactual confirmation from oracle runs (if provided)
        cf = cf_mode = None
        for mode, stage in (("oracle_text", "TEXT"), ("oracle_mentions", "DETECTION"),
                            ("oracle_regions", "LOCALIZATION")):
            st = oracle_status.get(mode, {}).get(mo.mention_id)
            if st is not None and st in safe_values:
                cf, cf_mode = stage, mode
                break
        # [T] text acquisition
        if text is not None and mo.mention_id in unrecognisable:
            ws = [rec_words.get(w) for seg in m.segments for w in seg.words]
            missing = all(w is None or w.recognised is None for w in ws)
            done("TEXT", "TEXT_MISSING" if missing else "TEXT_CORRUPTED",
                 "safe under oracle_text" if cf == "TEXT" else "")
            continue
        # [D] detection
        x = matches.get(mo.mention_id)
        if x is not None:
            if not x.detected:
                if m.entity and m.entity in detected_entities:
                    done("DETECTION", "DETECTION_ALIAS_PROPAGATION",
                         f"entity {m.entity} detected elsewhere; form {m.form or '?'}")
                else:
                    done("DETECTION", "DETECTION_MISS")
                continue
            if x.boundary == "under":
                done("DETECTION", "DETECTION_BOUNDARY", "missing " + ",".join(x.missing_components or ["words"]))
                continue
            if x.type_correct is False and x.pred_type:
                try:
                    r = resolve_fact(gt, policy, PiiType(x.pred_type), m.subtype, m.entity)
                    if r.requirement is not Requirement.MUST:
                        done("DETECTION", "DETECTION_TYPE", f"{m.type.value} detected as {x.pred_type} "
                                                            f"({r.requirement.value} under policy)")
                        continue
                except ValueError:
                    pass
        # [L] localization
        if loc is not None and mo.mention_id in coverage and coverage[mo.mention_id] < 0.98:
            pages = [p for p in mo.pages if p in diag]
            d = next((diag[p] for p in pages if diag[p].kind == "systematic"), None)
            # Regions meant for this page may have landed on a neighbouring one.
            moved = next((diag[p + off] for p in mo.pages for off in (-1, 1)
                          if p + off in diag and diag[p + off].hypothesis == f"page offset {off:+d}"), None)
            if moved is not None and (d is None or d.hypothesis is None or d.hypothesis.startswith("page offset")):
                done("LOCALIZATION", "WRONG_PAGE", f"regions placed on page {moved.page} ({moved.hypothesis})")
            elif d is not None and d.hypothesis and d.hypothesis.startswith("page offset"):
                done("LOCALIZATION", "WRONG_PAGE", d.hypothesis)
            elif d is not None:
                done("LOCALIZATION", "LOCALIZATION_SYSTEMATIC", d.hypothesis or "")
            else:
                done("LOCALIZATION", "LOCALIZATION_LOCAL", f"coverage {coverage[mo.mention_id]:.2f}")
            continue
        # [A] application
        if fid is not None and plan_regions_by_mention is not None:
            regs = plan_regions_by_mention.get(mo.mention_id, [])
            if regs and any(r in not_applied for r in regs) and mo.status in (
                    MentionStatus.LEAK_VISIBLE, MentionStatus.PARTIAL):
                done("APPLICATION", "APPLICATION_NOT_APPLIED", "planned region left unchanged in output")
                continue
            if regs and mo.status in (MentionStatus.LEAK_VISIBLE, MentionStatus.PARTIAL):
                if oracle_status.get("oracle_regions", {}).get(mo.mention_id) in safe_values:
                    done("APPLICATION", "INTERACTION", "own plan covers the mention and GT regions are applied "
                                                       "correctly, but the tool's own regions are not")
                else:
                    done("APPLICATION", "APPLICATION_OFFSET", "plan covers the mention but the output does not")
                continue
        # removal (file-level survival)
        if mo.status is MentionStatus.LEAK_RECOVERABLE:
            done("REMOVAL", "REMOVAL_INCOMPLETE", ",".join(mo.l2_sources))
            continue
        if mo.status is MentionStatus.LEAK_LATENT:
            done("REMOVAL", "REMOVAL_INCOMPLETE", "latent (recoverable by enhancement)")
            continue
        # every exposed stage looks right for this mention, yet a perfect upstream input fixes it
        if cf and (text is not None or det is not None or loc is not None):
            done(cf, "INTERACTION", f"each exposed stage looks correct on its own output, but the mention is "
                                    f"safe when {cf_mode} input is injected")
            continue
        # black-box inference
        mb = bbox(as_poly([pt for seg in m.segments for wid in seg.words for pt in words[wid][1].poly]
                          or [[0, 0], [0, 1], [1, 1]]))
        orphans = [o for p in ev.pages if p.index in mo.pages for o in p.orphan_regions]
        if any(_near(mb, o, 30.0) for o in orphans):
            done("LOCALIZATION", "LOCALIZATION_LOCAL", "orphan redaction next to the missed mention", "inferred")
        elif m.entity and safe_by_entity.get(m.entity):
            done("DETECTION", "DETECTION_MISS", f"entity {m.entity} protected elsewhere", "inferred")
        elif cf:
            done(cf, f"{cf}_FAILURE", "confirmed by oracle run")
        else:
            done("UNKNOWN", "UNKNOWN_STAGE", "no intermediate evidence", "inferred")

    # -- over-redaction ---------------------------------------------------------------------
    word_attr: list[WordAttribution] = []
    predicted = set(det.predicted_words) if det is not None else set()
    touched = set(loc.words_touched) if loc else set()
    replaced_boxes = []
    for mo in ev.mentions:
        if mo.status is MentionStatus.SAFE_REPLACED:
            m = mentions[mo.mention_id]
            for seg in m.segments:
                pts = [pt for wid in seg.words for pt in words[wid][1].poly]
                if pts:
                    replaced_boxes.append((seg.page, bbox(as_poly(pts))))
    for w in ev.words:
        if w.status is WordStatus.PRESERVED or w.status is WordStatus.NOT_EVALUATED:
            continue
        if w.requirement not in (Effective.ORDINARY.value, Effective.MUST_NOT.value):
            continue
        wb = bbox(as_poly(words[w.word_id][1].poly))
        if w.word_id in predicted:
            cat, c = "FP_DETECTION", "confirmed"
        elif w.word_id in touched:
            cat, c = "LOCALIZATION_SPILL", "confirmed"
        elif any(pg == w.page and _near(wb, b, 3.0) for pg, b in replaced_boxes):
            cat, c = "REPLACEMENT_OVERFLOW", "inferred"
        else:
            cat, c = "APPLICATION_SPILL", "inferred"
        word_attr.append(WordAttribution(w.word_id, w.requirement, cat, c))

    # -- funnel ------------------------------------------------------------------------------
    must = [m for m in ev.mentions if m.requirement == Requirement.MUST.value]
    funnel = {"required": len(must)}
    if text is not None:
        funnel["recognised"] = len(must) - len([m for m in must if m.mention_id in unrecognisable])
    if det is not None:
        funnel["detected"] = sum(1 for m in must if matches.get(m.mention_id) and matches[m.mention_id].detected
                                 and matches[m.mention_id].boundary != "under")
    if loc is not None:
        funnel["planned"] = sum(1 for m in must if coverage.get(m.mention_id, 0) >= 0.98)
    funnel["visually_safe"] = sum(1 for m in must if m.status in SAFE or m.status in (
        MentionStatus.LEAK_RECOVERABLE, MentionStatus.LEAK_LATENT))
    funnel["truly_safe"] = sum(1 for m in must if m.status in SAFE)
    counts = {"by_category": dict(Counter(a.category for a in out if a.category != "SAFE")),
              "by_stage": dict(Counter(a.stage for a in out if a.category != "SAFE")),
              "by_confidence": dict(Counter(a.confidence for a in out if a.category != "SAFE")),
              "over_redaction": dict(Counter(w.category for w in word_attr)),
              "tags": dict(Counter(t for a in out for t in a.tags))}
    return AttributionResult(out, word_attr, funnel, counts)
