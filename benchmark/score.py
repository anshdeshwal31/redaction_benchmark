"""Score one candidate output: the output evaluation always, every component metric for
which the candidate exposed the stage, then attribution. Missing stages stay ``None``
(reported N/A), never zero (PLAN.md §9)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .attribution import AttributionResult, attribute
from .evaluate import DocumentEvaluation, EvalConfig, MentionStatus, evaluate_output
from .gt.policy import resolve
from .metrics import (
    DetectionResult,
    FidelityResult,
    InkIndex,
    LocalizationResult,
    PseudonymResult,
    ResolutionResult,
    TextAcqResult,
    evaluate_detection,
    evaluate_plan,
    evaluate_pseudonyms,
    evaluate_resolution,
    evaluate_text,
    plan_fidelity,
    project_mentions,
    replacements_by_mention,
    response_from_artifacts,
    response_from_replacements,
)
from .schemas import GTDocument, PiiType, PolicyProfile, Requirement


@dataclass
class DocumentScore:
    doc_id: str
    evaluation: DocumentEvaluation
    text: TextAcqResult | None = None
    detection: DetectionResult | None = None
    resolution: ResolutionResult | None = None
    localization: LocalizationResult | None = None
    fidelity: FidelityResult | None = None
    pseudonyms: PseudonymResult | None = None
    attribution: AttributionResult | None = None
    stages_exposed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"doc_id": self.doc_id, "stages_exposed": self.stages_exposed, "evaluation": self.evaluation.to_dict()}
        for name in ("text", "detection", "resolution", "localization", "fidelity", "pseudonyms", "attribution"):
            v = getattr(self, name)
            d[name] = v.to_dict() if v is not None else None
        return d


def score_document(
    source: bytes,
    output: bytes,
    gt: GTDocument,
    policy: PolicyProfile,
    artifacts: dict | None = None,
    label_map: dict[str, PiiType] | None = None,
    config: EvalConfig | None = None,
    oracle: dict[str, dict[str, str]] | None = None,
    unstable: set[str] | None = None,
    ink_dpi: float = 100,
) -> DocumentScore:
    artifacts = artifacts or {}
    res = resolve(gt, policy)
    ev = evaluate_output(source, output, gt, policy, config)
    score = DocumentScore(gt.doc_id, ev, stages_exposed=sorted(artifacts))
    if not artifacts and not any(m.status is MentionStatus.SAFE_REPLACED for m in ev.mentions):
        score.attribution = attribute(gt, policy, res, ev, oracle=oracle, unstable=unstable)
        return score
    index = InkIndex.build(gt, source, ink_dpi)
    text = artifacts.get("text")
    if text is not None:
        score.text = evaluate_text(text, gt, res, index)
    best_pred: dict[str, str | None] = {}
    if artifacts.get("mentions") is not None:
        preds = project_mentions(artifacts["mentions"], index, label_map, text=text)
        score.detection = evaluate_detection(gt, res, preds)
        best_pred = {x.mention_id: x.best_pred for x in score.detection.matches}
    plan = artifacts.get("plan")
    if plan is not None:
        score.localization = evaluate_plan(plan, gt, res, index)
        if ev.integrity.opens:
            score.fidelity = plan_fidelity(plan, source, output, index)
    names = {m.id: m.entity for m in gt.mentions if m.type is PiiType.PERSON_NAME and m.entity
             and res.mentions[m.id].requirement is Requirement.MUST}
    if artifacts.get("pseudonyms") is not None or any(
            m.status is MentionStatus.SAFE_REPLACED for m in ev.mentions if m.mention_id in names):
        score.pseudonyms = evaluate_pseudonyms(gt, res, output)
    if artifacts.get("entities") is not None and best_pred:
        clusters = {mid: c.id for c in artifacts["entities"].clusters for mid in c.mentions}
        score.resolution = evaluate_resolution(
            names, response_from_artifacts({g: p for g, p in best_pred.items() if g in names}, clusters))
    elif score.pseudonyms is not None and score.pseudonyms.evaluable:
        inferred = {k: v for k, v in replacements_by_mention(score.pseudonyms).items() if k in names}
        if inferred:
            score.resolution = evaluate_resolution(names, response_from_replacements(inferred),
                                                   source="inferred_from_pseudonyms")
    regions_by_mention: dict[str, list[str]] | None = None
    if plan is not None and best_pred:
        by_pred: dict[str, list[str]] = {}
        for r in plan.regions:
            for pid in r.mentions:
                by_pred.setdefault(pid, []).append(r.id)
        regions_by_mention = {g: by_pred.get(p, []) for g, p in best_pred.items() if p}
    score.attribution = attribute(gt, policy, res, ev, text=score.text, det=score.detection,
                                  loc=score.localization, fid=score.fidelity, pseud=score.pseudonyms,
                                  oracle=oracle, unstable=unstable, plan_regions_by_mention=regions_by_mention)
    return score
