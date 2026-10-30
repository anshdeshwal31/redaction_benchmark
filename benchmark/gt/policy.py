"""Apply a policy profile to GT facts: what must, may, or must not be redacted.

Word-level effective requirement uses the precedence MUST > MAY > MUST_NOT > ORDINARY:
a word inside any required mention must go even if it is also part of a protected
span; a word covered only by protected spans must stay; uncovered words are
ORDINARY (their loss is collateral damage, weighted below protected damage).

A policy's rules digest hashes what it decides without its wording, so a wording-only
release (1.1.0 -> 1.1.1) is known to judge outputs by the same rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from ..schemas import (
    Action,
    EntityKind,
    GTClassified,
    GTDocument,
    GTMention,
    PolicyProfile,
    PolicyRule,
    Requirement,
    Role,
)


class Effective(str, Enum):
    MUST = "MUST"
    MAY = "MAY"
    MUST_NOT = "MUST_NOT"
    ORDINARY = "ORDINARY"


_RANK = {Effective.MUST: 3, Effective.MAY: 2, Effective.MUST_NOT: 1, Effective.ORDINARY: 0}


@dataclass(frozen=True)
class Resolved:
    requirement: Requirement
    action: Action
    rule_id: str | None
    pending: str | None


@dataclass
class Resolution:
    policy: str
    mentions: dict[str, Resolved] = field(default_factory=dict)
    classified: dict[str, Resolved] = field(default_factory=dict)
    words: dict[str, Effective] = field(default_factory=dict)
    regions: dict[str, Effective] = field(default_factory=dict)

    def mentions_with(self, requirement: Requirement) -> list[str]:
        return [m for m, r in self.mentions.items() if r.requirement is requirement]


def _rule_matches(rule: PolicyRule, pii_type, subtype: str | None, role: Role, kind: EntityKind | None) -> bool:
    w = rule.when
    if w.target != "mention":
        return False
    if w.type is not None and pii_type not in w.type:
        return False
    if w.subtype is not None and (subtype or "") not in w.subtype:
        return False
    if w.role is not None and role not in w.role:
        return False
    if w.entity_kind is not None and kind not in w.entity_kind:
        return False
    return True


def _rule_matches_mention(rule: PolicyRule, m: GTMention, role: Role, kind: EntityKind | None) -> bool:
    return _rule_matches(rule, m.type, m.subtype, role, kind)


def resolve_fact(gt: GTDocument, policy: PolicyProfile, pii_type, subtype: str | None,
                 entity_id: str | None) -> Resolved:
    """Requirement for a PII fact that is not a page mention (e.g. a metadata surface item)."""
    ent = next((e for e in gt.entities if e.id == entity_id), None) if entity_id else None
    role = ent.role if ent else Role.UNKNOWN
    kind = ent.kind if ent else None
    hit = next((r for r in policy.rules if _rule_matches(r, pii_type, subtype, role, kind)), None)
    if hit is None:
        return Resolved(policy.default_mention, Action.EITHER, None, None)
    return Resolved(hit.requirement, hit.action, hit.id, hit.pending)


def _rule_matches_classified(rule: PolicyRule, c: GTClassified) -> bool:
    w = rule.when
    return w.target == "classified" and (w.cls is None or c.cls in w.cls)


def _promote(store: dict[str, Effective], key: str, value: Effective) -> None:
    current = store.get(key, Effective.ORDINARY)
    if _RANK[value] > _RANK[current]:
        store[key] = value


def resolve(gt: GTDocument, policy: PolicyProfile) -> Resolution:
    entities = {e.id: e for e in gt.entities}
    res = Resolution(policy=f"{policy.name}@{policy.version}")

    for m in gt.mentions:
        ent = entities.get(m.entity) if m.entity else None
        role = ent.role if ent else Role.UNKNOWN
        kind = ent.kind if ent else None
        hit = next((r for r in policy.rules if _rule_matches_mention(r, m, role, kind)), None)
        if hit is None:
            res.mentions[m.id] = Resolved(policy.default_mention, Action.EITHER, None, None)
        else:
            res.mentions[m.id] = Resolved(hit.requirement, hit.action, hit.id, hit.pending)

    for c in gt.classified:
        hit = next((r for r in policy.rules if _rule_matches_classified(r, c)), None)
        if hit is None:
            res.classified[c.id] = Resolved(policy.default_classified, Action.EITHER, None, None)
        else:
            res.classified[c.id] = Resolved(hit.requirement, hit.action, hit.id, hit.pending)

    # Word/region effective requirements.
    for p in gt.pages:
        for w in p.words:
            res.words[w.id] = Effective.ORDINARY
        for r in p.regions:
            res.regions[r.id] = Effective.ORDINARY
    for c in gt.classified:
        eff = Effective(res.classified[c.id].requirement.value)
        for seg in c.segments:
            for wid in seg.words:
                _promote(res.words, wid, eff)
            for rid in seg.regions:
                _promote(res.regions, rid, eff)
    for m in gt.mentions:
        eff = Effective(res.mentions[m.id].requirement.value)
        for seg in m.segments:
            for wid in seg.words:
                _promote(res.words, wid, eff)
            for rid in seg.regions:
                _promote(res.regions, rid, eff)
        # Optional words (titles, possessive markers) never penalise either way.
        for wid in m.optional_words:
            if res.words.get(wid) in (Effective.ORDINARY, Effective.MUST_NOT):
                res.words[wid] = Effective(policy.optional_words.value)
    return res


def pending_rules(policy: PolicyProfile) -> list[PolicyRule]:
    return [r for r in policy.rules if r.pending]


# Prose that explains a policy without changing what it decides; left out of the rules digest.
PROSE_FIELDS = ("version", "description")
RULE_PROSE_FIELDS = ("note",)
DEVIATION_PROSE_FIELDS = ("client_answer", "applied", "reason", "effect")


def rules_digest(policy: PolicyProfile) -> str:
    """SHA-256 (hex) of the policy without its prose: the version, the description, every rule's note
    and every deviation's client answer, applied text, reason and effect. Its name, rules, defaults,
    consistency scope, deviation ids and the rules each deviation names stay in, so two files with the
    same digest judge every output by the same rules. (YAML comments are never part of the policy.)"""
    data = policy.model_dump(mode="json", by_alias=True)
    for key in PROSE_FIELDS:
        data.pop(key, None)
    for rule in data["rules"]:
        for key in RULE_PROSE_FIELDS:
            rule.pop(key, None)
    for dev in data["deviations"]:
        for key in DEVIATION_PROSE_FIELDS:
            dev.pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
