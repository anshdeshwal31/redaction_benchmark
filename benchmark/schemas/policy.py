"""Redaction policy profiles (PLAN.md §7.3).

A policy maps GT facts to requirements. Rules are evaluated in order; the first
matching rule wins. Rules tied to an unresolved open question carry ``pending`` and
should use ``MAY`` so that undecided categories neither reward nor penalise a tool.
A deliberate departure from the client's answers is recorded as a deviation, which
scoring copies into its results and every report shows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import Action, ClassifiedClass, EntityKind, PiiType, Requirement, Role, StrictModel

POLICY_SCHEMA = "rb.policy/0.1"


class PolicyMatch(StrictModel):
    target: Literal["mention", "classified"] = "mention"
    type: list[PiiType] | None = None
    subtype: list[str] | None = None
    role: list[Role] | None = None
    entity_kind: list[EntityKind] | None = None
    cls: list[ClassifiedClass] | None = None


class PolicyRule(StrictModel):
    id: str = Field(min_length=1)
    when: PolicyMatch
    requirement: Requirement
    action: Action = Action.EITHER
    pending: str | None = Field(None, description="Open question this rule depends on, e.g. 'Q3'")
    note: str | None = None


class PolicyDeviation(StrictModel):
    """Where the policy knowingly applies something other than the client's answer, and why."""

    id: str = Field(min_length=1)
    question: str = Field(description="Questionnaire item, e.g. 'Q4'")
    client_answer: str
    applied: str = Field(description="What the policy applies instead")
    reason: str
    effect: str = Field("", description="How scores differ from scoring by the client's answer")
    rules: list[str] = Field([], description="Ids of the rules that implement the departure")


class PolicyProfile(StrictModel):
    schema_id: Literal["rb.policy/0.1"] = Field(POLICY_SCHEMA, alias="schema")
    name: str
    version: str
    description: str = ""
    consistency_scope: Literal["document", "matter", "global", "pending"] = "pending"
    rules: list[PolicyRule]
    default_mention: Requirement = Requirement.MAY
    default_classified: Requirement = Requirement.MUST_NOT
    optional_words: Requirement = Requirement.MAY
    deviations: list[PolicyDeviation] = Field([], description="Deliberate departures from the client's answers; "
                                                              "highlighted in every report")

    @model_validator(mode="after")
    def _check_deviations(self) -> "PolicyProfile":
        rule_ids = {r.id for r in self.rules}
        seen: set[str] = set()
        for d in self.deviations:
            if d.id in seen:
                raise ValueError(f"deviation {d.id} is listed twice")
            seen.add(d.id)
            unknown = [r for r in d.rules if r not in rule_ids]
            if unknown:
                raise ValueError(f"deviation {d.id} names rules that are not in the policy: {unknown}")
        return self
