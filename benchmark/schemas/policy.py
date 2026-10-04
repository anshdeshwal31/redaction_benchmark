"""Redaction policy profiles (PLAN.md §7.3).

A policy maps GT facts to requirements. Rules are evaluated in order; the first
matching rule wins. Rules tied to an unresolved open question carry ``pending`` and
should use ``MAY`` so that undecided categories neither reward nor penalise a tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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
