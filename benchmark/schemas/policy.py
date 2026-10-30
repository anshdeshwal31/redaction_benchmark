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

# Shared roles a policy may enable, per mention type ('ID' stands for every ID_* type); read from GT by
# benchmark.gt.shared_roles (docs/metrics.md, "Shared words"). 'state' and 'pc_prefix' go together.
SHARED_ROLE_NAMES: dict[str, tuple[str, ...]] = {
    "ADDRESS": ("box_kind", "designator", "street_type", "corner", "via", "dx", "country", "capital_city_phrase",
                "state", "pc_prefix"),
    "LOCATION": ("state", "pc_prefix", "country", "capital_city_phrase"),
    "ORGANISATION": ("legal_form",),
    "EMAIL": ("email_domain",),
    "ID": ("id_prefix",),
    "PHONE": ("phone_prefix",),
}


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


class SharedRoles(StrictModel):
    """Structure a realistic fake may keep (eval-0.4): in mentions of these types (and subtypes), the
    words and characters holding these roles are left out of every residual and never count as the
    replacement. Roles are read from GT by position (benchmark.gt.shared_roles)."""

    types: list[PiiType] = Field(min_length=1)
    subtypes: list[str] | None = Field(None, description="None: every subtype")
    roles: list[str] = Field(min_length=1, description="Role names of SHARED_ROLE_NAMES for every listed type")
    note: str = ""

    @model_validator(mode="after")
    def _check_roles(self) -> "SharedRoles":
        for t in self.types:
            key = "ID" if t.value.startswith("ID_") else t.value
            unknown = [r for r in self.roles if r not in SHARED_ROLE_NAMES.get(key, ())]
            if unknown:
                raise ValueError(f"shared roles {unknown} are not roles of {t.value}")
        if ("state" in self.roles) != ("pc_prefix" in self.roles):
            raise ValueError("'state' and 'pc_prefix' are shared together (the state/postcode pair)")
        return self


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
    shared_roles: list[SharedRoles] = Field([], description="Structure a realistic fake may keep, per mention "
                                                            "type (eval-0.4); none by default")

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
