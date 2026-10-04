"""Shared types and the benchmark-owned taxonomy (PLAN.md §7.2).

The taxonomy belongs to the benchmark, not to any candidate. Candidates map their
own labels onto it through a fixed, published ``label_map`` in their manifest.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]
Polygon = Annotated[list[Point], Field(min_length=3, description="Vertices in the declared frame")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    """Base for every schema: unknown keys are errors, never silently dropped."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=False)


class PiiType(str, Enum):
    PERSON_NAME = "PERSON_NAME"
    SIGNATURE = "SIGNATURE"
    HANDWRITTEN_MARK = "HANDWRITTEN_MARK"  # identifying initials or handwriting regions
    IMAGE_PII = "IMAGE_PII"  # photos, avatars with initials, ...
    ADDRESS = "ADDRESS"
    LOCATION = "LOCATION"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    URL = "URL"
    DATE = "DATE"
    AGE = "AGE"
    ID_CLAIM = "ID_CLAIM"
    ID_LEGAL_REF = "ID_LEGAL_REF"
    ID_PROVIDER = "ID_PROVIDER"
    ID_GOVERNMENT = "ID_GOVERNMENT"  # Medicare, TFN, IHI, DVA, CRN, licence, passport
    ID_BUSINESS = "ID_BUSINESS"  # ABN, ACN
    ID_FINANCIAL = "ID_FINANCIAL"  # BSB, account
    ID_OTHER = "ID_OTHER"
    FILE_PATH = "FILE_PATH"
    ORGANISATION = "ORGANISATION"
    QUASI_IDENTIFIER = "QUASI_IDENTIFIER"  # occupation, physical measurements, family composition


class NameComponent(str, Enum):
    TITLE = "title"
    GIVEN = "given"
    MIDDLE = "middle"
    SURNAME = "surname"
    NICKNAME = "nickname"
    INITIALS = "initials"
    FORMER_SURNAME = "former_surname"
    SUFFIX = "suffix"


class EntityKind(str, Enum):
    PERSON = "person"
    ORGANISATION = "organisation"
    LOCATION = "location"
    OTHER = "other"


class Role(str, Enum):
    CLAIMANT = "claimant"
    FAMILY_MEMBER = "family_member"
    TREATING_CLINICIAN = "treating_clinician"
    EXAMINING_EXPERT = "examining_expert"
    LEGAL_REPRESENTATIVE = "legal_representative"
    INSURER_STAFF = "insurer_staff"
    EMPLOYER_STAFF = "employer_staff"
    WITNESS = "witness"
    ALLEGED_PERPETRATOR = "alleged_perpetrator"
    OFFICIAL = "official"  # e.g. Justice of the Peace
    ADMIN_STAFF = "admin_staff"  # e.g. typist initials in reference codes
    OTHER_PRIVATE = "other_private"
    ORG_EMPLOYER = "org_employer"
    ORG_CLINIC = "org_clinic"
    ORG_LAW_FIRM = "org_law_firm"
    ORG_INSURER = "org_insurer"
    ORG_SCHOOL = "org_school"
    ORG_GOVERNMENT = "org_government"
    ORG_OTHER = "org_other"
    UNKNOWN = "unknown"


class ClassifiedClass(str, Enum):
    """Non-PII content that matters for over-redaction (PLAN.md §1.1 finding 7, §3.4)."""

    MEDICAL_TERM = "MEDICAL_TERM"
    MEDICATION = "MEDICATION"
    MEDICAL_EPONYM = "MEDICAL_EPONYM"  # e.g. Hoover's sign
    LEGAL_CITATION = "LEGAL_CITATION"  # e.g. a case name
    HOMOGRAPH_COMMON_WORD = "HOMOGRAPH_COMMON_WORD"  # ordinary word equal to a surname in the doc
    BOILERPLATE = "BOILERPLATE"  # form instructions, example addresses
    PLACEHOLDER_PSEUDONYM = "PLACEHOLDER_PSEUDONYM"  # already de-identified, e.g. "Student A"
    OTHER_PROTECTED = "OTHER_PROTECTED"


class Requirement(str, Enum):
    MUST = "MUST"
    MAY = "MAY"
    MUST_NOT = "MUST_NOT"


class Action(str, Enum):
    REMOVE = "remove"
    REPLACE = "replace"
    EITHER = "either"


class WordSource(str, Enum):
    TEXT_LAYER = "text_layer"
    RASTER = "raster"
    VECTOR = "vector"
    HANDWRITING = "handwriting"


class Legibility(str, Enum):
    CLEAR = "clear"
    DEGRADED = "degraded"
    ILLEGIBLE = "illegible"


class RegionKind(str, Enum):
    SIGNATURE = "signature"
    INITIALS = "initials"
    HANDWRITING = "handwriting"
    PHOTO = "photo"
    AVATAR = "avatar"
    LOGO = "logo"
    STAMP = "stamp"
    OTHER = "other"
