"""Declarative content for synthetic medico-legal pages. All names, numbers and
addresses are fictional.

Tokens are whitespace-delimited words. Tags attach a token to GT annotations:

- ``m:<mention>``            required word of a mention
- ``m:<mention>:<component>`` required word with a name component (given, surname, ...)
- ``opt:<mention>``          optional word of a mention (titles)
- ``c:<classified>``         word of a classified (protected) span

``span=(start, end)`` restricts the mention to part of the word (e.g. "Lizzie." -> "Lizzie").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import (
    AttributeValue,
    ClassifiedClass,
    EntityKind,
    GTEntity,
    GTRelation,
    PiiType,
    Role,
)


@dataclass(frozen=True)
class Tok:
    text: str
    tags: tuple[str, ...] = ()
    span: tuple[int, int] | None = None


def T(text: str, *tags: str, span: tuple[int, int] | None = None) -> Tok:
    return Tok(text, tuple(tags), span)


@dataclass(frozen=True)
class Line:
    x: float
    y: float  # baseline
    size: float
    toks: tuple[Tok, ...]


@dataclass(frozen=True)
class Squiggle:
    """A signature-like polyline; becomes a GT region and (optionally) a mention."""

    region_id: str
    points: tuple[tuple[float, float], ...]
    mention: str | None = None


@dataclass(frozen=True)
class MentionSpec:
    type: PiiType
    entity: str | None = None
    subtype: str | None = None
    form: str | None = None
    conditions: tuple[str, ...] = ()


@dataclass
class DocSpec:
    name: str
    width: float
    height: float
    pages: list[list[Line]]
    squiggles: dict[int, list[Squiggle]] = field(default_factory=dict)
    mentions: dict[str, MentionSpec] = field(default_factory=dict)
    classified: dict[str, ClassifiedClass] = field(default_factory=dict)
    entities: list[GTEntity] = field(default_factory=list)
    relations: list[GTRelation] = field(default_factory=list)
    matter: str | None = None


def L(x: float, y: float, size: float, *toks: Tok) -> Line:
    return Line(x, y, size, tuple(toks))


def _header(m: str) -> Line:
    return L(72, 40, 9, T("SMITHERS,", f"m:{m}:surname"), T("Jane", f"m:{m}:given"), T("|"), T("Claim"),
             T("CLM-48213", f"m:clm{m[-1]}"))


def _footer(m: str, date_m: str, page_no: int) -> Line:
    return L(72, 805, 8, T("SMITHERS", f"m:{m}:surname"), T("Jane", f"m:{m}:given"),
             T("(01.02.2026)", f"m:{date_m}", span=(1, 11)), T("Page"), T(str(page_no)), T("of"), T("2"))


def medico_legal_spec() -> DocSpec:
    """Two-page report exercising titles, sub-word spans, homographs, shared surnames,
    nicknames, inverted running headers, signatures, identifiers and protected content."""
    p0 = [
        _header("hdr0"),
        L(72, 80, 16, T("Independent"), T("Medical"), T("Report")),
        L(72, 120, 11, T("Claimant:"), T("Ms", "opt:m1"), T("Jane", "m:m1:given"), T("Elizabeth", "m:m1:middle"),
          T("Smithers", "m:m1:surname")),
        L(72, 140, 11, T("Date"), T("of"), T("birth:"), T("12/03/1970", "m:dob1")),
        L(72, 160, 11, T("Address:"), T("14", "m:addr1"), T("Example", "m:addr1"), T("Street,", "m:addr1"),
          T("Faketown", "m:addr1"), T("QLD", "m:addr1"), T("4000", "m:addr1")),
        L(72, 180, 11, T("Phone:"), T("0400", "m:ph1"), T("123", "m:ph1"), T("456", "m:ph1"), T("Email:"),
          T("jane.smithers@example.com", "m:em1")),
        L(72, 210, 11, T("Dr", "opt:m3"), T("Peter", "m:m3:given"), T("Strong", "m:m3:surname"), T("noted"), T("a"),
          T("strong", "c:hg1"), T("grip"), T("and"), T("absent"), T("Hoover's", "c:ep1"), T("sign.", "c:ep1")),
        L(72, 230, 11, T("Ms", "opt:m2"), T("Smithers", "m:m2:surname"), T("takes"), T("Sertraline", "c:med1"),
          T("50mg", "c:med1"), T("daily"), T("since"), T("04/05/2021.", "m:date1", span=(0, 10))),
        L(72, 250, 11, T("Her"), T("husband"), T("Tom", "m:m4:given"), T("Smithers", "m:m4:surname"), T("attended"),
          T("with"), T("her.")),
        L(72, 270, 11, T("She"), T("is"), T("known"), T("to"), T("friends"), T("as"),
          T("Lizzie.", "m:m5:nickname", span=(0, 6))),
        L(72, 360, 11, T("Signed:"), T("Dr", "opt:m7"), T("Peter", "m:m7:given"), T("Strong", "m:m7:surname")),
        _footer("ftr0", "date2", 1),
    ]
    p1 = [
        _header("hdr1"),
        L(72, 120, 11, T("Ms", "opt:m8"), T("Smithers", "m:m8:surname"), T("worked"), T("at"), T("Acme", "m:org1"),
          T("Widgets", "m:org1"), T("Pty", "m:org1"), T("Ltd", "m:org1"), T("for"), T("six"), T("years.")),
        L(72, 140, 11, T("Medicare"), T("number"), T("2123", "m:gov1"), T("45670", "m:gov1"), T("1", "m:gov1")),
        L(72, 160, 11, T("Her"), T("GP,"), T("Dr", "opt:m9"), T("Mia", "m:m9:given"), T("Hart,", "m:m9:surname", span=(0, 4)),
          T("prescribed"), T("a"), T("short"), T("course"), T("of"), T("physiotherapy.")),
        L(72, 180, 11, T("No"), T("further"), T("treatment"), T("is"), T("recommended.")),
        _footer("ftr1", "date3", 2),
    ]
    mentions = {
        "hdr0": MentionSpec(PiiType.PERSON_NAME, "E1", form="SURNAME, Given", conditions=("running_header",)),
        "hdr1": MentionSpec(PiiType.PERSON_NAME, "E1", form="SURNAME, Given", conditions=("running_header",)),
        "ftr0": MentionSpec(PiiType.PERSON_NAME, "E1", form="SURNAME Given", conditions=("running_footer",)),
        "ftr1": MentionSpec(PiiType.PERSON_NAME, "E1", form="SURNAME Given", conditions=("running_footer",)),
        "clm0": MentionSpec(PiiType.ID_CLAIM, "E1", conditions=("running_header",)),
        "clm1": MentionSpec(PiiType.ID_CLAIM, "E1", conditions=("running_header",)),
        "m1": MentionSpec(PiiType.PERSON_NAME, "E1", form="Title Given Middle Surname"),
        "m2": MentionSpec(PiiType.PERSON_NAME, "E1", form="Title Surname"),
        "m3": MentionSpec(PiiType.PERSON_NAME, "E2", form="Title Given Surname", conditions=("homograph_surname",)),
        "m4": MentionSpec(PiiType.PERSON_NAME, "E3", form="Given Surname", conditions=("shared_surname",)),
        "m5": MentionSpec(PiiType.PERSON_NAME, "E1", form="Nickname"),
        "m7": MentionSpec(PiiType.PERSON_NAME, "E2", form="Title Given Surname"),
        "m8": MentionSpec(PiiType.PERSON_NAME, "E1", form="Title Surname"),
        "m9": MentionSpec(PiiType.PERSON_NAME, "E5", form="Title Given Surname"),
        "sig1": MentionSpec(PiiType.SIGNATURE, "E2"),
        "dob1": MentionSpec(PiiType.DATE, "E1", subtype="dob"),
        "date1": MentionSpec(PiiType.DATE, None, subtype="event"),
        "date2": MentionSpec(PiiType.DATE, None, subtype="document", conditions=("running_footer",)),
        "date3": MentionSpec(PiiType.DATE, None, subtype="document", conditions=("running_footer",)),
        "addr1": MentionSpec(PiiType.ADDRESS, "E1", subtype="residential"),
        "ph1": MentionSpec(PiiType.PHONE, "E1", subtype="mobile"),
        "em1": MentionSpec(PiiType.EMAIL, "E1"),
        "org1": MentionSpec(PiiType.ORGANISATION, "E4"),
        "gov1": MentionSpec(PiiType.ID_GOVERNMENT, "E1", subtype="medicare"),
    }
    entities = [
        GTEntity(id="E1", kind=EntityKind.PERSON, role=Role.CLAIMANT, canonical_name="Jane Elizabeth Smithers",
                 attributes={"gender": AttributeValue(value="F", evidence=["title:Ms"])}, matter="MX"),
        GTEntity(id="E2", kind=EntityKind.PERSON, role=Role.EXAMINING_EXPERT, canonical_name="Peter Strong",
                 attributes={"gender": AttributeValue(value="M", evidence=["given-name"])}, matter="MX"),
        GTEntity(id="E3", kind=EntityKind.PERSON, role=Role.FAMILY_MEMBER, canonical_name="Tom Smithers",
                 attributes={"gender": AttributeValue(value="M", evidence=["relation:husband"])}, matter="MX"),
        GTEntity(id="E4", kind=EntityKind.ORGANISATION, role=Role.ORG_EMPLOYER, canonical_name="Acme Widgets Pty Ltd",
                 matter="MX"),
        GTEntity(id="E5", kind=EntityKind.PERSON, role=Role.TREATING_CLINICIAN, canonical_name="Mia Hart",
                 attributes={"gender": AttributeValue(value="F", evidence=["given-name"])}, matter="MX"),
    ]
    return DocSpec(
        name="medico_legal",
        width=595.0,
        height=842.0,
        pages=[p0, p1],
        squiggles={0: [Squiggle("sig1", ((80, 330), (95, 300), (110, 335), (130, 305), (150, 330), (175, 310),
                                         (200, 325)), mention="sig1")]},
        mentions=mentions,
        classified={"hg1": ClassifiedClass.HOMOGRAPH_COMMON_WORD, "ep1": ClassifiedClass.MEDICAL_EPONYM,
                    "med1": ClassifiedClass.MEDICATION},
        entities=entities,
        relations=[GTRelation(type="spouse_of", source="E3", target="E1"),
                   GTRelation(type="employs", source="E4", target="E1")],
        matter="MX",
    )


def narrative_spec() -> DocSpec:
    """Landscape narrative table page (content later rotated 90° inside the image).

    Exercises first-name-only references, two different people sharing a first name,
    and a spelling variant of one person's name.
    """
    p0 = [
        L(60, 60, 11, T("Date"), T("Event"), T("details")),
        L(60, 90, 11, T("June"), T("2021"), T("Helen", "m:n1:given"), T("Kane", "m:n1:surname"), T("organised"),
          T("a"), T("party"), T("with"), T("Kellie-Ann.", "m:n2:given", span=(0, 10))),
        L(60, 110, 11, T("Later"), T("Helen", "m:n3:given"), T("said"), T("Kelli-Ann", "m:n4:given"), T("was"),
          T("away.")),
        L(60, 130, 11, T("Helen", "m:n5:given"), T("Hockings,", "m:n5:surname", span=(0, 8)), T("from"), T("the"),
          T("library,"), T("helped"), T("me.")),
        L(60, 150, 11, T("Investigator"), T("Mary", "m:n6:given"), T("Crane", "m:n6:surname"), T("0411", "m:n7"),
          T("222", "m:n7"), T("333", "m:n7")),
    ]
    mentions = {
        "n1": MentionSpec(PiiType.PERSON_NAME, "N1", form="Given Surname", conditions=("shared_given_name",)),
        "n2": MentionSpec(PiiType.PERSON_NAME, "N2", form="Given", conditions=("hyphenated",)),
        "n3": MentionSpec(PiiType.PERSON_NAME, "N1", form="Given", conditions=("shared_given_name",)),
        "n4": MentionSpec(PiiType.PERSON_NAME, "N2", form="Given", conditions=("spelling_variant",)),
        "n5": MentionSpec(PiiType.PERSON_NAME, "N3", form="Given Surname", conditions=("shared_given_name",)),
        "n6": MentionSpec(PiiType.PERSON_NAME, "N4", form="Given Surname"),
        "n7": MentionSpec(PiiType.PHONE, "N4", subtype="mobile"),
    }
    entities = [
        GTEntity(id="N1", kind=EntityKind.PERSON, role=Role.ALLEGED_PERPETRATOR, canonical_name="Helen Kane",
                 attributes={"gender": AttributeValue(value="F", evidence=["given-name"])}),
        GTEntity(id="N2", kind=EntityKind.PERSON, role=Role.EMPLOYER_STAFF, canonical_name="Kellie-Ann",
                 attributes={"gender": AttributeValue(value="F", evidence=["given-name"])}),
        GTEntity(id="N3", kind=EntityKind.PERSON, role=Role.EMPLOYER_STAFF, canonical_name="Helen Hockings",
                 attributes={"gender": AttributeValue(value="F", evidence=["given-name"])}),
        GTEntity(id="N4", kind=EntityKind.PERSON, role=Role.WITNESS, canonical_name="Mary Crane",
                 attributes={"gender": AttributeValue(value="F", evidence=["given-name"])}),
    ]
    return DocSpec(name="narrative", width=842.0, height=595.0, pages=[p0], mentions=mentions, entities=entities)
