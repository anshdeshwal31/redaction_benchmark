# Address case table

`docs/address_cases.json` is one tool-neutral table of fictional address cases with their expected
results under the project owner's rulings of 2026-10-22. It is the address policy by example: tools 1
and 2 copy it into their tests, the benchmark's GT splitter and role reader are tested against it, and
the annotation guidelines follow it.

- Version 0.3.0: 293 cases, 288 core and 5 stretch. 245 cases hold 265 address units; the other 48
  hold only kept places and look-alikes.
- Every case is fictional. No real ground truth was read for this table.
- The table was built in the address work of 2026-10-22 and is copied here unchanged (line endings
  apart). Edit it there and copy it again, so the two never drift.

## Decisions applied

The owner's rulings of 2026-10-22:

1. **Same-state fakes.** The state word stays as written. The fake suburb and postcode come from a real
   locality of that state. A setting can switch to other states.
2. **Place-only addresses: option (c).** A suburb, state and postcode written together with no street
   line is an address, a place-only unit.
   - The exception: right after a label or phrase that names another kind of place (birth, injury,
     accident or incident, declaration), the place stays, postcode included.
   - A place written with a postcode after a court's or hospital's name is replaced, and the name stays
     outside the unit. So places after a hospital, ward, court or occupation, or after 'Seen <date>',
     are replaced, and so is an accident place written as a full street address.
3. **Streets in prose: option (a).**
   - A street named in running text with no house number is kept.
   - So is a street with a suburb but no house number, state or postcode ('injured on Victoria Road,
     Ryde'): both words stay, with no address unit.
   - The same street followed by its suburb, state and postcode ('injured on Victoria Road, Ryde NSW
     2112') is a street name followed by its locality, an address in every option.
   - Only a place that directly follows the street is its locality: an optional comma, then the
     suburb, state and postcode. A place named later in the sentence, after other words, is not: the
     street stays kept, and the place follows decision 2 on its own.
4. **Tool 1's baselines.** They get fixed new versions labelled post-exposure. This changes no expected
   result here.
5. **Tool 2's profiles.** All profiles replace whole addresses as one unit. Units, roles, structure
   words, look-alikes and same-fake groups hold in every profile; `kept_places` hold in the client
   profile.
6. **Joel's tool.** It gets no changes and is graded on its own redacted text. Its prototype cases are
   used as inputs only.
7. **The official list.** The ABS suburb and postcode list (suburb list v2) is stored in tools 1 and 2
   as both the gazetteer and the fake pool.

Settled later on 2026-10-22: **overseas addresses** ('12 Queen Street, Auckland 1010, New Zealand')
are replaced whole, and the country word stays, like 'Australia'. The fake keeps the shape with no
foreign gazetteer: every number, the street name and type, the town and every postcode character
change. No Australian state is ever inferred from a postcode followed by a foreign country. The table
marks such units `overseas`; its 4 overseas cases are stretch.

Earlier rulings that still hold: a suburb, town, state or postcode on its own is kept; role words such
as 'the claimant' stay; every address of anyone is replaced, persons' and organisations' alike; the
whole-address rule is a recorded deviation from the client's answer (Joel's Q4); legislation and
guideline citations are never addresses. Court, hospital and public-body names next to a place follow
the policy's organisations rule, never the address rule. Traditional Place name lines and care-of
lines stay outside the address unit.

## Format

The top-level keys are `table`, `version`, `date`, `decisions`, `positions`, `roles`,
`structure_roles`, `fake_constraints` (the rules every fake follows across roles and units),
`own_rules` (the policy rule each look-alike and neighbour kind follows), `gazetteer`, `tiers`,
`families`, `coverage`, `case_count`, `cases` and `matters`.

**Positions** are `[line, word]`: the index of the line in the case's `lines`, then the index of the
word in that line's text split on whitespace. Punctuation stays attached to its word. Both count
from 0.

**A case:**

| Field | Meaning |
|---|---|
| `id` | Unique kebab-case name. |
| `family`, `also` | The coverage family the case belongs to, and the other families it also exercises. |
| `tier` | `core` or `stretch`. A stretch case follows from the owner's decisions, but the tools' planned rules do not reach it yet. |
| `description`, `notes` | What the case tests, and why its expected result is what it is where that is not obvious. |
| `sources` | Where the case came from. |
| `layout_matters` | True when page, row, column, band, box or gap tags carry meaning. |
| `lines` | The document, one entry per line or box, in reading order. |
| `expect` | The expected results. |

**A line:**

| Field | Meaning |
|---|---|
| `text` | The words, split on whitespace. |
| `page` | Page index, from 0. |
| `row` | Vertical slot on its page. Lines with the same row sit side by side (form boxes, table cells, a date in the right column). A skipped row number is a paragraph gap. |
| `x0`, `x1` | Left edge, in points on an A4 page (default 72), and right edge (form boxes only). |
| `band` | `body`, `letterhead`, `header` or `footer`. |
| `box` | `label` (a form label box), `value` (a form value box) or `cell` (a table cell). |
| `drawn` | `outlines`: the line is drawn as vector outlines with no text layer. |

**`expect`:**

- `addresses`: one entry per address unit.
  - `id` (`A1`, `A2`, ...) and `kind`: `street`, `street_numberless`, `corner`, `box`, `dx` or
    `place_only`.
  - `street_line` and `locality`: the unit's word positions, split as the GT splits them. The street
    line is every word from the unit's first word to the start of its place. The locality is the
    suburb, state, postcode and country, or the 'IN YOUR CAPITAL CITY' line that stands in for them.
    A DX line is all street line, and a place-only unit is all locality.
  - `structure`: the words that stay unchanged (roles listed in `structure_roles`).
  - `partial_structure`: words that are partly structure. Their `keep` letters stay and their
    `change` part follows its rule: a glued 'NSW2750' (keep 'NSW', change '2750'), or a level code in a
    hotel-style number 'B02/53' (keep 'B', change '02/53').
  - `words`: `[line, word, text, role]` for every word of the unit, in order.
  - `lines`, `pages`, `form`: how many lines the unit spans, which pages, and whether it is built from
    form value boxes.
  - `repeat_of`: the unit this one repeats elsewhere in the document.
  - `overseas`: true when the unit ends in a country other than Australia.
  - `splitter`: the GT splitter's result for this unit when the table was built (2026-10-22, before the
    splitter fixes of eval-0.4); the tests below compare with the current splitter instead.
- `kept_places`: places on their own that the client keeps. They must stay unchanged. Kinds: `suburb`,
  `city`, `state`, `postcode`, `country`, `suburb_and_state`, `place_of_birth`, `place_of_injury`,
  `place_of_accident`, `place_of_incident`, `place_of_declaration` and `street_in_prose`.
- `lookalikes`: words shaped like address parts that are not addresses (legislation, occupations,
  hospitals, courts, dates, clinical phrases and abbreviations, amounts, phones and others). No
  address unit may include them, and the address rule never changes them. Each carries `follows`: the
  policy rule that decides its final text (see `own_rules`).
- `not_address`: neighbours that belong to no unit, such as labels, person names, organisations,
  phones, ABNs, URLs, dates and page furniture. Each carries `follows`.
- `same_fake`: groups of unit ids that must get the same fake. Two units in a case get the same fake
  only when a group lists them both. **Matters** group cases into one matter, with their own
  `same_fake` and `different_fake` lists of `case/unit` names.

## Roles

| Role | Part | Stays | What the fake does |
|---|---|---|---|
| `building` | street | no | Building or property name words, the building word included ('Westpoint Tower', 'Kilbride Station'): pool words. |
| `designator` | street | yes | Australia Post's unit and floor words and their abbreviations, plus the building designators (list below). |
| `unit_number` | street | no | The number or letter after a unit, building, block, tower or wing word: the number rule. |
| `level_number` | street | no | The number after 'Level', 'Floor' or 'Fl', or before 'Floor' (an ordinal keeps its suffix): another realistic level, with no digit equal to the original's at its position. |
| `number` | street | no | House number, ranges and letter suffixes included: the number rule. |
| `unit_number+number` | street | no | A '5/3' word: both numbers follow the number rule, and the slash stays. A level-type code before the unit number ('B02/53') stays. |
| `street_name` | street | no | A pool name in the original's case. |
| `street_type` | street | no | Another canonical type in the original's style. AS 4590 and AS/NZS 4819 abbreviations map to one canonical type ('Cres' and 'Cr' are both Crescent). |
| `street_suffix` | street | no | A directional after the type ('Smith Street North'): another directional in the original's style. |
| `corner` | street | yes | 'Cnr' and '&', and their spelled-out forms 'Corner', 'of' and 'and' inside a corner form. |
| `box_kind` | street | yes | Australia Post's postal delivery types (list below). |
| `box_number` | street | no | The number rule. |
| `box_extra` | street | no | Words between the box number and the place that are not a road ('City East'). A road after the box number ('RMB 4600 Lakes Road') takes `street_name` and `street_type`. |
| `via` | street | yes | 'via' between a property name and its place ('Kilbride Station via Berrima'). |
| `dx` | street | yes | 'DX'. |
| `dx_number` | street | no | The number rule. |
| `dx_exchange` | street | no | A fake locality of the same state. |
| `suburb` | locality | no | A real locality of the same state and class. In an overseas unit it is the town. |
| `state` | locality | yes | Stays exactly as written, a doubled state ('NSW NSW') included. |
| `postcode` | locality | no | The fake suburb's own postcode. In an overseas unit every character changes. |
| `state+postcode` | locality | letters only | A glued 'NSW2750': the letters stay, the digits follow the postcode rule, and the fake stays glued. |
| `country` | locality | yes | 'Australia', or the country that ends an overseas address. |
| `capital_city_phrase` | locality | yes | 'IN YOUR CAPITAL CITY' written in place of a locality. It names no place. |

**Designators.** Unit types: Unit (U), Apartment (Apt), Flat (F), Factory (FY), House (HSE), Office
(OFF), Penthouse (PTHS), Room (RM), Shop, Studio (STU), Suite (SE, Ste), Townhouse (TNHS), Villa,
Warehouse (WE), Lot. Floor and level types: Level (L, Lvl), Floor (FL), Ground Floor (G), Lower Ground
Floor (LG), Upper Ground Floor (UG), Mezzanine (M), Basement (B); Ground, Lower Ground, Upper Ground,
Mezzanine and Basement need no number. Building designators: 'Building', 'Block', 'Tower' and 'Wing'
before a letter or number designator ('Building C', 'Tower 2'); a building name ('Westpoint Tower')
stays identifying.

**Box kinds.** PO Box, P.O. Box, Post Office Box, GPO Box, Locked Bag, Locked Mail Bag, Private Bag,
Private Mail Bag, Reply Paid, RMB (R.M.B.), RSD, CMB, PMB, MS, RMS, CMA, CPA, Care PO. Care PO, CMA, CPA
and CMB may stand without a number. MS, RMS, CMA and CPA count only in upper case, so 'diagnosed with
MS 3 years ago' and 'Ms Jane Citizen' are not boxes.

**Postcodes.** The original postcode's leading digits that match its state's prefix may stay in a
fake, matched from the left: '2' NSW and ACT, '3' VIC, '4' QLD, '5' SA, '6' WA, '7' TAS, '08' NT.
Every later digit differs at its position. An original outside its state's usual prefix (a PO-box
block such as NSW 1300, or a cross-border code) shares no digit.

## Using the table in tests

`tests/address_cases.py` loads the table and builds fictional GT from a case's layout: word ids
`w{line}_{word}`, one line and block id per case line, word boxes from `row` and `x0`.

- `build_gt(case)` makes one unsplit ADDRESS mention per unit, one segment per run of adjacent words
  on a line, as an annotator would before `rb gt split-addresses`.
  `tests/test_address_split.py::test_splitter_agrees_with_the_case_table` runs the splitter on it and
  compares the outcome (split, locality only, left whole) and the street-line words with the table.
  The units the splitter cannot reach yet are listed in the test (`SPLIT_ALLOWED`: the overseas
  units, 'IN YOUR CAPITAL CITY', street lines followed by a suburb alone, an OCR letter in the
  postcode, and a directional after the street type that only the postcode's owner decides); each of
  them must be flagged for the GT review.
- A split unit becomes a street line (ADDRESS, condition `address_street_line`) and a locality
  (LOCATION/`locality`, condition `address_locality`) with the id `<street id>-loc`.
- Kept places become LOCATION mentions with a place subtype. Under policy 1.1.1 the places-kept rule
  (MUST_NOT) knows only suburb, city, region, state, country, postcode and place_of_declaration; any
  other subtype falls to `default_mention` (MAY). So `place_of_birth`, `place_of_injury`,
  `place_of_accident` and `place_of_incident` become one LOCATION mention with subtype `suburb`
  (`city` for a capital) over the whole place, postcode included; `suburb_and_state` becomes `suburb`,
  or `suburb` plus `state`; `street_in_prose` is graded MAY.
- Look-alikes carry no address mention. Their own mentions follow their `follows` rule.

**What a tool's test asserts per case:**

1. **Units.** Each expected unit comes out as exactly one unit, with exactly its street-line and
   locality words. No other unit exists, and no unit touches a kept place, a look-alike or a
   `not_address` word.
2. **Replacement.** Every unit word outside `structure` changes under the fake's constraints.
   Structure words and the `keep` part of `partial_structure` words stay. Kept places and labels are
   unchanged.
3. **Consistency.** The `same_fake` groups and the matters hold, and units outside a group get
   different whole fakes. No fake word or number repeats an identifying word or number of any unit in
   the case.
4. **Verification.** Search the whole output for each unit's targets: the unit's text, each of its
   lines, its number with its street name, its box kind with its number, its suburb with its state
   and postcode, and its suburb, street-name and building words as whole tokens (never connecting
   words such as 'of' or 'the', never bare numbers, street types or lone postcodes). Kept places,
   labels, and look-alikes whose own rule keeps them are exempt; any other hit fails.
