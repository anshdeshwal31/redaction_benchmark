"""Named mock specs usable through the adapter CLI (and therefore through the runner)."""

from .mock import MockSpec

NAMED = {spec.name: spec for spec in [
    MockSpec("reference"),
    MockSpec("reference-raster", method="rasterize"),
    # Does everything the client's rules ask (the board's ceiling): a fake of the right kind for every
    # required item, one fake per person across a matter, drawn stand-ins for signatures and images,
    # and a searchable text layer. 300 DPI keeps the kept words undamaged at every evaluation DPI.
    MockSpec("reference-replace", method="rasterize", replace_all=True, pseudonym_key="entity", stand_ins=True,
             redacted_text_layer=True, raster_dpi=300),
    MockSpec("null", method="null"),
    MockSpec("overlay", method="overlay"),
    MockSpec("annotation", method="annotation"),
    MockSpec("translucent", method="translucent"),
    MockSpec("blur", method="blur"),
    MockSpec("incremental", incremental=True),
    MockSpec("keep-text-layer", method="rasterize", keep_text_layer=True),
    MockSpec("leak-metadata", leak_metadata=True),
    MockSpec("ocr-misses-names", ocr_drop_types=("PERSON_NAME",)),
    MockSpec("ocr-noise", ocr_char_error=0.1),
    MockSpec("boundary-shrink", boundary="shrink_last"),
    MockSpec("drops-headers", drop_conditions=("running_header", "running_footer")),
    MockSpec("type-swap", type_swap=(("PERSON_NAME", "ORGANISATION"),)),
    MockSpec("fp-protected", redact_protected=True),
    MockSpec("dpi-scale", scale=0.24),
    MockSpec("y-flip", y_flip=True),
    MockSpec("page-offset", page_offset=1),
    MockSpec("shift", shift_pt=(0, 6)),
    MockSpec("partial", partial=0.5),
    MockSpec("over-redact", dilate_pt=8),
    MockSpec("merge-same-given", merge_same_given=True),
    MockSpec("split-aliases", split_aliases=True),
    MockSpec("pseudonym-consistent", pseudonyms="consistent"),
    MockSpec("pseudonym-inconsistent", pseudonyms="inconsistent"),
    MockSpec("pseudonym-gender-flip", pseudonyms="gender_flip"),
    MockSpec("replace-contacts-valid", pseudonyms="consistent", replace_other="valid"),
    MockSpec("replace-contacts-invalid", pseudonyms="consistent", replace_other="invalid"),
    MockSpec("nondeterministic", seed=None, drop_fraction=0.3),
    MockSpec("corrupt", method="corrupt"),
    # eval-0.4: fakes drawn the way tools draw them (every required item faked, signatures given stand-ins).
    # Correct: fakes in place in the original's font and size; lines re-typeset with their fakes; whole
    # addresses faked as one flowing line each.
    MockSpec("replace-in-place", replace_all=True, stand_ins=True, fakes_in_place=True),
    MockSpec("retypeset", replace_all=True, stand_ins=True, retypeset=True),
    MockSpec("whole-address-replace", replace_all=True, stand_ins=True, fakes_in_place=True, address_whole=True),
    # Leaks: the originals re-typeset, a fake over the original, on a 30% white box, in a white box half a
    # line off, a text layer one line off, a locality kept (moved along its line), a locality erased.
    MockSpec("retypeset-keep", replace_all=True, retypeset_keep=True),
    MockSpec("overprint", replace_all=True, stand_ins=True, overprint=True),
    MockSpec("overprint-veil", replace_all=True, stand_ins=True, veil=0.3),
    MockSpec("shift-white", replace_all=True, stand_ins=True, shift_white=0.5),
    MockSpec("text-next-line", method="rasterize", replace_all=True, stand_ins=True, redacted_text_layer=True,
             raster_dpi=300, text_next_line=True),
    MockSpec("keeps-locality-moved", replace_all=True, stand_ins=True, fakes_in_place=True,
             keeps_locality_moved="line"),
    MockSpec("erases-locality", replace_all=True, stand_ins=True, fakes_in_place=True, erases_locality=True),
]}
