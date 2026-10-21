"""Oracle inputs materialised from GT for the controlled experiments (PLAN.md §4.2).

All artifacts are in the canonical ``display_pt`` frame. Only *required* (MUST) PII is
injected: the oracle answers "what a perfect upstream stage would hand you".
"""

from __future__ import annotations

from pathlib import Path

from ..geometry import as_poly, rect_poly
from ..gt.io import write_model
from ..gt.policy import resolve
from ..schemas import (
    Action,
    Cluster,
    EntitiesArtifact,
    GTDocument,
    InjectionKind,
    MentionsArtifact,
    PlanArtifact,
    PlanRegion,
    PolicyProfile,
    PredMention,
    RegionAnchor,
    Requirement,
    TextArtifact,
    TextPage,
    TextToken,
)


def _word_box(word, start, end):
    if word.char_polys and start is not None:
        import numpy as np

        pts = np.vstack([as_poly(p) for p in word.char_polys[start:end]])
        return rect_poly(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()).tolist()
    return word.poly


def oracle_text(gt: GTDocument) -> TextArtifact:
    pages = []
    for p in gt.pages:
        toks = [TextToken(id=w.id, text=w.text, poly=w.poly, order=w.order, line=w.line) for w in p.words]
        pages.append(TextPage(index=p.index, tokens=toks, detected_rotation=None))
    return TextArtifact(source="text_layer", pages=pages)


def _must_mentions(gt: GTDocument, policy: PolicyProfile):
    res = resolve(gt, policy)
    return [m for m in gt.mentions if res.mentions[m.id].requirement is Requirement.MUST]


def oracle_mentions(gt: GTDocument, policy: PolicyProfile) -> MentionsArtifact:
    words = gt.word_index()
    regions_idx = gt.region_index()
    out = []
    for m in _must_mentions(gt, policy):
        regions = []
        for seg in m.segments:
            for wid in seg.words:
                regions.append(RegionAnchor(page=seg.page, poly=_word_box(words[wid][1], seg.char_start, seg.char_end)))
            for rid in seg.regions:
                regions.append(RegionAnchor(page=seg.page, poly=regions_idx[rid][1].poly))
        tokens = [wid for seg in m.segments for wid in seg.words] or None
        out.append(PredMention(id=m.id, label=m.type.value, text=m.text or None, regions=regions, tokens=tokens))
    return MentionsArtifact(mentions=out)


def oracle_entities(gt: GTDocument, policy: PolicyProfile) -> EntitiesArtifact:
    clusters: dict[str, list[str]] = {}
    for m in _must_mentions(gt, policy):
        clusters.setdefault(m.entity or m.id, []).append(m.id)
    return EntitiesArtifact(clusters=[Cluster(id=k, mentions=v) for k, v in clusters.items()])


def oracle_regions(gt: GTDocument, policy: PolicyProfile) -> PlanArtifact:
    """GT regions with the policy's action: ``replace`` where a fake is required, else ``remove``.
    Each region names its mention, which the injected mentions.json describes."""
    res = resolve(gt, policy)
    regions = []
    for m in oracle_mentions(gt, policy).mentions:
        action = "replace" if res.mentions[m.id].action is Action.REPLACE else "remove"
        for k, r in enumerate(m.regions or []):
            regions.append(PlanRegion(id=f"{m.id}-{k}", page=r.page, poly=r.poly, action=action, mentions=[m.id]))
    return PlanArtifact(regions=regions)


FILENAMES = {InjectionKind.TEXT: "text.json", InjectionKind.MENTIONS: "mentions.json",
             InjectionKind.ENTITIES: "entities.json", InjectionKind.REGIONS: "plan.json"}


def materialise(kinds: list[InjectionKind], gt: GTDocument, policy: PolicyProfile, out_dir: Path) -> dict[str, Path]:
    makers = {InjectionKind.TEXT: lambda: oracle_text(gt),
              InjectionKind.MENTIONS: lambda: oracle_mentions(gt, policy),
              InjectionKind.ENTITIES: lambda: oracle_entities(gt, policy),
              InjectionKind.REGIONS: lambda: oracle_regions(gt, policy)}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in kinds:
        path = out_dir / FILENAMES[k]
        write_model(makers[k](), path)
        paths[k.value] = path
    return paths


# -- stage swapping: inject another candidate's artifacts --------------------------------------

_SOURCE_FILES = {InjectionKind.TEXT: ("text", "text.json"), InjectionKind.MENTIONS: ("mentions", "mentions.json"),
                 InjectionKind.ENTITIES: ("entities", "entities.json"), InjectionKind.REGIONS: ("plan", "plan.json")}


def _to_display(model, geoms):
    """Return a copy of a framed artifact with every polygon in display_pt."""
    from ..geometry import Frame

    frame = getattr(model, "frame", None)
    if frame is None or frame is Frame.DISPLAY_PT:
        return model
    dpi = model.dpi

    def conv(page, poly):
        return [tuple(pt) for pt in geoms[page].transform(as_poly(poly), frame, Frame.DISPLAY_PT, dpi).tolist()]

    data = model.model_dump(mode="python", by_alias=True)
    if "pages" in data:  # text
        for pg in data["pages"]:
            for tok in pg["tokens"]:
                tok["poly"] = conv(pg["index"], tok["poly"])
    if "mentions" in data:
        for m in data["mentions"]:
            for r in m.get("regions") or []:
                r["poly"] = conv(r["page"], r["poly"])
    if "regions" in data:  # plan
        for r in data["regions"]:
            r["poly"] = conv(r["page"], r["poly"])
    data["frame"] = Frame.DISPLAY_PT
    data["dpi"] = None
    return type(model).model_validate(data)


def materialise_from_run(kinds: list[InjectionKind], source_run_dir: Path, doc_id: str, input_pdf: Path,
                         out_dir: Path) -> dict[str, Path]:
    """Copy another run's stage artifacts for the same document (repeat 0) into the inject
    directory, normalised to display_pt and with labels mapped to benchmark types."""
    import json

    from ..geometry import load_page_geometries
    from ..schemas import CandidateManifest, PiiType
    from .artifacts import MODELS

    run = json.loads((source_run_dir / "run.json").read_text(encoding="utf-8"))
    src_manifest = CandidateManifest.model_validate(run["candidate"]["manifest"])
    item = next((i for i in run["items"] if i["doc_id"] == doc_id and i["repeat"] == 0), None)
    if item is None:
        raise FileNotFoundError(f"source run has no output for {doc_id}")
    workdir = source_run_dir / item["workdir"] / "out"
    geoms = load_page_geometries(input_pdf)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in kinds:
        kind_name, fname = _SOURCE_FILES[k]
        src = workdir / fname
        if not src.exists():
            raise FileNotFoundError(f"source run did not produce {fname} for {doc_id}")
        from ..schemas import ArtifactKind

        model = MODELS[ArtifactKind(kind_name)].model_validate(json.loads(src.read_text(encoding="utf-8")))
        model = _to_display(model, geoms)
        if k is InjectionKind.MENTIONS and src_manifest.label_map:
            for m in model.mentions:
                mapped = src_manifest.label_map.get(m.label)
                m.label = (mapped.value if isinstance(mapped, PiiType) else mapped) or m.label
        path = out_dir / FILENAMES[k]
        write_model(model, path)
        paths[k.value] = path
    return paths
