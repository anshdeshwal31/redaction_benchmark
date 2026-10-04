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
    regions = []
    for m in oracle_mentions(gt, policy).mentions:
        for k, r in enumerate(m.regions or []):
            regions.append(PlanRegion(id=f"{m.id}-{k}", page=r.page, poly=r.poly, action="remove", mentions=[m.id]))
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
