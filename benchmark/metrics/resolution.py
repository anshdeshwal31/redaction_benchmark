"""Entity-resolution metrics over PII-bearing mentions (PLAN.md §3.5).

Key = GT mention -> GT entity. Response = GT mention -> predicted cluster, obtained by
mapping each detected GT mention to the cluster of its best-matching prediction (or,
for black-box pseudonymisers, inferred from shared replacement values). Standard
CoNLL metrics (MUC, B³, CEAF-e) plus LEA and pairwise links, with the false merges and
false splits listed explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

Assignment = dict[str, str]  # mention id -> entity / cluster id


def _clusters(a: Assignment) -> list[set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for m, c in a.items():
        out[c].add(m)
    return list(out.values())


def _prf(p: float | None, r: float | None) -> dict:
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": None if p is None else round(p, 4), "recall": None if r is None else round(r, 4),
            "f1": None if f is None else round(f, 4)}


def muc(key: Assignment, resp: Assignment) -> dict:
    def score(a: Assignment, b: Assignment) -> float | None:
        num = den = 0
        for k in _clusters(a):
            if len(k) < 2:
                continue
            parts = {b.get(m, f"__single_{m}") for m in k}
            num += len(k) - len(parts)
            den += len(k) - 1
        return num / den if den else None

    return _prf(score(resp, key), score(key, resp))


def b_cubed(key: Assignment, resp: Assignment) -> dict:
    common = [m for m in key if m in resp]
    if not common:
        return _prf(None, None)
    kc = {m: {x for x in common if key[x] == key[m]} for m in common}
    rc = {m: {x for x in common if resp[x] == resp[m]} for m in common}
    p = float(np.mean([len(kc[m] & rc[m]) / len(rc[m]) for m in common]))
    r = float(np.mean([len(kc[m] & rc[m]) / len(kc[m]) for m in common]))
    return _prf(p, r)


def ceaf_e(key: Assignment, resp: Assignment) -> dict:
    K, R = _clusters(key), _clusters(resp)
    if not K or not R:
        return _prf(None, None)
    sim = np.array([[2 * len(k & r) / (len(k) + len(r)) for r in R] for k in K])
    rows, cols = linear_sum_assignment(-sim)
    total = float(sim[rows, cols].sum())
    return _prf(total / len(R), total / len(K))


def lea(key: Assignment, resp: Assignment) -> dict:
    def link(n: int) -> float:
        return n * (n - 1) / 2 if n > 1 else 1.0

    def score(a: Assignment, b: Assignment) -> float | None:
        A, B = _clusters(a), _clusters(b)
        num = den = 0.0
        for k in A:
            if len(k) == 1:
                (m,) = k
                res_k = 1.0 if any(len(r) == 1 and m in r for r in B) else 0.0
            else:
                res_k = sum(link(len(k & r)) if len(k & r) > 1 else 0.0 for r in B) / link(len(k))
            num += len(k) * res_k
            den += len(k)
        return num / den if den else None

    return _prf(score(resp, key), score(key, resp))


@dataclass
class ResolutionResult:
    mentions_scored: int
    muc: dict
    b_cubed: dict
    ceaf_e: dict
    lea: dict
    pairwise: dict
    false_merges: list[dict]
    false_splits: list[dict]
    source: str  # "entities_artifact" | "inferred_from_pseudonyms"

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_resolution(key: Assignment, resp: Assignment, source: str = "entities_artifact") -> ResolutionResult:
    common = sorted(m for m in key if m in resp)
    k = {m: key[m] for m in common}
    r = {m: resp[m] for m in common}
    kp = {(a, b) for a, b in combinations(common, 2) if k[a] == k[b]}
    rp = {(a, b) for a, b in combinations(common, 2) if r[a] == r[b]}
    p = len(kp & rp) / len(rp) if rp else None
    rec = len(kp & rp) / len(kp) if kp else None
    merges: dict[tuple[str, str], list] = defaultdict(list)
    for a, b in sorted(rp - kp):
        merges[tuple(sorted((k[a], k[b])))].append((a, b))
    splits: dict[str, list] = defaultdict(list)
    for a, b in sorted(kp - rp):
        splits[k[a]].append((a, b))
    return ResolutionResult(
        mentions_scored=len(common), muc=muc(k, r), b_cubed=b_cubed(k, r), ceaf_e=ceaf_e(k, r), lea=lea(k, r),
        pairwise=_prf(p, rec),
        false_merges=[{"entities": list(e), "pairs": len(v)} for e, v in sorted(merges.items())],
        false_splits=[{"entity": e, "pairs": len(v)} for e, v in sorted(splits.items())],
        source=source,
    )


def response_from_artifacts(best_pred: dict[str, str | None], clusters: dict[str, str]) -> Assignment:
    """GT mention -> cluster of its best prediction (undetected mentions are left out)."""
    return {g: clusters[p] for g, p in best_pred.items() if p is not None and p in clusters}


def response_from_replacements(replacements: dict[str, dict[str, str]]) -> Assignment:
    """Infer clusters from pseudonyms: mentions sharing any replacement value of the same
    component kind are linked (union-find). ``replacements``: mention -> {component: value}."""
    parent: dict[str, str] = {m: m for m in replacements}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[tuple[str, str], str] = {}
    for m, comps in replacements.items():
        for comp, value in comps.items():
            key = (comp, value.casefold())
            if key in owner:
                parent[find(m)] = find(owner[key])
            else:
                owner[key] = m
    return {m: find(m) for m in replacements}
