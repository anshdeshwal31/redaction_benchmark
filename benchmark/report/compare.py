"""Compare two scored runs mention by mention (PLAN.md §12, §13).

Transitions are keyed on GT ids (document, mention), so they are stable across
candidate versions: fixed, regressed, persistent (with category changes, e.g. a
detection miss that became a localization error is partial progress), stable safe.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..evaluate import HIDDEN_VALUES, SAFE_VALUES

# Transitions follow the policy's bar (a hidden but unreplaced mention still fails); leak
# rates follow privacy (hidden is not leaked).
SAFE = SAFE_VALUES
HIDDEN = HIDDEN_VALUES


def _must(results: Path) -> dict[tuple[str, str], dict]:
    rows = pq.read_table(results / "tables" / "mentions.parquet").to_pylist()
    return {(r["doc_id"], r["mention_id"]): r for r in rows
            if r["requirement"] == "MUST" and r["repeat"] == 0 and r["status"] != "NOT_EVALUATED"}


def compare(results_a: Path, results_b: Path, n_boot: int = 2000) -> dict:
    a, b = _must(results_a), _must(results_b)
    keys = sorted(set(a) & set(b))
    transitions = []
    for k in keys:
        sa, sb = a[k]["status"] in SAFE, b[k]["status"] in SAFE
        if not sa and sb:
            kind = "fixed"
        elif sa and not sb:
            kind = "regressed"
        elif sa and sb:
            kind = "stable_safe"
        else:
            kind = "persistent" if a[k].get("attr_category") == b[k].get("attr_category") else "category_changed"
        transitions.append({"doc_id": k[0], "mention_id": k[1], "kind": kind, "status_a": a[k]["status"],
                            "status_b": b[k]["status"], "category_a": a[k].get("attr_category"),
                            "category_b": b[k].get("attr_category")})
    # paired document-cluster bootstrap of the difference in leak rate (b - a)
    docs = sorted({k[0] for k in keys})
    per_doc = {d: [(a[k]["status"] not in HIDDEN, b[k]["status"] not in HIDDEN) for k in keys if k[0] == d]
               for d in docs}
    diffs = []
    if len(docs) >= 2:
        rng = np.random.default_rng(0)
        for _ in range(n_boot):
            sample = [per_doc[docs[i]] for i in rng.integers(0, len(docs), len(docs))]
            flat = [x for s in sample for x in s]
            diffs.append(np.mean([y for _, y in flat]) - np.mean([x for x, _ in flat]))
    leak_a = float(np.mean([a[k]["status"] not in HIDDEN for k in keys])) if keys else None
    leak_b = float(np.mean([b[k]["status"] not in HIDDEN for k in keys])) if keys else None
    fail_a = float(np.mean([a[k]["status"] not in SAFE for k in keys])) if keys else None
    fail_b = float(np.mean([b[k]["status"] not in SAFE for k in keys])) if keys else None
    return {
        "mentions_compared": len(keys),
        "only_in_a": len(set(a) - set(b)),
        "only_in_b": len(set(b) - set(a)),
        "counts": dict(Counter(t["kind"] for t in transitions)),
        "leak_rate_a": leak_a,
        "leak_rate_b": leak_b,
        "fail_rate_a": fail_a,
        "fail_rate_b": fail_b,
        "leak_rate_diff_ci95": (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))) if diffs else None,
        "transitions": [t for t in transitions if t["kind"] != "stable_safe"],
    }
