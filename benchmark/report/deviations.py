"""Departures from the client's answers (policy deviations) as a report section.

Scoring copies the policy's deviations into ``scoring.json``; the run report and the
comparison board show them near the top, so nobody reads a score as the client's own rules
when the policy applied something else on purpose. A board can hold runs scored under
different policies: each departure then names the runs it applies to. Runs scored under
another policy than the one their tool was given get a warning of their own.
"""

from __future__ import annotations

import html
from pathlib import Path

TITLE = "Departures from the client's answers"
MISMATCH_TITLE = "Scored under another policy than the tool was given"


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def policy_label(scoring: dict) -> str:
    """The policy a run was scored under: name@version, or the policy file's name for results
    scored before the version was recorded."""
    if scoring.get("policy_id"):
        return scoring["policy_id"]
    path = scoring.get("policy")
    return f"{Path(path).name} (version not recorded)" if path else "unknown"


def collect(scorings: list[dict]) -> list[dict]:
    """Deviations of several scored runs, once per (policy, deviation id), in first-seen order,
    each with the runs scored under that policy."""
    out: dict[tuple[str, str], dict] = {}
    for sc in scorings:
        policy = policy_label(sc)
        for d in sc.get("deviations") or []:
            entry = out.setdefault((policy, d.get("id", "")), dict(d, policy_id=policy, runs=[]))
            if sc.get("run_id") and sc["run_id"] not in entry["runs"]:
                entry["runs"].append(sc["run_id"])
    return list(out.values())


def section(deviations: list[dict], policies: dict[str, list[str]] | None = None) -> str:
    """HTML for the deviations (every value escaped); empty when there are none. ``policies`` maps
    every policy on the page to its runs: when they differ, each departure names its runs."""
    if not deviations:
        return ""
    with_devs = sorted({d.get("policy_id") or "" for d in deviations} - {""})
    mixed = len(policies or {}) > 1
    if mixed:
        listing = "; ".join(f"{_esc(p)} ({_esc(', '.join(runs))})" for p, runs in policies.items())
        intro = (f"The runs on this page were scored under different policies: {listing}. Each departure below "
                 f"belongs to one policy and applies only to the runs named in its row; the other runs' scores "
                 f"follow their own policy.")
    elif len(with_devs) > 1:
        intro = (f"The policies of these runs depart from the client's answers on purpose in {len(deviations)} "
                 f"place(s); each run's scores follow its own policy, not the answers below.")
    else:
        name = f"Policy {_esc(with_devs[0])}" if with_devs else "The policy"
        intro = (f"{name} departs from the client's answers on purpose in {len(deviations)} place(s); the scores "
                 f"follow the policy, not the answers below.")
    by_policy = mixed or len(with_devs) > 1
    rows = "".join(
        f"<tr><td>{_esc(d.get('question'))}</td><td>{_esc(d.get('client_answer'))}</td>"
        f"<td>{_esc(d.get('applied'))}</td><td>{_esc(d.get('reason'))}</td>"
        f"<td>{_esc(d.get('effect') or '-')}</td>"
        + (f"<td>{_esc(d.get('policy_id'))}</td>" if by_policy else "")
        + (f"<td>{_esc(', '.join(d.get('runs') or []))}</td>" if mixed else "") + "</tr>"
        for d in deviations)
    return (f'<section class="dev"><h2>{_esc(TITLE)}</h2>'
            f"<p><b>Read the scores with these in mind.</b> {intro}</p>"
            f"<div class=wrap><table><tr><th>question</th><th>client's answer</th><th>applied instead</th>"
            f"<th>why</th><th>effect on the scores</th>" + ("<th>policy</th>" if by_policy else "")
            + ("<th>runs</th>" if mixed else "") + f"</tr>{rows}</table></div></section>")


def mismatch_section(scorings: list[dict]) -> str:
    """A warning for every run scored under another policy than the one its tool was given
    (scoring.json ``policy_warning``); empty when there is none."""
    warned = [(sc.get("run_id") or "", sc["policy_warning"]) for sc in scorings if sc.get("policy_warning")]
    if not warned:
        return ""
    named = len(scorings) > 1
    items = "".join(f"<li>{f'<b>{_esc(run)}</b>: ' if named and run else ''}{_esc(text)}</li>" for run, text in warned)
    return f'<section class="dev"><h2>{_esc(MISMATCH_TITLE)}</h2><ul>{items}</ul></section>'
