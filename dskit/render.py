"""Markdown rendering shared by all skills."""

from __future__ import annotations

from .evidence import Ledger

SEV_ORDER = {"critical": 0, "high": 1, "moderate": 2, "info": 3}


def finding_block(led: Ledger, f) -> str:
    lines = [f"**{f.id}** · {f.kind.upper()} · severity `{f.severity}` · confidence `{f.confidence}`  ",
             f"{f.statement}  "]
    if f.risk:
        lines.append(f"- *Risk:* {f.risk}")
    if f.possible_explanation:
        lines.append(f"- *Possible explanation:* {f.possible_explanation}")
    if f.recommended_check:
        lines.append(f"- *Recommended check:* {f.recommended_check}")
    ev = ", ".join(f"`{e}`" for e in f.evidence)
    vis = [v.id for v in led.visuals if set(v.evidence) & set(f.evidence)]
    lines.append(f"- *Evidence:* {ev}" + (f" · *Visual:* {', '.join(vis)}" if vis else ""))
    return "\n".join(lines)


def findings_section(led: Ledger, title: str, findings) -> str:
    findings = sorted(findings, key=lambda f: SEV_ORDER[f.severity])
    if not findings:
        return f"## {title}\n\nNone.\n"
    return f"## {title}\n\n" + "\n\n".join(finding_block(led, f) for f in findings) + "\n"


def visuals_section(led: Ledger) -> str:
    if not led.visuals:
        return "## Visual Evidence\n\nNo visual was generated: none would add information beyond the tables above.\n"
    out = ["## Visual Evidence\n"]
    for v in led.visuals:
        out.append(f"**{v.id}** — {v.purpose}  \nEvidence: {', '.join(f'`{e}`' for e in v.evidence)}\n\n"
                   f"![{v.id}]({v.path})\n")
    return "\n".join(out)


def evidence_section(led: Ledger) -> str:
    rows = ["## Evidence\n", "| ID | Source | Location | Observation | Method | Confidence |",
            "|---|---|---|---|---|---|"]
    for e in led.evidence:
        rows.append(f"| {e.id} | {e.source} | {e.location} | {_esc(e.observation)} | {_esc(e.method)} | {e.confidence} |")
    return "\n".join(rows) + "\n"


def list_section(title: str, items: list[str], empty: str = "None recorded.") -> str:
    if not items:
        return f"## {title}\n\n{empty}\n"
    return f"## {title}\n\n" + "\n".join(f"- {i}" for i in items) + "\n"


def recommendations_section(led: Ledger, title: str = "Recommended Next Actions") -> str:
    if not led.recommendations:
        return f"## {title}\n\nNone.\n"
    return f"## {title}\n\n" + "\n".join(
        f"- **{r.id}** {r.action} *(based on {', '.join(r.based_on)})*" for r in led.recommendations) + "\n"


def decision_section(led: Ledger, title: str = "Decision") -> str:
    why = "\n".join(f"- {r}" for r in led.decision_rationale)
    return f"## {title}\n\n**`{led.decision}`** (confidence: `{led.decision_confidence}`)\n\n{why}\n"


def _esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
