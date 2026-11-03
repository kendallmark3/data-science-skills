"""Skill 4 - Evidence-Backed Reporting.

Builds a report from evidence that already exists. It never adds evidence of its
own: a claim without resolvable evidence is listed as unsupported, not written up
as a finding; a claim is never stated more confidently than its best evidence;
conflicting claims are shown side by side.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import render, viz
from .evidence import CONFIDENCE, Ledger, StopCondition, load_run, max_confidence, min_confidence

SKILL_LABEL = {"data-profiling-quality": "Data profiling", "experiment-analysis": "Experiment analysis",
               "model-evaluation-review": "Model evaluation", "evidence-backed-reporting": "Supplied claims"}
KEY_INFO = {"primary_effect", "practical_significance", "production_projection"}
PROSE = {"high": "The evidence shows", "medium": "The evidence suggests", "low": "Weak evidence indicates"}


def build_report(out_dir: str | Path, runs: list[str | Path] = (), claims_file: str | Path | None = None,
                 intent: str | None = None) -> Ledger:
    out = Path(out_dir)
    pool: dict[str, dict] = {}
    candidates, run_visuals, run_recs, sources = [], [], [], []

    for run in runs:
        rd = Path(run)
        if not (rd / "findings.json").exists():
            raise StopCondition(f"{rd} is not a skill output directory (no findings.json)")
        fj, ev = load_run(rd)
        for e in ev:
            pool.setdefault(e["id"], e)
        sources.append({"skill": fj["skill"], "dir": str(rd), "decision": fj.get("decision"),
                        "confidence": fj.get("decision_confidence"), "intent": fj.get("intent"),
                        "status": fj.get("status", "COMPLETE"), "gaps": fj.get("gaps", [])})
        for f in fj["findings"]:
            if f["severity"] != "info" or f["category"] in KEY_INFO:
                candidates.append({"source_id": f["id"], "claim": f["statement"], "evidence": f["evidence"],
                                   "confidence": f["confidence"], "kind": f["kind"], "severity": f["severity"],
                                   "skill": fj["skill"], "category": f["category"]})
        run_visuals += [dict(v, dir=str(rd), skill=fj["skill"]) for v in fj.get("visuals", [])]
        run_recs += [dict(r, skill=fj["skill"]) for r in fj.get("recommendations", [])]

    claims_doc, cdir = {}, None
    if claims_file:
        cp = Path(claims_file)
        if not cp.exists():
            raise StopCondition(f"Claims file not found: {cp}")
        cdir = cp.parent
        claims_doc = yaml.safe_load(cp.read_text()) or {}
        for ef in claims_doc.get("evidence_files", []):
            for e in json.loads((cdir / ef).read_text()):
                pool.setdefault(e["id"], dict(e, source=e["source"]))
        for c in claims_doc.get("claims", []):
            candidates.append({"source_id": c["id"], "claim": c["claim"], "evidence": c.get("evidence", []),
                               "confidence": c.get("confidence", "medium"), "kind": c.get("kind", "observation"),
                               "severity": c.get("severity", "moderate"), "skill": "evidence-backed-reporting",
                               "category": c.get("topic", "claim"), "topic": c.get("topic"), "direction": c.get("direction")})
    if not candidates and not sources:
        raise StopCondition("Nothing to report: no skill outputs and no claims were supplied.")

    led = Ledger("evidence-backed-reporting", "RP",
                 intent=intent or claims_doc.get("intent") or "; ".join(s["intent"] for s in sources if s["intent"]))
    led.summary["sources"] = sources

    # ---- validate every candidate claim against the evidence pool ---------------
    unsupported, conflicts, trace, src_to_rp = [], [], {}, {}
    topics: dict[str, list[dict]] = {}
    for c in candidates:
        if c.get("topic") and c.get("direction"):
            topics.setdefault(c["topic"], []).append(c)
    conflicted = {id(c) for group in topics.values() if len({x["direction"] for x in group}) > 1 for c in group}

    for c in candidates:
        found = [e for e in c["evidence"] if e in pool]
        missing = [e for e in c["evidence"] if e not in pool]
        if not found:
            reason = "no evidence cited" if not c["evidence"] else f"cited evidence {missing} does not exist in any supplied source"
            unsupported.append({"id": c["source_id"], "claim": c["claim"], "reason": reason, "stated_confidence": c["confidence"]})
            continue
        for e in found:
            led.import_evidence(pool[e])
        ceiling = max_confidence(*(pool[e].get("confidence", "medium") for e in found))
        conf = min_confidence(c["confidence"], ceiling)
        notes = []
        if conf != c["confidence"]:
            notes.append(f"confidence reduced from {c['confidence']} to {conf}: the strongest supporting evidence is {ceiling}")
        if missing:
            notes.append(f"cited evidence {missing} could not be found and was ignored")
        if id(c) in conflicted:
            conf = "low"
            notes.append(f"conflicts with another claim on topic '{c['topic']}'")
        sev = c["severity"]
        fid = led.add_finding(c["claim"], found, kind=c["kind"], category=c["category"], severity=sev, confidence=conf)
        src_to_rp[c["source_id"]] = fid
        trace[fid] = {"source_finding": c["source_id"], "source_skill": c["skill"], "notes": notes,
                      "evidence": [{"id": e, "source": pool[e]["source"], "location": pool[e]["location"]} for e in found]}
        if id(c) in conflicted:
            conflicts.append(fid)
    led.summary.update(traceability=trace, unsupported_claims=unsupported)
    led.summary["conflicts"] = [{"topic": t, "findings": [src_to_rp[x["source_id"]] for x in g if x["source_id"] in src_to_rp],
                                 "directions": {src_to_rp.get(x["source_id"], x["source_id"]): x["direction"] for x in g}}
                                for t, g in topics.items() if len({x["direction"] for x in g}) > 1]

    # ---- curate visuals --------------------------------------------------------
    excluded_visuals = []
    vis_dir = out / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    used_ev = {e for f in led.findings for e in f.evidence}
    claim_visuals = [dict(v, dir=str(cdir), skill="evidence-backed-reporting") for v in claims_doc.get("visuals", [])]
    for v in run_visuals + claim_visuals:
        srcfile = Path(v["dir"]) / v["path"]
        missing = [e for e in v["evidence"] if e not in pool]
        if missing:
            excluded_visuals.append({"id": v["id"], "path": v["path"], "reason": f"cites evidence {missing} that does not exist"})
            continue
        if not srcfile.exists():
            excluded_visuals.append({"id": v["id"], "path": v["path"], "reason": "image file not found"})
            continue
        if not set(v["evidence"]) & used_ev:
            continue  # valid, but supports nothing in this report
        for e in v["evidence"]:
            led.import_evidence(pool[e])
        dest = f"{v['skill'].split('-')[0]}-{Path(v['path']).name}"
        shutil.copy(srcfile, vis_dir / dest)
        led.add_visual(v.get("type", "supplied_visual"), f"visuals/{dest}", v["evidence"],
                       f"{v['purpose']} (from {SKILL_LABEL.get(v['skill'], v['skill'])}, {v['id']})")
    led.summary["excluded_visuals"] = excluded_visuals
    if len(sources) >= 2:
        _decision_summary_visual(led, out, sources)

    # ---- recommendations: only those whose basis survived ---------------------------
    withheld = []
    for r in run_recs:
        basis = [src_to_rp[b] for b in r["based_on"] if b in src_to_rp]
        if basis:
            led.recommend(f"{r['action']} *({SKILL_LABEL.get(r['skill'], r['skill'])})*", basis)
    unsupported_ids = {u["id"] for u in unsupported}
    for r in claims_doc.get("recommendations", []):
        bad = [b for b in r["based_on"] if b in unsupported_ids or b not in src_to_rp]
        if bad:
            withheld.append({"action": r["action"], "reason": f"rests on unsupported claim(s) {bad}"})
            continue
        basis = [src_to_rp[b] for b in r["based_on"]]
        weak = [b for b in basis if next(f for f in led.findings if f.id == b).confidence == "low"]
        led.recommend(r["action"] + (" *(caveat: rests on low-confidence evidence)*" if weak else ""), basis)
    led.summary["withheld_recommendations"] = withheld

    for s in sources:
        led.gaps += [f"[{SKILL_LABEL.get(s['skill'], s['skill'])}] {g}" for g in s["gaps"]]
    if unsupported:
        led.gaps.append(f"{len(unsupported)} claim(s) could not be traced to evidence and were excluded from the findings.")
    confs = [s["confidence"] for s in sources if s.get("confidence") in CONFIDENCE]
    led.decision = "REPORTED_WITH_EXCLUSIONS" if (unsupported or excluded_visuals or withheld) else "REPORTED"
    led.decision_confidence = min_confidence(*confs) if confs else ("low" if conflicts else "medium")
    if conflicts and led.decision_confidence == "high":
        led.decision_confidence = "medium"
    led.write(out, _report(led))
    return led


def _decision_summary_visual(led, out, sources):
    sev = ["critical", "high", "moderate"]
    colors = {"critical": viz.CRITICAL, "high": viz.SERIOUS, "moderate": viz.WARNING}
    rows = []
    for s in sources:
        fs = [f for fid, t in led.summary["traceability"].items() if t["source_skill"] == s["skill"]
              for f in led.findings if f.id == fid]
        rows.append((f"{SKILL_LABEL.get(s['skill'], s['skill'])}\n{s['decision']}", {k: sum(f.severity == k for f in fs) for k in sev}, fs))
    fig, ax = viz.figure(7.5, 0.7 * len(rows) + 1.4)
    left = np.zeros(len(rows))
    for k in sev:
        vals = np.array([r[1][k] for r in rows])
        ax.barh(range(len(rows)), vals, left=left, color=colors[k], label=k, height=0.55, edgecolor=viz.SURFACE, linewidth=2)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v:
                ax.text(l0 + v / 2, i, str(v), ha="center", va="center", fontsize=9,
                        color=viz.INK if k == "moderate" else "#ffffff")
        left += vals
    for i, r in enumerate(rows):
        if not any(r[1].values()):
            ax.text(0.1, i, "no moderate-or-worse findings", va="center", fontsize=9, color=viz.MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("findings carried into this report")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Decision and risk summary by source")
    ev = sorted({e for r in rows for f in r[2] for e in f.evidence})
    path = viz.save(fig, out, "decision-summary.png", f"Counts of moderate+ findings per source skill; {len(ev)} evidence items "
                    "(listed in findings.json).")
    if ev:
        led.add_visual("decision_summary", path, ev, "Summarize each source's decision and the severity of what it found")


def _finding_md(led, f) -> str:
    t = led.summary["traceability"][f.id]
    lead = "Inference" if f.kind == "inference" else PROSE[f.confidence]
    lines = [f"### {f.id} · confidence `{f.confidence}` · {f.kind}",
             f"**Claim:** {lead}: {f.statement[0].lower() + f.statement[1:] if lead != 'Inference' else f.statement}  ",
             f"**Source:** {SKILL_LABEL.get(t['source_skill'], t['source_skill'])} finding `{t['source_finding']}`  ",
             "**Evidence:**"]
    for e in f.evidence:
        ev = led.get_evidence(e)
        lines.append(f"- `{e}` — {ev.source}, {ev.location}: {ev.observation} *({ev.method}; {ev.confidence})*")
    vis = [v for v in led.visuals if set(v.evidence) & set(f.evidence) and v.type != "decision_summary"]
    if vis:
        lines.append("**Visual:** " + ", ".join(f"[{v.id}]({v.path})" for v in vis))
    for n in t["notes"]:
        lines.append(f"> ⚠ {n}")
    return "\n".join(lines)


def _report(led) -> str:
    F = led.findings
    S = led.summary
    key = sorted([f for f in F if f.id not in {c for x in S["conflicts"] for c in x["findings"]}],
                 key=lambda f: (render.SEV_ORDER[f.severity], -CONFIDENCE.index(f.confidence)))
    exec_lines = [f"- **{SKILL_LABEL.get(s['skill'], s['skill'])}:** `{s['decision']}` (confidence `{s['confidence']}`)" for s in S["sources"]]
    top = key[:5]
    exec_lines += [f"- {f.id} ({f.severity}, {f.confidence} confidence): {f.statement}" for f in top]
    if S["unsupported_claims"]:
        exec_lines.append(f"- {len(S['unsupported_claims'])} claim(s) were excluded because no evidence supports them.")
    if S["conflicts"]:
        exec_lines.append(f"- {len(S['conflicts'])} topic(s) have conflicting evidence; see Conflicts.")
    conflicts_md = "\n\n".join(
        f"**Topic `{c['topic']}`** — the evidence disagrees:\n" + "\n".join(
            f"- {fid} says *{d}*: {next(f.statement for f in F if f.id == fid)} (evidence: "
            f"{', '.join(next(f.evidence for f in F if f.id == fid))}, confidence `low`)" for fid, d in c["directions"].items())
        + "\n\nNeither claim is adopted. Resolve with a measurement both sides accept." for c in S["conflicts"]) or "None."
    unsup = "\n".join(f"- `{u['id']}` — \"{u['claim']}\" — **not presented as a finding**: {u['reason']}." for u in S["unsupported_claims"]) or "None."
    exv = "\n".join(f"- `{v['id']}` ({v['path']}) — excluded: {v['reason']}." for v in S["excluded_visuals"]) or "None."
    withheld = "\n".join(f"- ~~{w['action']}~~ — withheld: {w['reason']}." for w in S["withheld_recommendations"]) or "None."
    implications = [f for f in key if f.kind == "inference"]
    parts = [
        "# Evidence-Backed Report\n",
        "## Executive Summary\n\n" + "\n".join(exec_lines) + "\n",
        f"## Question / Intent\n\n{led.intent}\n",
        "## Key Findings\n\nObservations below are measured; inferences are labelled. Confidence never exceeds the strongest evidence cited.\n\n"
        + "\n\n".join(_finding_md(led, f) for f in key) + "\n",
        render.visuals_section(led),
        "## Implications\n\n" + ("\n".join(f"- {f.id} (inference, `{f.confidence}`): {f.statement}" for f in implications) or "None drawn.") + "\n",
        "## Risks\n\n" + ("\n".join(f"- {f.id} `{f.severity}`: {f.statement}" for f in key if f.severity in ("critical", "high")) or "None.") + "\n",
        f"## Conflicts\n\n{conflicts_md}\n",
        f"## Unsupported Claims\n\n{unsup}\n\nExcluded visuals:\n\n{exv}\n",
        render.list_section("Unknowns", led.gaps),
        render.recommendations_section(led, "Recommendations") + f"\nWithheld recommendations:\n\n{withheld}\n",
        f"## Confidence\n\nOverall `{led.decision_confidence}` — "
        + ("the lowest confidence among source decisions" if S["sources"] else "no skill decisions were supplied")
        + ("; lowered because evidence conflicts." if S["conflicts"] else ".") + "\n",
        render.evidence_section(led).replace("## Evidence", "## Evidence Appendix", 1),
    ]
    return "\n".join(parts)
