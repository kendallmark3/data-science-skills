"""Command-line entry point: `dsk <skill> ...` (or `python -m dskit <skill> ...`).

Exit codes: 0 = analysis completed, 2 = stopped on a stop condition (an honest
INSUFFICIENT_EVIDENCE / INVALID result was written instead of a fabricated one).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import evaluation, experiment, profiling, reporting
from .evidence import StopCondition, write_stopped


def _yaml(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise StopCondition(f"Context file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    data.setdefault("_source", p.name)
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsk", description="Evidence-first data-science skills")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="Data Profiling & Quality")
    p.add_argument("data")
    p.add_argument("--context", help="YAML: intended_use, target, id_columns, time_column, split_column, schema")
    p.add_argument("--out", required=True)

    e = sub.add_parser("experiment", help="Experiment Analysis")
    e.add_argument("data")
    e.add_argument("--meta", required=True, help="YAML: hypothesis, primary_metric, expected_split, practical_threshold, ...")
    e.add_argument("--out", required=True)

    m = sub.add_parser("evaluate", help="Model & Evaluation Review")
    m.add_argument("data")
    m.add_argument("--claim", required=True, help="YAML: claim, mode (classification|llm), requirements, segments, ...")
    m.add_argument("--upstream", nargs="*", default=[], help="profiling output dirs whose findings bear on this evaluation")
    m.add_argument("--out", required=True)

    r = sub.add_parser("report", help="Evidence-Backed Reporting")
    r.add_argument("runs", nargs="*", help="output directories from the other skills")
    r.add_argument("--claims", help="YAML of hand-authored claims, evidence files and visuals")
    r.add_argument("--intent")
    r.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    skill = {"profile": "data-profiling-quality", "experiment": "experiment-analysis",
             "evaluate": "model-evaluation-review", "report": "evidence-backed-reporting"}[a.cmd]
    try:
        if a.cmd == "profile":
            led = profiling.profile(a.data, a.out, _yaml(a.context))
        elif a.cmd == "experiment":
            meta = _yaml(a.meta)
            meta["_meta_source"] = meta.pop("_source")
            led = experiment.analyze(a.data, meta, a.out)
        elif a.cmd == "evaluate":
            claim = _yaml(a.claim)
            claim["_claim_source"] = claim.pop("_source")
            led = evaluation.review(a.data, claim, a.out, a.upstream)
        else:
            led = reporting.build_report(a.out, a.runs, a.claims, a.intent)
    except StopCondition as stop:
        write_stopped(a.out, skill, stop)
        print(f"STOPPED ({stop.decision}): {stop.reason}", file=sys.stderr)
        return 2
    sev = {}
    for f in led.findings:
        sev[f.severity] = sev.get(f.severity, 0) + 1
    print(f"{skill}: {led.decision} (confidence {led.decision_confidence}) · findings {sev} · "
          f"visuals {len(led.visuals)} · wrote {a.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
