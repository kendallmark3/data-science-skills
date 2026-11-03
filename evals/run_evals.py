"""Scenario-based evaluation harness.

    python evals/run_evals.py                      # run every scenario, grade, run negative controls
    python evals/run_evals.py --only report_mixed_evidence
    python evals/run_evals.py --grade <out_dir> --expected <scenario>
        # grade any output directory (e.g. one Claude produced while following a SKILL.md)

Expectations live in evals/expected/<scenario>.yaml and are semantic: a required
finding matches on category plus any of several phrases, not on an exact string.
Negative controls in evals/negative/ are deliberately bad outputs; the harness
fails if the grader accepts any of them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dskit import cli  # noqa: E402

EVALS = ROOT / "evals"
OUT = ROOT / "output" / "evals"


def _match(f: dict, spec: dict) -> bool:
    text = f["statement"].lower()
    if "category" in spec:
        cats = spec["category"] if isinstance(spec["category"], list) else [spec["category"]]
        if f["category"] not in cats:
            return False
    if "not_category" in spec and f["category"] in spec["not_category"]:
        return False
    if "severity" in spec and f["severity"] not in spec["severity"]:
        return False
    if "any_of" in spec and not any(p.lower() in text for p in spec["any_of"]):
        return False
    for key in ("kind", "confidence"):
        if key in spec and f[key] != spec[key]:
            return False
    return True


def grade(out_dir: Path, exp: dict) -> list[str]:
    """Return a list of failures (empty = pass)."""
    fails = []
    fj = out_dir / "findings.json"
    if not fj.exists():
        return [f"no findings.json in {out_dir}"]
    F = json.loads(fj.read_text())
    E = json.loads((out_dir / "evidence.json").read_text())
    eids = {e["id"] for e in E}
    findings, visuals, recs = F.get("findings", []), F.get("visuals", []), F.get("recommendations", [])

    if F.get("decision") not in exp.get("acceptable_decisions", [F.get("decision")]):
        fails.append(f"decision {F.get('decision')} not in {exp['acceptable_decisions']}")
    if F.get("decision") in exp.get("forbidden_decisions", []):
        fails.append(f"forbidden decision {F.get('decision')}")
    if "expected_status" in exp and F.get("status") != exp["expected_status"]:
        fails.append(f"status {F.get('status')} != {exp['expected_status']}")

    # universal traceability invariants
    for f in findings:
        if not f.get("evidence") or not set(f["evidence"]) <= eids:
            fails.append(f"{f['id']} cites missing evidence {f.get('evidence')}")
    fids = {f["id"] for f in findings}
    for r in recs:
        if not r.get("based_on") or not set(r["based_on"]) <= fids:
            fails.append(f"recommendation {r['id']} not based on findings")
    for v in visuals:
        if not set(v.get("evidence", [])) or not set(v["evidence"]) <= eids:
            fails.append(f"visual {v['id']} cites missing evidence")
        if not (out_dir / v["path"]).exists():
            fails.append(f"visual file missing: {v['path']}")

    for spec in exp.get("required_findings", []):
        if not any(_match(f, spec) for f in findings):
            fails.append(f"missing required finding {spec}")
    for spec in exp.get("forbidden_findings", []):
        if "recommendation_pattern" in spec:
            bad = [r["action"] for r in recs if re.search(spec["recommendation_pattern"], r["action"], re.I)]
        else:
            bad = [f["id"] for f in findings if _match(f, {k: v for k, v in spec.items() if k != "why"})]
        if bad:
            fails.append(f"forbidden: {spec.get('why', spec)} → {bad}")
    for spec in exp.get("required_evidence", []):
        if not any(e["source"] == spec["source"] and spec.get("location_contains", "") in e["location"] for e in E):
            fails.append(f"missing evidence from {spec}")
    types = {v["type"] for v in visuals}
    for t in exp.get("expected_visuals", []):
        if t not in types:
            fails.append(f"missing visual {t}")
    for t in exp.get("forbidden_visuals", []):
        if t in types:
            fails.append(f"forbidden visual {t}")
    if exp.get("forbidden_visuals_any") and visuals:
        fails.append(f"no visual was warranted, got {sorted(types)}")
    summ = F.get("summary", {})
    got_unsup = {u["id"] for u in summ.get("unsupported_claims", [])}
    for cid in exp.get("required_unsupported", []):
        if cid not in got_unsup:
            fails.append(f"claim {cid} should be marked unsupported")
    got_conf = {c["topic"] for c in summ.get("conflicts", [])}
    for t in exp.get("required_conflicts", []):
        if t not in got_conf:
            fails.append(f"conflict on {t} not surfaced")
    return fails


def run_scenario(sc: dict, outs: dict) -> Path:
    out = OUT / sc["scenario"]
    i = sc["inputs"]
    args = {"profile": lambda: ["profile", i["data"]] + (["--context", i["context"]] if "context" in i else []),
            "experiment": lambda: ["experiment", i["data"], "--meta", i["meta"]],
            "evaluate": lambda: ["evaluate", i["data"], "--claim", i["claim"]],
            "report": lambda: ["report", *[str(outs[r]) for r in i.get("runs", [])]]
            + (["--claims", i["claims"]] if "claims" in i else []) + ["--intent", sc["intent"]]}[sc["skill"]]()
    args = [str(ROOT / a) if (ROOT / a).exists() and not Path(a).is_absolute() else a for a in args]
    cli.main(args + ["--out", str(out)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--grade", help="grade an existing output directory")
    ap.add_argument("--expected", help="scenario whose expectations to grade against (with --grade)")
    a = ap.parse_args()

    if a.grade:
        exp = yaml.safe_load((EVALS / "expected" / f"{a.expected}.yaml").read_text())
        fails = grade(Path(a.grade), exp)
        print("PASS" if not fails else "FAIL\n  " + "\n  ".join(fails))
        return 0 if not fails else 1

    scenarios = [yaml.safe_load(p.read_text()) for p in sorted((EVALS / "scenarios").glob("*.yaml"))]
    scenarios.sort(key=lambda s: s["skill"] == "report")  # reports consume other scenarios' outputs
    if a.only:
        scenarios = [s for s in scenarios if s["scenario"] == a.only or a.only in s["inputs"].get("runs", [])
                     ] or sys.exit(f"unknown scenario {a.only}")
    outs, results = {}, []
    import contextlib
    import io
    for sc in scenarios:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            outs[sc["scenario"]] = run_scenario(sc, outs)
        exp = yaml.safe_load((EVALS / "expected" / f"{sc['scenario']}.yaml").read_text())
        fails = grade(outs[sc["scenario"]], exp)
        decision = json.loads((outs[sc["scenario"]] / "findings.json").read_text()).get("decision")
        results.append({"scenario": sc["scenario"], "skill": sc["skill"], "decision": decision,
                        "pass": not fails, "failures": fails})

    negatives = []
    for neg in sorted((EVALS / "negative").glob("*/")):
        meta = yaml.safe_load((neg / "graded_against.yaml").read_text())
        exp = yaml.safe_load((EVALS / "expected" / f"{meta['scenario']}.yaml").read_text())
        fails = grade(neg, exp)
        negatives.append({"negative_control": neg.name, "graded_against": meta["scenario"], "rejected": bool(fails),
                          "reasons": fails})

    width = max(len(r["scenario"]) for r in results)
    print(f"{'scenario':<{width}}  {'skill':<10}  {'decision':<26}  result")
    for r in results:
        print(f"{r['scenario']:<{width}}  {r['skill']:<10}  {str(r['decision']):<26}  {'PASS' if r['pass'] else 'FAIL'}")
        for f in r["failures"]:
            print(f"    - {f}")
    print()
    for n in negatives:
        print(f"negative control {n['negative_control']:<30} {'REJECTED (good)' if n['rejected'] else 'ACCEPTED (bad!)'}"
              f"  [{len(n['reasons'])} violations]")
    passed = sum(r["pass"] for r in results)
    ok = passed == len(results) and all(n["rejected"] for n in negatives)
    print(f"\n{passed}/{len(results)} scenarios passed; {sum(n['rejected'] for n in negatives)}/{len(negatives)} negative controls rejected")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"scenarios": results, "negative_controls": negatives}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
