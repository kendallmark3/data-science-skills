---
name: evidence-backed-reporting
description: Turn analytical findings into a concise report where every conclusion traces to evidence. Consumes outputs of the data-profiling, experiment-analysis and model-evaluation skills and/or a claims file; caps each claim's confidence at its strongest evidence, shows conflicting evidence side by side, excludes unsupported claims and visuals, and withholds recommendations that rest on them. Use when the user asks for a write-up, readout, executive summary, decision memo or review document from analysis results, or asks to check whether a report's claims are supported.
---

# Evidence-Backed Reporting

**Intent:** a reviewer can see what was observed, where it came from, what was inferred, which visual supports it, how confident the analysis is, and what is still unknown.

## Inputs

- Output directories from the other skills (each holds `findings.json`, `evidence.json` and `visuals/`).
- Optionally, a **claims YAML** (see `examples/claims.example.yaml`): claims with evidence IDs, stated confidence, `topic`/`direction` for conflict detection, evidence files, visuals and proposed recommendations. Use it for notes, other people's analyses, or claims in a draft report.

If the user hands you a draft report to check, extract its claims into a claims file, map each one to the evidence that exists, and run the engine. Do not supply evidence yourself to make a claim pass.

## Engine location

The commands below use `dsk`. If `dsk` is not on PATH, run `python3 -m dskit` with the same arguments from this plugin's root: the folder two levels above this SKILL.md, which contains `pyproject.toml`. Don't search the filesystem for the engine, and don't `pip install` anything without asking the user first.

## How to run

```bash
dsk report <run_dir> [<run_dir> ...] [--claims claims.yaml] --intent "<decision question>" --out <out_dir>
```

Then read `findings.json` → `summary.traceability`, `unsupported_claims`, `conflicts`, `withheld_recommendations`.

## What the engine enforces

| Situation | Treatment |
|---|---|
| A claim cites no evidence, or only IDs that don't exist | listed under **Unsupported Claims**, never as a finding |
| A claim is stated more confidently than its evidence | confidence lowered to the strongest evidence, with a note |
| Two claims on the same topic point in opposite directions | both shown under **Conflicts** at `low` confidence; neither is adopted |
| A visual cites evidence that doesn't exist, or its file is missing | excluded, with the reason |
| A recommendation rests on an unsupported claim | **withheld**, with the reason |
| A finding is an inference | written as "Inference: …", never in observation wording |

Prose follows confidence: *high* → "The evidence shows", *medium* → "The evidence suggests", *low* → "Weak evidence indicates".

## Your part

The engine produces a faithful, mechanical report. You may then edit it for the audience:
- tighten the executive summary for an executive, product, engineering or scientific reader
- order findings by what matters for the decision
- add implications, labelled as inference, citing the finding IDs they rest on

While editing, **don't**:
- add a number, cause or forecast that has no evidence ID
- raise a finding's confidence
- drop a conflict, an unsupported claim or a gap because it is inconvenient
- add charts. Reuse the source skills' visuals. A new summary visual is justified only when it changes how the decision reads, and it must cite evidence IDs.

## Outputs

`report.md` (Executive Summary, Question/Intent, Key Findings, Visual Evidence, Implications, Risks, Conflicts, Unsupported Claims, Unknowns, Recommendations, Confidence, Evidence Appendix), plus `findings.json`, `evidence.json` and `visuals/` (copies of the curated source visuals, plus `decision-summary.png` when there are two or more sources).

## Stop conditions

No inputs, an input directory that isn't a skill output, or a request to state something the evidence doesn't support. In the last case, produce the report with that claim listed as unsupported and tell the user why.
