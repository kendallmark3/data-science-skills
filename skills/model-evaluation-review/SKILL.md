---
name: model-evaluation-review
description: Review whether a model or LLM/AI-system evaluation actually supports a readiness claim (READY, READY_WITH_CONDITIONS, NOT_READY, INSUFFICIENT_EVIDENCE). Recomputes metrics from predictions, breaks aggregates down by segment, checks baselines, calibration, coverage against production traffic, and for LLM systems groundedness, unsupported claims, refusals, judge-human agreement, latency, cost and version reproducibility. Use when the user asks "is this model ready?", shares eval results or a benchmark, or wants a model/LLM eval reviewed.
---

# Model & Evaluation Review

**Intent:** answer one question: *does this evaluation prove the claim being made?* The review is about the quality of the evaluation, not about how high the headline number is.

## Inputs

- **Evaluation data** (CSV):
  - classification: one row per case with label, score, optional baseline score and segment columns
  - LLM: one row per case with `task_success`, and optionally `grounded`, `unsupported_claim`, `format_valid`, `should_refuse`/`refused`, `human_task_success`, `latency_ms`, `cost_usd`, `model_version`, and a category column
- **Claim YAML** (see `examples/claim.example.yaml`): the claim in words, `mode`, `requirements`, `segment_columns`, `critical_segments`, `production_mix`.

A review without a claim has nothing to check against, so it stops. If the user gives only metrics ("94% accuracy!"), ask what decision those metrics are meant to support, or write down the claim they implied.

## Engine location

The commands below use `dsk`. If `dsk` is not on PATH, run `python3 -m dskit` with the same arguments from this plugin's root: the folder two levels above this SKILL.md, which contains `pyproject.toml`. Don't search the filesystem for the engine, and don't `pip install` anything without asking the user first.

## How to run

1. Inspect the data and confirm the label and score columns.
2. Run:
   ```bash
   dsk evaluate <eval.csv> --claim <claim.yaml> --out <out_dir> [--upstream <profiling_out_dir>]
   ```
   Pass `--upstream` when the training or eval data was profiled. Leakage and contamination findings from that run are carried in, with their original evidence IDs, as evaluation risks.
3. Read `findings.json`. Look at the segment results and coverage before the aggregate.
4. Add reasoning where it helps, labelled as inference with evidence IDs. Examples: why a segment might fail, what a representative eval set would look like, which requirement is missing from the claim.
5. Reply with the decision and the claim it was judged against, what is strong, what is weak, what is hidden by the aggregate, and the next evaluation to run.

## What the engine checks

- **Classification:** recomputes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier and ECE from the raw scores, so supplied numbers are never trusted as given. Also compares against the majority-class and supplied baselines, computes per-segment recall with Wilson CIs, compares the eval mix with production, recomputes recall at the production mix (an inference), and flags implausible AUC as possible leakage.
- **LLM / generative:** rates with Wilson CIs for task success, groundedness, unsupported claims and format validity. Also counts *successful-but-ungrounded* answers (the weakness a task-success headline hides), missed refusals and over-refusals, judge-human agreement (Cohen's κ; κ < 0.6 marks task success as unvalidated and lowers its confidence), p95 latency, cost, mixed model versions, per-category groundedness, and coverage.

## Decision rules

- fewer than 100 cases → `INSUFFICIENT_EVIDENCE`
- a stated requirement fails overall, or a **critical** segment is confirmed below it (CI upper bound < requirement) → `NOT_READY`
- requirements pass, but there are moderate or high risks (non-critical segment failure, coverage gap, weak judge, calibration, upstream leakage, mixed versions) → `READY_WITH_CONDITIONS`, with the conditions listed
- otherwise → `READY`

A segment that is below the requirement but too small to confirm is reported as *uncertain*, not as a failure.

## Outputs

`report.md`, `findings.json`, `evidence.json`, `visuals/`:
- `segment-performance.png` (recall by segment with CIs and the requirement line)
- `confusion-matrix.png` (overall vs the failing segment; only when a segment fails)
- `baseline-vs-candidate.png` (only when a baseline is supplied)
- `success-vs-groundedness.png` and `failure-modes.png` (LLM mode)

## Rules

- A strong aggregate never overrides a confirmed failure in a segment that matters.
- An unvalidated judge makes every judged metric low-confidence. Say so.
- Requirements that the claim implies but nobody evaluated (safety, fairness, robustness) are **gaps**, not passes.

See `references/method.md`.
