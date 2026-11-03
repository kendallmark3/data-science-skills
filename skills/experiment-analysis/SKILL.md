---
name: experiment-analysis
description: Evaluate an A/B test or controlled experiment and decide what the evidence supports (SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE, INVALID_EXPERIMENT). Checks design, sample ratio mismatch, effect size with confidence intervals, statistical vs practical significance, power, guardrails, segments with multiple-comparison correction, duration and novelty. Use when the user shares experiment results, asks "did the test win?", "should we ship the treatment?", "is this lift significant?", or wants an experiment readout.
---

# Experiment Analysis

**Intent:** say what the experiment actually supports. A small p-value is not a rollout decision.

## Inputs

- **Data:** either one row per unit (`variant`, metric columns, optional `assigned_date`, segment columns) or daily aggregates (`date`, `variant`, `users`, `conversions`); for aggregates set `format: aggregate`, `primary_metric` to the successes column and `count_column` to the exposed-units column.
- **Metadata YAML** (see `examples/meta.example.yaml`). The decision depends on these fields:
  - `hypothesis`, `primary_metric`, `assignment`, `expected_split`: without these the design can't be assessed, so the result is `INVALID_EXPERIMENT`
  - `practical_threshold`: the smallest effect worth acting on. Without it the result is `INCONCLUSIVE`, because "significant" can't be turned into "worth it"
  - `guardrails`, `segment_columns`, `time_column`, `planned_duration_days`, `stopping_rule`, `alpha`

If the user describes the design in chat, write it into the metadata file. If they can't say what effect would be worth shipping, ask them. That is a business judgement, and you shouldn't make it up.

## Engine location

The commands below use `dsk`. If `dsk` is not on PATH, run `python3 -m dskit` with the same arguments from this plugin's root: the folder two levels above this SKILL.md, which contains `pyproject.toml`. Don't search the filesystem for the engine, and don't `pip install` anything without asking the user first.

## How to run

1. Inspect the data and metadata. Confirm which column is the primary metric and which arm is control.
2. Run:
   ```bash
   dsk experiment <data.csv> --meta <meta.yaml> --out <out_dir>
   ```
3. Read `findings.json`. Check the design first (SRM, attrition, stopping), then the effect, then guardrails and segments.
4. Add reasoning only where it helps: likely causes of an SRM, whether a segment split looks post-hoc, or what a better next experiment would be. Label it as inference.
5. Reply with the decision, the effect and CI next to the practical threshold, any threats to validity, and the recommended next step.

## Method follows the problem

The engine chooses the test from the metric: binary metrics get a two-proportion Wald CI and pooled z-test; continuous metrics get Welch's t-test. Segments are Holm-corrected, and heterogeneity is tested with Cochran's Q. If the problem needs something else (ratio metrics, clustered units, sequential designs, CUPED), say the engine's method doesn't fit. Compute the right method and record it as evidence with its method stated, rather than forcing the default.

## Decision rules (in order)

1. Design fields missing → `INVALID_EXPERIMENT`
2. SRM (χ² p < 0.001), differential attrition, or early stopping → `INVALID_EXPERIMENT`. The effect is **not** plotted.
3. No practical threshold → `INCONCLUSIVE`
4. Significant, but the whole CI is below the threshold → `NOT_SUPPORTED` (real but trivial)
5. Treatment is significantly worse, or the CI rules out the threshold → `NOT_SUPPORTED`
6. Primary metric wins but a guardrail breaks its tolerance → `INCONCLUSIVE` (contradictory metrics)
7. The CI lower bound clears the threshold → `SUPPORTED`
8. Otherwise (the CI spans both trivial and worthwhile effects) → `INCONCLUSIVE`, with the n per arm needed

## Outputs

`report.md`, `findings.json`, `evidence.json`, `visuals/`:
- `effect-vs-threshold.png`: the estimate and CI against the practical threshold and each guardrail's tolerance
- `sample-ratio.png`: drawn instead of the effect chart when SRM invalidates the comparison
- `segment-effects.png`: only when segments differ (Q p < 0.05)

## Rules

- Never write "significant" without the effect size and CI next to it.
- Never recommend a rollout unless the decision is `SUPPORTED`.
- Segment wins are hypotheses for a follow-up test, not conclusions, unless the segment was pre-specified.
- An absent stopping rule is a gap. Mention it.

See `references/method.md` for formulas and thresholds.
