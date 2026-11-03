---
name: data-profiling-quality
description: Decide whether a tabular dataset (CSV, JSON, Parquet) is trustworthy enough for its intended analytical or modeling use. Finds missingness, duplicate IDs, impossible values, malformed dates, label typos, constant columns, target leakage and train/test contamination, with evidence IDs and charts. Use when the user asks to profile, audit, sanity-check or "look at" a dataset, asks whether data is clean or ready for a model, or before building a model or analysis on a new table.
---

# Data Profiling & Quality

**Intent:** decide whether this data can be trusted for *this* use. That is a different question from describing it.

## Inputs

- **Required:** a table (`.csv`, `.json`, `.jsonl`, `.parquet` if `pyarrow` is installed).
- **Optional context** (YAML; see `examples/context.example.yaml`). Each field you supply unlocks checks:
  - `intended_use`: sets severity in context
  - `target`: leakage and class-balance checks
  - `id_columns`: entity-level duplicate checks
  - `split_column`: train/test contamination
  - `time_column`, `as_of`: malformed and future dates, freshness
  - `schema`: per column `type`, `min`, `max`, `allowed`
  - `required_columns`: raises the severity of missingness in those columns

If the user mentioned any of this in conversation (for example "churned is the label"), write it into a context file. Don't ask for context you can infer safely. Missing context is recorded as a gap, and the analysis still runs.

## Engine location

The commands below use `dsk`. If `dsk` is not on PATH, run `python3 -m dskit` with the same arguments from this plugin's root: the folder two levels above this SKILL.md, which contains `pyproject.toml`. Don't search the filesystem for the engine, and don't `pip install` anything without asking the user first.

## How to run

1. Look at the file first (`head`, column names) so you know what you are profiling.
2. Run the engine:
   ```bash
   dsk profile <data> --context <context.yaml> --out <out_dir>
   # or: python -m dskit profile ...
   ```
3. Read `<out_dir>/findings.json`, not only `report.md`. Exit code `2` means a stop condition fired. Report the reason and don't work around it.
4. Add your reasoning only where it adds value: domain explanations, which findings matter for the stated use, and what to check next. Label everything you add as an **inference** and give the evidence ID it rests on.
5. Reply with the fitness decision, the critical and high findings (each with its evidence ID), the visuals worth opening, and the gaps.

## Outputs

`report.md`, `findings.json` (findings, decision, gaps, recommendations), `evidence.json`, `visuals/`. Charts are generated only when a finding needs one:

| Visual | When |
|---|---|
| `missingness.png` | any column has missing values |
| `category-anomalies.png` | typo or case/whitespace label variants found |
| `range-<col>.png` | out-of-range or impossible values found |
| `quality-scorecard.png` | at least one moderate-or-worse finding |

## Decision vocabulary

`FIT_FOR_USE` · `FIT_WITH_REMEDIATION` (high-severity issues) · `NOT_FIT_AS_IS` (critical: leakage, contamination) · `INSUFFICIENT_EVIDENCE` (unreadable or empty).

## Rules

- Each finding separates **observation**, **risk**, **possible explanation** and **recommended check**. Don't merge them into one sentence.
- Outliers that fall within a valid range are *informational*. Never call them defects without domain evidence.
- Leakage is an **inference**: it says the column is *probably* written after the outcome. Say "probable" and recommend confirming the timing.
- Don't run extra statistical tests or draw extra charts beyond what the engine produced unless a finding needs them. If you do, register them as evidence and cite the IDs.
- Never invent a defect that has no evidence ID behind it. A clean dataset should come back clean.

## Stop conditions

Stop and report rather than guess when the file can't be read, parses to an empty table, or the user asks for a conclusion (for example "the data is unbiased") that the checks can't support. Say which evidence would be needed.

See `references/method.md` for the exact rules and thresholds.
