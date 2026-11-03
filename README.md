# Data Science Skills

Four reusable Claude Code skills for everyday data-science work. Each one turns realistic inputs into inspectable artifacts (a report, structured findings, an evidence ledger and charts), and every conclusion traces back to the evidence behind it.

| Skill | Question it answers | Decision vocabulary |
|---|---|---|
| `data-profiling-quality` | Is this dataset trustworthy enough for its intended use? | `FIT_FOR_USE` · `FIT_WITH_REMEDIATION` · `NOT_FIT_AS_IS` · `INSUFFICIENT_EVIDENCE` |
| `experiment-analysis` | What does this experiment actually support? | `SUPPORTED` · `NOT_SUPPORTED` · `INCONCLUSIVE` · `INVALID_EXPERIMENT` |
| `model-evaluation-review` | Does this evaluation prove the readiness claim? | `READY` · `READY_WITH_CONDITIONS` · `NOT_READY` · `INSUFFICIENT_EVIDENCE` |
| `evidence-backed-reporting` | What can a reviewer rely on, and what is still unknown? | `REPORTED` · `REPORTED_WITH_EXCLUSIONS` |

**New here? Start with [HOWTO.md](HOWTO.md):** setup, how to run each skill, where input comes from, and where output goes.

## What this is

- Four **skills** (`skills/*/SKILL.md`). Each tells Claude when to use it, what inputs it needs, how to run it, how to read the result, and when to stop.
- A small deterministic **engine** (`dskit/`) that does the calculations, so numbers are reproducible and tested instead of produced by a language model.
- A **shared evidence contract**. Findings cite evidence IDs, visuals cite evidence IDs, and recommendations cite findings. The code refuses dangling references.
- **Fixtures** with deliberately planted problems, **tests**, and a scenario-based **evaluation harness** with negative controls.

## What this is not

- Not four giant prompts. The prompts are short; the analysis is code.
- Not a replacement for scientific or domain judgement. The skills surface evidence and risk. People decide.
- Not a claim that statistical output equals truth. A p-value is not a rollout decision, and the skills enforce that.
- Not a dashboard, web app, database or orchestration layer.

## Installation

```bash
git clone <this repo> data-science-skills && cd data-science-skills
pip install -e ".[dev]"        # installs the `dsk` command (+ pytest, scikit-learn for tests)
```

Then expose the skills to Claude Code. Pick one option.

**Option A: as a plugin for one session** (no copying):

```bash
claude --plugin-dir /path/to/data-science-skills
```

**Option B: as personal skills, always on** (symlinks keep them in sync with the repo):

```bash
mkdir -p ~/.claude/skills
for s in skills/*/; do ln -sfn "$PWD/$s" ~/.claude/skills/$(basename "$s"); done
```

**Option C: for one project only.** Use the same loop with `<project>/.claude/skills` as the target.

Check with `claude --plugin-dir . plugin details data-science-skills`. It should list the four skills. Claude picks a skill up automatically when your request matches its description. You can also name it, for example "use the experiment-analysis skill on …".

## Usage

The way you'd normally use these is to ask Claude. Each skill runs the engine and then adds interpretation, labelled as inference.

> "Profile `data/customers.csv` before we train the churn model. `churned` is the label, `customer_id` the key."
> "Here's the pricing-test export. Did the new page win? We needed +1pp upgrades to justify it."
> "Our support assistant scores 93% task success. Is it ready to answer customers without review?"
> "Write the readout for Thursday from these three analyses, and flag anything in my notes the data doesn't support."

You can also run the engine directly:

```bash
dsk profile    fixtures/datasets/customers.csv --context fixtures/datasets/customers.context.yaml --out output/profile
dsk experiment fixtures/experiments/checkout_button_daily.csv --meta fixtures/experiments/checkout_button.meta.yaml --out output/experiment
dsk evaluate   fixtures/evaluations/churn_model_predictions.csv --claim fixtures/evaluations/churn_model.claim.yaml \
               --upstream output/profile --out output/evaluation
dsk report     output/profile output/experiment output/evaluation --intent "Launch targeted retention offers?" --out output/report
dsk report     --claims fixtures/evidence/review_claims.yaml --out output/report-claims
```

Without installing, `python -m dskit …` from the repo root works the same way. Exit code `2` means a stop condition fired. An honest `INSUFFICIENT_EVIDENCE` or `STOPPED` result is written instead of a guess.

Templates for the context, metadata and claim files are in each skill's `examples/` folder.

## Outputs

Every run writes the same shape:

```
<out>/
├── report.md        human-readable, in the section order each skill specifies
├── findings.json    decision, findings, visuals, assumptions, gaps, recommendations
├── evidence.json    every measured fact: source, location, observation, method, confidence
└── visuals/         only the charts a finding needs (PNG)
```

A finding separates **observation** from **inference**. Its confidence is capped at the strongest evidence it cites. Recommendations sit in their own list and point to the findings they rest on. The contract is in `skills/*/references/evidence-contract.md`.

## Testing

```bash
pytest                        # 49 tests: calculations vs scikit-learn/scipy, seeded-issue detection, traceability, stop conditions
python evals/run_evals.py     # 14 scenarios + 2 negative controls
```

The eval harness grades on meaning: required findings, forbidden findings, required evidence, expected visuals and the acceptable range of decisions. It does not match exact strings. It can also grade output that Claude produced while following a skill:

```bash
python evals/run_evals.py --grade path/to/claude_output --expected statistically_significant_but_trivial
```

`fixtures/generate_fixtures.py` rebuilds every fixture deterministically. `fixtures/SEEDS.md` lists what was planted in each.

## Architecture

```
Intent ─→ Skill (SKILL.md: when, inputs, rules, stop conditions)
            │
            ▼
         Evidence ─→ Analysis (dskit: deterministic) ─→ Finding (observation | inference, capped confidence)
                                                          │
                                                          ▼
                                                    Visual Evidence (only when it explains a finding)
                                                          │
                                                          ▼
                                                      Decision ─→ Validation (pytest + evals)
```

Claude's reasoning sits on top of the engine. It sets up context from the conversation, decides what is worth deepening, explains likely causes, and writes for the audience. Anything it adds is labelled as inference and cites evidence IDs. The skills compose through files, so `evidence-backed-reporting` can consume any mix of the other three, and `model-evaluation-review --upstream` carries profiling leakage findings into an evaluation with their original evidence IDs.

## Repository layout

```
.claude-plugin/plugin.json   lets Claude Code load the repo as a plugin
skills/<skill>/SKILL.md      the four skills (+ references/, examples/)
dskit/                       evidence model, four engines, CLI, chart style
fixtures/                    seeded datasets, experiments, evaluations, claims
tests/                       pytest suite, including the cross-skill workflow
evals/                       scenarios/, expected/, negative/, run_evals.py
output/                      generated example runs (git-ignored)
```
