# How to Run the Skills

This guide covers what to install, how to start a skill, what input you give it, and where the results land.

---

## 1. One-time setup

```bash
git clone https://github.com/<you>/data-science-skills.git
cd data-science-skills
pip install -e ".[dev]"
```

This installs the `dsk` command and the test tools. If `pip` refuses to install into your system Python ("externally managed environment"), create a virtual environment first:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Then make the skills visible to Claude Code. Pick one:

| Option | Command | When to use it |
|---|---|---|
| Plugin, one session | `claude --plugin-dir /path/to/data-science-skills` | trying it out |
| Always on, all projects | `for s in skills/*/; do ln -sfn "$PWD/$s" ~/.claude/skills/$(basename "$s"); done` | everyday use |
| One project only | the same loop, targeting `<your-project>/.claude/skills/` | team repos |

Check it worked:

```bash
claude --plugin-dir . plugin details data-science-skills   # should list 4 skills
```

---

## 2. Two ways to run a skill

### A. Ask Claude (the normal way)

Open Claude Code in the folder that holds your data and ask in plain English. Claude picks the right skill from your request, builds the context file from what you said, runs the engine, and explains the result.

> "Profile `data/customers.csv` before we train the churn model. `churned` is the label and `customer_id` is the key."

> "Here's our A/B test export, `tests/checkout.csv`. 50/50 split, 21 days. Finance needs +1pp to break even. Should we ship?"

> "Our support bot scores 93% task success in `evals/run_12.csv`. Is it ready to answer customers without review?"

> "Write the readout for Thursday from `output/profile`, `output/experiment` and `output/eval`. Flag anything in `notes.yaml` the data doesn't back up."

Mention where to put the results ("put the outputs in `analysis/checkout`") and Claude passes that as `--out`.

### B. Run the engine yourself

You get the same calculations without Claude's interpretation layer. This is useful in scripts or CI.

```bash
dsk profile    <data>        --context <context.yaml>  --out <folder>
dsk experiment <data.csv>    --meta    <meta.yaml>     --out <folder>
dsk evaluate   <eval.csv>    --claim   <claim.yaml>    --out <folder>  [--upstream <profile-folder>]
dsk report     <folder> ...  [--claims <claims.yaml>]  --out <folder>  [--intent "<question>"]
```

`python3 -m dskit …` run from the repo root does the same thing if `dsk` isn't on your PATH.

---

## 3. Where the input comes from

**Your files.** Each skill takes **a data file you already have** plus **a small YAML file describing it**. When you work through Claude, Claude writes the YAML from your conversation. When you run `dsk` yourself, copy a template from the skill's `examples/` folder.

| Skill | Data file (yours) | Description file | Template |
|---|---|---|---|
| Data Profiling & Quality | any table: CSV, JSON/JSONL, Parquet* | context: intended use, target, ID columns, split column, schema (all optional) | `skills/data-profiling-quality/examples/context.example.yaml` |
| Experiment Analysis | one row per user (`variant`, metric columns) **or** daily totals (`date`, `variant`, `users`, `conversions`) | meta: hypothesis, primary metric, expected split, **practical threshold**, guardrails | `skills/experiment-analysis/examples/meta.example.yaml` |
| Model & Evaluation Review | predictions (`y_true`, `y_score`, segments) **or** LLM eval rows (`task_success`, `grounded`, …) | claim: the sentence being tested, requirements, critical segments, production mix | `skills/model-evaluation-review/examples/claim.example.yaml` |
| Evidence-Backed Reporting | output folders from the other three skills | optional claims file for notes or draft-report statements | `skills/evidence-backed-reporting/examples/claims.example.yaml` |

\* Parquet needs `pip install -e ".[parquet]"`.

**Practice data.** `fixtures/` holds realistic sample inputs with planted problems, so you can see each skill catch something. `fixtures/SEEDS.md` lists what was planted.

```
fixtures/datasets/      customers.csv (11 planted defects), customers_clean.csv (none), unreadable.csv
fixtures/experiments/   6 experiments: trivial win, real win, broken split, guardrail breach, underpowered, no design
fixtures/evaluations/   churn model (weak segment), support-bot LLM eval (strict + pilot claims)
fixtures/evidence/      claims file with strong, weak, conflicting and unsupported claims
```

---

## 4. Where the output goes

Everything is written to the folder you pass as `--out`. Nothing goes anywhere else.

```
<out>/
├── report.md        read this first: decision, findings, evidence table, gaps, next actions
├── findings.json    the same content as structured data, for scripts or the reporting skill
├── evidence.json    every measured fact: which file, which column/rows, what was observed, how
└── visuals/         charts, only when a finding needs one (PNG)
```

How to read it:
- Every finding has an ID like `DP-F-004` and cites evidence like `DP-E-007`. Look the evidence up in `evidence.json`, or in the table in `report.md`, to see the exact file and column it came from.
- **observation** means measured. **inference** means reasoned from the evidence. They are never mixed.
- The **decision** is at the top of `report.md` and in `findings.json → decision`.
- **Exit code `2`** means the skill stopped rather than guess (unreadable file, missing claim, no labels). `report.md` says why.

The `output/` folder in this repo is git-ignored, so example runs stay out of commits. Point `--out` wherever you like in your own projects.

---

## 5. Quick recipes (using the practice data)

```bash
# Data quality
dsk profile fixtures/datasets/customers.csv \
  --context fixtures/datasets/customers.context.yaml --out output/profile
open output/profile/report.md

# Experiment: significant but too small to matter
dsk experiment fixtures/experiments/checkout_button_daily.csv \
  --meta fixtures/experiments/checkout_button.meta.yaml --out output/experiment

# Model readiness, carrying in the data-quality findings
dsk evaluate fixtures/evaluations/churn_model_predictions.csv \
  --claim fixtures/evaluations/churn_model.claim.yaml --upstream output/profile --out output/eval

# One report across all three
dsk report output/profile output/experiment output/eval \
  --intent "Should we launch model-targeted retention offers?" --out output/report
```

---

## 6. Checking that it works

```bash
pytest                       # 49 tests
python3 evals/run_evals.py   # 14 scenarios + 2 deliberately bad outputs that must be rejected
```

To grade the output of a real Claude session against a scenario's expectations:

```bash
python3 evals/run_evals.py --grade output/live/checkout-button --expected statistically_significant_but_trivial
```

Scenario names are the file names in `evals/expected/`.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `dsk: command not found` | `pip install -e .`, or run `python3 -m dskit …` from the repo root |
| Claude doesn't pick up the skill | check the plugin is loaded (section 1), or name the skill: "use the experiment-analysis skill" |
| Experiment comes back `INCONCLUSIVE` with "no practical threshold" | add `practical_threshold` to the meta file. Only you can say what lift is worth shipping |
| Evaluation stops: "No claim supplied" | add a `claim:` sentence. A review needs to know what it is proving |
| Report lists your claim under "Unsupported" | the claim cites no evidence ID, or one that doesn't exist. Add the evidence to an evidence file and cite its ID |
