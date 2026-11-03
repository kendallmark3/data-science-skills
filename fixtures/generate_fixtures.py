"""Regenerate every fixture deterministically.

    python fixtures/generate_fixtures.py

Each fixture has a known, intentionally seeded condition. SEEDS.md documents
what was planted so tests and evals can check detection without guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent
rng = np.random.default_rng(20261103)


def _dump_yaml(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False, width=100))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def _base_customers(n: int) -> pd.DataFrame:
    tenure = rng.integers(1, 72, n)
    spend = np.round(rng.gamma(4, 15, n) + 10, 2)
    tickets = rng.poisson(1.2, n)
    logit = -1.2 - 0.03 * tenure + 0.35 * tickets + 0.004 * spend
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    signup = pd.Timestamp("2026-10-15") - pd.to_timedelta(tenure * 30 + rng.integers(0, 30, n), "D")
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "email": [f"user{i}@example.com" for i in range(1, n + 1)],
        "age": rng.integers(18, 86, n),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "plan": rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.32, 0.18]),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "monthly_spend": spend,
        "tenure_months": tenure,
        "support_tickets": tickets,
        "churned": churned,
        "split": rng.choice(["train", "test"], n, p=[0.8, 0.2]),
    })


def make_datasets() -> dict:
    d = ROOT / "datasets"
    d.mkdir(exist_ok=True)
    seeds = {}

    clean = _base_customers(1000)
    clean["customer_id"] = [f"K{i:05d}" for i in range(1, 1001)]
    clean.to_csv(d / "customers_clean.csv", index=False)

    df = _base_customers(1188)
    # SEED: suspiciously constant feature
    df["account_currency"] = "USD"
    # SEED: target leakage - only populated after a customer has churned
    df["days_since_cancellation"] = np.where(df["churned"] == 1, rng.integers(1, 90, len(df)), np.nan)
    # SEED: 14.8% missing email (176 of 1188 base rows)
    miss = rng.choice(len(df), 176, replace=False)
    df.loc[miss, "email"] = np.nan
    # SEED: impossible ages
    bad_age_idx = [11, 404, 977]
    df.loc[bad_age_idx, "age"] = [-3, 142, 230]
    # SEED: malformed dates
    bad_date_idx = [21, 222, 333, 444, 555]
    df.loc[bad_date_idx, "signup_date"] = ["2024-13-45", "31/02/2024", "not_a_date", "2024/07/1x", "20240"]
    # SEED: category typo and inconsistent casing/whitespace
    typo_idx = [5, 105, 205, 305]
    case_idx = [6, 106, 206]
    df.loc[typo_idx, "plan"] = "premuim"
    df.loc[case_idx, "plan"] = "Premium "
    # SEED: mixed numeric formatting
    fmt_idx = list(range(50, 1050, 50))
    df["monthly_spend"] = df["monthly_spend"].astype(object)
    for i in fmt_idx:
        df.at[i, "monthly_spend"] = f"${float(df.at[i, 'monthly_spend']):,.2f}"
    # SEED: 12 duplicated customer IDs; 6 exact copies, 6 copies placed in the other split
    dup_src = [30, 130, 230, 430, 630, 830, 31, 131, 231, 431, 631, 831]
    dups = df.loc[dup_src].copy()
    flip = dups.index[6:]
    dups.loc[flip, "split"] = dups.loc[flip, "split"].map({"train": "test", "test": "train"})
    df = pd.concat([df, dups], ignore_index=True)
    df.to_csv(d / "customers.csv", index=False)

    _dump_yaml(d / "customers.context.yaml", {
        "intended_use": "Train and evaluate a churn model that drives retention outreach",
        "target": "churned",
        "id_columns": ["customer_id"],
        "time_column": "signup_date",
        "split_column": "split",
        "schema": {
            "age": {"type": "integer", "min": 16, "max": 110},
            "monthly_spend": {"type": "float", "min": 0},
            "plan": {"type": "category", "allowed": ["basic", "standard", "premium"]},
            "signup_date": {"type": "date"},
            "churned": {"type": "integer", "allowed": [0, 1]},
        },
        "required_columns": ["email"],
    })
    _dump_yaml(d / "customers_clean.context.yaml", {
        "intended_use": "Train and evaluate a churn model",
        "target": "churned", "id_columns": ["customer_id"], "time_column": "signup_date",
        "split_column": "split",
    })
    (d / "unreadable.csv").write_bytes(b"\x00\x01\x02 not a table \xff\xfe")

    seeds["customers.csv"] = {
        "duplicate_customer_ids": 12, "exact_duplicate_rows": 6, "ids_in_both_splits": 6,
        "missing_email_rows": 176, "impossible_age_rows": 3, "malformed_date_rows": 5,
        "category_typo": "premuim (4 rows)", "case_whitespace_variant": "'Premium ' (3 rows)",
        "mixed_numeric_format": f"monthly_spend '$x,xxx.xx' strings ({len(fmt_idx)} rows)",
        "constant_feature": "account_currency", "leakage_column": "days_since_cancellation",
    }
    seeds["customers_clean.csv"] = "No seeded defects; used to check that nothing is invented."
    return seeds


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def _units(n_c: int, n_t: int, p_c: float, p_t: float, start: str, days: int,
           guard_c: float = 0.04, guard_t: float = 0.04) -> pd.DataFrame:
    def arm(n, p, g, name):
        conv = np.zeros(n, int)
        conv[: round(n * p)] = 1
        rng.shuffle(conv)
        ref = np.zeros(n, int)
        ref[: round(n * g)] = 1
        rng.shuffle(ref)
        return pd.DataFrame({
            "variant": name, "converted": conv, "refunded": ref,
            "page_load_ms": np.round(rng.normal(820, 140, n)).astype(int),
            "device": rng.choice(["mobile", "desktop"], n, p=[0.55, 0.45]),
            "assigned_date": (pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), "D")).strftime("%Y-%m-%d"),
        })
    df = pd.concat([arm(n_c, p_c, guard_c, "control"), arm(n_t, p_t, guard_t, "treatment")], ignore_index=True)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    df.insert(0, "user_id", [f"U{i:06d}" for i in range(len(df))])
    return df[["user_id", "variant", "assigned_date", "device", "converted", "refunded", "page_load_ms"]]


def make_experiments() -> dict:
    d = ROOT / "experiments"
    d.mkdir(exist_ok=True)
    common_guard = [{"metric": "page_load_ms", "direction": "lower_is_better", "max_relative_degradation": 0.05}]

    # A: significant but operationally trivial (daily aggregates, ~1M users per arm)
    rows = []
    for day in pd.date_range("2026-09-01", periods=21):
        for variant, rate in (("control", 0.1000), ("treatment", 0.1030)):
            users = int(rng.integers(48_500, 51_500))
            conv = int(round(users * rate * rng.normal(1, 0.01)))
            rows.append({"date": day.strftime("%Y-%m-%d"), "variant": variant, "users": users, "conversions": conv})
    pd.DataFrame(rows).to_csv(d / "checkout_button_daily.csv", index=False)
    _dump_yaml(d / "checkout_button.meta.yaml", {
        "name": "Checkout button color", "hypothesis": "A green checkout button increases checkout conversion.",
        "format": "aggregate", "primary_metric": "conversions", "metric_type": "binary",
        "unit": "user", "expected_split": {"control": 0.5, "treatment": 0.5},
        "practical_threshold": {"absolute": 0.01, "rationale": "Below +1pp the redesign does not repay migration cost"},
        "alpha": 0.05, "planned_duration_days": 21, "stopping_rule": "fixed horizon, 21 days",
        "assignment": "user-level hash randomization",
    })

    # B: meaningful effect, adequate evidence
    _units(10_000, 10_000, 0.120, 0.148, "2026-08-04", 14).to_csv(d / "retention_offer_units.csv", index=False)
    _dump_yaml(d / "retention_offer.meta.yaml", {
        "name": "Retention offer", "hypothesis": "Offering one free month to at-risk customers increases renewal (converted).",
        "format": "units", "primary_metric": "converted", "metric_type": "binary", "unit": "user",
        "expected_split": {"control": 0.5, "treatment": 0.5},
        "practical_threshold": {"absolute": 0.01, "rationale": "+1pp renewal covers the offer cost"},
        "secondary_metrics": [], "guardrails": common_guard + [
            {"metric": "refunded", "direction": "lower_is_better", "max_absolute_degradation": 0.01}],
        "segment_columns": ["device"], "time_column": "assigned_date",
        "alpha": 0.05, "planned_duration_days": 14, "stopping_rule": "fixed horizon, 14 days",
        "assignment": "user-level hash randomization",
    })

    # C: sample ratio mismatch - invalid experiment despite an attractive lift
    _units(10_000, 8_600, 0.110, 0.135, "2026-09-08", 14).to_csv(d / "pricing_page_units.csv", index=False)
    _dump_yaml(d / "pricing_page.meta.yaml", {
        "name": "Pricing page redesign", "hypothesis": "The redesigned pricing page increases plan upgrades.",
        "format": "units", "primary_metric": "converted", "metric_type": "binary", "unit": "user",
        "expected_split": {"control": 0.5, "treatment": 0.5},
        "practical_threshold": {"absolute": 0.01}, "guardrails": common_guard,
        "segment_columns": ["device"], "time_column": "assigned_date",
        "alpha": 0.05, "planned_duration_days": 14, "stopping_rule": "fixed horizon, 14 days",
        "assignment": "user-level hash randomization",
    })

    # D: primary metric improves, guardrail breaks - contradictory metrics
    _units(9_000, 9_000, 0.200, 0.232, "2026-07-01", 14, guard_c=0.040, guard_t=0.068).to_csv(
        d / "free_shipping_units.csv", index=False)
    _dump_yaml(d / "free_shipping.meta.yaml", {
        "name": "Free shipping threshold", "hypothesis": "Lowering the free-shipping threshold increases purchases.",
        "format": "units", "primary_metric": "converted", "metric_type": "binary", "unit": "user",
        "expected_split": {"control": 0.5, "treatment": 0.5},
        "practical_threshold": {"absolute": 0.01},
        "guardrails": [{"metric": "refunded", "direction": "lower_is_better", "max_absolute_degradation": 0.01}],
        "segment_columns": ["device"], "time_column": "assigned_date",
        "alpha": 0.05, "planned_duration_days": 14, "stopping_rule": "fixed horizon, 14 days",
        "assignment": "user-level hash randomization",
    })

    # E: underpowered - inconclusive
    _units(900, 900, 0.200, 0.215, "2026-10-01", 10).to_csv(d / "onboarding_email_units.csv", index=False)
    _dump_yaml(d / "onboarding_email.meta.yaml", {
        "name": "Onboarding email", "hypothesis": "A day-2 onboarding email increases activation.",
        "format": "units", "primary_metric": "converted", "metric_type": "binary", "unit": "user",
        "expected_split": {"control": 0.5, "treatment": 0.5},
        "practical_threshold": {"absolute": 0.02},
        "alpha": 0.05, "planned_duration_days": 10, "stopping_rule": "fixed horizon, 10 days",
        "assignment": "user-level hash randomization", "time_column": "assigned_date",
    })

    # F: design evidence missing - no hypothesis, no assignment, no threshold
    _dump_yaml(d / "no_design.meta.yaml", {"name": "Homepage hero test", "format": "units"})
    return {
        "checkout_button": "A: +0.30pp lift on 10% base, n≈1M/arm. Significant; below the 1pp practical threshold → NOT_SUPPORTED",
        "retention_offer": "B: +2.8pp lift on 12% base, n=10k/arm, guardrails flat → SUPPORTED",
        "pricing_page": "C: 10,000 vs 8,600 on a 50/50 design (SRM) → INVALID_EXPERIMENT",
        "free_shipping": "D: +3.2pp conversion but refunds +2.8pp vs 1pp tolerance → INCONCLUSIVE (contradictory)",
        "onboarding_email": "E: +1.5pp on 900/arm, CI spans 0 and threshold → INCONCLUSIVE",
        "no_design": "F: metadata missing hypothesis/metric/assignment → INVALID_EXPERIMENT (stop)",
    }


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------
def make_evaluations() -> dict:
    d = ROOT / "evaluations"
    d.mkdir(exist_ok=True)

    # Churn model: strong globally, weak for new customers.
    n = 6000
    seg = rng.choice(["established", "new"], n, p=[0.8, 0.2])
    region = rng.choice(["north", "south", "east", "west"], n)
    y = (rng.random(n) < np.where(seg == "new", 0.30, 0.22)).astype(int)
    sep = np.where(seg == "new", 0.35, 2.6)                      # weak separation for new customers
    z = rng.normal(0, 1, n) + y * sep - np.where(seg == "new", 0.2, 1.3)
    score = 1 / (1 + np.exp(-1.6 * z))
    base_z = rng.normal(0, 1, n) + y * 1.0 - 0.5
    base = 1 / (1 + np.exp(-1.2 * base_z))
    pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)], "tenure_segment": seg, "region": region,
        "y_true": y, "y_score": np.round(score, 5), "baseline_score": np.round(base, 5),
    }).to_csv(d / "churn_model_predictions.csv", index=False)
    _dump_yaml(d / "churn_model.claim.yaml", {
        "claim": "The churn model is ready to drive retention outreach for all customers.",
        "mode": "classification", "task": "binary churn prediction",
        "label_column": "y_true", "score_column": "y_score", "baseline_score_column": "baseline_score",
        "threshold": 0.5,
        "requirements": {"min_recall": 0.75, "min_precision": 0.55, "min_roc_auc": 0.80},
        "segment_columns": ["tenure_segment", "region"],
        "critical_segments": {"tenure_segment": ["new"]},
        "production_mix": {"tenure_segment": {"established": 0.65, "new": 0.35}},
    })

    # LLM support assistant: high task success, poor groundedness on policy questions.
    m = 240
    cat = np.repeat(["billing", "technical", "account", "policy"], m // 4)
    p_ground = {"billing": 0.88, "technical": 0.85, "account": 0.86, "policy": 0.48}
    grounded = np.array([rng.random() < p_ground[c] for c in cat]).astype(int)
    unsupported = ((grounded == 0) & (rng.random(m) < 0.55)).astype(int)
    success = (rng.random(m) < np.where(grounded == 1, 0.97, 0.80)).astype(int)
    should_refuse = (rng.random(m) < 0.08).astype(int)
    refused = np.where(should_refuse == 1, (rng.random(m) < 0.75).astype(int), (rng.random(m) < 0.02).astype(int))
    human = np.full(m, np.nan)
    hs = rng.choice(m, 60, replace=False)
    human[hs] = np.where(rng.random(60) < 0.72, success[hs], 1 - success[hs])
    pd.DataFrame({
        "case_id": [f"Q{i:04d}" for i in range(m)], "category": cat,
        "task_success": success, "grounded": grounded, "unsupported_claim": unsupported,
        "format_valid": (rng.random(m) < 0.985).astype(int), "should_refuse": should_refuse, "refused": refused,
        "latency_ms": np.round(rng.lognormal(7.1, 0.35, m)).astype(int),
        "cost_usd": np.round(rng.normal(0.0042, 0.0008, m), 5),
        "human_task_success": human,
        "model_version": np.where(np.arange(m) < 200, "assistant-2026-09-12", "assistant-2026-10-01"),
    }).to_csv(d / "support_assistant_llm_eval.csv", index=False)
    claim = {
        "claim": "The support assistant is ready to answer customer questions without human review.",
        "mode": "llm", "task": "customer support question answering over the help-center corpus",
        "segment_columns": ["category"],
        "requirements": {"min_task_success": 0.90, "min_groundedness": 0.85, "max_unsupported_claim_rate": 0.05,
                         "min_format_valid": 0.95, "max_p95_latency_ms": 3000, "max_cost_per_case_usd": 0.01},
        "production_mix": {"category": {"billing": 0.30, "technical": 0.30, "account": 0.25, "policy": 0.15}},
    }
    _dump_yaml(d / "support_assistant.claim.yaml", claim)
    lenient = json.loads(json.dumps(claim))
    lenient["claim"] = "The support assistant is ready for an agent-assist pilot where a human approves every reply."
    lenient["requirements"].update({"min_groundedness": 0.70, "max_unsupported_claim_rate": 0.20})
    _dump_yaml(d / "support_assistant_pilot.claim.yaml", lenient)
    return {
        "churn_model": "AUC strong overall; tenure_segment=new has weak separation (recall far below 0.75), "
                       "and new customers are 20% of eval vs 35% of production → not unconditionally READY",
        "support_assistant (strict)": "task success ~0.93 but groundedness ~0.77, unsupported-claim rate above 5% → NOT_READY",
        "support_assistant_pilot (lenient)": "same data, pilot thresholds → READY_WITH_CONDITIONS",
    }


# ---------------------------------------------------------------------------
# Reporting fixture
# ---------------------------------------------------------------------------
def make_evidence() -> dict:
    d = ROOT / "evidence"
    (d / "visuals").mkdir(parents=True, exist_ok=True)
    supplied = [
        {"id": "X-E-001", "source": "retention_offer_results.csv", "location": "primary metric, both arms",
         "observation": "Renewal 14.8% treatment vs 12.0% control (n=10,000 each); diff +2.8pp, 95% CI [1.8, 3.8]pp",
         "method": "two-proportion Wald CI", "confidence": "high", "value": 0.028},
        {"id": "X-E-002", "source": "retention_offer_results.csv", "location": "segment=tablet",
         "observation": "Tablet users: +9pp lift, n=140 per arm, 95% CI [-1.5, 19.5]pp",
         "method": "two-proportion Wald CI", "confidence": "low", "value": 0.09},
        {"id": "X-E-003", "source": "ops_dashboard_export.csv", "location": "weekly support tickets, weeks 32-35",
         "observation": "Support tickets per 1k customers fell from 41 to 37 after launch",
         "method": "weekly count / active customers", "confidence": "medium", "value": -4},
        {"id": "X-E-004", "source": "cs_team_survey.md", "location": "Q3 agent survey, item 4",
         "observation": "7 of 9 agents report more offer-related tickets since launch",
         "method": "agent self-report survey", "confidence": "low", "value": 7},
    ]
    (d / "supplied_evidence.json").write_text(json.dumps(supplied, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.bar(["control", "treatment"], [12.0, 14.8], color=["#c3c2b7", "#2a78d6"])
    ax.set_ylabel("Renewal rate (%)")
    ax.set_title("Renewal by arm (X-E-001)", loc="left")
    fig.tight_layout()
    fig.savefig(d / "visuals" / "renewal_by_arm.png")
    plt.close(fig)

    _dump_yaml(d / "review_claims.yaml", {
        "intent": "Decide whether to expand the retention offer to all at-risk customers.",
        "evidence_files": ["supplied_evidence.json"],
        "claims": [
            {"id": "C-1", "claim": "The retention offer increased renewal by 2.8 percentage points.",
             "evidence": ["X-E-001"], "confidence": "high", "topic": "renewal", "direction": "up"},
            {"id": "C-2", "claim": "Tablet users respond far more strongly to the offer than other users.",
             "evidence": ["X-E-002"], "confidence": "high", "topic": "tablet_effect", "direction": "up"},
            {"id": "C-3", "claim": "Support ticket volume decreased after launch.",
             "evidence": ["X-E-003"], "confidence": "medium", "topic": "support_tickets", "direction": "down"},
            {"id": "C-4", "claim": "Support ticket volume increased after launch.",
             "evidence": ["X-E-004"], "confidence": "medium", "topic": "support_tickets", "direction": "up"},
            {"id": "C-5", "claim": "Expanding the offer will add $2M in annual revenue.",
             "evidence": [], "confidence": "high", "topic": "revenue", "direction": "up"},
            {"id": "C-6", "claim": "Churned customers who received the offer were 3x more likely to return.",
             "evidence": ["X-E-099"], "confidence": "high", "topic": "winback", "direction": "up"},
        ],
        "visuals": [
            {"id": "XV-1", "path": "visuals/renewal_by_arm.png", "evidence": ["X-E-001"],
             "purpose": "Renewal rate by experiment arm"},
            {"id": "XV-2", "path": "visuals/revenue_projection.png", "evidence": ["X-E-077"],
             "purpose": "Projected revenue from expansion"},
        ],
        "recommendations": [
            {"action": "Expand the retention offer to all at-risk customers.", "based_on": ["C-1", "C-5"]},
            {"action": "Run a powered follow-up test for tablet users before targeting them.", "based_on": ["C-2"]},
        ],
    })
    return {"review_claims.yaml": "C-1 strong · C-2 weak evidence stated as high confidence · C-3/C-4 conflict · "
                                  "C-5 no evidence · C-6 cites missing evidence · XV-2 cites missing evidence · "
                                  "recommendation 1 leans on unsupported C-5"}


def main() -> None:
    seeds = {"datasets": make_datasets(), "experiments": make_experiments(),
             "evaluations": make_evaluations(), "evidence": make_evidence()}
    lines = ["# Seeded fixture conditions", "", "Generated by `generate_fixtures.py` (fixed RNG seed).", ""]
    for group, items in seeds.items():
        lines.append(f"## {group}")
        for k, v in items.items():
            if isinstance(v, dict):
                lines.append(f"- **{k}**")
                lines += [f"  - {kk}: {vv}" for kk, vv in v.items()]
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")
    (ROOT / "SEEDS.md").write_text("\n".join(lines))
    print("fixtures regenerated")


if __name__ == "__main__":
    main()
