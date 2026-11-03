import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dskit.experiment import DECISIONS, continuous_effect, holm, required_n_per_arm

from .conftest import load3, FIX, assert_traceable, load

CASES = {
    "exp_trivial": {"NOT_SUPPORTED"},
    "exp_supported": {"SUPPORTED"},
    "exp_srm": {"INVALID_EXPERIMENT"},
    "exp_contradictory": {"INCONCLUSIVE"},
    "exp_underpowered": {"INCONCLUSIVE"},
    "exp_no_design": {"INVALID_EXPERIMENT"},
}


@pytest.mark.parametrize("name,expected", CASES.items())
def test_decisions(runs, name, expected):
    f, ev, out = load3(runs[name])
    assert f["decision"] in DECISIONS
    assert f["decision"] in expected, f"{name}: {f['decision']} — {f['decision_rationale']}"
    assert_traceable(f, ev, out)


def test_effect_size_and_ci_match_independent_calculation(runs):
    df = pd.read_csv(FIX / "experiments" / "retention_offer_units.csv")
    g = df.groupby("variant")["converted"].agg(["sum", "count"])
    pc, pt = g.loc["control", "sum"] / g.loc["control", "count"], g.loc["treatment", "sum"] / g.loc["treatment", "count"]
    se = np.sqrt(pc * (1 - pc) / g.loc["control", "count"] + pt * (1 - pt) / g.loc["treatment", "count"])
    # pooled z-test is equivalent to the uncorrected chi-square on the 2x2 table
    table = [[g.loc[a, "sum"], g.loc[a, "count"] - g.loc[a, "sum"]] for a in ("control", "treatment")]
    p = stats.chi2_contingency(table, correction=False).pvalue
    s = load(runs["exp_supported"])[0]["summary"]["primary"]
    assert s["diff"] == pytest.approx(pt - pc, abs=1e-12)
    assert s["ci"][0] == pytest.approx(pt - pc - 1.959964 * se, abs=1e-6)
    assert s["ci"][1] == pytest.approx(pt - pc + 1.959964 * se, abs=1e-6)
    assert s["p_value"] == pytest.approx(p, rel=1e-6)


def test_aggregate_input_matches_counts(runs):
    df = pd.read_csv(FIX / "experiments" / "checkout_button_daily.csv")
    g = df.groupby("variant")[["users", "conversions"]].sum()
    rate = g["conversions"] / g["users"]
    s = load(runs["exp_trivial"])[0]["summary"]["primary"]
    assert s["diff"] == pytest.approx(rate["treatment"] - rate["control"], abs=1e-12)


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    c, t = pd.Series(rng.normal(10, 2, 300)), pd.Series(rng.normal(10.5, 3, 250))
    e = continuous_effect("m", c, t)
    assert e.p == pytest.approx(stats.ttest_ind(t, c, equal_var=False).pvalue)
    ci = stats.ttest_ind(t, c, equal_var=False).confidence_interval()
    assert (e.lo, e.hi) == pytest.approx((ci.low, ci.high))


def test_significance_does_not_imply_rollout(runs):
    f = load(runs["exp_trivial"])[0]
    assert f["summary"]["primary"]["p_value"] < 0.001                    # highly significant...
    assert f["summary"]["primary"]["ci"][1] < f["summary"]["practical_threshold"]  # ...but trivial
    assert f["decision"] == "NOT_SUPPORTED"
    assert not any(r["action"].lower().startswith("roll out") for r in f["recommendations"])
    assert any(x["category"] == "practical_significance" and x["kind"] == "inference" for x in f["findings"])


def test_only_supported_recommends_rollout(runs):
    for name in CASES:
        f = load(runs[name])[0]
        rollout = any(r["action"].lower().startswith("roll out") for r in f["recommendations"])
        assert rollout == (f["decision"] == "SUPPORTED"), name


def test_malformed_experiment_flagged(runs):
    f = load(runs["exp_srm"])[0]
    srm = [x for x in f["findings"] if x["category"] == "srm"]
    assert srm and srm[0]["severity"] == "critical"
    # effect is not visualised when it cannot be interpreted
    assert [v["type"] for v in f["visuals"]] == ["sample_ratio_mismatch"]
    primary = next(x for x in f["findings"] if x["category"] == "primary_effect")
    assert primary["confidence"] == "low" and "not interpretable" in primary["statement"]
    nd = load(runs["exp_no_design"])[0]
    assert nd["findings"][0]["category"] == "design" and nd["visuals"] == []


def test_contradictory_metrics_identified(runs):
    f = load(runs["exp_contradictory"])[0]
    g = [x for x in f["findings"] if x["category"] == "guardrail" and x["severity"] == "high"]
    assert g and "refunded" in g[0]["statement"]
    assert any("contradict" in r for r in f["decision_rationale"])


def test_underpowered_reports_required_n(runs):
    f = load(runs["exp_underpowered"])[0]
    assert any(x["category"] == "power" for x in f["findings"])
    n = f["summary"]["n_needed_per_arm"]
    assert n > 900 and f"{n:,}" in f["recommendations"][0]["action"]


def test_effect_visual_reflects_values(runs):
    f, ev, _ = load3(runs["exp_trivial"])
    v = next(v for v in f["visuals"] if v["type"] == "treatment_control_effect")
    primary_ev = next(x for x in f["findings"] if x["category"] == "primary_effect")["evidence"][0]
    assert primary_ev in v["evidence"]


def test_holm_and_power_helpers():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    n = required_n_per_arm(0.10, 0.01)
    se = np.sqrt(0.1 * 0.9 / n + 0.11 * 0.89 / n)
    assert (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se == pytest.approx(0.01, rel=0.01)
