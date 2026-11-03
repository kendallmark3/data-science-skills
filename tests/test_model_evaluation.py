import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from dskit.evaluation import DECISIONS, classification_metrics, cohen_kappa, review, wilson
from dskit.evidence import StopCondition

from .conftest import load3, FIX, assert_traceable, load, y

PRED = FIX / "evaluations" / "churn_model_predictions.csv"


def test_metrics_match_sklearn():
    df = pd.read_csv(PRED)
    m = classification_metrics(df["y_true"], df["y_score"], 0.5)
    pred = (df["y_score"] >= 0.5).astype(int)
    assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(df["y_true"], df["y_score"]))
    assert m["pr_auc"] == pytest.approx(skm.average_precision_score(df["y_true"], df["y_score"]))
    assert m["precision"] == pytest.approx(skm.precision_score(df["y_true"], pred))
    assert m["recall"] == pytest.approx(skm.recall_score(df["y_true"], pred))
    assert m["f1"] == pytest.approx(skm.f1_score(df["y_true"], pred))
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(df["y_true"], pred))
    assert m["brier"] == pytest.approx(skm.brier_score_loss(df["y_true"], df["y_score"]))


def test_wilson_and_kappa():
    assert wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    assert cohen_kappa(a, b) == pytest.approx(skm.cohen_kappa_score(a, b))


def test_segment_failure_hidden_by_aggregate(runs):
    f, ev, out = load3(runs["eval_churn"])
    assert_traceable(f, ev, out)
    overall = f["summary"]["overall"]["recall"]
    assert overall >= 0.75                                   # aggregate passes...
    seg = [x for x in f["findings"] if x["category"] == "segment_failure"]
    assert len(seg) == 1 and "tenure_segment=new" in seg[0]["statement"]  # ...segment does not
    assert seg[0]["severity"] == "critical"
    assert f["decision"] in {"NOT_READY", "READY_WITH_CONDITIONS"} and f["decision"] != "READY"
    assert f["decision"] == "NOT_READY"


def test_segment_visual_matches_data(runs):
    f, ev, _ = load3(runs["eval_churn"])
    df = pd.read_csv(PRED)
    new = df[df["tenure_segment"] == "new"]
    recall_new = ((new["y_score"] >= 0.5) & (new["y_true"] == 1)).sum() / new["y_true"].sum()
    v = next(v for v in f["visuals"] if v["type"] == "segment_performance")
    seg_ev = next(e for e in ev if e["location"] == "segment tenure_segment=new")
    assert seg_ev["id"] in v["evidence"]
    assert seg_ev["value"]["recall"] == pytest.approx(recall_new)
    assert any(v["type"] == "confusion_matrix" for v in f["visuals"])


def test_coverage_gap_and_projection(runs):
    f, _, _ = load3(runs["eval_churn"])
    assert any(x["category"] == "coverage" and "new" in x["statement"] for x in f["findings"])
    proj = next(x for x in f["findings"] if x["category"] == "production_projection")
    assert proj["kind"] == "inference" and proj["confidence"] == "medium"


def test_upstream_profiling_risks_carried_in(runs):
    f, ev, _ = load3(runs["eval_churn"])
    up = [x for x in f["findings"] if x["category"] == "upstream_data_risk"]
    assert len(up) == 2
    ids = {e["id"] for e in ev}
    assert all(e.startswith("DP-E-") and e in ids for x in up for e in x["evidence"])
    assert any(e["source"] == "customers.csv" for e in ev)


def test_llm_strong_success_poor_groundedness(runs):
    f, ev, out = load3(runs["eval_llm"])
    assert_traceable(f, ev, out)
    assert f["summary"]["rates"]["task_success"] >= 0.90
    assert f["decision"] == "NOT_READY"
    cats = {x["category"] for x in f["findings"] if x["severity"] != "info"}
    assert {"requirement_failed", "hidden_failure", "judge_validity", "segment_failure"} <= cats
    ts = next(x for x in f["findings"] if x["statement"].startswith("`task_success`"))
    assert ts["confidence"] == "low"        # unvalidated judge lowers confidence in the headline
    assert {"llm_success_vs_groundedness", "failure_mode_distribution"} <= {v["type"] for v in f["visuals"]}


def test_llm_decision_depends_on_thresholds(runs):
    pilot = load(runs["eval_llm_pilot"])[0]
    assert pilot["decision"] == "READY_WITH_CONDITIONS"
    assert pilot["recommendations"] and all(d in DECISIONS for d in (pilot["decision"],))


def test_decision_rationale_cites_real_findings(runs):
    for name in ("eval_churn", "eval_llm", "eval_llm_pilot"):
        f = load(runs[name])[0]
        ids = {x["id"] for x in f["findings"]}
        cited = {r.split(":")[0] for r in f["decision_rationale"] if r.startswith("ME-F-")}
        assert cited and cited <= ids, name


def test_stop_and_insufficient_evidence(tmp_path):
    claim = y(FIX / "evaluations" / "churn_model.claim.yaml")
    with pytest.raises(StopCondition):
        review(PRED, {k: v for k, v in claim.items() if k != "claim"}, tmp_path / "a")
    small = pd.read_csv(PRED).head(60)
    small.to_csv(tmp_path / "small.csv", index=False)
    led = review(tmp_path / "small.csv", claim, tmp_path / "b")
    assert led.decision == "INSUFFICIENT_EVIDENCE"
    no_labels = pd.read_csv(PRED).drop(columns=["y_true"])
    no_labels.to_csv(tmp_path / "nolabel.csv", index=False)
    with pytest.raises(StopCondition):
        review(tmp_path / "nolabel.csv", claim, tmp_path / "c")
