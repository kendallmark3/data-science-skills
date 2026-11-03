"""raw dataset → profiling → experiment → model evaluation → evidence-backed report, via the CLI."""

import json

import pandas as pd
import pytest

from dskit import cli

from .conftest import FIX, assert_traceable

ORIGINAL_SOURCES = {"customers.csv", "retention_offer_units.csv", "churn_model_predictions.csv",
                    "retention_offer.meta.yaml", "churn_model.claim.yaml"}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipeline")
    steps = [
        ["profile", str(FIX / "datasets/customers.csv"), "--context", str(FIX / "datasets/customers.context.yaml"),
         "--out", str(d / "1-profile")],
        ["experiment", str(FIX / "experiments/retention_offer_units.csv"), "--meta",
         str(FIX / "experiments/retention_offer.meta.yaml"), "--out", str(d / "2-experiment")],
        ["evaluate", str(FIX / "evaluations/churn_model_predictions.csv"), "--claim",
         str(FIX / "evaluations/churn_model.claim.yaml"), "--upstream", str(d / "1-profile"), "--out", str(d / "3-evaluation")],
        ["report", str(d / "1-profile"), str(d / "2-experiment"), str(d / "3-evaluation"),
         "--intent", "Should we launch model-targeted retention offers?", "--out", str(d / "4-report")],
    ]
    for s in steps:
        assert cli.main(s) == 0, s
    return d


def j(p):
    return json.loads(p.read_text())


def test_each_stage_is_traceable(pipeline):
    for stage in ("1-profile", "2-experiment", "3-evaluation", "4-report"):
        assert_traceable(j(pipeline / stage / "findings.json"), j(pipeline / stage / "evidence.json"), pipeline / stage)


def test_report_traces_to_original_sources(pipeline):
    f = j(pipeline / "4-report/findings.json")
    ev = {e["id"]: e for e in j(pipeline / "4-report/evidence.json")}
    assert len(f["summary"]["sources"]) == 3
    for fid, t in f["summary"]["traceability"].items():
        for e in t["evidence"]:
            assert e["id"] in ev
            assert ev[e["id"]]["source"] in ORIGINAL_SOURCES, (fid, ev[e["id"]]["source"])


def test_leakage_survives_two_hops(pipeline):
    """Leakage found in profiling → carried into the evaluation → appears in the report with original evidence."""
    rep = j(pipeline / "4-report/findings.json")
    ev = {e["id"]: e for e in j(pipeline / "4-report/evidence.json")}
    hop = [x for x in rep["findings"] if x["category"] == "upstream_data_risk" and "days_since_cancellation" in x["statement"]]
    assert hop, "upstream leakage finding missing from report"
    src = [ev[e] for e in hop[0]["evidence"]]
    assert all(e["id"].startswith("DP-E-") and e["source"] == "customers.csv" for e in src)
    raw = pd.read_csv(FIX / "datasets/customers.csv")
    assert any(f"populated in {raw['days_since_cancellation'].notna().sum()} rows" in e["observation"] for e in src)


def test_decisions_and_visuals_carried(pipeline):
    rep = j(pipeline / "4-report/findings.json")
    decisions = {s["skill"]: s["decision"] for s in rep["summary"]["sources"]}
    assert decisions == {"data-profiling-quality": "NOT_FIT_AS_IS", "experiment-analysis": "SUPPORTED",
                         "model-evaluation-review": "NOT_READY"}
    paths = {v["path"] for v in rep["visuals"]}
    assert {"visuals/model-segment-performance.png", "visuals/experiment-effect-vs-threshold.png",
            "visuals/decision-summary.png"} <= paths
    md = (pipeline / "4-report/report.md").read_text()
    for p in paths:
        assert (pipeline / "4-report" / p).exists() and p in md
