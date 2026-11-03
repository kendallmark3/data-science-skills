import json

import pandas as pd
import pytest

from dskit import cli
from dskit.evidence import Ledger, StopCondition, TraceabilityError
from dskit.profiling import profile

from .conftest import load3, FIX, assert_traceable, load

RAW = FIX / "datasets" / "customers.csv"


@pytest.fixture(scope="module")
def prof(runs):
    return load(runs["profile"]) + (runs["profile"],)


def by_cat(findings, cat):
    return [f for f in findings["findings"] if f["category"] == cat]


def test_seeded_missingness_detected(prof):
    f, ev, _ = prof
    raw = pd.read_csv(RAW, dtype=str)
    email = [x for x in by_cat(f, "missingness") if "`email`" in x["statement"]]
    assert email and email[0]["severity"] == "high"
    e = next(e for e in ev if e["id"] == email[0]["evidence"][0])
    assert e["value"] == pytest.approx(raw["email"].isna().mean(), abs=1e-6)


def test_duplicates_detected(prof):
    f, ev, _ = prof
    dup = by_cat(f, "duplicates")
    assert any("6 rows are exact duplicates" in x["statement"] for x in dup)
    assert any("12 values repeat" in x["statement"] for x in dup)
    contamination = by_cat(f, "contamination")
    assert contamination and "6 `customer_id` values" in contamination[0]["statement"]


def test_impossible_values_detected(prof):
    f, _, _ = prof
    rv = by_cat(f, "range_violation")
    assert rv and "[-3, 142, 230]" in rv[0]["statement"] and rv[0]["severity"] == "high"


def test_dates_categories_formats_constant(prof):
    f, _, _ = prof
    assert any("5 malformed dates" in x["statement"] for x in by_cat(f, "date_anomaly"))
    assert any("premuim" in x["statement"] for x in by_cat(f, "category_anomaly"))
    fmt = " ".join(x["statement"] for x in by_cat(f, "formatting"))
    assert "Premium " in fmt and "monthly_spend" in fmt and "20 values" in fmt
    assert any("account_currency" in x["statement"] for x in by_cat(f, "constant"))


def test_leakage_detected_as_inference(prof):
    f, _, _ = prof
    leak = by_cat(f, "leakage")
    assert len(leak) == 1 and "days_since_cancellation" in leak[0]["statement"]
    assert leak[0]["severity"] == "critical" and leak[0]["kind"] == "inference"
    # structural missingness is reported as leakage, not double-counted as missingness
    assert not any("days_since_cancellation" in x["statement"] for x in by_cat(f, "missingness"))


def test_evidence_and_visuals_traceable(prof):
    f, ev, out = prof
    assert_traceable(f, ev, out)
    types = {v["type"] for v in f["visuals"]}
    assert {"missingness_chart", "category_frequency", "range_violation", "quality_scorecard"} <= types
    miss_v = next(v for v in f["visuals"] if v["type"] == "missingness_chart")
    email_ev = next(x for x in by_cat(f, "missingness") if "`email`" in x["statement"])["evidence"][0]
    assert email_ev in miss_v["evidence"]


def test_no_unsupported_critical_findings(prof):
    f, _, _ = prof
    crit = {x["category"] for x in f["findings"] if x["severity"] == "critical"}
    assert crit == {"leakage", "contamination"}  # exactly the seeded critical problems
    assert f["decision"] == "NOT_FIT_AS_IS"


def test_clean_dataset_not_flagged(runs):
    f, ev, out = load3(runs["profile_clean"])
    assert_traceable(f, ev, out)
    assert [x for x in f["findings"] if x["severity"] != "info"] == []
    assert f["decision"] == "FIT_FOR_USE"
    assert f["visuals"] == []          # nothing worth drawing


def test_unreadable_file_stops(tmp_path):
    with pytest.raises(StopCondition):
        profile(FIX / "datasets" / "unreadable.csv", tmp_path / "x", {})
    code = cli.main(["profile", str(FIX / "datasets" / "unreadable.csv"), "--out", str(tmp_path / "cli")])
    assert code == 2
    out = json.loads((tmp_path / "cli" / "findings.json").read_text())
    assert out["status"] == "STOPPED" and out["decision"] == "INSUFFICIENT_EVIDENCE" and out["findings"] == []


def test_missing_context_is_a_gap_not_an_error(tmp_path):
    led = profile(RAW, tmp_path / "noctx", {})
    assert any("No target column" in g for g in led.gaps)
    assert not led.findings_by("leakage")      # cannot claim leakage without a target


def test_ledger_refuses_dangling_references():
    led = Ledger("t", "T")
    with pytest.raises(TraceabilityError):
        led.add_finding("x", ["T-E-999"], category="c")
    with pytest.raises(TraceabilityError):
        led.add_finding("x", [], category="c")
    e = led.add_evidence("s", "l", "o", "m", confidence="low")
    f = led.add_finding("x", [e], category="c", confidence="high")
    assert led.findings[0].confidence == "low"   # capped by evidence
    with pytest.raises(TraceabilityError):
        led.recommend("do it", [])
    with pytest.raises(TraceabilityError):
        led.add_visual("v", "p.png", ["T-E-404"], "p")
    led.recommend("do it", [f])
