import re

import pytest

from dskit.evidence import StopCondition
from dskit.reporting import build_report

from .conftest import assert_traceable, load


@pytest.fixture(scope="module")
def rep(runs):
    f, ev = load(runs["report_claims"])
    md = (runs["report_claims"] / "report.md").read_text()
    return f, ev, md, runs["report_claims"]


def section(md, title):
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1) if m else ""


def finding_for(f, claim_id):
    fid = next((k for k, t in f["summary"]["traceability"].items() if t["source_finding"] == claim_id), None)
    return next((x for x in f["findings"] if x["id"] == fid), None)


def test_findings_reference_evidence(rep):
    f, ev, _, out = rep
    assert_traceable(f, ev, out)
    for fid, t in f["summary"]["traceability"].items():
        assert t["evidence"] and all(e["source"] for e in t["evidence"])


def test_unsupported_claims_excluded(rep):
    f, _, md, _ = rep
    unsupported = {u["id"] for u in f["summary"]["unsupported_claims"]}
    assert unsupported == {"C-5", "C-6"}
    statements = " ".join(x["statement"] for x in f["findings"])
    assert "$2M" not in statements and "3x more likely" not in statements
    assert "$2M" not in section(md, "Key Findings") and "$2M" in section(md, "Unsupported Claims")
    assert "not presented as a finding" in section(md, "Unsupported Claims")


def test_low_confidence_evidence_stays_low(rep):
    f, _, md, _ = rep
    c2 = finding_for(f, "C-2")
    assert c2["confidence"] == "low"                     # stated "high", evidence is low
    block = md[md.index(f"### {c2['id']}"):].split("###")[1]
    assert "Weak evidence indicates" in block
    assert "The evidence shows" not in block
    assert "confidence reduced from high to low" in block
    assert finding_for(f, "C-1")["confidence"] == "high"


def test_conflicts_shown_not_resolved(rep):
    f, _, md, _ = rep
    conflicts = f["summary"]["conflicts"]
    assert len(conflicts) == 1 and conflicts[0]["topic"] == "support_tickets"
    for fid in conflicts[0]["findings"]:
        assert next(x for x in f["findings"] if x["id"] == fid)["confidence"] == "low"
        assert f"### {fid}" not in section(md, "Key Findings")
    assert "Neither claim is adopted" in section(md, "Conflicts")


def test_recommendations_distinct_and_withheld_when_unsupported(rep):
    f, _, md, _ = rep
    withheld = f["summary"]["withheld_recommendations"]
    assert len(withheld) == 1 and "C-5" in withheld[0]["reason"]
    assert all("Expand the retention offer" not in r["action"] for r in f["recommendations"])
    assert "Expand the retention offer" not in section(md, "Key Findings")
    rec_actions = {r["action"] for r in f["recommendations"]}
    assert not rec_actions & {x["statement"] for x in f["findings"]}


def test_visual_traceability(rep):
    f, ev, _, out = rep
    assert [v["purpose"].split(" (")[0] for v in f["visuals"]] == ["Renewal rate by experiment arm"]
    excluded = f["summary"]["excluded_visuals"]
    assert len(excluded) == 1 and excluded[0]["id"] == "XV-2"
    assert f["decision"] == "REPORTED_WITH_EXCLUSIONS"


def test_inference_labelled_in_prose(runs):
    md = (runs["report_pipeline"] / "report.md").read_text()
    f, _ = load(runs["report_pipeline"])
    for x in f["findings"]:
        if x["kind"] == "inference":
            assert f"**Claim:** Inference: {x['statement']}" in md or f"### {x['id']}" not in section(md, "Key Findings")


def test_reporting_stops_without_inputs(tmp_path):
    with pytest.raises(StopCondition):
        build_report(tmp_path / "x", [], None)
    with pytest.raises(StopCondition):
        build_report(tmp_path / "y", [tmp_path], None)
