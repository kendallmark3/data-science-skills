from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dskit import evaluation, experiment, profiling, reporting

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"


def load(out: Path) -> tuple[dict, list[dict]]:
    return json.loads((out / "findings.json").read_text()), json.loads((out / "evidence.json").read_text())


def load3(out: Path) -> tuple[dict, list[dict], Path]:
    return (*load(out), out)


def y(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    """Run every skill once on the fixtures; tests inspect the written artifacts."""
    base = tmp_path_factory.mktemp("runs")
    out = {}

    def prof(name, data, ctx):
        d = base / name
        profiling.profile(FIX / "datasets" / data, d, y(FIX / "datasets" / ctx))
        out[name] = d

    def exp(name, data, meta):
        d = base / name
        m = y(FIX / "experiments" / meta)
        m["_meta_source"] = meta
        experiment.analyze(FIX / "experiments" / data, m, d)
        out[name] = d

    def ev(name, data, claim, upstream=None):
        d = base / name
        c = y(FIX / "evaluations" / claim)
        c["_claim_source"] = claim
        evaluation.review(FIX / "evaluations" / data, c, d, upstream)
        out[name] = d

    prof("profile", "customers.csv", "customers.context.yaml")
    prof("profile_clean", "customers_clean.csv", "customers_clean.context.yaml")
    exp("exp_trivial", "checkout_button_daily.csv", "checkout_button.meta.yaml")
    exp("exp_supported", "retention_offer_units.csv", "retention_offer.meta.yaml")
    exp("exp_srm", "pricing_page_units.csv", "pricing_page.meta.yaml")
    exp("exp_contradictory", "free_shipping_units.csv", "free_shipping.meta.yaml")
    exp("exp_underpowered", "onboarding_email_units.csv", "onboarding_email.meta.yaml")
    exp("exp_no_design", "onboarding_email_units.csv", "no_design.meta.yaml")
    ev("eval_churn", "churn_model_predictions.csv", "churn_model.claim.yaml", [out["profile"]])
    ev("eval_llm", "support_assistant_llm_eval.csv", "support_assistant.claim.yaml")
    ev("eval_llm_pilot", "support_assistant_llm_eval.csv", "support_assistant_pilot.claim.yaml")
    d = base / "report_claims"
    reporting.build_report(d, [], FIX / "evidence" / "review_claims.yaml")
    out["report_claims"] = d
    d = base / "report_pipeline"
    reporting.build_report(d, [out["profile"], out["exp_supported"], out["eval_churn"]],
                           intent="Should we launch model-targeted retention offers?")
    out["report_pipeline"] = d
    return out


def assert_traceable(findings: dict, evidence: list[dict], out: Path) -> None:
    """Shared invariants every skill output must satisfy."""
    ids = {e["id"] for e in evidence}
    fids = {f["id"] for f in findings["findings"]}
    for f in findings["findings"]:
        assert f["evidence"], f"{f['id']} has no evidence"
        assert set(f["evidence"]) <= ids, f"{f['id']} cites unknown evidence"
        assert f["kind"] in ("observation", "inference")
    for v in findings["visuals"]:
        assert v["evidence"] and set(v["evidence"]) <= ids, f"{v['id']} cites unknown evidence"
        p = out / v["path"]
        assert p.exists() and p.stat().st_size > 5_000, f"{v['path']} missing or empty"
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for r in findings["recommendations"]:
        assert r["based_on"] and set(r["based_on"]) <= fids, f"{r['id']} not based on findings"
        assert r["action"] not in {f["statement"] for f in findings["findings"]}
