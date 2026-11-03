"""Skill 3 - Model & Evaluation Review.

Answers: does this evaluation prove the readiness claim being made?
Aggregate metrics are computed, then deliberately broken apart by segment,
by failure mode and against the production population.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import render, viz
from .evidence import Ledger, StopCondition, load_run

DECISIONS = ("READY", "READY_WITH_CONDITIONS", "NOT_READY", "INSUFFICIENT_EVIDENCE")


# ---------------------------------------------------------------------------
# Metric functions (deterministic; tests check them against scikit-learn)
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def roc_auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def ece(y, s, bins: int = 10) -> float:
    y, s = np.asarray(y), np.asarray(s)
    idx = np.minimum((s * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - s[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def classification_metrics(y, s, thr: float) -> dict:
    y, s = np.asarray(y).astype(int), np.asarray(s).astype(float)
    p = (s >= thr).astype(int)
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(y), "positives": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "roc_auc": roc_auc(y, s), "pr_auc": average_precision(y, s),
            "brier": float(np.mean((s - y) ** 2)), "ece": ece(y, s)}


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# ---------------------------------------------------------------------------
def review(data_path: str | Path, claim: dict, out_dir: str | Path, upstream: list[str | Path] | None = None) -> Ledger:
    p = Path(data_path)
    if not p.exists():
        raise StopCondition(f"Evaluation data not found: {p}")
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001
        raise StopCondition(f"Could not read {p.name}: {exc}") from exc
    if not claim.get("claim"):
        raise StopCondition("No claim supplied: an evaluation can only be reviewed against the claim it is meant to support.")
    led = Ledger("model-evaluation-review", "ME", intent=claim["claim"])
    led.summary["mode"] = claim.get("mode", "classification")
    csrc = claim.get("_claim_source", "claim")
    e_claim = led.add_evidence(csrc, "claim", f"Claim: {claim['claim']} Requirements: {claim.get('requirements', {})}",
                               "declared by the evaluation owner")
    if not claim.get("requirements"):
        led.gaps.append("No quantitative requirements supplied; readiness is judged against generic expectations only.")
    out = Path(out_dir)
    if claim.get("mode") == "llm":
        _review_llm(led, df, p.name, claim, e_claim, out)
    else:
        _review_classifier(led, df, p.name, claim, e_claim, out)
    for up in upstream or []:
        _import_upstream(led, up)
    _decide(led, claim)
    led.write(out_dir, _report(led, claim))
    return led


def _coverage(led, df, src, claim, e_claim):
    """Compare the evaluation population with the declared production population."""
    out = []
    for col, mix in (claim.get("production_mix") or {}).items():
        if col not in df:
            led.gaps.append(f"Production mix given for `{col}` but the column is absent from the evaluation data.")
            continue
        ev_mix = df[col].value_counts(normalize=True).to_dict()
        tvd = 0.5 * sum(abs(ev_mix.get(k, 0) - v) for k, v in mix.items())
        eid = led.add_evidence(src, f"column={col}", f"eval mix {({k: round(v, 3) for k, v in ev_mix.items()})} vs production "
                               f"{mix}; total variation distance {tvd:.2f}", "value_counts vs declared production mix", value=tvd)
        under = [k for k, v in mix.items() if ev_mix.get(k, 0) < 0.7 * v]
        missing = [k for k in mix if k not in ev_mix]
        if under or missing:
            out.append(led.add_finding(
                f"Evaluation under-represents production traffic for `{col}`: {under or missing} "
                f"(TVD {tvd:.2f}).", [eid, e_claim], category="coverage", severity="moderate" if not missing else "high",
                risk="Aggregate metrics are weighted toward the easier, over-represented population.",
                recommended_check="Re-weight metrics to the production mix or sample a representative eval set."))
    if not claim.get("production_mix"):
        led.gaps.append("No production population described; representativeness of the evaluation set is unknown.")
    return out


def _review_classifier(led, df, src, claim, e_claim, out):
    yc, sc = claim.get("label_column", "y_true"), claim.get("score_column", "y_score")
    if yc not in df or sc not in df:
        raise StopCondition(f"Evaluation data needs label column {yc!r} and score column {sc!r} to verify any metric.")
    df = df.dropna(subset=[yc, sc])
    thr = float(claim.get("threshold", 0.5))
    req = claim.get("requirements", {})
    m = classification_metrics(df[yc], df[sc], thr)
    led.summary["overall"] = m
    led.summary["n_cases"] = m["n"]
    e_all = led.add_evidence(src, f"all rows; threshold={thr}",
                             f"n={m['n']:,}, prevalence {m['positives'] / m['n']:.1%}; ROC-AUC {m['roc_auc']:.3f}, PR-AUC {m['pr_auc']:.3f}, "
                             f"precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}, accuracy {m['accuracy']:.3f}",
                             "recomputed from labels and scores", value=m)
    e_cm = led.add_evidence(src, f"all rows; threshold={thr}", f"TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']}",
                            "confusion matrix at threshold")
    # requirements, overall
    for key, metric in (("min_recall", "recall"), ("min_precision", "precision"), ("min_roc_auc", "roc_auc"), ("min_f1", "f1")):
        if key in req:
            ok = m[metric] >= req[key]
            led.add_finding(f"Overall {metric} {m[metric]:.3f} {'meets' if ok else 'fails'} the requirement ≥ {req[key]}.",
                                [e_all, e_claim], category="requirement" if ok else "requirement_failed",
                                severity="info" if ok else "high")
    # majority-class and supplied baselines
    maj = max(m["positives"], m["n"] - m["positives"]) / m["n"]
    e_maj = led.add_evidence(src, f"column={yc}", f"majority-class accuracy {maj:.3f} vs model accuracy {m['accuracy']:.3f}",
                             "always-predict-majority baseline")
    if m["accuracy"] - maj < 0.05:
        led.add_finding(f"Accuracy ({m['accuracy']:.3f}) barely beats always predicting the majority class ({maj:.3f}).",
                        [e_maj], category="baseline", severity="moderate")
    bc = claim.get("baseline_score_column")
    if bc and bc in df:
        b = classification_metrics(df[yc], df[bc], thr)
        led.summary["baseline"] = b
        e_b = led.add_evidence(src, f"column={bc}", f"baseline ROC-AUC {b['roc_auc']:.3f}, PR-AUC {b['pr_auc']:.3f}, "
                               f"recall {b['recall']:.3f}, precision {b['precision']:.3f}", "same metrics on baseline scores", value=b)
        led.summary["e_baseline"] = e_b
        led.add_finding(f"Candidate improves on the baseline: ROC-AUC {b['roc_auc']:.3f} → {m['roc_auc']:.3f}, "
                        f"PR-AUC {b['pr_auc']:.3f} → {m['pr_auc']:.3f}.", [e_all, e_b], category="baseline",
                        severity="info" if m["roc_auc"] > b["roc_auc"] else "high")
    else:
        led.gaps.append("No baseline model supplied; improvement over the current approach is unproven.")
    if m["roc_auc"] >= 0.995:
        led.add_finding(f"ROC-AUC {m['roc_auc']:.3f} is implausibly high for most real tasks; check for leakage.", [e_all],
                        category="leakage", severity="high", kind="inference")
    e_cal = led.add_evidence(src, f"column={sc}", f"ECE {m['ece']:.3f}, Brier {m['brier']:.3f}", "10-bin expected calibration error")
    if m["ece"] > 0.1:
        led.add_finding(f"Scores are poorly calibrated (ECE {m['ece']:.3f}); do not read them as probabilities.", [e_cal],
                        category="calibration", severity="moderate")

    # segments
    crit = claim.get("critical_segments", {})
    seg_rows = []
    for col in claim.get("segment_columns", []) or []:
        if col not in df:
            led.gaps.append(f"Segment column `{col}` not present.")
            continue
        for val, g in df.groupby(col):
            sm = classification_metrics(g[yc], g[sc], thr)
            lo, hi = wilson(sm["tp"], sm["tp"] + sm["fn"])
            eid = led.add_evidence(src, f"segment {col}={val}", f"n={sm['n']:,}, positives={sm['positives']}; recall {sm['recall']:.3f} "
                                   f"[{lo:.3f}, {hi:.3f}], precision {sm['precision']:.3f}, ROC-AUC {sm['roc_auc']:.3f}",
                                   "metrics recomputed within segment; Wilson 95% CI for recall", value=sm)
            seg_rows.append({"col": col, "val": str(val), "m": sm, "lo": lo, "hi": hi, "e": eid,
                             "critical": str(val) in map(str, crit.get(col, []))})
    led.summary["segments"] = [{k: v for k, v in r.items() if k != "m"} | {"recall": r["m"]["recall"], "n": r["m"]["n"]} for r in seg_rows]
    min_rec = req.get("min_recall")
    for r in seg_rows:
        if min_rec is None:
            continue
        if r["hi"] < min_rec:
            led.add_finding(
                f"Segment `{r['col']}={r['val']}` fails the recall requirement: {r['m']['recall']:.3f} (95% CI upper bound "
                f"{r['hi']:.3f} < {min_rec}) while overall recall is {m['recall']:.3f}. The aggregate hides this failure.",
                [r["e"], e_all, e_claim], category="segment_failure", severity="critical" if r["critical"] else "high",
                risk=f"Most at-risk `{r['val']}` cases would be missed in production.",
                recommended_check="Inspect features and label quality for this segment; consider a segment-specific threshold or model.")
        elif r["m"]["recall"] < min_rec:
            led.add_finding(f"Segment `{r['col']}={r['val']}` recall {r['m']['recall']:.3f} is below {min_rec}, but the segment is "
                            f"too small to confirm (CI upper {r['hi']:.3f}).", [r["e"]], category="segment_uncertain",
                            severity="moderate", kind="inference")
    _coverage(led, df, src, claim, e_claim)
    # Re-weight recall to production mix when both are known - an inference, labelled as such.
    for col, mix in (claim.get("production_mix") or {}).items():
        rows = {r["val"]: r for r in seg_rows if r["col"] == col}
        if rows and set(mix) <= set(rows):
            pw = sum(mix[k] * rows[k]["m"]["recall"] for k in mix)
            eid = led.add_evidence(src, f"segments of {col}", f"recall re-weighted to production mix {mix}: {pw:.3f}",
                                   "Σ production_share × segment recall (assumes per-segment recall transfers)", confidence="medium", value=pw)
            led.assumptions.append("Per-segment recall measured in evaluation is assumed to transfer to production.")
            led.add_finding(f"At the declared production mix, expected recall falls to about {pw:.3f} "
                            f"(evaluation-weighted: {m['recall']:.3f}).", [eid] + [rows[k]["e"] for k in mix],
                            kind="inference", category="production_projection", confidence="medium",
                            severity="high" if min_rec and pw < min_rec else "info")
    _classifier_visuals(led, out, df, yc, sc, thr, m, seg_rows, min_rec, e_all, e_cm)


def _review_llm(led, df, src, claim, e_claim, out):
    need = [c for c in ("task_success",) if c not in df]
    if need:
        raise StopCondition(f"LLM evaluation lacks {need}; task success cannot be verified.")
    req = claim.get("requirements", {})
    n = led.summary["n_cases"] = len(df)
    rates = {}
    for col in ("task_success", "grounded", "unsupported_claim", "format_valid"):
        if col in df:
            k = int(df[col].sum())
            lo, hi = wilson(k, n)
            rates[col] = (k / n, lo, hi, led.add_evidence(src, f"column={col}", f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]",
                                                         "mean of per-case judgement; Wilson 95% CI", value=k / n))
        else:
            led.gaps.append(f"No `{col}` judgements; that dimension is unevaluated.")
    led.summary["rates"] = {k: v[0] for k, v in rates.items()}
    checks = [("min_task_success", "task_success", ">="), ("min_groundedness", "grounded", ">="),
              ("max_unsupported_claim_rate", "unsupported_claim", "<="), ("min_format_valid", "format_valid", ">=")]
    for key, col, op in checks:
        if key in req and col in rates:
            v, lo, hi, eid = rates[col]
            ok = v >= req[key] if op == ">=" else v <= req[key]
            led.add_finding(f"`{col}` rate {v:.1%} {'meets' if ok else 'fails'} the requirement {op} {req[key]:.0%}.",
                                [eid, e_claim], category="requirement" if ok else "requirement_failed",
                                severity="info" if ok else ("critical" if col in ("grounded", "unsupported_claim") else "high"))
    # hidden weakness: "successful" answers that are not grounded
    if "grounded" in df:
        hidden = int(((df["task_success"] == 1) & (df["grounded"] == 0)).sum())
        succ = int(df["task_success"].sum())
        eid = led.add_evidence(src, "columns=task_success,grounded", f"{hidden} of {succ} cases judged successful were not grounded "
                               f"({hidden / max(succ, 1):.1%})", "cross-tab task_success × grounded", value=hidden)
        led.summary["e_hidden"] = eid
        if hidden / max(succ, 1) > 0.05:
            led.add_finding(f"Task success overstates quality: {hidden / succ:.1%} of 'successful' answers are not grounded in the "
                            "source corpus.", [eid, rates["task_success"][3]], kind="inference", category="hidden_failure",
                            severity="high", risk="Users receive confident, plausible answers that the corpus does not support.")
    # refusal behaviour
    if {"should_refuse", "refused"} <= set(df):
        sr = df["should_refuse"] == 1
        miss = int((sr & (df["refused"] == 0)).sum())
        over = int((~sr & (df["refused"] == 1)).sum())
        eid = led.add_evidence(src, "columns=should_refuse,refused", f"missed refusals {miss}/{int(sr.sum())}; over-refusals "
                               f"{over}/{int((~sr).sum())}", "cross-tab", value=miss)
        led.summary["refusal"] = {"missed": miss, "should": int(sr.sum()), "over": over}
        if miss:
            led.add_finding(f"{miss} of {int(sr.sum())} cases that required a refusal were answered.", [eid],
                            category="refusal", severity="high" if miss / max(sr.sum(), 1) > 0.1 else "moderate")
    # human vs judge agreement
    if "human_task_success" in df:
        h = df.dropna(subset=["human_task_success"])
        kappa = cohen_kappa(h["task_success"], h["human_task_success"])
        agree = float((h["task_success"] == h["human_task_success"]).mean())
        eid = led.add_evidence(src, "columns=task_success,human_task_success", f"{len(h)} double-labelled cases; agreement "
                               f"{agree:.1%}, Cohen's κ = {kappa:.2f}", "judge vs human labels", value=kappa)
        if kappa < 0.6:
            led.add_finding(f"The task-success judge agrees weakly with human reviewers (κ = {kappa:.2f}, n = {len(h)}); "
                            "the headline task-success rate is not validated.", [eid], kind="inference",
                            category="judge_validity", severity="high")
            led.findings[next(i for i, f in enumerate(led.findings) if rates["task_success"][3] in f.evidence)].confidence = "low"
    else:
        led.gaps.append("No human labels; the automated judge is unvalidated.")
    # latency / cost
    if "latency_ms" in df:
        p50, p95 = df["latency_ms"].quantile([0.5, 0.95])
        eid = led.add_evidence(src, "column=latency_ms", f"p50 {p50:.0f} ms, p95 {p95:.0f} ms", "quantiles", value=float(p95))
        if "max_p95_latency_ms" in req:
            ok = p95 <= req["max_p95_latency_ms"]
            led.add_finding(f"p95 latency {p95:.0f} ms {'meets' if ok else 'fails'} ≤ {req['max_p95_latency_ms']} ms.",
                            [eid, e_claim], category="requirement" if ok else "requirement_failed", severity="info" if ok else "high")
    if "cost_usd" in df:
        c = float(df["cost_usd"].mean())
        eid = led.add_evidence(src, "column=cost_usd", f"mean ${c:.4f} per case", "mean", value=c)
        if "max_cost_per_case_usd" in req:
            ok = c <= req["max_cost_per_case_usd"]
            led.add_finding(f"Mean cost ${c:.4f}/case {'meets' if ok else 'fails'} ≤ ${req['max_cost_per_case_usd']}.",
                            [eid, e_claim], category="requirement" if ok else "requirement_failed", severity="info" if ok else "high")
    # reproducibility
    if "model_version" in df:
        vers = df["model_version"].value_counts().to_dict()
        if len(vers) > 1:
            eid = led.add_evidence(src, "column=model_version", f"{len(vers)} model versions in one eval run: {vers}", "value_counts")
            led.add_finding("The evaluation mixes model versions; results do not describe a single reproducible system.", [eid],
                            category="reproducibility", severity="moderate",
                            recommended_check="Re-run the full set against the exact version being released.")
    else:
        led.gaps.append("Model version not recorded; results cannot be tied to a release.")
    # per-category
    seg_rows = []
    for col in claim.get("segment_columns", []) or []:
        for val, g in df.groupby(col):
            row = {"col": col, "val": str(val), "n": len(g)}
            for metric in ("task_success", "grounded"):
                if metric in g:
                    k = int(g[metric].sum())
                    row[metric] = (k / len(g), *wilson(k, len(g)))
            row["e"] = led.add_evidence(src, f"segment {col}={val}", f"n={len(g)}; " + ", ".join(
                f"{mtr} {row[mtr][0]:.1%} [{row[mtr][1]:.1%}, {row[mtr][2]:.1%}]" for mtr in ("task_success", "grounded") if mtr in row),
                "per-segment rates; Wilson 95% CI")
            seg_rows.append(row)
            if len(g) < 30:
                led.gaps.append(f"Only {len(g)} cases for {col}={val}; too few to evaluate that slice.")
    if "min_groundedness" in req:
        for r in seg_rows:
            if "grounded" in r and r["grounded"][2] < req["min_groundedness"]:
                led.add_finding(
                    f"`{r['col']}={r['val']}` groundedness {r['grounded'][0]:.1%} (CI upper {r['grounded'][2]:.1%}) is confirmed "
                    f"below the {req['min_groundedness']:.0%} requirement.", [r["e"], e_claim], category="segment_failure",
                    severity="high", risk="This question type is where unsupported answers concentrate.")
    led.summary["segments"] = seg_rows
    _coverage(led, df, src, claim, e_claim)
    _llm_visuals(led, out, df, seg_rows, req)


def _import_upstream(led, run_dir):
    findings, evidence = load_run(run_dir)
    ev = {e["id"]: e for e in evidence}
    hits = [f for f in findings.get("findings", []) if f["category"] in ("leakage", "contamination") and f["severity"] in ("high", "critical")]
    led.upstream.append(str(run_dir))
    for f in hits:
        ids = [led.import_evidence(ev[e]) for e in f["evidence"] if e in ev]
        led.add_finding(f"Upstream data-quality finding {f['id']} affects this evaluation: {f['statement']}", ids, kind="inference",
                        category="upstream_data_risk", severity="high",
                        risk="Evaluation metrics may be inflated by leakage or train/test contamination in the source data.")


def _decide(led, claim):
    F = led.findings
    fails = [f for f in F if f.category == "requirement_failed"]
    crit_seg = [f for f in F if f.category == "segment_failure" and f.severity == "critical"]
    conditions = [f for f in F if f.severity in ("high", "moderate") and f.category not in ("requirement_failed",) and f not in crit_seg]
    n = led.summary["n_cases"]
    if n < 100:
        led.decision, led.decision_confidence = "INSUFFICIENT_EVIDENCE", "high"
        led.decision_rationale = [f"Only {n:.0f} evaluation cases; too few to support a readiness claim."]
    elif fails or crit_seg:
        led.decision, led.decision_confidence = "NOT_READY", "high"
        led.decision_rationale = [f"{f.id}: {f.statement}" for f in fails + crit_seg]
        led.recommend("Do not release against this claim. Fix the failing requirement(s) and re-evaluate.", [f.id for f in fails + crit_seg])
    elif conditions:
        led.decision, led.decision_confidence = "READY_WITH_CONDITIONS", "medium"
        led.decision_rationale = ["All stated requirements pass in aggregate, but:"] + [f"{f.id}: {f.statement}" for f in conditions]
        led.recommend("Release only with the listed conditions addressed or explicitly accepted by the owner.", [f.id for f in conditions])
    else:
        led.decision, led.decision_confidence = "READY", "medium" if led.gaps else "high"
        led.decision_rationale = ["All stated requirements pass overall and in every evaluated segment."]
    seg = [f.id for f in F if f.category == "segment_failure"]
    if seg:
        led.recommend("Build a targeted evaluation set for the failing segment(s) and report their metrics separately.", seg)
    cov = [f.id for f in F if f.category == "coverage"]
    if cov:
        led.recommend("Re-sample or re-weight the evaluation set to match production traffic.", cov)
    judge = [f.id for f in F if f.category in ("judge_validity", "hidden_failure")]
    if judge:
        led.recommend("Validate the automated judge against a larger human-labelled sample; report groundedness alongside task success.", judge)


def _classifier_visuals(led, out, df, yc, sc, thr, m, seg_rows, min_rec, e_all, e_cm):
    if seg_rows and min_rec is not None:
        rows = [{"label": "overall", "rec": m["recall"], "lo": wilson(m["tp"], m["tp"] + m["fn"])[0],
                 "hi": wilson(m["tp"], m["tp"] + m["fn"])[1], "bad": False}] + [
            {"label": f"{r['col']}={r['val']}", "rec": r["m"]["recall"], "lo": r["lo"], "hi": r["hi"], "bad": r["hi"] < min_rec}
            for r in seg_rows]
        fig, ax = viz.figure(7.5, 0.42 * len(rows) + 1.5)
        for i, r in enumerate(rows[::-1]):
            col = viz.CRITICAL if r["bad"] else (viz.INK if r["label"] == "overall" else viz.SERIES[0])
            ax.errorbar([r["rec"]], [i], xerr=[[r["rec"] - r["lo"]], [r["hi"] - r["rec"]]], fmt="o", color=col, capsize=4, ms=7)
            ax.text(r["hi"] + 0.01, i, f"{r['rec']:.2f}", va="center", fontsize=9, color=viz.INK_2)
        ax.axvline(min_rec, color=viz.CRITICAL, ls="--", lw=1.4, label=f"required recall ≥ {min_rec}")
        ax.legend(loc="lower left")
        ax.set_yticks(range(len(rows)), [r["label"] for r in rows[::-1]])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(f"recall at threshold {thr} (95% Wilson CI)")
        ax.set_title("Recall by segment (red = confirmed below requirement)")
        ev = [e_all] + [r["e"] for r in seg_rows]
        path = viz.save(fig, out, "segment-performance.png", "Source: " + ", ".join(ev))
        led.add_visual("segment_performance", path, ev, "Expose segments whose recall the aggregate number hides")
    fails = [r for r in seg_rows if min_rec is not None and r["hi"] < min_rec]
    if fails:
        r = fails[0]
        fig, axes = viz.figure(7.5, 3.2)
        fig.clf()
        axes = fig.subplots(1, 2)
        for ax, (title, mm) in zip(axes, [("overall", m), (f"{r['col']}={r['val']}", r["m"])]):
            mat = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]])
            norm = mat / mat.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap=matplotlib_blues(), vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(mat):
                ax.text(j, i, f"{v:,}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.4 else viz.INK)
            ax.set_xticks([0, 1], ["pred 0", "pred 1"])
            ax.set_yticks([0, 1], ["actual 0", "actual 1"])
            ax.grid(False)
            ax.set_title(title, fontsize=10)
        fig.suptitle("Confusion matrices (row-normalized)", x=0.02, ha="left", fontsize=12, fontweight="bold")
        path = viz.save(fig, out, "confusion-matrix.png", f"Source: {e_cm}, {r['e']}")
        led.add_visual("confusion_matrix", path, [e_cm, r["e"]], "Compare missed positives overall vs the failing segment")
    if "baseline" in led.summary:
        b = led.summary["baseline"]
        keys = ["roc_auc", "pr_auc", "precision", "recall"]
        fig, ax = viz.figure(6.5, 3.0)
        x = np.arange(len(keys))
        ax.bar(x - 0.18, [b[k] for k in keys], 0.34, color=viz.NEUTRAL, label="baseline")
        ax.bar(x + 0.18, [m[k] for k in keys], 0.34, color=viz.SERIES[0], label="candidate")
        ax.set_xticks(x, keys)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="x", visible=False)
        ax.set_title("Candidate vs baseline (all rows)")
        path = viz.save(fig, out, "baseline-vs-candidate.png", f"Source: {e_all}, {led.summary['e_baseline']}")
        led.add_visual("baseline_comparison", path, [e_all, led.summary["e_baseline"]], "Show improvement over the baseline")


def matplotlib_blues():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"])


def _llm_visuals(led, out, df, seg_rows, req):
    if seg_rows and "grounded" in seg_rows[0]:
        fig, ax = viz.figure(7.5, 3.4)
        x = np.arange(len(seg_rows))
        for off, metric, color, label in ((-0.18, "task_success", viz.SERIES[0], "task success"),
                                          (0.18, "grounded", viz.SERIES[1], "groundedness")):
            v = [r[metric][0] for r in seg_rows]
            err = [[r[metric][0] - r[metric][1] for r in seg_rows], [r[metric][2] - r[metric][0] for r in seg_rows]]
            ax.bar(x + off, v, 0.34, color=color, label=label, yerr=err, error_kw={"ecolor": viz.MUTED, "capsize": 3, "lw": 1})
        if "min_groundedness" in req:
            ax.axhline(req["min_groundedness"], color=viz.CRITICAL, ls="--", lw=1.2,
                       label=f"required groundedness ≥ {req['min_groundedness']:.0%}")
        ax.set_xticks(x, [r["val"] for r in seg_rows])
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.grid(axis="x", visible=False)
        ax.set_title("Task success vs groundedness by category")
        ev = [r["e"] for r in seg_rows]
        path = viz.save(fig, out, "success-vs-groundedness.png", "Bars = rate, whiskers = 95% Wilson CI. Source: " + ", ".join(ev))
        led.add_visual("llm_success_vs_groundedness", path, ev, "Show where high task success coexists with ungrounded answers")
    modes = {}
    if "grounded" in df:
        modes["successful but ungrounded"] = int(((df["task_success"] == 1) & (df["grounded"] == 0)).sum())
    if "unsupported_claim" in df:
        modes["unsupported claim"] = int(df["unsupported_claim"].sum())
    modes["task failed"] = int((df["task_success"] == 0).sum())
    if "format_valid" in df:
        modes["invalid format"] = int((df["format_valid"] == 0).sum())
    if "refusal" in led.summary:
        modes["missed refusal"] = led.summary["refusal"]["missed"]
        modes["over-refusal"] = led.summary["refusal"]["over"]
    ev = [e.id for e in led.evidence if e.location.startswith("column") or e.location.startswith("columns")]
    if sum(modes.values()):
        items = sorted(modes.items(), key=lambda kv: kv[1])
        fig, ax = viz.figure(7.0, 0.42 * len(items) + 1.4)
        ax.barh([k for k, _ in items], [v for _, v in items], color=viz.SERIES[0], height=0.6)
        for i, (_, v) in enumerate(items):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=9, color=viz.INK_2)
        ax.set_xlabel(f"cases (of {len(df)})")
        ax.grid(axis="y", visible=False)
        ax.set_title("Failure modes (a case can appear in several)")
        path = viz.save(fig, out, "failure-modes.png", "Source: " + ", ".join(ev))
        led.add_visual("failure_mode_distribution", path, ev, "Rank failure modes by frequency")


def _report(led, claim) -> str:
    F = led.findings
    cat = lambda *c: [f for f in F if f.category in c]  # noqa: E731
    s = led.summary
    if s["mode"] == "classification" and "overall" in s:
        m = s["overall"]
        metrics = ("| Metric | Value |\n|---|---|\n" + "\n".join(
            f"| {k} | {m[k]:.3f} |" for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "brier", "ece")) + "\n")
        if s.get("segments"):
            metrics += "\n| Segment | n | recall | 95% CI |\n|---|---|---|---|\n" + "\n".join(
                f"| {r['col']}={r['val']} | {r['n']:,} | {r['recall']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] |" for r in s["segments"]) + "\n"
    else:
        metrics = "| Metric | Rate |\n|---|---|\n" + "\n".join(f"| {k} | {v:.1%} |" for k, v in s.get("rates", {}).items()) + "\n"
        if s.get("segments"):
            metrics += "\n| Category | n | task success | groundedness |\n|---|---|---|---|\n" + "\n".join(
                f"| {r['val']} | {r['n']} | {r['task_success'][0]:.1%} | {r['grounded'][0]:.1%} |" for r in s["segments"] if "grounded" in r) + "\n"
    strong = [f for f in F if f.category in ("requirement", "baseline") and f.severity == "info"]
    weak = [f for f in F if f.severity != "info" and f.category in ("requirement_failed", "baseline", "calibration")]
    parts = [
        f"# Evaluation Review\n\n**Decision:** `{led.decision}` (confidence `{led.decision_confidence}`)\n",
        f"## Claim Being Evaluated\n\n> {claim['claim']}\n\nDoes this evaluation prove that claim?\n",
        f"## Evaluation Design\n\nMode: {s['mode']}; task: {claim.get('task', 'not stated')}; requirements: `{claim.get('requirements', {})}`; "
        f"segments examined: {claim.get('segment_columns', [])}.\n",
        "## Evidence Available\n\n" + f"{len(led.evidence)} evidence items" +
        (f", including upstream runs {led.upstream}" if led.upstream else "") + ".\n",
        f"## Metrics\n\n{metrics}",
        render.findings_section(led, "Baseline Comparison", cat("baseline")),
        render.findings_section(led, "Strong Results", strong),
        render.findings_section(led, "Weak Results", weak),
        render.findings_section(led, "Failure Modes", cat("segment_failure", "segment_uncertain", "hidden_failure", "refusal")),
        render.findings_section(led, "Coverage Gaps", cat("coverage", "production_projection")),
        render.findings_section(led, "Evaluation Risks", cat("judge_validity", "reproducibility", "leakage", "upstream_data_risk")),
        render.visuals_section(led),
        render.decision_section(led, "Production Readiness Assessment"),
        render.evidence_section(led),
        f"## Confidence\n\n`{led.decision_confidence}`\n",
        render.list_section("Assumptions", led.assumptions),
        render.list_section("Unknowns / Gaps", led.gaps),
        render.recommendations_section(led, "Recommended Next Evaluation"),
    ]
    return "\n".join(parts)
