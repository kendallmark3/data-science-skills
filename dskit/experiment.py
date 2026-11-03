"""Skill 2 - Experiment Analysis.

Answers: what does this experiment's evidence actually support?
Design validity is checked before effects are interpreted; statistical and
practical significance are judged separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import render, viz
from .evidence import Ledger, StopCondition

DECISIONS = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "INVALID_EXPERIMENT")
SRM_ALPHA = 0.001


@dataclass
class Effect:
    metric: str
    kind: str            # binary | continuous
    n_c: int
    n_t: int
    mean_c: float
    mean_t: float
    diff: float
    lo: float
    hi: float
    p: float
    se: float

    @property
    def rel(self) -> float:
        return self.diff / self.mean_c if self.mean_c else float("nan")

    @property
    def significant(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def fmt(self, x: float) -> str:
        return f"{x * 100:+.2f}pp" if self.kind == "binary" else f"{x:+.3g}"

    def level(self, x: float) -> str:
        return f"{x:.2%}" if self.kind == "binary" else f"{x:.4g}"


def binary_effect(metric, n_c, x_c, n_t, x_t, alpha=0.05) -> Effect:
    pc, pt = x_c / n_c, x_t / n_t
    se = np.sqrt(pc * (1 - pc) / n_c + pt * (1 - pt) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)
    pool = (x_c + x_t) / (n_c + n_t)
    se0 = np.sqrt(pool * (1 - pool) * (1 / n_c + 1 / n_t))
    p = 2 * stats.norm.sf(abs(pt - pc) / se0) if se0 > 0 else 1.0
    d = pt - pc
    return Effect(metric, "binary", int(n_c), int(n_t), pc, pt, d, d - z * se, d + z * se, float(p), float(se))


def continuous_effect(metric, c: pd.Series, t: pd.Series, alpha=0.05) -> Effect:
    c, t = c.dropna().astype(float), t.dropna().astype(float)
    vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = np.sqrt(vc + vt)
    df = (vc + vt) ** 2 / (vc ** 2 / (len(c) - 1) + vt ** 2 / (len(t) - 1))
    q = stats.t.ppf(1 - alpha / 2, df)
    d = t.mean() - c.mean()
    p = stats.ttest_ind(t, c, equal_var=False).pvalue
    return Effect(metric, "continuous", len(c), len(t), c.mean(), t.mean(), d, d - q * se, d + q * se, float(p), float(se))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def required_n_per_arm(p_c: float, delta: float, alpha=0.05, power=0.8) -> int:
    p_t = min(max(p_c + delta, 1e-6), 1 - 1e-6)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z ** 2 * (p_c * (1 - p_c) + p_t * (1 - p_t)) / delta ** 2))


def _is_binary(s: pd.Series) -> bool:
    return set(pd.unique(s.dropna())) <= {0, 1}


def analyze(data_path: str | Path, meta: dict, out_dir: str | Path) -> Ledger:
    src = Path(data_path).name
    msrc = meta.get("_meta_source", "experiment metadata")
    led = Ledger("experiment-analysis", "EX", intent=meta.get("hypothesis") or "Evaluate experiment " + meta.get("name", ""))
    alpha = float(meta.get("alpha", 0.05))
    out = Path(out_dir)

    # ---- design evidence ---------------------------------------------------------
    missing_design = [k for k in ("hypothesis", "primary_metric", "assignment", "expected_split") if not meta.get(k)]
    e_design = led.add_evidence(msrc, "design fields",
                                f"present: {[k for k in ('hypothesis', 'primary_metric', 'assignment', 'expected_split', 'practical_threshold', 'stopping_rule', 'guardrails') if meta.get(k)]}; "
                                f"missing: {missing_design}", "metadata field check")
    if missing_design:
        f = led.add_finding(f"Experiment design evidence is incomplete: missing {missing_design}.", [e_design],
                            category="design", severity="critical",
                            risk="Without a pre-declared metric, hypothesis and assignment method, any result is post-hoc.",
                            recommended_check="Recover the pre-registration or experiment brief before interpreting results.")
        led.decision, led.decision_confidence = "INVALID_EXPERIMENT", "high"
        led.decision_rationale = [f"{f}: the design cannot be assessed, so the result cannot be interpreted."]
        led.gaps.append(f"Design fields missing: {missing_design}.")
        led.recommend("Do not interpret this experiment; reconstruct the design (hypothesis, primary metric, assignment) first.", [f])
        led.write(out, _report(led, None, [], []))
        return led

    df = _load(data_path, meta)
    metric, arms = meta["primary_metric"], meta.get("arms", ["control", "treatment"])
    vcol = meta.get("variant_column", "variant")
    if set(arms) - set(df[vcol].unique()):
        raise StopCondition(f"Arms {arms} not all present in {src} column {vcol!r}")
    if not meta.get("practical_threshold"):
        led.gaps.append("No practical (business) threshold supplied; practical significance cannot be judged.")
    if not meta.get("stopping_rule"):
        led.gaps.append("No stopping rule recorded; cannot rule out optional stopping (peeking).")

    # ---- per-arm aggregates & SRM -----------------------------------------------
    agg = meta.get("format") == "aggregate"
    ncol = meta.get("count_column", "users")
    counts = {a: int(df.loc[df[vcol] == a, ncol].sum()) if agg else int((df[vcol] == a).sum()) for a in arms}
    split = meta["expected_split"]
    total = sum(counts.values())
    chi = stats.chisquare([counts[a] for a in arms], [split[a] * total for a in arms])
    e_srm = led.add_evidence(src, f"column={vcol}", f"allocation {counts} vs expected {split}; chi-square p = {chi.pvalue:.2g}",
                             "chi-square goodness of fit on assignment counts", value=float(chi.pvalue))
    srm = chi.pvalue < SRM_ALPHA
    led.summary.update(counts=counts, srm_p=float(chi.pvalue))
    f_srm = None
    if srm:
        f_srm = led.add_finding(
            f"Sample ratio mismatch: observed allocation {counts} is incompatible with the designed {split} split "
            f"(p = {chi.pvalue:.1e}).", [e_srm], category="srm", severity="critical",
            risk="Assignment or logging is broken, so the arms are not comparable and any effect estimate is biased.",
            possible_explanation="Redirect failures, bot filtering applied to one arm, or a logging bug in the treatment path.",
            recommended_check="Audit assignment and exposure logging per arm before re-running.")

    # ---- primary effect -------------------------------------------------------------
    c_mask, t_mask = df[vcol] == arms[0], df[vcol] == arms[1]
    if agg:
        eff = binary_effect(metric, counts[arms[0]], df.loc[c_mask, metric].sum(), counts[arms[1]], df.loc[t_mask, metric].sum(), alpha)
    elif _is_binary(df[metric]):
        eff = binary_effect(metric, c_mask.sum(), df.loc[c_mask, metric].sum(), t_mask.sum(), df.loc[t_mask, metric].sum(), alpha)
    else:
        eff = continuous_effect(metric, df.loc[c_mask, metric], df.loc[t_mask, metric], alpha)
    method = "two-proportion Wald CI; pooled z-test" if eff.kind == "binary" else "Welch t-test and CI"
    e_eff = led.add_evidence(src, f"metric={metric}",
                             f"{arms[0]} {eff.level(eff.mean_c)} (n={eff.n_c:,}) vs {arms[1]} {eff.level(eff.mean_t)} (n={eff.n_t:,}); "
                             f"diff {eff.fmt(eff.diff)} [{eff.fmt(eff.lo)}, {eff.fmt(eff.hi)}] {1 - alpha:.0%} CI; "
                             f"relative {eff.rel:+.1%}; p = {eff.p:.2g}", method,
                             confidence="low" if srm else "high", value=eff.diff)
    led.summary["primary"] = {"metric": metric, "control": eff.mean_c, "treatment": eff.mean_t, "diff": eff.diff,
                              "ci": [eff.lo, eff.hi], "relative": eff.rel, "p_value": eff.p, "method": method}
    sig_word = "statistically significant" if eff.significant else "not statistically significant"
    f_eff = led.add_finding(f"Primary metric `{metric}` changed by {eff.fmt(eff.diff)} ({eff.rel:+.1%} relative), "
                            f"{sig_word} at α={alpha} (CI {eff.fmt(eff.lo)} to {eff.fmt(eff.hi)}).",
                            [e_eff], category="primary_effect", severity="info")

    # practical threshold
    thr = meta.get("practical_threshold") or {}
    thr_abs = thr.get("absolute") if "absolute" in thr else (thr["relative"] * eff.mean_c if "relative" in thr else None)
    f_prac = None
    if srm:
        led.findings[-1].statement += " This estimate is not interpretable because of the sample ratio mismatch."
    if thr_abs is not None and not srm:
        e_thr = led.add_evidence(msrc, "practical_threshold", f"minimum worthwhile effect {eff.fmt(thr_abs)}"
                                 + (f" ({thr['rationale']})" if thr.get("rationale") else ""), "declared in experiment metadata")
        if eff.significant and eff.hi < thr_abs and eff.diff > 0:
            f_prac = led.add_finding(f"The effect is real but operationally trivial: even the upper CI bound ({eff.fmt(eff.hi)}) "
                                     f"is below the practical threshold ({eff.fmt(thr_abs)}).", [e_eff, e_thr],
                                     kind="inference", category="practical_significance", severity="high")
        elif eff.lo >= thr_abs:
            f_prac = led.add_finding(f"The whole CI ({eff.fmt(eff.lo)} to {eff.fmt(eff.hi)}) clears the practical threshold "
                                     f"({eff.fmt(thr_abs)}).", [e_eff, e_thr], kind="inference",
                                     category="practical_significance", severity="info")
        elif eff.diff >= thr_abs:
            f_prac = led.add_finding(f"Point estimate clears the practical threshold ({eff.fmt(thr_abs)}) but the CI lower bound "
                                     f"({eff.fmt(eff.lo)}) does not.", [e_eff, e_thr], kind="inference",
                                     category="practical_significance", severity="moderate")
        led.summary["practical_threshold"] = thr_abs

    # power / MDE
    e_pow = None
    if eff.kind == "binary":
        mde = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)) * eff.se
        need = required_n_per_arm(eff.mean_c, thr_abs) if thr_abs else None
        e_pow = led.add_evidence(src, f"metric={metric}", f"minimum detectable effect at 80% power ≈ {eff.fmt(mde)}"
                                 + (f"; n per arm needed to detect {eff.fmt(thr_abs)}: {need:,}" if need else ""),
                                 "normal approximation, two-sided", value=float(mde))
        led.summary["mde_80"] = float(mde)
        led.summary["n_needed_per_arm"] = need
        ambiguous = thr_abs is not None and eff.lo < thr_abs <= eff.hi
        if thr_abs and mde > thr_abs and ambiguous and not srm:
            led.add_finding(f"Underpowered: the minimum detectable effect ({eff.fmt(mde)}) exceeds the practical threshold "
                            f"({eff.fmt(thr_abs)}); about {need:,} users per arm are needed.", [e_pow, e_thr],
                            kind="inference", category="power", severity="high")

    # ---- missing data / attrition --------------------------------------------------
    if not agg:
        mr = {a: float(df.loc[df[vcol] == a, metric].isna().mean()) for a in arms}
        if any(mr.values()):
            table = [[df.loc[df[vcol] == a, metric].isna().sum(), df.loc[df[vcol] == a, metric].notna().sum()] for a in arms]
            p = stats.chi2_contingency(table).pvalue
            e_m = led.add_evidence(src, f"metric={metric}", f"missing outcome rate by arm {mr}; chi-square p = {p:.2g}",
                                   "null rate by arm", value=mr)
            if p < 0.01:
                led.add_finding("Differential attrition: outcome missingness differs by arm.", [e_m], category="attrition",
                                severity="high", risk="Survivors in each arm are no longer comparable.")

    # ---- guardrails --------------------------------------------------------------------
    breaches, guard_effects = [], []
    for g in meta.get("guardrails", []) or []:
        gm = g["metric"]
        if gm not in df:
            led.gaps.append(f"Guardrail `{gm}` declared but not present in {src}.")
            continue
        ge = (binary_effect(gm, c_mask.sum(), df.loc[c_mask, gm].sum(), t_mask.sum(), df.loc[t_mask, gm].sum(), alpha)
              if _is_binary(df[gm]) else continuous_effect(gm, df.loc[c_mask, gm], df.loc[t_mask, gm], alpha))
        sign = 1 if g.get("direction", "lower_is_better") == "lower_is_better" else -1
        tol = g["max_absolute_degradation"] if "max_absolute_degradation" in g else g.get("max_relative_degradation", 0) * ge.mean_c
        worse = sign * ge.diff
        e_g = led.add_evidence(src, f"guardrail={gm}", f"{arms[0]} {ge.level(ge.mean_c)} vs {arms[1]} {ge.level(ge.mean_t)}; "
                               f"diff {ge.fmt(ge.diff)} [{ge.fmt(ge.lo)}, {ge.fmt(ge.hi)}]; tolerance {ge.fmt(sign * tol)}",
                               "two-proportion CI" if ge.kind == "binary" else "Welch CI", value=ge.diff)
        guard_effects.append((ge, sign * tol, e_g))
        if ge.significant and worse > tol:
            breaches.append(led.add_finding(
                f"Guardrail `{gm}` degraded by {ge.fmt(ge.diff)}, beyond its tolerance of {ge.fmt(sign * tol)}.", [e_g],
                category="guardrail", severity="high",
                risk="The primary win may be paid for with the guarded outcome."))
        elif sign * (ge.hi if sign == 1 else ge.lo) > tol:
            led.add_finding(f"Guardrail `{gm}` shows no confirmed degradation, but the CI does not rule out exceeding the tolerance.",
                            [e_g], category="guardrail", severity="moderate", kind="inference")
        else:
            led.add_finding(f"Guardrail `{gm}` held (diff {ge.fmt(ge.diff)}, tolerance {ge.fmt(sign * tol)}).", [e_g],
                            category="guardrail", severity="info")

    # ---- segments & multiple comparisons -------------------------------------------------
    seg_results = []
    for sc in meta.get("segment_columns", []) or []:
        if sc not in df or not _is_binary(df[metric]):
            continue
        for val, g in df.groupby(sc):
            cm, tm = g[vcol] == arms[0], g[vcol] == arms[1]
            if cm.sum() < 30 or tm.sum() < 30:
                continue
            seg_results.append((sc, val, binary_effect(metric, cm.sum(), g.loc[cm, metric].sum(), tm.sum(), g.loc[tm, metric].sum(), alpha)))
    if seg_results:
        adj = holm([r[2].p for r in seg_results])
        e_seg = []
        for (sc, val, se), pa in zip(seg_results, adj):
            e_seg.append(led.add_evidence(src, f"segment {sc}={val}", f"diff {se.fmt(se.diff)} [{se.fmt(se.lo)}, {se.fmt(se.hi)}]; "
                                          f"n={se.n_c:,}/{se.n_t:,}; Holm-adjusted p = {pa:.2g}",
                                          "two-proportion CI; Holm correction across segments", value=se.diff))
        w = np.array([1 / r[2].se ** 2 for r in seg_results])
        d = np.array([r[2].diff for r in seg_results])
        q = float((w * (d - (w * d).sum() / w.sum()) ** 2).sum())
        qp = float(stats.chi2.sf(q, len(seg_results) - 1)) if len(seg_results) > 1 else 1.0
        e_q = led.add_evidence(src, f"segments={[r[0] for r in seg_results][0]}", f"Cochran's Q = {q:.2f}, p = {qp:.2g} "
                               f"across {len(seg_results)} segments ({len(seg_results)} tests, Holm-corrected)",
                               "inverse-variance heterogeneity test")
        led.summary["segments"] = {f"{r[0]}={r[1]}": {"diff": r[2].diff, "ci": [r[2].lo, r[2].hi], "p_holm": pa}
                                   for r, pa in zip(seg_results, adj)}
        opposite = [(r, pa, e) for r, pa, e in zip(seg_results, adj, e_seg) if pa < alpha and np.sign(r[2].diff) != np.sign(eff.diff)]
        if qp < 0.05 or opposite:
            led.add_finding(f"Treatment effect differs across segments (Cochran's Q p = {qp:.2g}).", [e_q] + e_seg,
                            category="segment_heterogeneity", severity="moderate", kind="inference",
                            recommended_check="Confirm the segment split was pre-specified before acting on it.")
        else:
            led.add_finding(f"No evidence the effect differs across segments (Q p = {qp:.2g}); segment estimates are consistent "
                            "with the overall effect.", [e_q] + e_seg, category="segments", severity="info")

    # ---- duration / novelty ----------------------------------------------------------------
    tcol = meta.get("time_column") or ("date" if "date" in df else None)
    trend = None
    if tcol and tcol in df:
        days = pd.to_datetime(df[tcol])
        span = int((days.max() - days.min()).days) + 1
        planned = meta.get("planned_duration_days")
        e_dur = led.add_evidence(src, f"column={tcol}", f"{span} days observed ({days.min().date()} to {days.max().date()})"
                                 + (f"; planned {planned}" if planned else ""), "max(date) - min(date) + 1", value=span)
        if span < 7:
            led.add_finding(f"Experiment ran {span} days, less than one weekly cycle.", [e_dur], category="duration",
                            severity="moderate", risk="Day-of-week effects and novelty are not averaged out.")
        if planned and span < planned:
            led.add_finding(f"Experiment stopped after {span} of {planned} planned days.", [e_dur], category="stopping",
                            severity="high", risk="Early stopping inflates false-positive rates.")
        mid = days.min() + (days.max() - days.min()) / 2
        halves = []
        for part in (days <= mid, days > mid):
            g = df[part]
            cm, tm = g[vcol] == arms[0], g[vcol] == arms[1]
            if eff.kind == "binary":
                nc_, nt_ = (g.loc[cm, ncol].sum(), g.loc[tm, ncol].sum()) if agg else (cm.sum(), tm.sum())
                halves.append(binary_effect(metric, nc_, g.loc[cm, metric].sum(), nt_, g.loc[tm, metric].sum(), alpha))
        if len(halves) == 2:
            z = (halves[0].diff - halves[1].diff) / np.sqrt(halves[0].se ** 2 + halves[1].se ** 2)
            pz = float(2 * stats.norm.sf(abs(z)))
            e_nov = led.add_evidence(src, f"column={tcol}", f"effect first half {halves[0].fmt(halves[0].diff)} vs second half "
                                     f"{halves[1].fmt(halves[1].diff)}; difference p = {pz:.2g}", "z-test on half-period effects")
            trend = (halves, e_nov)
            if pz < 0.05 and halves[0].diff > halves[1].diff:
                led.add_finding("Effect decays over the experiment (possible novelty effect).", [e_nov], category="novelty",
                                severity="moderate", kind="inference",
                                recommended_check="Extend the test or analyze returning users separately.")
    else:
        led.gaps.append("No time column; duration, novelty and seasonality could not be assessed.")

    _decide(led, eff, srm, f_srm, f_eff, f_prac, thr_abs, breaches)
    _visuals(led, out, eff, e_eff, thr_abs, guard_effects, seg_results, srm, e_srm, counts, split, arms)
    led.write(out, _report(led, eff, guard_effects, seg_results))
    return led


def _load(path, meta) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise StopCondition(f"Experiment data not found: {p}")
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001
        raise StopCondition(f"Could not read {p.name}: {exc}") from exc
    need = [meta.get("variant_column", "variant"), meta["primary_metric"]]
    if meta.get("format") == "aggregate":
        need.append(meta.get("count_column", "users"))
    missing = [c for c in need if c not in df]
    if missing:
        raise StopCondition(f"{p.name} lacks required columns {missing}")
    return df


def _decide(led, eff, srm, f_srm, f_eff, f_prac, thr, breaches):
    severe = [f for f in led.findings if f.severity in ("high", "critical") and f.category in ("attrition", "stopping")]
    if srm:
        led.decision, led.decision_confidence = "INVALID_EXPERIMENT", "high"
        led.decision_rationale = [f"{f_srm}: allocation does not match the design, so the observed effect "
                                  f"({eff.fmt(eff.diff)}) is not interpretable."]
        led.recommend("Do not act on this result. Fix assignment/exposure logging, then re-run the experiment.", [f_srm])
        return
    if severe:
        led.decision, led.decision_confidence = "INVALID_EXPERIMENT", "medium"
        led.decision_rationale = [f"{f.id}: {f.statement}" for f in severe]
        led.recommend("Re-run with the design defect fixed before interpreting the effect.", [f.id for f in severe])
        return
    if thr is None:
        led.decision, led.decision_confidence = "INCONCLUSIVE", "medium"
        led.decision_rationale = ["No practical threshold was declared, so statistical significance cannot be turned into a decision."]
        led.recommend("Agree the minimum worthwhile effect with the business owner, then re-read this result.", [f_eff])
        return
    if eff.significant and eff.diff > 0 and eff.hi < thr:
        led.decision, led.decision_confidence = "NOT_SUPPORTED", "high"
        led.decision_rationale = [f"{f_eff}: the effect is statistically significant (p = {eff.p:.2g}).",
                                  f"{f_prac}: the entire CI is below the practical threshold, so the hypothesis of a "
                                  "worthwhile improvement is not supported. Significance is not a rollout signal."]
        led.recommend("Do not roll out on the basis of this effect; it is real but too small to matter.", [f_eff, f_prac])
        return
    if eff.hi < 0 or (eff.significant and eff.diff < 0):
        led.decision, led.decision_confidence = "NOT_SUPPORTED", "high"
        led.decision_rationale = [f"{f_eff}: the treatment performed worse than control."]
        led.recommend("Do not roll out; the treatment underperformed control.", [f_eff])
        return
    if not eff.significant and eff.hi < thr:
        led.decision, led.decision_confidence = "NOT_SUPPORTED", "medium"
        led.decision_rationale = [f"{f_eff}: no significant effect, and the CI excludes the practical threshold."]
        led.recommend("Stop pursuing this change; a worthwhile effect is ruled out.", [f_eff])
        return
    if breaches:
        led.decision, led.decision_confidence = "INCONCLUSIVE", "medium"
        led.decision_rationale = [f"{f_eff}: the primary metric improved.",
                                  *[f"{b}: but a guardrail broke its tolerance — the metrics contradict each other." for b in breaches]]
        led.recommend("Hold rollout. Quantify the net value of the primary gain against the guardrail loss with the business owner.",
                      [f_eff, *breaches])
        return
    if eff.lo >= thr:
        threats = [f for f in led.findings if f.severity in ("moderate", "high") and f.category not in ("practical_significance",)]
        led.decision = "SUPPORTED"
        led.decision_confidence = "medium" if threats or led.gaps else "high"
        led.decision_rationale = [f"{f_eff}: significant improvement.", f"{f_prac}: the CI clears the practical threshold.",
                                  "All declared guardrails held."]
        led.recommend("Roll out the treatment, keeping guardrail monitoring in place for the first weeks.", [f_eff, f_prac])
        return
    if eff.significant and eff.diff >= thr:
        led.decision, led.decision_confidence = "SUPPORTED", "medium"
        led.decision_rationale = [f"{f_eff}: significant, and the point estimate clears the threshold; "
                                  "the CI lower bound does not, so the size of the gain is uncertain."]
        led.recommend("Roll out cautiously or extend the test to tighten the estimate.", [f_eff, f_prac] if f_prac else [f_eff])
        return
    power = [f.id for f in led.findings if f.category == "power"]
    led.decision, led.decision_confidence = "INCONCLUSIVE", "high" if power else "medium"
    led.decision_rationale = [f"{f_eff}: the CI ({eff.fmt(eff.lo)} to {eff.fmt(eff.hi)}) includes both zero/trivial effects "
                              f"and the practical threshold ({eff.fmt(thr)})."]
    n_need = led.summary.get("n_needed_per_arm")
    led.recommend(f"Re-run with ≈{n_need:,} users per arm to detect the practical threshold at 80% power."
                  if n_need else "Collect more data before deciding.", [f_eff, *power])


def _visuals(led, out, eff, e_eff, thr, guard_effects, seg_results, srm, e_srm, counts, split, arms):
    if srm:
        # The effect is not interpretable; drawing it would imply evidence that does not exist.
        fig, ax = viz.figure(6.5, 3.0)
        total = sum(counts.values())
        y = np.arange(len(arms))
        ax.barh(y + 0.18, [split[a] * total for a in arms], height=0.34, color=viz.NEUTRAL, label="expected by design")
        ax.barh(y - 0.18, [counts[a] for a in arms], height=0.34, color=viz.CRITICAL, label="observed")
        for i, a in enumerate(arms):
            ax.text(counts[a], i - 0.18, f" {counts[a]:,}", va="center", fontsize=9, color=viz.INK_2)
            ax.text(split[a] * total, i + 0.18, f" {split[a] * total:,.0f}", va="center", fontsize=9, color=viz.MUTED)
        ax.set_yticks(y, arms)
        ax.set_xlabel("users assigned")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Sample ratio mismatch (p = {led.summary['srm_p']:.1e})")
        path = viz.save(fig, out, "sample-ratio.png", f"Effect estimate deliberately not plotted: arms are not comparable. Source: {e_srm}")
        led.add_visual("sample_ratio_mismatch", path, [e_srm], "Show that allocation departs from the design")
        return
    panels = [(eff, thr, e_eff, "primary")] + [(g, t, e, "guardrail") for g, t, e in guard_effects]
    fig, axes = viz.figure(7.5, 1.25 * len(panels) + 0.9)
    fig.clf()
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    for ax, (e, t, _, role) in zip(axes, panels):
        scale = 100 if e.kind == "binary" else 1
        unit = "pp" if e.kind == "binary" else ""
        color = viz.SERIES[0] if role == "primary" else viz.SERIES[1]
        ax.errorbar([e.diff * scale], [0], xerr=[[(e.diff - e.lo) * scale], [(e.hi - e.diff) * scale]], fmt="o",
                    color=color, ms=8, capsize=5, lw=2)
        ax.axvline(0, color=viz.AXIS, lw=1)
        if t is not None:
            ax.axvline(t * scale, color=viz.CRITICAL if role == "guardrail" else viz.GOOD, lw=1.5, ls="--")
            ax.text(t * scale, 0.42, " tolerance" if role == "guardrail" else " practical threshold", fontsize=8,
                    color=viz.INK_2, va="center")
        lo_x = min(e.lo, 0, t if t is not None else 0) * scale
        hi_x = max(e.hi, 0, t if t is not None else 0) * scale
        pad = (hi_x - lo_x) * 0.15 or 1
        ax.set_xlim(lo_x - pad, hi_x + pad)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{role}: {e.metric}  —  {e.fmt(e.diff)} [{e.fmt(e.lo)}, {e.fmt(e.hi)}]", fontsize=10)
        ax.set_xlabel(f"treatment − control ({unit or 'units'})", fontsize=9)
    ev = [e_eff] + [g[2] for g in guard_effects]
    path = viz.save(fig, out, "effect-vs-threshold.png", "Dot = estimate, bar = 95% CI, dashed = decision line. Source: " + ", ".join(ev))
    led.add_visual("treatment_control_effect", path, ev,
                   "Compare the effect and its uncertainty with the practical threshold and guardrail tolerances")
    het = led.findings_by("segment_heterogeneity")
    if het and seg_results:
        fig, ax = viz.figure(7.0, 0.55 * len(seg_results) + 1.4)
        for i, (sc, val, se) in enumerate(seg_results):
            ax.errorbar([se.diff * 100], [i], xerr=[[(se.diff - se.lo) * 100], [(se.hi - se.diff) * 100]], fmt="o",
                        color=viz.SERIES[0], capsize=4)
        ax.axvline(eff.diff * 100, color=viz.MUTED, ls=":", lw=1.2)
        ax.axvline(0, color=viz.AXIS, lw=1)
        ax.set_yticks(range(len(seg_results)), [f"{sc}={val}" for sc, val, _ in seg_results])
        ax.set_xlabel("treatment − control (pp)")
        ax.set_title("Effect by segment (dotted = overall)")
        path = viz.save(fig, out, "segment-effects.png", "Source: " + ", ".join(het[0].evidence))
        led.add_visual("segment_comparison", path, het[0].evidence, "Show how the effect varies by segment")


def _report(led, eff, guard_effects, seg_results) -> str:
    F = led.findings
    cat = lambda *c: [f for f in F if f.category in c]  # noqa: E731
    threats = [f for f in F if f.category in ("srm", "attrition", "duration", "stopping", "novelty", "power", "design")]
    observed = ""
    if eff:
        s = led.summary
        observed = (f"| Arm | n | {eff.metric} |\n|---|---|---|\n| control | {eff.n_c:,} | {eff.level(eff.mean_c)} |\n"
                    f"| treatment | {eff.n_t:,} | {eff.level(eff.mean_t)} |\n\n"
                    f"Difference {eff.fmt(eff.diff)} ({eff.rel:+.1%} relative), p = {eff.p:.2g}; method: {s['primary']['method']}.\n")
    parts = [
        f"# Experiment Analysis: {led.intent}\n",
        f"**Decision:** `{led.decision}` (confidence `{led.decision_confidence}`)\n",
        f"## Experiment Question\n\nDid the treatment produce a worthwhile improvement in the primary metric?\n",
        f"## Hypothesis\n\n{led.intent}\n",
        render.findings_section(led, "Experiment Design Assessment", cat("design", "srm", "attrition", "stopping", "duration")),
        f"## Observed Results\n\n{observed or 'Not computed.'}",
        render.findings_section(led, "Effect Size", cat("primary_effect")),
        "## Uncertainty\n\n" + (f"{1 - 0.05:.0%} CI for the difference: {eff.fmt(eff.lo)} to {eff.fmt(eff.hi)}. "
                                + (f"Minimum detectable effect at 80% power ≈ {eff.fmt(led.summary['mde_80'])}." if 'mde_80' in led.summary else "")
                                if eff else "Not computed.") + "\n",
        "## Statistical Interpretation\n\n" + (("The difference is statistically distinguishable from zero." if eff.significant
                                                else "The difference is not statistically distinguishable from zero.")
                                               + " Statistical significance says the effect is probably not zero; it does not say the effect is large enough to act on.\n" if eff else "Not applicable.\n"),
        render.findings_section(led, "Practical / Business Interpretation", cat("practical_significance", "power")),
        render.findings_section(led, "Threats to Validity", threats),
        render.findings_section(led, "Segment Findings", cat("segments", "segment_heterogeneity")),
        render.findings_section(led, "Guardrail Findings", cat("guardrail")),
        render.visuals_section(led),
        render.evidence_section(led),
        render.decision_section(led),
        f"## Confidence\n\n`{led.decision_confidence}`. Gaps that limit it are listed below.\n",
        render.list_section("Unknowns / Gaps", led.gaps),
        render.recommendations_section(led, "Recommended Next Experiment / Action"),
    ]
    return "\n".join(parts)
