"""Skill 1 - Data Profiling & Quality.

Answers: is this dataset trustworthy enough for its intended use?
Every check emits evidence; only checks with something to say emit findings.
"""

from __future__ import annotations

import difflib
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import render, viz
from .evidence import Ledger, StopCondition

DECISIONS = ("FIT_FOR_USE", "FIT_WITH_REMEDIATION", "NOT_FIT_AS_IS", "INSUFFICIENT_EVIDENCE")

# Plausibility rules applied even without a schema. Kept tiny on purpose.
PLAUSIBLE = {"age": (0, 120)}
ID_PATTERN = re.compile(r"(^id$|_id$|^id_|_key$|^uuid$)", re.I)
CURRENCY = re.compile(r"[\$€£,\s]")


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise StopCondition(f"Input file not found: {p}")
    try:
        if p.suffix == ".csv":
            df = pd.read_csv(p, dtype=str, keep_default_na=True)
        elif p.suffix in (".json", ".jsonl"):
            df = pd.read_json(p, lines=p.suffix == ".jsonl", dtype=False).astype("string").astype(object)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p).astype("string").astype(object)
        else:
            raise StopCondition(f"Unsupported file type {p.suffix!r}; use CSV, JSON/JSONL or Parquet")
    except StopCondition:
        raise
    except Exception as exc:  # noqa: BLE001 - any parse failure is a stop condition
        raise StopCondition(f"Could not read {p.name} as a table: {type(exc).__name__}: {exc}") from exc
    if df.empty or df.shape[1] < 2:
        raise StopCondition(f"{p.name} parsed to {df.shape[0]} rows x {df.shape[1]} columns; nothing to profile")
    return df.replace({"": np.nan})


def _infer(raw: pd.Series, hint: str | None) -> dict:
    """Infer a column's type from its raw string values."""
    nn = raw.dropna()
    info = {"type": "empty", "numeric": None, "dates": None, "needs_cleaning": 0}
    if nn.empty:
        return info
    direct = pd.to_numeric(nn, errors="coerce")
    cleaned = pd.to_numeric(nn.str.replace(CURRENCY, "", regex=True), errors="coerce")
    iso = pd.to_datetime(nn, format="%Y-%m-%d", errors="coerce")
    if hint == "date" or iso.notna().mean() >= 0.9:
        info.update(type="date", dates=pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce"))
    elif direct.notna().mean() == 1:
        info.update(type="numeric", numeric=pd.to_numeric(raw, errors="coerce"))
    elif cleaned.notna().mean() >= 0.98 and direct.notna().mean() >= 0.5:
        info.update(type="numeric", needs_cleaning=int(direct.isna().sum()),
                    numeric=pd.to_numeric(raw.str.replace(CURRENCY, "", regex=True), errors="coerce"))
    else:
        uniq = nn.nunique()
        info["type"] = "categorical" if (uniq <= 50 or uniq / len(nn) < 0.5) else "text"
    return info


def profile(path: str | Path, out_dir: str | Path, context: dict | None = None) -> Ledger:
    ctx = context or {}
    src = Path(path).name
    led = Ledger("data-profiling-quality", "DP", intent=ctx.get("intended_use", "General-purpose analysis"))
    df = load_table(path)
    n, k = df.shape
    schema = ctx.get("schema", {})
    target, split_col = ctx.get("target"), ctx.get("split_column")
    if not ctx.get("intended_use"):
        led.gaps.append("No intended use supplied; severities assume general analytical use.")
    if not schema:
        led.gaps.append("No expected schema supplied; validity checks use inferred types and generic plausibility rules only.")
    if not target:
        led.gaps.append("No target column supplied; leakage and class-balance checks were not run.")

    e_shape = led.add_evidence(src, "table", f"{n} rows x {k} columns", "row/column count")
    info = {c: _infer(df[c], (schema.get(c) or {}).get("type") if c != ctx.get("time_column") else "date")
            for c in df.columns}
    types = {c: i["type"] for c, i in info.items()}
    e_types = led.add_evidence(src, "all columns", "; ".join(f"{c}={t}" for c, t in types.items()),
                               "type inference on raw string values (numeric / ISO date / categorical / text)")
    led.summary.update(rows=n, columns=k, inferred_types=types)

    row_flags = {"validity": pd.Series(False, index=df.index), "consistency": pd.Series(False, index=df.index)}
    dim_evidence = {"completeness": [], "uniqueness": [], "validity": [], "consistency": [], "leakage": []}
    tgt = pd.to_numeric(df[target], errors="coerce") if target in df else None
    if target and target not in df:
        led.gaps.append(f"Target column {target!r} not found; leakage and balance checks skipped.")

    # ---- missingness ------------------------------------------------------
    miss = df.isna().mean()
    led.summary["missing_fraction"] = {c: round(float(v), 4) for c, v in miss.items() if v > 0}
    miss_ev = {}
    for col, frac in miss[miss > 0].sort_values(ascending=False).items():
        cnt = int(df[col].isna().sum())
        eid = led.add_evidence(src, f"column={col}", f"{frac:.1%} missing ({cnt} of {n} rows)",
                               "null count / row count", value=frac)
        miss_ev[col] = eid
        dim_evidence["completeness"].append(eid)
    structural = set()
    if tgt is not None:
        for col in miss_ev:
            if col == target:
                continue
            present = df[col].notna()
            rates = tgt.groupby(present).mean()
            if len(rates) == 2 and {round(rates[True], 6), round(rates[False], 6)} == {0.0, 1.0}:
                structural.add(col)
    for col, eid in miss_ev.items():
        frac = miss[col]
        if col in structural:
            continue  # reported as leakage below; missingness here is a symptom, not the defect
        sev = "high" if frac >= 0.3 else "moderate" if frac >= 0.05 else "info"
        required = col in ctx.get("required_columns", [])
        if required and sev != "high":
            sev = {"info": "moderate", "moderate": "high"}[sev]
        led.add_finding(
            f"`{col}` is {frac:.1%} missing" + (" and is marked as required." if required else "."), [eid],
            category="missingness", severity=sev,
            risk="Analyses that need this field will silently drop or bias rows." if sev != "info" else None,
            possible_explanation="Optional field at capture, or a failed join/extract.",
            recommended_check=f"Check whether missingness in `{col}` is random or concentrated in a segment or time period.")

    # ---- duplicates -------------------------------------------------------
    exact = int(df.duplicated().sum())
    e_dup = led.add_evidence(src, "all columns", f"{exact} exact duplicate rows", "DataFrame.duplicated()", value=exact)
    dim_evidence["uniqueness"].append(e_dup)
    if exact:
        led.add_finding(f"{exact} rows are exact duplicates of another row.", [e_dup], category="duplicates",
                        severity="high", risk="Inflates counts and weights duplicated entities twice.",
                        possible_explanation="Repeated load or a join that fanned out.",
                        recommended_check="Trace duplicated rows to the ingestion batch that produced them.")
    id_cols = ctx.get("id_columns") or [c for c in df.columns if ID_PATTERN.search(c) and df[c].nunique() / max(df[c].notna().sum(), 1) >= 0.95]
    for col in id_cols:
        if col not in df:
            continue
        dup_mask = df[col].duplicated(keep=False) & df[col].notna()
        n_dup_ids = int(df.loc[dup_mask, col].nunique())
        eid = led.add_evidence(src, f"column={col}", f"{n_dup_ids} identifier values appear more than once "
                               f"({int(dup_mask.sum())} rows involved)", "value_counts() > 1", value=n_dup_ids)
        dim_evidence["uniqueness"].append(eid)
        if n_dup_ids:
            led.add_finding(f"Identifier `{col}` is not unique: {n_dup_ids} values repeat across {int(dup_mask.sum())} rows.",
                            [eid], category="duplicates", severity="high",
                            risk="Entity-level metrics and joins will double count.",
                            possible_explanation="Re-ingested records or entity updates appended instead of merged.",
                            recommended_check=f"Decide the canonical row per `{col}` (latest, first, or merge).")
        if split_col and split_col in df:
            per = df.dropna(subset=[col]).groupby(col)[split_col].nunique()
            leaked = int((per > 1).sum())
            eid = led.add_evidence(src, f"columns={col},{split_col}", f"{leaked} `{col}` values appear in more than one split",
                                   "groupby(id)[split].nunique() > 1", value=leaked)
            dim_evidence["leakage"].append(eid)
            if leaked:
                led.add_finding(f"Train/test contamination: {leaked} `{col}` values appear in both splits.", [eid],
                                category="contamination", severity="critical",
                                risk="Evaluation metrics will be optimistic because the model sees test entities in training.",
                                possible_explanation="Splitting by row instead of by entity after duplicates were introduced.",
                                recommended_check=f"Re-split by `{col}` (group split) and recompute evaluation metrics.")
    if not id_cols:
        led.gaps.append("No identifier column supplied or detected; entity-level duplication not assessed.")

    # ---- per-column validity ------------------------------------------------
    today = pd.Timestamp(ctx.get("as_of", date.today().isoformat()))
    for col in df.columns:
        ci, rule = info[col], schema.get(col, {})
        raw = df[col]
        if ci["needs_cleaning"]:
            bad = raw.notna() & pd.to_numeric(raw, errors="coerce").isna()
            ex = raw[bad].head(3).tolist()
            eid = led.add_evidence(src, f"column={col}", f"{ci['needs_cleaning']} values need currency/thousand-separator "
                                   f"stripping to parse as numbers, e.g. {ex}", "to_numeric before vs after stripping [$,€£ ,]",
                                   value=ci["needs_cleaning"])
            row_flags["consistency"] |= bad
            dim_evidence["consistency"].append(eid)
            led.add_finding(f"`{col}` mixes numeric formats ({ci['needs_cleaning']} values stored as formatted strings).",
                            [eid], category="formatting", severity="moderate",
                            risk="Naive parsing drops or mis-types these rows.",
                            possible_explanation="Values from a second source system or a manual export.",
                            recommended_check="Normalize at ingestion and confirm the formatted values share the same unit/currency.")
        if ci["type"] == "numeric":
            num = ci["numeric"]
            lo, hi = rule.get("min"), rule.get("max")
            plaus = PLAUSIBLE.get(col.lower())
            if lo is None and hi is None and plaus:
                lo, hi = plaus
                led.assumptions.append(f"`{col}` checked against generic plausibility range {plaus} (no schema supplied).")
            if lo is not None or hi is not None:
                viol = num.notna() & (((num < lo) if lo is not None else False) | ((num > hi) if hi is not None else False))
                if viol.any():
                    vals = sorted(num[viol].unique().tolist())
                    impossible = bool(plaus and ((num[viol] < plaus[0]) | (num[viol] > plaus[1])).any())
                    eid = led.add_evidence(src, f"column={col}; rows={df.index[viol].tolist()[:10]}",
                                           f"{int(viol.sum())} values outside [{lo}, {hi}]: {vals[:10]}",
                                           "range check against schema" if rule else "range check against plausibility rule",
                                           value=int(viol.sum()))
                    row_flags["validity"] |= viol
                    dim_evidence["validity"].append(eid)
                    led.add_finding(f"`{col}` has {int(viol.sum())} out-of-range values {vals[:5]}"
                                    + (" that are physically impossible." if impossible else "."), [eid],
                                    category="range_violation", severity="high" if impossible else "moderate",
                                    risk="Corrupts averages and model inputs; signals an upstream entry or unit defect.",
                                    possible_explanation="Sentinel codes, typos, or unit mix-ups at entry.",
                                    recommended_check=f"Look up the source records for these `{col}` values; null them or correct them.")
            if rule.get("allowed") is not None:
                bad = num.notna() & ~num.isin(rule["allowed"])
                if bad.any():
                    eid = led.add_evidence(src, f"column={col}", f"{int(bad.sum())} values not in {rule['allowed']}",
                                           "allowed-values check", value=int(bad.sum()))
                    row_flags["validity"] |= bad
                    led.add_finding(f"`{col}` contains values outside its allowed set.", [eid],
                                    category="schema_violation", severity="high")
            # outliers: reported, never called defects
            nn = num.dropna()
            if len(nn) >= 30 and nn.nunique() > 10:
                q1, q3 = nn.quantile([0.25, 0.75])
                iqr = q3 - q1
                out = nn[(nn < q1 - 3 * iqr) | (nn > q3 + 3 * iqr)]
                if lo is not None or hi is not None:
                    out = out[(out >= (lo if lo is not None else -np.inf)) & (out <= (hi if hi is not None else np.inf))]
                if len(out):
                    eid = led.add_evidence(src, f"column={col}", f"{len(out)} values beyond 3xIQR fences "
                                           f"(max {out.max():g}; median {nn.median():g})", "Tukey fences, k=3", value=len(out))
                    led.add_finding(f"`{col}` has {len(out)} extreme value{'s' if len(out) != 1 else ''}; they are in a valid range and may be genuine.",
                                    [eid], category="outliers", severity="info",
                                    possible_explanation="Heavy-tailed but real behaviour.",
                                    recommended_check="Decide whether downstream models need robust scaling or capping.")
        elif ci["type"] == "date":
            dts = ci["dates"]
            malformed = raw.notna() & dts.isna()
            if malformed.any():
                eid = led.add_evidence(src, f"column={col}; rows={df.index[malformed].tolist()[:10]}",
                                       f"{int(malformed.sum())} values do not parse as ISO dates: {raw[malformed].head(5).tolist()}",
                                       "pd.to_datetime(format='%Y-%m-%d')", value=int(malformed.sum()))
                row_flags["validity"] |= malformed
                dim_evidence["validity"].append(eid)
                led.add_finding(f"`{col}` has {int(malformed.sum())} malformed dates.", [eid], category="date_anomaly",
                                severity="high" if malformed.mean() > 0.05 else "moderate",
                                risk="Rows fall out of any time-based split, cohort or freshness calculation.",
                                possible_explanation="Mixed locale formats or free-text entry.",
                                recommended_check="Recover the original timestamps from the source system.")
            future = dts > today
            if future.any():
                eid = led.add_evidence(src, f"column={col}", f"{int(future.sum())} dates after {today.date()}",
                                       "date > as_of", value=int(future.sum()))
                row_flags["validity"] |= future
                led.add_finding(f"`{col}` has {int(future.sum())} dates in the future.", [eid], category="date_anomaly",
                                severity="moderate", recommended_check="Check for timezone or year-entry errors.")
            if dts.notna().any():
                led.add_evidence(src, f"column={col}", f"range {dts.min().date()} to {dts.max().date()}; latest record is "
                                 f"{(today - dts.max()).days} days before {today.date()}", "min/max of parsed dates",
                                 value=(today - dts.max()).days)
        elif ci["type"] == "categorical" and col not in (target, split_col):
            _check_categories(led, src, df, col, rule, row_flags, dim_evidence)

    # ---- constant columns -----------------------------------------------------
    for col in df.columns:
        nn = df[col].dropna()
        if nn.empty:
            continue
        top = nn.value_counts(normalize=True).iloc[0]
        if nn.nunique() == 1:
            eid = led.add_evidence(src, f"column={col}", f"single value {nn.iloc[0]!r} in all {len(nn)} non-null rows",
                                   "nunique() == 1")
            led.add_finding(f"`{col}` is constant ({nn.iloc[0]!r}); it carries no information.", [eid],
                            category="constant", severity="moderate",
                            risk="Wasted feature; may also mean the extract was filtered more narrowly than intended.",
                            possible_explanation="Single-market extract or a defaulted field.",
                            recommended_check="Confirm the extract covers the intended population; drop the column otherwise.")
        elif top >= 0.99:
            eid = led.add_evidence(src, f"column={col}", f"top value covers {top:.1%} of rows", "value_counts(normalize)")
            led.add_finding(f"`{col}` is near-constant ({top:.1%} one value).", [eid], category="constant", severity="info")

    # ---- leakage & balance ---------------------------------------------------
    if tgt is not None:
        for col in structural:
            present = df[col].notna()
            eid = led.add_evidence(src, f"columns={col},{target}",
                                   f"`{col}` is populated in {int(present.sum())} rows; {target} mean is "
                                   f"{tgt[present].mean():.2f} when present vs {tgt[~present].mean():.2f} when absent",
                                   "target rate grouped by notnull(column)")
            dim_evidence["leakage"].append(eid)
            led.add_finding(f"Probable target leakage: whether `{col}` is populated perfectly predicts `{target}`.",
                            [eid, miss_ev[col]], category="leakage", severity="critical", kind="inference",
                            risk="A model will learn the label from this column and fail in production, where it is not yet known.",
                            possible_explanation=f"`{col}` is recorded after the outcome `{target}` occurs.",
                            recommended_check=f"Confirm when `{col}` is written relative to the prediction time; exclude it from features.")
        for col, ci in info.items():
            if col in (target, *structural) or ci["type"] != "numeric" or col in id_cols:
                continue
            x = ci["numeric"]
            ok = x.notna() & tgt.notna()
            if ok.sum() > 30 and x[ok].nunique() > 1:
                r = float(np.corrcoef(x[ok], tgt[ok])[0, 1])
                if abs(r) >= 0.9:
                    eid = led.add_evidence(src, f"columns={col},{target}", f"Pearson r = {r:.3f}", "correlation with target")
                    dim_evidence["leakage"].append(eid)
                    led.add_finding(f"`{col}` is almost perfectly correlated with `{target}` (r={r:.2f}); possible leakage.",
                                    [eid], category="leakage", severity="high", kind="inference",
                                    recommended_check=f"Confirm `{col}` is available at prediction time.")
        vc = tgt.value_counts(normalize=True)
        minority = float(vc.min()) if len(vc) > 1 else 0.0
        e_bal = led.add_evidence(src, f"column={target}", ", ".join(f"{k:g}: {v:.1%}" for k, v in vc.items()),
                                 "value_counts(normalize)", value=minority)
        led.summary["target_balance"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        if minority < 0.15:
            led.add_finding(f"Target `{target}` is imbalanced (minority class {minority:.1%}).", [e_bal],
                            category="imbalance", severity="high" if minority < 0.05 else "moderate",
                            recommended_check="Use precision/recall or PR-AUC, not accuracy; consider stratified splits.")

    # ---- redundant numeric pairs ----------------------------------------------
    nums = {c: i["numeric"] for c, i in info.items() if i["type"] == "numeric" and c != target and c not in structural}
    if len(nums) >= 2:
        corr = pd.DataFrame(nums).corr()
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if abs(corr.loc[a, b]) >= 0.95:
                    eid = led.add_evidence(src, f"columns={a},{b}", f"Pearson r = {corr.loc[a, b]:.3f}", "pairwise correlation")
                    led.add_finding(f"`{a}` and `{b}` are nearly redundant (r={corr.loc[a, b]:.2f}).", [eid],
                                    category="correlation", severity="info")

    _scorecard(led, df, id_cols, row_flags, dim_evidence, miss)
    led.summary["target"] = target
    _visuals(led, df, info, miss_ev, out_dir)
    _decide(led)
    _recommend(led)
    led.write(out_dir, _report(led, src, e_shape, e_types))
    return led


def _check_categories(led, src, df, col, rule, row_flags, dim_evidence):
    vals = df[col].dropna()
    allowed = rule.get("allowed")
    norm = vals.str.strip().str.lower()
    # case / whitespace variants: different raw strings that normalize to the same label
    variants = {k: sorted(g.unique().tolist()) for k, g in vals.groupby(norm) if g.nunique() > 1}
    if variants:
        bad = df[col].notna() & df[col].isin([v for vs in variants.values() for v in vs]) & (df[col] != df[col].str.strip().str.lower())
        eid = led.add_evidence(src, f"column={col}", f"labels differing only by case/whitespace: {variants}",
                               "group by strip().lower(), nunique > 1", value=int(bad.sum()))
        row_flags["consistency"] |= bad
        dim_evidence["consistency"].append(eid)
        led.add_finding(f"`{col}` has inconsistent label formatting {list(variants.values())}.", [eid],
                        category="formatting", severity="moderate",
                        risk="Group-bys split one category into several.",
                        recommended_check="Normalize case and whitespace at ingestion.")
    if allowed is not None:
        invalid = sorted(set(norm.unique()) - {str(a).lower() for a in allowed})
        reference = [str(a).lower() for a in allowed]
    else:
        freq = norm.value_counts()
        reference = [v for v, c in freq.items() if c >= 0.01 * len(norm)]
        invalid = [v for v, c in freq.items() if c < 0.01 * len(norm)
                   and difflib.get_close_matches(v, [r for r in reference if freq[r] >= 10 * c], 1, 0.8)]
    if invalid:
        matches = {v: (difflib.get_close_matches(v, reference, 1, 0.75) or [None])[0] for v in invalid}
        mask = df[col].notna() & df[col].str.strip().str.lower().isin(invalid)
        eid = led.add_evidence(src, f"column={col}", f"{int(mask.sum())} rows with unexpected labels; closest valid label: {matches}",
                               "allowed-values check" if allowed else "rare label within edit distance of a frequent label",
                               value=int(mask.sum()))
        row_flags["validity"] |= mask
        dim_evidence["validity"].append(eid)
        typo = any(m for m in matches.values())
        led.add_finding(f"`{col}` contains unexpected labels {invalid}" + (" that look like typos of valid labels." if typo else "."),
                        [eid], category="category_anomaly", severity="moderate", kind="inference" if typo else "observation",
                        possible_explanation="Free-text entry or a mapping table missing a value.",
                        recommended_check=f"Map {matches} after confirming with the data owner.")


def _scorecard(led, df, id_cols, row_flags, dim_evidence, miss):
    n = len(df)
    present = [c for c in id_cols if c in df]
    repeated = df.duplicated(subset=present) if present else df.duplicated()
    scores = {
        "Completeness": 1 - float(miss.mean()),
        "Uniqueness": 1 - float(repeated.mean()),
        "Validity": 1 - float(row_flags["validity"].mean()),
        "Consistency": 1 - float(row_flags["consistency"].mean()),
    }
    leak = [f for f in led.findings if f.category in ("leakage", "contamination")]
    led.summary["scorecard"] = {k: round(v, 4) for k, v in scores.items()}
    led.summary["leakage_check"] = "FAIL" if leak else "PASS" if dim_evidence["leakage"] else "NOT_RUN"
    led.summary["_score_evidence"] = dim_evidence


def _visuals(led, df, info, miss_ev, out_dir):
    out = Path(out_dir)
    material = [f for f in led.findings if f.severity != "info"]
    # 1. missingness profile - only when something is missing
    shown = [c for c in miss_ev if c != led.summary.get("target")]
    if shown:
        fig, ax = viz.figure(7.5, 0.45 * len(shown) + 1.4)
        fr = [float(df[c].isna().mean()) for c in shown][::-1]
        cols = shown[::-1]
        flagged = {f.evidence[0] for f in led.findings if f.category == "missingness" and f.severity == "high"}
        colors = [viz.CRITICAL if miss_ev[c] in flagged else viz.SERIES[0] for c in cols]
        ax.barh(cols, [x * 100 for x in fr], color=colors, height=0.6)
        for i, x in enumerate(fr):
            ax.text(x * 100 + 0.8, i, f"{x:.1%}", va="center", fontsize=9, color=viz.INK_2)
        ax.set_xlabel("% of rows missing")
        ax.set_xlim(0, max(fr) * 100 * 1.18)
        ax.grid(axis="y", visible=False)
        ax.set_title("Missing values by column")
        path = viz.save(fig, out, "missingness.png", "Red = high-severity finding. Source: evidence " + ", ".join(miss_ev[c] for c in shown))
        led.add_visual("missingness_chart", path, [miss_ev[c] for c in shown], "Show which fields are incomplete and by how much")
    # 2. category anomalies - the bad labels next to the valid ones
    cat_f = [f for f in led.findings if f.category in ("category_anomaly",) or
             (f.category == "formatting" and "label" in f.statement)]
    cat_cols = []
    for f in cat_f:
        col = led.get_evidence(f.evidence[0]).location.split("=")[1]
        if col not in cat_cols:
            cat_cols.append(col)
    if cat_cols:
        col = cat_cols[0]
        vc = df[col].fillna("<missing>").value_counts()
        valid = set(df[col].dropna().str.strip().str.lower().value_counts().loc[lambda s: s >= 0.01 * len(df)].index)
        labels = [repr(v) if v != v.strip() else v for v in vc.index]
        is_bad = [not (v in valid and v == v.strip().lower()) and v != "<missing>" for v in vc.index]
        fig, ax = viz.figure(7.5, 3.6)
        ax.bar(labels, vc.values, color=[viz.CRITICAL if b else viz.SERIES[0] for b in is_bad], width=0.6)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9, color=viz.INK_2)
        ax.set_yscale("log")
        ax.set_ylabel("rows (log scale)")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"`{col}` label frequencies — anomalous labels in red")
        ev = [e for f in cat_f for e in f.evidence if led.get_evidence(e).location == f"column={col}"]
        path = viz.save(fig, out, "category-anomalies.png", "Source: evidence " + ", ".join(ev))
        led.add_visual("category_frequency", path, ev, f"Make typo and formatting variants in `{col}` visible against valid labels")
    # 3. impossible / out-of-range values
    rv = [f for f in led.findings if f.category == "range_violation"]
    if rv:
        f = rv[0]
        col = led.get_evidence(f.evidence[0]).location.split(";")[0].split("=")[1]
        x = info[col]["numeric"].dropna()
        fig, ax = viz.figure(7.5, 2.6)
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(x))
        ev = led.get_evidence(f.evidence[0])
        bad_vals = set(map(float, re.findall(r"-?\d+(?:\.\d+)?", ev.observation.split(":")[-1])))
        bad = x.isin(bad_vals)
        ax.scatter(x[~bad], jitter[~bad], s=8, color=viz.SERIES[0], alpha=0.35, linewidths=0)
        ax.scatter(x[bad], jitter[bad], s=40, color=viz.CRITICAL, zorder=3)
        for v, j in zip(x[bad], jitter[bad]):
            ax.annotate(f"{v:g}", (v, j), xytext=(4, 6), textcoords="offset points", fontsize=9, color=viz.INK)
        ax.set_yticks([])
        ax.set_xlabel(col)
        ax.set_title(f"`{col}` values — out-of-range values in red")
        path = viz.save(fig, out, f"range-{col}.png", f"Source: evidence {f.evidence[0]}")
        led.add_visual("range_violation", path, f.evidence, f"Show impossible `{col}` values relative to the valid distribution")
    # 4. scorecard - only when there is something material to summarize
    if material:
        sc = led.summary["scorecard"]
        ev = sorted({e for v in led.summary["_score_evidence"].values() for e in v})
        fig, ax = viz.figure(7.5, 3.0)
        names = list(sc)[::-1]
        affected = [(1 - sc[k]) * 100 for k in names]
        colors = [viz.GOOD if v == 0 else viz.WARNING if v < 1 else viz.CRITICAL for v in affected]
        ax.barh(names, affected, color=colors, height=0.55)
        for i, v in enumerate(affected):
            ax.text(v + max(affected) * 0.01 + 0.02, i, f"{v:.1f}%", va="center", fontsize=9, color=viz.INK_2)
        ax.set_xlim(0, max(max(affected) * 1.15, 1))
        ax.set_xlabel("% of rows (cells, for completeness) affected")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Quality issues by dimension · leakage check: {led.summary['leakage_check']}")
        path = viz.save(fig, out, "quality-scorecard.png",
                        "Green 0% · amber <1% · red ≥1%. Source: evidence " + ", ".join(ev))
        led.add_visual("quality_scorecard", path, ev, "Summarize quality dimensions and the leakage check in one view")
    led.summary.pop("_score_evidence", None)


def _decide(led):
    crit = [f for f in led.findings if f.severity == "critical"]
    high = [f for f in led.findings if f.severity == "high"]
    if crit:
        led.decision, led.decision_confidence = "NOT_FIT_AS_IS", "high"
        led.decision_rationale = [f"{f.id}: {f.statement}" for f in crit]
    elif high:
        led.decision, led.decision_confidence = "FIT_WITH_REMEDIATION", "medium"
        led.decision_rationale = [f"{f.id}: {f.statement}" for f in high]
    else:
        led.decision, led.decision_confidence = "FIT_FOR_USE", "medium"
        led.decision_rationale = ["No critical or high-severity issue was detected by the checks that were run."]
        if led.gaps:
            led.decision_rationale.append("Confidence is limited by the gaps listed below.")


def _recommend(led):
    actions = {
        "leakage": "Remove leakage columns from the feature set and document when each field is written.",
        "contamination": "Re-split train/test by entity identifier and recompute any evaluation done on this split.",
        "duplicates": "Deduplicate by identifier with an explicit rule (latest record wins) before analysis.",
        "range_violation": "Null or correct out-of-range values after checking the source records.",
        "date_anomaly": "Repair or quarantine malformed dates; re-derive any time-based split.",
        "category_anomaly": "Map unexpected labels to valid ones with data-owner sign-off.",
        "formatting": "Normalize formats (numbers, label case/whitespace) at ingestion.",
        "missingness": "Investigate the mechanism behind high missingness before imputing or dropping.",
        "constant": "Drop constant columns and confirm the extract's population filter.",
        "imbalance": "Evaluate with class-sensitive metrics and stratified splits.",
    }
    for cat, action in actions.items():
        fs = [f.id for f in led.findings if f.category == cat and f.severity != "info"]
        if fs:
            led.recommend(action, fs)


def _report(led, src, e_shape, e_types) -> str:
    F = led.findings
    by = lambda *sev: [f for f in F if f.severity in sev]  # noqa: E731
    sc = led.summary.get("scorecard", {})
    sc_rows = "\n".join(f"| {k} | {v:.1%} |" for k, v in sc.items())
    parts = [
        f"# Data Profile: `{src}`\n",
        f"**Intended use:** {led.intent}  \n**Fitness decision:** `{led.decision}` (confidence `{led.decision_confidence}`)\n",
        f"## Dataset Summary\n\n{led.summary['rows']} rows × {led.summary['columns']} columns (`{e_shape}`). "
        f"{len(F)} findings: {len(by('critical'))} critical, {len(by('high'))} high, {len(by('moderate'))} moderate, "
        f"{len(by('info'))} informational.\n",
        render.findings_section(led, "Critical Findings", by("critical")),
        render.findings_section(led, "High-Risk Quality Issues", by("high")),
        render.findings_section(led, "Moderate Issues", by("moderate")),
        render.findings_section(led, "Informational Observations", by("info")),
        "## Schema / Type Findings\n\n" + "| Column | Inferred type |\n|---|---|\n" +
        "\n".join(f"| {c} | {t} |" for c, t in led.summary["inferred_types"].items()) + f"\n\nEvidence: `{e_types}`\n",
        "## Missingness Analysis\n\n" + ("\n".join(f"- `{c}`: {v:.1%}" for c, v in led.summary["missing_fraction"].items())
                                         or "No missing values.") + "\n",
        "## Duplicate Analysis\n\n" + ("\n".join(f"- {f.id}: {f.statement}" for f in F if f.category in ("duplicates", "contamination"))
                                       or "No duplicates detected.") + "\n",
        "## Potential Leakage\n\n" + ("\n".join(f"- {f.id}: {f.statement}" for f in F if f.category == "leakage")
                                      or ("No leakage signal detected." if led.summary.get("target_balance") else "Not assessed (no target).")) + "\n",
        f"## Data Quality Scorecard\n\n| Dimension | Rows/cells unaffected |\n|---|---|\n{sc_rows}\n| Leakage/contamination check | {led.summary['leakage_check']} |\n",
        render.visuals_section(led),
        render.evidence_section(led),
        render.list_section("Assumptions", led.assumptions),
        render.list_section("Unknowns / Gaps", led.gaps),
        render.recommendations_section(led),
    ]
    return "\n".join(parts)
