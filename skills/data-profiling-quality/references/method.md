# Profiling method and thresholds

Every rule is in `dskit/profiling.py`. The thresholds are defaults chosen to be conservative. Change them in code instead of quietly reinterpreting them in prose.

| Check | Method | Severity |
|---|---|---|
| Type inference | raw strings → numeric / ISO date (≥90% parse) / categorical (≤50 levels or <50% unique) / text | evidence only |
| Missingness | null count / rows | ≥30% high · ≥5% moderate · >0 info; one level higher for `required_columns` |
| Structural missingness | target rate grouped by `notnull(col)` is exactly {0, 1} | reported as **leakage** (critical) instead of missingness |
| Exact duplicates | `DataFrame.duplicated()` | high |
| Duplicate identifiers | value_counts > 1 on `id_columns` (or detected `*_id` columns ≥95% unique) | high |
| Train/test contamination | identifier present in more than one split | critical |
| Mixed numeric formats | parses only after stripping `$ € £ , space` | moderate |
| Range violations | schema `min`/`max`; with no schema, generic plausibility (`age` 0–120, recorded as an assumption) | high if physically impossible, else moderate |
| Allowed values | schema `allowed` | moderate (categorical) / high (numeric codes) |
| Label anomalies | case/whitespace variants; rare labels (<1%) within edit-distance ratio 0.8 of a label ≥10× more frequent | moderate; a typo match is an *inference* |
| Malformed dates | non-null values that fail `%Y-%m-%d` | >5% high, else moderate |
| Future dates | date > `as_of` (default today) | moderate |
| Constant columns | `nunique == 1` → moderate; top value ≥99% → info | |
| Numeric leakage | \|Pearson r\| with target ≥ 0.9 | high (inference) |
| Class imbalance | minority < 15% moderate, < 5% high | |
| Outliers | Tukey fences at 3×IQR, excluding values already flagged as range violations | info only |
| Redundant features | \|r\| ≥ 0.95 between numeric features | info |

## Fitness decision

- any **critical** → `NOT_FIT_AS_IS`
- any **high** → `FIT_WITH_REMEDIATION`
- otherwise → `FIT_FOR_USE` (gaps are listed, and confidence stays `medium` because only these checks were run)

## What this method cannot see

Sampling bias, label noise, semantic errors in values that are plausible but wrong, and whether the extract covers the population you intend. When the intended use depends on any of these, list it under gaps.
