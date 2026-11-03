# Evaluation-review method

All code is in `dskit/evaluation.py`. The metric functions are checked against scikit-learn in `tests/test_model_evaluation.py`.

| Quantity | Method |
|---|---|
| ROC-AUC | Mann–Whitney rank statistic |
| PR-AUC | average precision (step-wise, as in scikit-learn) |
| Calibration | 10-bin expected calibration error; Brier score |
| Proportion CIs | Wilson score interval, 95% |
| Confirmed segment failure | Wilson upper bound < requirement |
| Coverage | total variation distance between the eval and production mix; a segment is under-represented if its eval share < 0.7 × its production share |
| Production projection | Σ production_share × segment recall. This is an **inference**: it assumes per-segment recall transfers to production |
| Judge validity | Cohen's κ between the automated judge and human labels; κ < 0.6 is weak |
| Hidden failure (LLM) | cases with task_success = 1 and grounded = 0, as a share of successes; > 5% is flagged |

## Claim YAML fields

```yaml
claim: <the sentence being tested>
mode: classification | llm
label_column / score_column / baseline_score_column / threshold   # classification
requirements: {min_recall, min_precision, min_roc_auc, min_f1}      # classification
requirements: {min_task_success, min_groundedness, max_unsupported_claim_rate,
               min_format_valid, max_p95_latency_ms, max_cost_per_case_usd}  # llm
segment_columns: [...]
critical_segments: {column: [values]}       # failing one of these blocks READY
production_mix: {column: {value: share}}
```

## Not covered

Fairness metrics across protected attributes, adversarial or robustness testing, temporal drift (that needs time-stamped production data), and statistical comparison of two models on the same cases (for example a paired bootstrap). If the claim depends on any of these, record it as a gap.
