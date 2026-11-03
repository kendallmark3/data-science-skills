# Examples

| Fixture | Condition | Expected |
|---|---|---|
| `churn_model` | AUC 0.91 overall; new-customer recall 0.57 (CI upper bound 0.62) vs ≥0.75; new customers are 20% of the eval set vs 35% of production | `NOT_READY` (critical segment) |
| `support_assistant` (strict) | task success 93% but groundedness 78%, unsupported claims 16% | `NOT_READY` |
| `support_assistant_pilot` (lenient thresholds, same data) | requirements pass; weak judge, policy-category groundedness, missed refusals | `READY_WITH_CONDITIONS` |

```bash
dsk evaluate fixtures/evaluations/churn_model_predictions.csv \
  --claim fixtures/evaluations/churn_model.claim.yaml --out output/evaluation-churn

dsk evaluate fixtures/evaluations/support_assistant_llm_eval.csv \
  --claim fixtures/evaluations/support_assistant.claim.yaml --out output/evaluation-llm
```

**Typical prompt to Claude:** "Our RAG assistant scores 93% on the eval set. Is it ready to answer customers without review? Results are in `eval_run_1012.csv`."
