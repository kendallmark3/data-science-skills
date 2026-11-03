# Examples

**Report from a claims file.** It contains strong, weak, conflicting and unsupported claims, plus a visual that cites missing evidence.

```bash
dsk report --claims fixtures/evidence/review_claims.yaml --out output/report-claims
```

Expected:
- C-1 is reported at `high` confidence.
- C-2 drops from `high` to `low`.
- C-3 and C-4 appear under Conflicts.
- C-5 (no evidence) and C-6 (cites missing evidence) are listed as unsupported.
- The visual XV-2 is excluded.
- "Expand the offer to all" is withheld because it rests on C-5.

**Report across skills.** This is the cross-skill pipeline.

```bash
dsk profile fixtures/datasets/customers.csv --context fixtures/datasets/customers.context.yaml --out output/pipeline/1-profile
dsk experiment fixtures/experiments/retention_offer_units.csv --meta fixtures/experiments/retention_offer.meta.yaml --out output/pipeline/2-experiment
dsk evaluate fixtures/evaluations/churn_model_predictions.csv --claim fixtures/evaluations/churn_model.claim.yaml \
  --upstream output/pipeline/1-profile --out output/pipeline/3-evaluation
dsk report output/pipeline/1-profile output/pipeline/2-experiment output/pipeline/3-evaluation \
  --intent "Should we launch model-targeted retention offers?" --out output/pipeline/4-report
```

**Typical prompt to Claude:** "Write the readout for Thursday's review from these three analyses. Flag anything in my draft notes (`notes.yaml`) that the data doesn't back up."
