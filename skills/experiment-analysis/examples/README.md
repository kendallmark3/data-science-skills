# Examples

Each fixture plants one condition (see `fixtures/SEEDS.md`):

| Fixture | Condition | Expected |
|---|---|---|
| `checkout_button` | +0.35pp on ~1M users per arm; threshold +1pp | `NOT_SUPPORTED` (significant but trivial) |
| `retention_offer` | +2.8pp, CI [1.9, 3.7]pp; guardrails flat | `SUPPORTED` |
| `pricing_page` | 10,000 vs 8,600 on a 50/50 design | `INVALID_EXPERIMENT` (SRM) |
| `free_shipping` | +3.2pp conversion, refunds +2.8pp vs a 1pp tolerance | `INCONCLUSIVE` (contradictory metrics) |
| `onboarding_email` | 900 per arm, CI spans 0 and the threshold | `INCONCLUSIVE` (underpowered) |
| `no_design` | metadata missing hypothesis, metric and assignment | `INVALID_EXPERIMENT` |

```bash
dsk experiment fixtures/experiments/checkout_button_daily.csv \
  --meta fixtures/experiments/checkout_button.meta.yaml --out output/experiment-checkout
```

**Typical prompt to Claude:** "Here's the export from our pricing test (`pricing_test.csv`). Did the new page win? We needed at least +1pp upgrades to justify it."
