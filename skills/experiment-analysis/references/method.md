# Experiment method

All code is in `dskit/experiment.py`.

| Quantity | Method |
|---|---|
| Sample ratio mismatch | χ² goodness of fit of assignment counts against `expected_split`; invalid if p < 0.001 |
| Binary effect | diff = p_t − p_c; Wald CI with unpooled SE; p-value from the pooled two-proportion z-test |
| Continuous effect | Welch's t-test; CI with Welch–Satterthwaite df |
| Practical significance | the CI is compared with `practical_threshold.absolute`, or `relative × control mean` |
| Power | MDE₈₀ = (z₁₋α/₂ + z₀.₈) × SE; n per arm needed to detect the threshold, by normal approximation |
| Guardrails | breach when the CI excludes zero **and** the degradation exceeds tolerance; a CI that reaches past the tolerance with no confirmed breach is reported as moderate |
| Segments | per-segment binary effect (segments with ≥30 per arm), Holm-adjusted p-values, Cochran's Q for heterogeneity |
| Novelty | effect in the first half of days vs the second half, z-test on the difference |
| Duration | observed span < 7 days → day-of-week risk; span < `planned_duration_days` → early-stopping risk |
| Attrition | missing primary outcome by arm, χ² test; p < 0.01 → invalid |

## Reading the result

- **Statistical significance** answers "is the effect probably not zero?"
- **Practical significance** answers "is it big enough to act on?" Only the business can set the threshold, and the engine never sets it.
- **Power** is reported as a finding only when the result is ambiguous, meaning the CI straddles the threshold. Once a CI has settled the question, power doesn't change the decision.

## Limits

- Units are assumed independent. Clustered assignment needs cluster-robust SEs, and the engine doesn't provide them.
- The engine has no sequential-testing correction. If anyone peeked at the results, treat the p-value as optimistic.
- Ratio metrics (revenue per session, for example) need the delta method. They are not covered.
