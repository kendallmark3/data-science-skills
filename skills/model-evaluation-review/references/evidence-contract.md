# Shared evidence contract

All four skills write the same structures (`dskit/evidence.py`).

```jsonc
// evidence.json — one item per measured fact
{"id": "DP-E-001", "source": "customers.csv", "location": "column=email",
 "observation": "14.8% missing (178 of 1200 rows)", "method": "null count / row count",
 "confidence": "high", "value": 0.148}

// findings.json → findings[]
{"id": "DP-F-001", "statement": "`email` is 14.8% missing and is marked as required.",
 "kind": "observation" | "inference", "category": "missingness",
 "severity": "info" | "moderate" | "high" | "critical", "confidence": "low" | "medium" | "high",
 "evidence": ["DP-E-001"], "risk": "...", "possible_explanation": "...", "recommended_check": "..."}

// findings.json → visuals[]
{"id": "DP-V-001", "type": "missingness_chart", "path": "visuals/missingness.png",
 "evidence": ["DP-E-001", "DP-E-002"], "purpose": "Show which fields are incomplete"}

// findings.json → recommendations[]
{"id": "DP-R-001", "action": "...", "based_on": ["DP-F-001"]}
```

These invariants are enforced in code, not left to convention:

1. A finding must cite at least one existing evidence ID.
2. A finding's confidence is capped at the highest confidence among its evidence.
3. A visual must cite at least one existing evidence ID.
4. A recommendation must cite at least one existing finding.
5. Evidence imported from another skill keeps its original ID (`DP-…`, `EX-…`, `ME-…`), so a report finding can be traced back to the raw file.

ID prefixes: `DP` data profiling · `EX` experiment · `ME` model evaluation · `RP` report · `X` supplied by a person.

`findings.json` also has `decision`, `decision_confidence`, `decision_rationale`, `assumptions`, `gaps`, `summary` and `upstream`. A run that stopped has `"status": "STOPPED"` and a `stop_reason`.
