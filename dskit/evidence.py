"""Shared evidence model used by all four skills.

The contract is deliberately small:

    Evidence        something measured, with its source, location and method
    Finding         a statement that cites one or more Evidence IDs
                    kind = "observation" (directly measured) or "inference" (reasoned)
    Visual          a chart that cites the Evidence it draws
    Assumption      something taken as true without verification
    Gap             something the analysis needed but did not have
    Recommendation  an action, citing the Findings that justify it

The Ledger refuses dangling references: a finding cannot cite evidence that does
not exist, a visual cannot be registered without evidence, and a recommendation
cannot exist without findings. That is the mechanism that keeps recommendations
from masquerading as evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIDENCE = ("low", "medium", "high")
SEVERITY = ("info", "moderate", "high", "critical")
KINDS = ("observation", "inference")

_CONF_RANK = {c: i for i, c in enumerate(CONFIDENCE)}


def min_confidence(*levels: str) -> str:
    return min(levels, key=lambda c: _CONF_RANK[c])


def max_confidence(*levels: str) -> str:
    return max(levels, key=lambda c: _CONF_RANK[c])


class TraceabilityError(ValueError):
    """Raised when something tries to cite evidence or findings that do not exist."""


class StopCondition(Exception):
    """Raised when the analysis must stop rather than produce a fabricated answer."""

    def __init__(self, reason: str, decision: str = "INSUFFICIENT_EVIDENCE"):
        super().__init__(reason)
        self.reason = reason
        self.decision = decision


def _jsonable(v: Any) -> Any:
    """Convert numpy scalars and similar to plain Python for JSON output."""
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and v != v:
        return None  # NaN is not valid JSON
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclass
class Evidence:
    id: str
    source: str
    location: str
    observation: str
    method: str
    confidence: str = "high"
    value: Any = None


@dataclass
class Finding:
    id: str
    statement: str
    kind: str
    category: str
    severity: str
    confidence: str
    evidence: list[str]
    risk: str | None = None
    possible_explanation: str | None = None
    recommended_check: str | None = None


@dataclass
class Visual:
    id: str
    type: str
    path: str
    evidence: list[str]
    purpose: str


@dataclass
class Recommendation:
    id: str
    action: str
    based_on: list[str]


@dataclass
class Ledger:
    """Collects evidence, findings and visuals for one skill run."""

    skill: str
    prefix: str
    intent: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    visuals: list[Visual] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    decision: str | None = None
    decision_confidence: str | None = None
    decision_rationale: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    upstream: list[str] = field(default_factory=list)

    # -- registration -----------------------------------------------------
    def _next(self, kind: str, items: list) -> str:
        return f"{self.prefix}-{kind}-{len(items) + 1:03d}"

    def add_evidence(self, source: str, location: str, observation: str, method: str,
                     confidence: str = "high", value: Any = None) -> str:
        assert confidence in CONFIDENCE, confidence
        eid = self._next("E", self.evidence)
        self.evidence.append(Evidence(eid, source, location, observation, method,
                                      confidence, _jsonable(value)))
        return eid

    def import_evidence(self, ev: dict) -> str:
        """Bring evidence from an upstream skill run, preserving its original ID."""
        if not any(e.id == ev["id"] for e in self.evidence):
            self.evidence.append(Evidence(**{k: ev.get(k) for k in Evidence.__dataclass_fields__}))
        return ev["id"]

    def add_finding(self, statement: str, evidence: list[str], *, kind: str = "observation",
                    category: str, severity: str = "info", confidence: str | None = None,
                    risk: str | None = None, possible_explanation: str | None = None,
                    recommended_check: str | None = None) -> str:
        assert kind in KINDS and severity in SEVERITY, (kind, severity)
        self._check_evidence(evidence)
        # A finding can never be more confident than its strongest evidence.
        ceiling = max_confidence(*(self.get_evidence(e).confidence for e in evidence))
        conf = min_confidence(confidence or ceiling, ceiling)
        fid = self._next("F", self.findings)
        self.findings.append(Finding(fid, statement, kind, category, severity, conf,
                                     list(evidence), risk, possible_explanation, recommended_check))
        return fid

    def add_visual(self, vtype: str, path: str, evidence: list[str], purpose: str) -> str:
        self._check_evidence(evidence)
        vid = self._next("V", self.visuals)
        self.visuals.append(Visual(vid, vtype, path, list(evidence), purpose))
        return vid

    def recommend(self, action: str, based_on: list[str]) -> str:
        if not based_on:
            raise TraceabilityError(f"Recommendation without supporting findings: {action!r}")
        known = {f.id for f in self.findings}
        missing = [b for b in based_on if b not in known]
        if missing:
            raise TraceabilityError(f"Recommendation cites unknown findings {missing}")
        rid = self._next("R", self.recommendations)
        self.recommendations.append(Recommendation(rid, action, list(based_on)))
        return rid

    def _check_evidence(self, ids: list[str]) -> None:
        if not ids:
            raise TraceabilityError("Every finding and visual must cite at least one evidence ID")
        known = {e.id for e in self.evidence}
        missing = [i for i in ids if i not in known]
        if missing:
            raise TraceabilityError(f"Unknown evidence IDs: {missing}")

    def get_evidence(self, eid: str) -> Evidence:
        return next(e for e in self.evidence if e.id == eid)

    def findings_by(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    # -- output -------------------------------------------------------------
    def to_dict(self) -> dict:
        return _jsonable({
            "skill": self.skill,
            "intent": self.intent,
            "decision": self.decision,
            "decision_confidence": self.decision_confidence,
            "decision_rationale": self.decision_rationale,
            "summary": self.summary,
            "findings": [asdict(f) for f in self.findings],
            "visuals": [asdict(v) for v in self.visuals],
            "assumptions": self.assumptions,
            "gaps": self.gaps,
            "recommendations": [asdict(r) for r in self.recommendations],
            "upstream": self.upstream,
        })

    def write(self, out_dir: str | Path, report_md: str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "findings.json").write_text(json.dumps(self.to_dict(), indent=2))
        (out / "evidence.json").write_text(
            json.dumps([_jsonable(asdict(e)) for e in self.evidence], indent=2))
        (out / "report.md").write_text(report_md)
        return out


def write_stopped(out_dir: str | Path, skill: str, stop: StopCondition, intent: str = "") -> Path:
    """Write an honest stop report instead of a fabricated result."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"skill": skill, "intent": intent, "decision": stop.decision, "status": "STOPPED",
               "stop_reason": stop.reason, "findings": [], "visuals": [], "gaps": [stop.reason],
               "assumptions": [], "recommendations": [], "upstream": []}
    (out / "findings.json").write_text(json.dumps(payload, indent=2))
    (out / "evidence.json").write_text("[]")
    (out / "report.md").write_text(
        f"# {skill}: STOPPED\n\n**Decision:** `{stop.decision}`\n\n"
        f"The analysis stopped instead of producing an unsupported result.\n\n"
        f"**Reason:** {stop.reason}\n")
    return out


def load_run(run_dir: str | Path) -> tuple[dict, list[dict]]:
    run = Path(run_dir)
    findings = json.loads((run / "findings.json").read_text())
    evidence = json.loads((run / "evidence.json").read_text())
    return findings, evidence
