"""Shared evidence + finding records used by all three scoring layers.

Every scoring conclusion (L1/L2/L3) carries one or more Evidence items
so the report can highlight the specific turn that triggered the
decision. This is the foundation of the explainability promise in the
design doc.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Evidence:
    """One pointer into the conversation that justifies a finding.

    ``turn_index`` is the index inside the conversation list.
    ``excerpt`` is the literal text shown in the report (may be shortened).
    ``reason`` is a short human-readable explanation.
    """

    turn_index: int
    role: Literal["user", "agent"]
    excerpt: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """A single scoring conclusion produced by any of the three layers."""

    finding_id: str
    layer: Literal["L1", "L2", "L3"]
    category: str
    status: Literal["pass", "partial", "fail", "violated"]
    score: float
    confidence: float = 1.0
    rationale: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "layer": self.layer,
            "category": self.category,
            "status": self.status,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "evidence": [e.to_dict() for e in self.evidence],
            "extra": self.extra,
        }


def excerpt_of(text: str, max_chars: int = 80) -> str:
    """Trim a long turn text for evidence display."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
