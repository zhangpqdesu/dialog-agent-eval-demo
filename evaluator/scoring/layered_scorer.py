"""LayeredScorer — orchestrates L1+L2+L3 and computes the final score.

Output is a ``ScoringResult`` dataclass with:
  - per-layer findings (L1/L2/L3 Finding lists)
  - aggregated layer scores
  - overall_score (penalty-style)
  - confidence (mean L3 confidence)
  - inconsistency_flags (where L1/L2/L3 disagree)
  - needs_human_review (bool)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from config import ScoringCfg
from evaluator.scoring.evidence import Finding
from evaluator.scoring.l1_rule_gate import run_rule_gate
from evaluator.scoring.l2_flow_coverage import run_flow_coverage
from evaluator.scoring.l3_semantic_judge import DIMENSIONS, run_semantic_judge


@dataclass
class ScoringResult:
    overall_score: float
    layer_scores: dict[str, float | None]
    confidence: float
    needs_human_review: bool
    findings: list[Finding] = field(default_factory=list)
    inconsistency_flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "layer_scores": {
                k: (None if v is None else round(v, 2))
                for k, v in self.layer_scores.items()
            },
            "confidence": round(self.confidence, 4),
            "needs_human_review": self.needs_human_review,
            "inconsistency_flags": self.inconsistency_flags,
            "findings": [f.to_dict() for f in self.findings],
            "meta": self.meta,
        }


def _l1_penalty(findings: list[Finding], penalty_per: float) -> float:
    return sum(penalty_per for f in findings if f.status == "violated")


def _layer_pass_ratio(findings: list[Finding]) -> float:
    if not findings:
        return 1.0
    return sum(f.score for f in findings) / len(findings)


def _detect_inconsistencies(
    l1: list[Finding], l2: list[Finding], l3: list[Finding]
) -> list[str]:
    flags: list[str] = []
    l1_flow_violated = any(
        f.status == "violated" and f.category == "flow" for f in l1
    )
    l2_all_pass = bool(l2) and all(f.status == "pass" for f in l2)
    if l1_flow_violated and l2_all_pass:
        flags.append("L1 flow violation but L2 all pass")

    l1_constraint_violated = any(
        f.status == "violated" and f.category == "constraint" for f in l1
    )
    l3_constraint_aware = next(
        (f for f in l3 if f.finding_id == "L3.constraint_awareness"), None
    )
    if l1_constraint_violated and l3_constraint_aware and l3_constraint_aware.score >= 0.75:
        flags.append("L1 constraint violated but L3 constraint_awareness >= 4/5")

    return flags


class LayeredScorer:
    def __init__(
        self,
        scoring_cfg: ScoringCfg,
        l2_llm: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
        l3_llm: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = scoring_cfg
        self.l2_llm = l2_llm
        self.l3_llm = l3_llm

    def score(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        conversation: list[dict[str, Any]],
    ) -> ScoringResult:
        l1 = run_rule_gate(example, scenario, conversation)
        l2 = run_flow_coverage(example, conversation, llm_complete_json=self.l2_llm)
        l3_skipped = self.l3_llm is None
        l3 = (
            run_semantic_judge(example, conversation, self.l3_llm, self.cfg.l3_samples)
            if not l3_skipped else []
        )

        all_findings: list[Finding] = []
        all_findings.extend(l1)
        all_findings.extend(l2)
        all_findings.extend(l3)

        l1_penalty = _l1_penalty(l1, self.cfg.l1_penalty_per_violation)
        l2_ratio = _layer_pass_ratio(l2)
        l3_ratio = _layer_pass_ratio(l3) if l3 else 0.0

        denom = self.cfg.l2_weight + self.cfg.l3_weight
        weighted = 0.0
        if denom > 0:
            if l3:
                weighted = (self.cfg.l2_weight * l2_ratio + self.cfg.l3_weight * l3_ratio) / denom
            else:
                weighted = l2_ratio
        overall = max(0.0, weighted * 100 - l1_penalty)

        layer_scores: dict[str, float | None] = {
            "L1_violation_count": float(sum(1 for f in l1 if f.status == "violated")),
            "L2_pass_ratio": l2_ratio,
            # Distinguish "L3 ran and scored 0" from "L3 was skipped".
            "L3_mean": None if l3_skipped else l3_ratio,
        }

        l3_confs = [f.confidence for f in l3]
        if l3_confs:
            confidence = sum(l3_confs) / len(l3_confs)
        elif l3_skipped:
            # Cap confidence when we never asked the judge: callers must
            # not mistake "no semantic check" for "high-confidence pass".
            confidence = 0.5
        else:
            # L3 ran but produced no findings (edge case: empty dimensions).
            confidence = 1.0
        flags = _detect_inconsistencies(l1, l2, l3)
        if l3_skipped:
            flags.append("L3 skipped (no judge available)")
        needs_review = (
            confidence < self.cfg.confidence_threshold
            or bool(flags)
            or l3_skipped
        )

        return ScoringResult(
            overall_score=overall,
            layer_scores=layer_scores,
            confidence=confidence,
            needs_human_review=needs_review,
            findings=all_findings,
            inconsistency_flags=flags,
            meta={
                "l1_penalty_total": l1_penalty,
                "l1_penalty_per_violation": self.cfg.l1_penalty_per_violation,
                "l2_weight": self.cfg.l2_weight,
                "l3_weight": self.cfg.l3_weight,
                "l3_samples": self.cfg.l3_samples,
                "l3_skipped": l3_skipped,
                "confidence_threshold": self.cfg.confidence_threshold,
                "dimensions": list(DIMENSIONS.keys()),
            },
        )
