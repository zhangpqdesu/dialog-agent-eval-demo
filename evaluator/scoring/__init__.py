"""Layered scoring package.

L1 — RuleGate: deterministic 0/1 checks for hard constraints.
L2 — FlowCoverage: rule-based evidence + LLM semantic verification.
L3 — SemanticJudge: multi-dimension LLM judging with sample voting.

All findings share the same Evidence schema so the report layer can
link a score back to specific turns of the conversation.
"""
from evaluator.scoring.evidence import Evidence, Finding
from evaluator.scoring.l1_rule_gate import RuleGate, run_rule_gate
from evaluator.scoring.l2_flow_coverage import FlowCoverage, run_flow_coverage
from evaluator.scoring.l3_semantic_judge import SemanticJudge, run_semantic_judge
from evaluator.scoring.layered_scorer import LayeredScorer, ScoringResult

__all__ = [
    "Evidence",
    "Finding",
    "RuleGate",
    "run_rule_gate",
    "FlowCoverage",
    "run_flow_coverage",
    "SemanticJudge",
    "run_semantic_judge",
    "LayeredScorer",
    "ScoringResult",
]
