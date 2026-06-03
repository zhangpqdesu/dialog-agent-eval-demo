"""Tests for the layered scoring stack (L1/L2/L3 + LayeredScorer)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import ScoringCfg
from evaluator.scoring import (
    LayeredScorer,
    run_flow_coverage,
    run_rule_gate,
    run_semantic_judge,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    payload = json.loads(
        (ROOT / "data" / "processed" / "dialog_instruction_eval_examples.json").read_text(encoding="utf-8")
    )
    return payload["examples"][0]


# ── L1: RuleGate ──

def test_l1_max_chars_violation(example):
    conv = [
        {"role": "user", "text": "是我，你说。"},
        {"role": "agent", "text": "你好" * 50},  # way over 30 chars
    ]
    findings = run_rule_gate(example, {"profile_id": "cooperative_user"}, conv)
    by_id = {f.finding_id: f for f in findings}
    assert by_id["L1.max_chars_per_turn"].status == "violated"
    assert by_id["L1.max_chars_per_turn"].evidence[0].turn_index == 1


def test_l1_max_chars_pass(example):
    conv = [
        {"role": "user", "text": "嗯。"},
        {"role": "agent", "text": "今天合同生效了。"},
    ]
    findings = run_rule_gate(example, {"profile_id": "cooperative_user"}, conv)
    by_id = {f.finding_id: f for f in findings}
    assert by_id["L1.max_chars_per_turn"].status == "pass"


def test_l1_opening_line_present(example):
    conv = [
        {"role": "user", "text": "你说。"},
        {"role": "agent", "text": "你好，飞毛腿合同今天生效。"},
    ]
    findings = run_rule_gate(example, {"profile_id": "cooperative_user"}, conv)
    by_id = {f.finding_id: f for f in findings}
    assert by_id["L1.must_use_opening_line"].status == "pass"


def test_l1_opening_line_missing(example):
    conv = [
        {"role": "user", "text": "你说。"},
        {"role": "agent", "text": "嗯哼，今天天气真好。"},
    ]
    findings = run_rule_gate(example, {"profile_id": "cooperative_user"}, conv)
    by_id = {f.finding_id: f for f in findings}
    assert by_id["L1.must_use_opening_line"].status == "violated"


def test_l1_driving_endcall_pass(example):
    conv = [
        {"role": "user", "text": "我在开车。"},
        {"role": "agent", "text": "好的，稍后再打，注意安全。"},
    ]
    findings = run_rule_gate(example, {"profile_id": "driving_user"}, conv)
    by_id = {f.finding_id: f for f in findings}
    assert by_id["L1.driving_user_endcall"].status == "pass"


def test_l1_driving_endcall_violated(example):
    conv = [
        {"role": "user", "text": "我在开车。"},
        {"role": "agent", "text": "今天飞毛腿合同生效了。"},
    ]
    findings = run_rule_gate(example, {"profile_id": "driving_user"}, conv)
    by_id = {f.finding_id: f for f in findings}
    assert by_id["L1.driving_user_endcall"].status == "violated"


# ── L2: FlowCoverage (keyword-only mode, deterministic) ──

def test_l2_keyword_only_partial_coverage(example):
    conv = [
        {"role": "user", "text": "嗯。"},
        {"role": "agent", "text": "飞毛腿合同生效，可以开始配送了。"},
        {"role": "user", "text": "好。"},
        {"role": "agent", "text": "记得注意安全。"},
    ]
    findings = run_flow_coverage(example, conv, llm_complete_json=None)
    assert any(f.status in {"pass", "partial"} for f in findings)
    assert all(f.finding_id.startswith("L2.") for f in findings)


def test_l2_with_llm_calls_judge(example):
    captured = []

    def fake_llm(messages):
        captured.append(messages)
        return {"verdict": "pass", "rationale": "覆盖到了",
                "best_turn_index": 1}

    conv = [
        {"role": "user", "text": "嗯。"},
        {"role": "agent", "text": "飞毛腿合同生效，能开始配送吗？"},
    ]
    findings = run_flow_coverage(example, conv, llm_complete_json=fake_llm)
    assert len(captured) == len(example["instruction_core"]["call_flow"]["steps"])
    for f in findings:
        assert f.status == "pass"
        assert f.score == 1.0


# ── L3: SemanticJudge (mocked LLM, multi-sample) ──

def test_l3_multi_sample_median_and_confidence(example):
    sequence = iter([
        {"score": 4, "rationale": "ok", "evidence_turn_indices": [1]},
        {"score": 5, "rationale": "great", "evidence_turn_indices": [1]},
        {"score": 4, "rationale": "ok", "evidence_turn_indices": [1]},
    ] * 5)  # 5 dimensions * 3 samples = 15 calls

    def fake_llm(messages):
        return next(sequence)

    conv = [
        {"role": "user", "text": "嗯。"},
        {"role": "agent", "text": "你好，飞毛腿合同生效，今天能配送吗？"},
    ]
    findings = run_semantic_judge(example, conv, fake_llm, n_samples=3)
    assert len(findings) == 5
    for f in findings:
        assert f.layer == "L3"
        assert 0.0 <= f.score <= 1.0
        assert f.confidence >= 0.5  # stdev small -> high confidence


def test_l3_handles_llm_error_with_neutral_score(example):
    def bad_llm(messages):
        raise RuntimeError("network down")

    conv = [{"role": "user", "text": "嗯。"},
            {"role": "agent", "text": "你好。"}]
    findings = run_semantic_judge(example, conv, bad_llm, n_samples=2)
    assert all(f.score == (3 - 1) / 4 for f in findings)  # neutral 3/5
    # confidence is high because all samples identical (stdev=0)
    assert all(f.confidence > 0.9 for f in findings)


# ── LayeredScorer ──

def test_layered_scorer_no_llm_falls_back_to_l1_plus_l2(example):
    scorer = LayeredScorer(ScoringCfg(l3_samples=1), l2_llm=None, l3_llm=None)
    conv = [
        {"role": "user", "text": "嗯。"},
        {"role": "agent", "text": "你好，飞毛腿合同生效，能配送吗？"},
        {"role": "user", "text": "行。"},
        {"role": "agent", "text": "好的，注意安全，先这样。"},
    ]
    result = scorer.score(example, {"profile_id": "cooperative_user"}, conv)
    assert 0 <= result.overall_score <= 100
    layers = {f.layer for f in result.findings}
    assert "L1" in layers and "L2" in layers
    assert "L3" not in layers


def test_layered_scorer_penalises_l1_violations(example):
    scorer = LayeredScorer(
        ScoringCfg(l3_samples=1, l1_penalty_per_violation=20.0),
    )
    bad_conv = [
        {"role": "user", "text": "嗯。"},
        # Both opening missing AND too long
        {"role": "agent", "text": "x" * 100},
    ]
    result = scorer.score(example, {"profile_id": "cooperative_user"}, bad_conv)
    violated = [f for f in result.findings if f.layer == "L1" and f.status == "violated"]
    assert len(violated) >= 2  # at least max_chars + opening
    # base weighted score - 2*20 penalty should drop significantly
    assert result.overall_score < 60


def test_layered_scorer_includes_dimensions_in_meta(example):
    scorer = LayeredScorer(ScoringCfg(l3_samples=1))
    result = scorer.score(example, {"profile_id": "cooperative_user"},
                          [{"role": "user", "text": "嗯。"},
                           {"role": "agent", "text": "你好。"}])
    assert "dimensions" in result.meta
    assert set(result.meta["dimensions"]) == {
        "naturalness", "accuracy", "empathy_handling",
        "goal_pursuit", "constraint_awareness",
    }


def test_layered_scorer_flags_low_confidence_for_review(example):
    # L3 with wildly varying samples -> low confidence
    sequence = iter([
        {"score": 5, "rationale": "great", "evidence_turn_indices": []},
        {"score": 1, "rationale": "bad", "evidence_turn_indices": []},
        {"score": 5, "rationale": "great", "evidence_turn_indices": []},
    ] * 5)

    def jumpy(messages):
        return next(sequence)

    scorer = LayeredScorer(
        ScoringCfg(l3_samples=3, confidence_threshold=0.9),
        l3_llm=jumpy,
    )
    result = scorer.score(example, {"profile_id": "cooperative_user"},
                          [{"role": "user", "text": "嗯。"},
                           {"role": "agent", "text": "你好。"}])
    assert result.needs_human_review is True
    assert result.confidence < 0.9
