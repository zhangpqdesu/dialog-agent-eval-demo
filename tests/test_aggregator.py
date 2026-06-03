"""Tests for :mod:`evaluator.aggregator`.

We feed in synthetic detail dicts that mirror the exact shape the
runner CLI writes to disk so the aggregator's contract is locked
against the actual output format.
"""
from __future__ import annotations

from evaluator.aggregator import aggregate


def _detail(
    *,
    agent: str,
    scenario_id: str,
    instruction_id: str,
    profile_id: str,
    overall_score: float,
    confidence: float = 1.0,
    needs_human_review: bool = False,
    l3_skipped: bool = False,
    findings: list[dict] | None = None,
    inconsistency_flags: list[str] | None = None,
) -> dict:
    return {
        "agent_name": agent,
        "scenario": {
            "scenario_id": scenario_id,
            "instruction_id": instruction_id,
            "profile_id": profile_id,
            "persona": "test",
        },
        "conversation": [],
        "report": {
            "rule_report": {"overall_score": overall_score},
            "layered_report": {
                "overall_score": overall_score,
                "layer_scores": {
                    "L1_violation_count": 0.0,
                    "L2_pass_ratio": 1.0,
                    "L3_mean": None if l3_skipped else 0.75,
                },
                "confidence": confidence,
                "needs_human_review": needs_human_review,
                "inconsistency_flags": inconsistency_flags or [],
                "findings": findings or [],
                "meta": {"l3_skipped": l3_skipped},
            },
        },
    }


def test_matrix_groups_by_agent_instruction_profile_and_keeps_scenarios():
    details = [
        _detail(agent="A", scenario_id="1_p1", instruction_id="1", profile_id="p1", overall_score=80),
        _detail(agent="A", scenario_id="1_p1_dup", instruction_id="1", profile_id="p1", overall_score=60),
        _detail(agent="A", scenario_id="1_p2", instruction_id="1", profile_id="p2", overall_score=90),
        _detail(agent="B", scenario_id="2_p1", instruction_id="2", profile_id="p1", overall_score=50),
    ]
    out = aggregate(details)
    cell = out["matrix"]["A"]["1"]["p1"]
    # Two scenarios under the same (agent, instruction, profile) must be
    # averaged, not silently dropped — that's the whole reason we store
    # cells as objects instead of bare scalars.
    assert cell["count"] == 2
    assert cell["score_mean"] == 70.0
    assert sorted(cell["scenario_ids"]) == ["1_p1", "1_p1_dup"]
    assert out["matrix"]["A"]["1"]["p2"]["score_mean"] == 90.0
    assert out["matrix"]["B"]["2"]["p1"]["score_mean"] == 50.0


def test_radar_emits_null_dimensions_when_judge_was_unavailable():
    details = [
        _detail(
            agent="offline_only",
            scenario_id="1_p1", instruction_id="1", profile_id="p1",
            overall_score=70, l3_skipped=True, confidence=0.5,
            needs_human_review=True,
        ),
    ]
    out = aggregate(details)
    assert "offline_only" in out["radar"]
    # No L3 findings → every dimension must be None (not 0.0), so the
    # front-end can show "not measured" instead of misleading zeros.
    assert out["radar"]["offline_only"] == {}


def test_radar_averages_l3_dimensions_per_agent():
    findings_a1 = [
        {"finding_id": "L3.naturalness", "layer": "L3", "score": 0.8, "status": "pass"},
        {"finding_id": "L3.accuracy", "layer": "L3", "score": 1.0, "status": "pass"},
    ]
    findings_a2 = [
        {"finding_id": "L3.naturalness", "layer": "L3", "score": 0.4, "status": "fail"},
        {"finding_id": "L3.accuracy", "layer": "L3", "score": 0.6, "status": "partial"},
    ]
    details = [
        _detail(agent="A", scenario_id="s1", instruction_id="1", profile_id="p1",
                overall_score=80, findings=findings_a1),
        _detail(agent="A", scenario_id="s2", instruction_id="1", profile_id="p2",
                overall_score=60, findings=findings_a2),
    ]
    out = aggregate(details)
    assert out["radar"]["A"]["naturalness"] == 0.6
    assert out["radar"]["A"]["accuracy"] == 0.8


def test_failure_modes_combines_all_three_layers_and_caps_top_n():
    findings = [
        {"finding_id": "L1.r1_max_chars", "layer": "L1", "status": "violated", "score": 0},
        {"finding_id": "L2.step_3", "layer": "L2", "status": "fail", "score": 0},
        {"finding_id": "L2.step_4", "layer": "L2", "status": "partial", "score": 0.5},
        {"finding_id": "L3.accuracy", "layer": "L3", "status": "fail", "score": 0.2},
        # A passing L3 finding must NOT enter failure_modes.
        {"finding_id": "L3.empathy_handling", "layer": "L3", "status": "pass", "score": 0.9},
    ]
    details = [
        _detail(agent="A", scenario_id="s1", instruction_id="1", profile_id="p1",
                overall_score=40, findings=findings),
        _detail(agent="A", scenario_id="s2", instruction_id="1", profile_id="p2",
                overall_score=50, findings=[findings[0], findings[2]]),
    ]
    out = aggregate(details, failure_top_n=10)
    by_id = {row["finding_id"]: row for row in out["failure_modes"]}
    assert by_id["L1.r1_max_chars"]["count"] == 2  # both scenarios
    assert by_id["L1.r1_max_chars"]["layer"] == "L1"
    assert by_id["L2.step_4"]["count"] == 2
    assert by_id["L2.step_3"]["count"] == 1
    assert by_id["L3.accuracy"]["count"] == 1
    assert "L3.empathy_handling" not in by_id

    # top_n cap must actually trim.
    trimmed = aggregate(details, failure_top_n=2)
    assert len(trimmed["failure_modes"]) == 2


def test_low_confidence_picks_up_review_skip_and_threshold():
    details = [
        _detail(agent="A", scenario_id="ok", instruction_id="1", profile_id="p1",
                overall_score=80, confidence=0.95),  # excluded
        _detail(agent="A", scenario_id="threshold", instruction_id="1", profile_id="p1",
                overall_score=70, confidence=0.4),  # confidence below threshold
        _detail(agent="A", scenario_id="reviewed", instruction_id="1", profile_id="p1",
                overall_score=80, confidence=0.9, needs_human_review=True),
        _detail(agent="A", scenario_id="skipped", instruction_id="1", profile_id="p1",
                overall_score=70, l3_skipped=True, confidence=0.5,
                needs_human_review=True),
    ]
    out = aggregate(details, low_confidence_threshold=0.6)
    sids = {r["scenario_id"] for r in out["low_confidence"]}
    assert "ok" not in sids
    assert sids == {"threshold", "reviewed", "skipped"}
    # Output is sorted ascending by confidence so the most uncertain
    # rows surface first in the report.
    confs = [r["confidence"] for r in out["low_confidence"]]
    assert confs == sorted(confs)


def test_aggregate_handles_empty_input():
    out = aggregate([])
    assert out["matrix"] == {}
    assert out["radar"] == {}
    assert out["failure_modes"] == []
    assert out["low_confidence"] == []
    assert out["totals"]["scenarios"] == 0
    assert out["totals"]["agents"] == []
