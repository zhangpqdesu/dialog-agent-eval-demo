"""Tests for :mod:`report.static_html`.

These exercise the renderer contract (must produce non-empty HTML
containing the expected anchors and core data so the JS chart layer
has something to bind to) and the no-data / unicode edge cases.
"""
from __future__ import annotations

from report.static_html import render


def _detail(**kw):
    base = {
        "agent_name": "A",
        "scenario": {
            "scenario_id": "1_p1",
            "instruction_id": "1",
            "profile_id": "p1",
            "persona": "test",
        },
        "conversation": [{"role": "user", "text": "你好"}],
        "report": {
            "rule_report": {"overall_score": 80},
            "layered_report": {
                "overall_score": 80,
                "layer_scores": {"L1_violation_count": 0, "L2_pass_ratio": 1.0, "L3_mean": 0.8},
                "confidence": 0.9,
                "needs_human_review": False,
                "inconsistency_flags": [],
                "findings": [
                    {"finding_id": "L3.naturalness", "layer": "L3", "status": "pass",
                     "score": 0.8, "rationale": "自然"},
                ],
                "meta": {"l3_skipped": False},
            },
        },
    }
    base.update(kw)
    return base


def test_render_contains_core_sections_and_data():
    summary = {
        "run_name": "smoke",
        "started_at": "2024-01-01T00:00:00",
        "finished_at": "2024-01-01T00:01:00",
        "seed": 42,
        "judge_provider": "kimi",
        "judge_available": False,
        "results": [{"agent_name": "A", "scenario_id": "1_p1", "overall_score": 80}],
    }
    details = [_detail()]
    aggregate_dict = {
        "matrix": {"A": {"1": {"p1": {"score_mean": 80.0, "count": 1, "scenario_ids": ["1_p1"]}}}},
        "radar": {"A": {"naturalness": 0.8}},
        "failure_modes": [
            {"layer": "L1", "finding_id": "L1.r1_max_chars", "count": 3, "examples": []},
        ],
        "low_confidence": [
            {"agent_name": "A", "scenario_id": "1_p1", "instruction_id": "1",
             "profile_id": "p1", "overall_score": 80, "confidence": 0.4,
             "needs_human_review": True, "l3_skipped": False, "inconsistency_flags": []},
        ],
        "totals": {"scenarios": 1, "agents": ["A"], "low_confidence_threshold": 0.6},
    }

    out = render(summary, details, aggregate_dict)
    assert isinstance(out, str) and out.startswith("<!doctype html>")
    # Each of the three charts needs its anchor div so ECharts can bind.
    assert 'id="chart-matrix"' in out
    assert 'id="chart-radar"' in out
    assert 'id="chart-failures"' in out
    # Aggregate data must be embedded in the inline JSON blob so the JS
    # side can render without needing a follow-up fetch.
    assert "L1.r1_max_chars" in out
    assert "1_p1" in out
    # The KPI row should reflect summary stats.
    assert "平均总分" in out
    # Low-confidence section is included.
    assert "低置信度场景" in out
    # ECharts CDN is loaded.
    assert "echarts" in out


def test_render_handles_empty_run_without_crashing():
    summary = {"run_name": "empty", "results": []}
    out = render(summary, [], {"matrix": {}, "radar": {}, "failure_modes": [], "low_confidence": [], "totals": {}})
    assert isinstance(out, str) and "<html" in out
    # Empty failure-mode / radar sections should still render their
    # anchor divs (the JS shows a "no data" message in-place).
    assert 'id="chart-failures"' in out
    assert 'id="chart-radar"' in out


def test_render_caps_detail_panels_at_max_detail():
    summary = {"run_name": "many", "results": []}
    details = [_detail(agent_name=f"A{i}") for i in range(10)]
    out = render(summary, details, {"matrix": {}, "radar": {}, "failure_modes": [], "low_confidence": [], "totals": {}}, max_detail=3)
    # Only 3 case panels should be inlined to keep the file small.
    assert out.count('<details class="case">') == 3


def test_render_escapes_html_to_prevent_xss():
    summary = {"run_name": "<script>x</script>", "results": []}
    out = render(summary, [], {"matrix": {}, "radar": {}, "failure_modes": [], "low_confidence": [], "totals": {}})
    # The user-controlled run_name must be HTML-escaped in the body.
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;" in out
