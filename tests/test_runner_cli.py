"""End-to-end smoke test for the runner CLI using the offline log adapter.

This exercises: config loading -> persona scenarios -> offline adapter ->
existing rule scorer -> JSON output. No LLM keys needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.eval_config import load_eval_config
from runner.cli import run


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def offline_setup(tmp_path):
    # Reuse existing seed instructions and personas (instruction_id "1")
    log = tmp_path / "log.json"
    log.write_text(
        json.dumps({
            "conversations": [
                {
                    "instruction_id": "1",
                    "scenario_id": "1_cooperative_user",
                    "turns": [
                        {"role": "user", "text": "是我，你说。"},
                        {"role": "agent", "text": "你好，飞毛腿合同今天生效了，能开始配送吗？"},
                        {"role": "user", "text": "行。"},
                        {"role": "agent", "text": "好的，注意安全，先这样。"},
                    ],
                }
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""run_name: smoke_test
instructions:
  source: {ROOT / 'data' / 'processed' / 'dialog_instruction_eval_examples.json'}
  filter_ids: ["1"]
personas:
  source: {ROOT / 'data' / 'personas.json'}
  filter_profile_ids: [cooperative_user]
agents_under_test:
  - name: replayed
    type: offline_log
    log_path: {log}
scoring:
  l3_samples: 1
output:
  dir: {tmp_path / 'out'}
  formats: [json]
""",
        encoding="utf-8",
    )
    return cfg_path, tmp_path / "out"


def test_runner_end_to_end_offline(offline_setup):
    cfg_path, out_dir = offline_setup
    cfg = load_eval_config(cfg_path)
    summary = run(cfg)
    assert summary["run_name"] == "smoke_test"
    # Reproducibility metadata is now persisted with every run.
    assert summary["seed"] == cfg.seed
    assert summary["judge_provider"] in {"kimi", "deepseek"}
    # No LLM credentials in CI ⇒ judge unavailable ⇒ L3 must be flagged as skipped.
    assert summary["judge_available"] is False
    assert len(summary["results"]) == 1
    r = summary["results"][0]
    assert r["agent_name"] == "replayed"
    assert r["scenario_id"] == "1_cooperative_user"
    assert isinstance(r["overall_score"], (int, float))
    # When the judge is missing, every result must self-flag for review with
    # a capped confidence — otherwise downstream dashboards would treat
    # "no semantic check" as "passed with high confidence".
    assert r["needs_human_review"] is True
    assert r["l3_skipped"] is True
    assert r["confidence"] <= 0.5

    files = sorted(p.name for p in out_dir.iterdir())
    assert "summary.json" in files
    assert "replayed__1_cooperative_user.json" in files

    detail = json.loads((out_dir / "replayed__1_cooperative_user.json").read_text(encoding="utf-8"))
    assert detail["agent_name"] == "replayed"
    assert detail["conversation"][0]["role"] == "user"
    # Both report shapes must be present: legacy `rule_report` (consumed by
    # api_server.py and batch runners) and the new `layered_report`.
    assert "rule_report" in detail["report"]
    layered = detail["report"]["layered_report"]
    assert layered is not None
    assert "overall_score" in layered
    assert "findings" in layered
    assert layered["meta"]["l3_skipped"] is True
    assert layered["needs_human_review"] is True
    # L3_mean should be `null` (not 0.0) when L3 was skipped, so callers
    # can distinguish "judged poorly" from "never judged".
    assert layered["layer_scores"]["L3_mean"] is None
