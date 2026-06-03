"""Integration tests for the layered ``/runs`` API surface.

We submit an inline YAML config that points at an offline conversation
log so no LLM credentials are required. The test exercises the full
queue → run → aggregate flow that the dashboard polls.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api_server
from api_server import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_layered_runs():
    # Each test gets a clean slate so list/get assertions are deterministic.
    api_server._layered_runs.clear()
    yield
    api_server._layered_runs.clear()


@pytest.fixture()
def inline_yaml(tmp_path):
    log = tmp_path / "log.json"
    log.write_text(
        json.dumps(
            {
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
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return f"""run_name: api_smoke
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
"""


def _wait_for_terminal(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    """Poll GET /runs/{id} until state leaves the queued/running bucket."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["state"] in {"succeeded", "failed"}:
            return last
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish in time; last state={last}")


def test_post_runs_rejects_invalid_yaml():
    client = TestClient(app)
    resp = client.post("/runs", json={"yaml_config": "not: [valid"})
    assert resp.status_code == 400
    assert "invalid yaml_config" in resp.json()["detail"]
    # The malformed submission must not leak into the listing.
    assert client.get("/runs").json()["runs"] == []


def test_post_runs_rejects_non_mapping_yaml():
    client = TestClient(app)
    resp = client.post("/runs", json={"yaml_config": "- just\n- a\n- list\n"})
    assert resp.status_code == 400


def test_get_run_404_for_unknown_id():
    client = TestClient(app)
    assert client.get("/runs/does-not-exist").status_code == 404
    assert client.get("/runs/does-not-exist/report").status_code == 404


def test_layered_run_end_to_end(inline_yaml):
    client = TestClient(app)
    resp = client.post("/runs", json={"yaml_config": inline_yaml})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]
    assert resp.json()["state"] == "queued"

    final = _wait_for_terminal(client, run_id)
    assert final["state"] == "succeeded", final
    assert final["run_name"].startswith("api_smoke__")
    # Offline => no judge => result rows must self-flag.
    assert final["judge_available"] is False
    assert final["result_count"] == 1
    assert final["summary"]["results"][0]["needs_human_review"] is True

    listing = client.get("/runs").json()["runs"]
    assert any(r["run_id"] == run_id for r in listing)

    report = client.get(f"/runs/{run_id}/report").json()
    agg = report["aggregate"]
    # Aggregator output contract (see evaluator/aggregator.py).
    for key in ("matrix", "radar", "failure_modes", "low_confidence", "totals"):
        assert key in agg
    assert agg["totals"]["scenarios"] == 1
    # The single offline scenario was L3-skipped, so it must land in the
    # low-confidence bucket for the dashboard to surface.
    assert agg["low_confidence"], agg


def test_report_endpoint_409_before_completion(inline_yaml):
    client = TestClient(app)
    # Pre-register a stuck run so we can check the 409 path without a race.
    api_server._layered_runs["frozen"] = {
        "state": "queued",
        "started_at": "2024-01-01T00:00:00",
        "finished_at": None,
        "error": None,
        "run_name": None,
        "output_dir": None,
        "summary": None,
        "aggregate": None,
    }
    resp = client.get("/runs/frozen/report")
    assert resp.status_code == 409
    assert resp.json()["detail"]["state"] == "queued"
