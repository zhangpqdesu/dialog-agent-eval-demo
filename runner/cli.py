"""Config-driven evaluation runner.

Entry point::

    python -m runner.cli --config config/eval_config.example.yaml
    python -m runner.cli --config <path> --scenario 1_cooperative_user

Drives the W1+W2 pipeline end-to-end:
  config -> instructions + personas -> for each (agent, instruction, persona):
    spin up adapter, simulate conversation, score with BOTH the legacy
    ``AutoScorer`` (kept for back-compat consumers — emitted as
    ``rule_report``) and the new layered scorer (emitted as
    ``layered_report``), write JSON report.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.adapters import AgentAdapter, make_adapter
from agent.adapters.offline_log import OfflineLogAdapter
from config import EvalConfig, load_eval_config
from evaluator.aggregator import aggregate
from evaluator.auto_scorer import AutoScorer
from evaluator.judge_factory import make_judge
from evaluator.scoring.layered_scorer import LayeredScorer
from llm.deepseek_client import load_dotenv
from report.static_html import render as render_html
from simulator.llm_user_simulator import LLMUserSimulator
from simulator.user_simulator import UserSimulator


def _load_instructions(cfg: EvalConfig) -> dict[str, Any]:
    payload = json.loads(Path(cfg.instructions.source).read_text(encoding="utf-8"))
    if cfg.instructions.filter_ids:
        wanted = set(cfg.instructions.filter_ids)
        payload["examples"] = [
            ex for ex in payload["examples"] if str(ex["instruction_id"]) in wanted
        ]
    return payload


def _load_personas(cfg: EvalConfig) -> list[dict[str, Any]]:
    payload = json.loads(Path(cfg.personas.source).read_text(encoding="utf-8"))
    personas = payload.get("personas", payload if isinstance(payload, list) else [])
    if cfg.personas.filter_profile_ids:
        wanted = set(cfg.personas.filter_profile_ids)
        personas = [p for p in personas if p.get("profile_id") in wanted]
    if not cfg.personas.include_red_team:
        personas = [p for p in personas if not p.get("profile_id", "").startswith("rt_")]
    return personas


def _load_scenarios() -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "user_simulator_scenarios.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scenarios": []}


def _persona_to_scenario(persona: dict[str, Any]) -> dict[str, Any]:
    """Promote a persona record into a scenario dict the runner can consume."""
    return {
        "scenario_id": persona.get("id") or f"{persona.get('instruction_id','x')}_{persona['profile_id']}",
        "id": persona.get("id"),
        "instruction_id": str(persona["instruction_id"]),
        "profile_id": persona["profile_id"],
        "persona": persona["name"],
        "name": persona["name"],
        "goal": persona.get("goal", ""),
        "style": persona.get("style", []),
        "initial_utterance": persona.get("initial_utterance", "你好，请说。"),
        "max_turns": persona.get("max_turns", 6),
    }


def run_conversation(
    adapter: AgentAdapter,
    example: dict[str, Any],
    scenario: dict[str, Any],
    *,
    user_mode: str = "llm",
    state_scenarios: dict[str, dict[str, Any]] | None = None,
    user_simulator_client: Any = None,
) -> list[dict[str, str]]:
    """Drive one conversation between adapter and a user simulator.

    user_mode:
      - "llm": use LLMUserSimulator (default, requires LLM keys)
      - "state": use the legacy rule-based UserSimulator (needs prebuilt scenario)
      - "offline": adapter must be OfflineLogAdapter; replay both sides
    """
    if user_mode == "offline":
        if not isinstance(adapter, OfflineLogAdapter):
            raise ValueError("user_mode='offline' requires an OfflineLogAdapter")
        user_turns = adapter.user_turns(str(example["instruction_id"]), scenario["scenario_id"])
        conversation: list[dict[str, str]] = []
        for i, u in enumerate(user_turns):
            conversation.append({"role": "user", "text": u})
            agent_reply = adapter.respond(example, scenario, conversation)
            if not agent_reply:
                break
            conversation.append({"role": "agent", "text": agent_reply})
        return conversation

    if user_mode == "state":
        if state_scenarios is None or scenario["scenario_id"] not in state_scenarios:
            raise ValueError(
                f"user_mode='state' needs a prebuilt scenario for {scenario['scenario_id']}"
            )
        sim = UserSimulator([state_scenarios[scenario["scenario_id"]]])
        session = sim.create_session(scenario["scenario_id"])
        conversation = [{"role": "user", "text": session.start()}]
        while True:
            agent_reply = adapter.respond(example, scenario, conversation)
            conversation.append({"role": "agent", "text": agent_reply})
            step = session.reply(agent_reply)
            if step.get("user_reply"):
                conversation.append({"role": "user", "text": step["user_reply"]})
            if step.get("finished"):
                break
        return conversation

    # default: LLM-driven simulator
    user = LLMUserSimulator(client=user_simulator_client)
    initial = user.start(scenario)
    conversation = [{"role": "user", "text": initial}]
    max_turns = scenario.get("max_turns", 6)
    for _ in range(max_turns):
        agent_reply = adapter.respond(example, scenario, conversation)
        conversation.append({"role": "agent", "text": agent_reply})
        step = user.reply(scenario, example, conversation)
        if step.get("user_reply"):
            conversation.append({"role": "user", "text": step["user_reply"]})
        if step.get("finished"):
            break
    return conversation


def _build_adapter(cfg_entry, **kw) -> AgentAdapter:
    return make_adapter(cfg_entry, **kw)


def run(cfg: EvalConfig, scenario_filter: str | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    # Reproducibility hook — recorded in summary.json so reruns can match seeds.
    random.seed(cfg.seed)
    instructions = _load_instructions(cfg)
    example_index = {ex["instruction_id"]: ex for ex in instructions["examples"]}
    personas = _load_personas(cfg)
    scenarios = _load_scenarios()
    state_scenarios = {s["scenario_id"]: s for s in scenarios.get("scenarios", [])}

    # Legacy regex/keyword scorer — still consumed by api_server.py and the
    # batch runners, so we keep emitting its output verbatim as `rule_report`.
    legacy_scorer = AutoScorer(instructions)
    # New layered scorer. Judge is shared by L2 (when it wants help) and L3.
    judge = make_judge(cfg.scoring, dotenv_path=ROOT / ".env")
    layered = LayeredScorer(cfg.scoring, l2_llm=judge, l3_llm=judge)

    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "judge_provider": cfg.scoring.judge_provider,
        "judge_available": judge is not None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }
    # Keep detail dicts in memory so we can hand them to the aggregator /
    # HTML renderer without re-reading the JSON we just wrote.
    details: list[dict[str, Any]] = []

    for agent_cfg in cfg.agents_under_test:
        adapter = _build_adapter(agent_cfg)
        for persona in personas:
            scenario = _persona_to_scenario(persona)
            if scenario_filter and scenario["scenario_id"] != scenario_filter:
                continue
            instruction_id = scenario["instruction_id"]
            if instruction_id not in example_index:
                continue
            example = example_index[instruction_id]

            user_mode = "offline" if agent_cfg.type == "offline_log" else "llm"
            try:
                conversation = run_conversation(
                    adapter,
                    example,
                    scenario,
                    user_mode=user_mode,
                    state_scenarios=state_scenarios,
                )
            except Exception as exc:  # noqa: BLE001 — top-level isolation per case
                print(f"[error] {agent_cfg.name}/{scenario['scenario_id']}: {exc}", file=sys.stderr)
                continue

            rule_report = legacy_scorer.score_conversation(instruction_id, scenario, conversation)
            layered_result = layered.score(example, scenario, conversation).to_dict()
            output = {
                "run_name": cfg.run_name,
                "agent_name": agent_cfg.name,
                "agent_type": agent_cfg.type,
                "agent_model": agent_cfg.model,
                "scenario": {
                    "scenario_id": scenario["scenario_id"],
                    "instruction_id": instruction_id,
                    "profile_id": scenario["profile_id"],
                    "persona": scenario["persona"],
                },
                "conversation": conversation,
                "report": {"rule_report": rule_report, "layered_report": layered_result},
            }
            details.append(output)
            fname = f"{agent_cfg.name}__{scenario['scenario_id']}.json"
            (out_dir / fname).write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary["results"].append({
                "agent_name": agent_cfg.name,
                "scenario_id": scenario["scenario_id"],
                "instruction_id": instruction_id,
                "profile_id": scenario["profile_id"],
                # `overall_score` is the headline number consumers expect;
                # layered's [0,100] score replaces the legacy rule_report total
                # for ranking/aggregation purposes.
                "overall_score": layered_result["overall_score"],
                "rule_overall_score": rule_report["overall_score"],
                "confidence": layered_result["confidence"],
                "needs_human_review": layered_result["needs_human_review"],
                "inconsistency_flags": layered_result["inconsistency_flags"],
                "l3_skipped": layered_result["meta"].get("l3_skipped", False),
                "file": fname,
            })

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Always write aggregate.json — it's tiny and the dashboard reads it
    # directly without needing to re-aggregate per-request.
    aggregate_dict = aggregate(details)
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate_dict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if "html" in (cfg.output.formats or []):
        # Single-file HTML — easy to ship as a CI artefact or e-mail.
        (out_dir / "report.html").write_text(
            render_html(summary, details, aggregate_dict),
            encoding="utf-8",
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-runner")
    parser.add_argument("--config", required=True, help="Path to eval_config.yaml")
    parser.add_argument("--scenario", default=None, help="Optional: run only this scenario_id")
    args = parser.parse_args(argv)

    cfg = load_eval_config(args.config)
    summary = run(cfg, scenario_filter=args.scenario)
    print(json.dumps(
        {
            "run_name": summary["run_name"],
            "result_count": len(summary["results"]),
            "output_dir": str(cfg.resolved_output_dir()),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
