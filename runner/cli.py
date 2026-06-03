"""Config-driven evaluation runner.

Entry point::

    python -m runner.cli --config config/eval_config.example.yaml
    python -m runner.cli --config <path> --scenario 1_cooperative_user

Drives the W1 pipeline end-to-end:
  config -> instructions + personas -> for each (agent, instruction, persona):
    spin up adapter, simulate conversation, score with the existing
    AutoScorer (layered scoring is W2), write JSON report.
"""
from __future__ import annotations

import argparse
import json
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
from evaluator.auto_scorer import AutoScorer
from llm.deepseek_client import load_dotenv
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
    instructions = _load_instructions(cfg)
    example_index = {ex["instruction_id"]: ex for ex in instructions["examples"]}
    personas = _load_personas(cfg)
    scenarios = _load_scenarios()
    state_scenarios = {s["scenario_id"]: s for s in scenarios.get("scenarios", [])}
    scorer = AutoScorer(instructions)

    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": cfg.run_name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }

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

            report = scorer.score_conversation(instruction_id, scenario, conversation)
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
                "report": {"rule_report": report, "layered_report": None},
            }
            fname = f"{agent_cfg.name}__{scenario['scenario_id']}.json"
            (out_dir / fname).write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary["results"].append({
                "agent_name": agent_cfg.name,
                "scenario_id": scenario["scenario_id"],
                "instruction_id": instruction_id,
                "overall_score": report["overall_score"],
                "file": fname,
            })

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
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
