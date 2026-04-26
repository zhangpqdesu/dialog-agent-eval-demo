#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.api_agent import APIAgent
from evaluator.auto_scorer import AutoScorer
from evaluator.llm_judge import LLMJudge
from llm.deepseek_client import load_dotenv
from simulator.user_simulator import UserSimulator
from simulator.llm_user_simulator import LLMUserSimulator


class RuleBasedAgent:
    def __init__(self, eval_examples: dict[str, Any]) -> None:
        self.examples = {
            example["instruction_id"]: example
            for example in eval_examples["examples"]
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "RuleBasedAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload)

    def respond(
        self,
        instruction_id: str,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
        step_index: int,
    ) -> str:
        example = self.examples[instruction_id]
        constraints = example["instruction_core"]["constraints"]
        steps = example["instruction_core"]["call_flow"]["steps"]
        user_text = history[-1]["text"] if history else ""
        profile_id = scenario["profile_id"]

        if profile_id == "driving_user":
            return "那我稍后再打，不打扰您了。"

        if profile_id == "off_topic_user" and "别的" in user_text:
            fallback = constraints.get("out_of_scope_reply")
            if fallback:
                return fallback

        if profile_id == "busy_user" and ("忙" in user_text or step_index == 0):
            return "就1分钟，我简单说重点。"

        if profile_id == "refusing_user" and ("不想" in user_text or "做不到" in user_text):
            if "注意安全" in user_text:
                return "行，理解你，注意安全。"
            return "我理解，但名额和合同会受影响。"

        if "价格" in user_text or "费用" in user_text:
            for point in example["instruction_core"]["knowledge_points"]:
                if "价格" in point or "费用" in point:
                    return self._truncate(point, constraints)

        if "退出" in user_text:
            for point in example["instruction_core"]["knowledge_points"]:
                if "退出" in point:
                    return self._truncate(point, constraints)

        if step_index < len(steps):
            return self._truncate(steps[step_index]["instruction"], constraints)

        if example["instruction_core"]["knowledge_points"]:
            kp_index = min(step_index - len(steps), len(example["instruction_core"]["knowledge_points"]) - 1)
            return self._truncate(example["instruction_core"]["knowledge_points"][kp_index], constraints)

        return self._truncate("先这样，有问题再联系。", constraints)

    def _truncate(self, text: str, constraints: dict[str, Any]) -> str:
        limit = constraints.get("max_chars_per_turn")
        if not limit:
            return text
        compact = text.replace("**", "").replace("\n", " ")
        if len(compact) <= limit:
            return compact
        return compact[: max(limit - 1, 1)] + "。"


def load_eval_examples() -> dict[str, Any]:
    eval_data_path = ROOT / "data" / "processed" / "dialog_instruction_eval_examples.json"
    return json.loads(eval_data_path.read_text(encoding="utf-8"))


def build_example_index(eval_examples: dict[str, Any]) -> dict[str, Any]:
    return {
        example["instruction_id"]: example
        for example in eval_examples["examples"]
    }


def run_one_evaluation(
    scenario_id: str,
    output_path: str | Path | None = None,
    agent_mode: str = "api",
    user_mode: str = "api",
    judge_mode: str = "rule",
    agent_model: str | None = None,
    user_model: str | None = None,
) -> dict[str, Any]:
    scenario_path = ROOT / "data" / "processed" / "user_simulator_scenarios.json"

    load_dotenv(ROOT / ".env")
    eval_examples = load_eval_examples()
    example_index = build_example_index(eval_examples)
    resolved_agent_model = agent_model or os.environ.get("DEEPSEEK_AGENT_MODEL") or "deepseek-chat"
    resolved_user_model = user_model or os.environ.get("DEEPSEEK_USER_MODEL") or "deepseek-chat"
    simulator = UserSimulator.from_file(scenario_path)
    scorer = AutoScorer(eval_examples)
    rule_agent = RuleBasedAgent(eval_examples)
    api_agent = APIAgent(model=resolved_agent_model) if agent_mode == "api" else None
    llm_user = (
        LLMUserSimulator(model=resolved_user_model)
        if user_mode == "api"
        else None
    )
    llm_judge = LLMJudge() if judge_mode in {"api", "hybrid"} else None

    scenario = simulator.scenarios[scenario_id]
    instruction_id = scenario["instruction_id"]
    example = example_index[instruction_id]
    session = simulator.create_session(scenario_id) if user_mode == "rule" else None

    initial_user = session.start() if session else llm_user.start(scenario)
    conversation = [{"role": "user", "text": initial_user}]
    step_index = 0
    finished = False

    while not finished:
        if agent_mode == "api":
            agent_reply = api_agent.respond(example, scenario, conversation)
        else:
            agent_reply = rule_agent.respond(instruction_id, scenario, conversation, step_index)
        conversation.append({"role": "agent", "text": agent_reply})
        if user_mode == "api":
            result = llm_user.reply(scenario, example, conversation)
        else:
            result = session.reply(agent_reply)
        if result.get("user_reply"):
            conversation.append({"role": "user", "text": result["user_reply"]})
        step_index += 1
        finished = bool(result.get("finished")) or step_index >= scenario["max_turns"]
        if finished:
            break

    rule_report = scorer.score_conversation(instruction_id, scenario, conversation)
    llm_report = None
    if llm_judge is not None:
        llm_report = llm_judge.judge(example, scenario, conversation, rule_report)

    output = {
        "scenario": {
            "scenario_id": scenario["scenario_id"],
            "instruction_id": instruction_id,
            "profile_id": scenario["profile_id"],
            "persona": scenario["persona"],
        },
        "modes": {
            "agent_mode": agent_mode,
            "user_mode": user_mode,
            "judge_mode": judge_mode,
            "agent_model": resolved_agent_model,
            "user_model": resolved_user_model,
        },
        "conversation": conversation,
        "report": {
            "rule_report": rule_report,
            "llm_report": llm_report,
        },
    }

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return output


def demo() -> None:
    output_path = ROOT / "outputs" / "demo_evaluation_report.json"
    result = run_one_evaluation(
        "1_cooperative_user",
        output_path=output_path,
        agent_mode="api",
        user_mode="api",
        judge_mode="rule",
        agent_model="deepseek-chat",
        user_model="deepseek-chat",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    demo()
