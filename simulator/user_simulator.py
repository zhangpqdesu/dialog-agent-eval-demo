#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KEYWORD_LIBRARY = {
    "rider_operations": {
        "task": ["合同", "配送", "飞毛腿", "上线"],
        "faq": ["退出", "奖励", "多少单", "排名", "派单"],
        "safety": ["安全", "注意安全"],
        "close": ["再见", "挂了", "拜拜"],
    },
    "merchant_education_saas": {
        "task": ["低延迟", "标准直播", "发布页", "发课"],
        "faq": ["价格", "费用", "怎么开通", "控制台", "企业微信"],
        "close": ["再见", "挂了", "拜拜"],
    },
}


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


@dataclass
class ScenarioState:
    state_id: str
    intent: str
    default_response: str
    expected_agent_signals: list[str]
    fallback_response: str
    transition_to: str | None
    terminal: bool = False


class ScenarioRuntime:
    def __init__(self, scenario: dict[str, Any], seed: int | None = None) -> None:
        self.scenario = scenario
        self.random = random.Random(seed)
        self.states = {
            state["state_id"]: ScenarioState(**state)
            for state in scenario["states"]
        }
        self.current_state_id = scenario["initial_state_id"]
        self.turn_index = 0
        self.finished = False

    def start(self) -> str:
        opening = self.scenario["initial_user_utterance"]
        return opening

    def reply(self, agent_message: str) -> dict[str, Any]:
        if self.finished:
            return {
                "state_id": self.current_state_id,
                "user_reply": "",
                "finished": True,
                "reason": "scenario_already_finished",
            }

        self.turn_index += 1
        state = self.states[self.current_state_id]
        matched = any(signal in agent_message for signal in state.expected_agent_signals)
        reply = state.default_response if matched else state.fallback_response

        if state.terminal:
            self.finished = True
        elif state.transition_to:
            self.current_state_id = state.transition_to

        if self.turn_index >= self.scenario["max_turns"]:
            self.finished = True

        return {
            "state_id": state.state_id,
            "user_reply": reply,
            "finished": self.finished,
            "matched_expected_signal": matched,
        }


class UserSimulator:
    def __init__(self, scenarios: list[dict[str, Any]], seed: int | None = None) -> None:
        self.scenarios = {scenario["scenario_id"]: scenario for scenario in scenarios}
        self.seed = seed

    @classmethod
    def from_file(cls, path: str | Path, seed: int | None = None) -> "UserSimulator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["scenarios"], seed=seed)

    def create_session(self, scenario_id: str) -> ScenarioRuntime:
        scenario = self.scenarios[scenario_id]
        return ScenarioRuntime(scenario, seed=self.seed)


def demo() -> None:
    root = Path(__file__).resolve().parents[1]
    scenario_path = root / "data" / "processed" / "user_simulator_scenarios.json"
    simulator = UserSimulator.from_file(scenario_path)

    preferred = [
        "1_cooperative_user",
        "1_refusing_user",
        "2_driving_user",
    ]
    scenario_id = next((item for item in preferred if item in simulator.scenarios), None)
    if scenario_id is None:
        scenario_id = next(iter(simulator.scenarios))

    session = simulator.create_session(scenario_id)
    print(f"scenario_id={scenario_id}")
    print(f"user: {session.start()}")

    canned_agent_messages = [
        "你好，我是站长，今天飞毛腿合同已经生效了，你今天能开始配送吗？",
        "单日合同和多日合同都有要求，连续完成配送会更稳一些。",
        "如果有问题我也可以继续帮你解释，注意安全。",
        "那先这样，回头有问题再联系。",
    ]

    for message in canned_agent_messages:
        result = session.reply(message)
        print(f"agent: {message}")
        print(f"user: {result['user_reply']}")
        if result["finished"]:
            break


if __name__ == "__main__":
    demo()
