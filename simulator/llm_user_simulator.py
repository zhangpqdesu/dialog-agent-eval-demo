#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from llm.kimi_client import KimiClient


class LLMUserSimulator:
    def __init__(self, client: KimiClient | None = None, model: str | None = None) -> None:
        self.client = client or KimiClient(model=model)

    def start(self, scenario: dict[str, Any]) -> str:
        return scenario["initial_user_utterance"]

    def reply(
        self,
        scenario: dict[str, Any],
        example: dict[str, Any],
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        system_prompt = (
            "你在扮演外呼场景中的真实用户。"
            "请严格按照给定 persona、goal、style 和 must_cover 来回应。"
            "只生成下一轮用户回复，并判断是否结束通话。"
            "输出必须是 JSON。"
        )
        user_prompt = {
            "scenario": scenario,
            "instruction_task": example["instruction_core"]["task"],
            "instruction_constraints": example["instruction_core"]["constraints"],
            "conversation_history": history,
            "output_schema": {
                "user_reply": "string",
                "finished": "boolean",
                "reason": "string",
            },
        }
        result = self.client.complete_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_prompt)},
            ],
            temperature=0.5,
            max_tokens=512,
        )
        result.setdefault("user_reply", "")
        result.setdefault("finished", False)
        result.setdefault("reason", "")
        return result
