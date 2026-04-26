#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from llm.deepseek_client import DeepSeekClient


class LLMUserSimulator:
    def __init__(self, client: DeepSeekClient | None = None, model: str | None = None) -> None:
        self.client = client or DeepSeekClient(model=model)

    def start(self, scenario: dict[str, Any]) -> str:
        # 兼容旧字段名 initial_user_utterance 和新字段名 initial_utterance
        return scenario.get("initial_utterance") or scenario.get("initial_user_utterance", "你好，请说。")

    def reply(
        self,
        scenario: dict[str, Any],
        example: dict[str, Any],
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        goal = scenario.get("goal", "")
        style = scenario.get("style", [])
        name = scenario.get("name") or scenario.get("persona", "用户")

        style_str = "、".join(style) if style else "自然对话"

        system_prompt = (
            f"你在扮演外呼场景中的真实用户，角色名：{name}。\n"
            f"你的目标：{goal}\n"
            f"你的风格：{style_str}\n"
            "请严格按照角色目标和风格回应，只生成下一轮用户回复，并判断是否结束通话。"
            "输出必须是 JSON。"
        )
        user_prompt = {
            "task": example["instruction_core"]["task"],
            "constraints": example["instruction_core"]["constraints"],
            "conversation_history": history,
            "output_schema": {
                "user_reply": "string",
                "finished": "boolean — 通话是否结束",
                "reason": "string — 结束原因（未结束则留空）",
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
