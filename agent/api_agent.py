#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from llm.deepseek_client import DeepSeekClient


class APIAgent:
    def __init__(self, client: DeepSeekClient | None = None, model: str | None = None) -> None:
        self.client = client or DeepSeekClient(model=model)

    def respond(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        system_prompt = (
            "你是被测外呼任务执行 Agent。"
            "你必须严格遵守给定任务指令、流程、知识点和约束。"
            "你的输出必须是用户可直接听到的一句话，不要输出解释。"
        )
        user_prompt = {
            "instruction_core": example["instruction_core"],
            "scenario_meta": {
                "scenario_id": scenario.get("id") or scenario.get("scenario_id", ""),
                "profile_id": scenario.get("profile_id", "custom"),
                "persona": scenario.get("name") or scenario.get("persona", "用户"),
            },
            "conversation_history": history,
            "output_requirement": {
                "format": "plain_text",
                "need": "只输出下一轮 agent 回复",
            },
        }
        return self.client.complete_text(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_prompt)},
            ],
            temperature=0.2,
            max_tokens=512,
        ).strip()
