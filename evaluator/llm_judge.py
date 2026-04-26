#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from llm.deepseek_client import DeepSeekClient


class LLMJudge:
    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()

    def judge(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        conversation: list[dict[str, str]],
        rule_report: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = (
            "你是复杂外呼任务评测专家。"
            "请基于任务指令、成功标准、失败条件和对话记录，输出结构化评测结论。"
            "不要复述全部对话，只输出 JSON。"
        )
        user_prompt = {
            "instruction_id": example["instruction_id"],
            "task": example["instruction_core"]["task"],
            "success_criteria": example["success_criteria"],
            "failure_conditions": example["failure_conditions"],
            "scenario": {
                "scenario_id": scenario.get("id") or scenario.get("scenario_id", ""),
                "profile_id": scenario.get("profile_id", "custom"),
                "persona": scenario.get("name") or scenario.get("persona", "用户"),
            },
            "conversation": conversation,
            "rule_report": rule_report,
            "output_schema": {
                "overall_score": "0-100 number",
                "summary": "string",
                "strengths": ["string"],
                "weaknesses": ["string"],
                "llm_findings": [
                    {
                        "criterion_or_condition": "string",
                        "judgement": "pass|fail|partial",
                        "reason": "string",
                    }
                ],
            },
        }
        return self.client.complete_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_prompt)},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
