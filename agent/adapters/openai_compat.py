"""Adapter for any OpenAI-compatible chat completion endpoint.

Re-uses the ``openai`` client. The full structured instruction is rendered
as a system message; conversation history is converted to chat turns.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from agent.adapters.base import AgentAdapter
from config.eval_config import AgentCfg


_SYSTEM_TEMPLATE = """你是被测外呼任务执行 Agent。你必须严格遵守给定的任务指令、流程、知识点和约束。\
你的输出必须是用户可直接听到的一句话，不要解释。

任务角色：{role}
任务目标：{task}
开场白：{opening_line}
必传信息：
{required_information}
约束：
{constraints}
知识点：
{knowledge_points}
"""


def _fmt_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "（无）"


class OpenAICompatAdapter(AgentAdapter):
    def __init__(
        self,
        cfg: AgentCfg,
        client: OpenAI | None = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> None:
        super().__init__(cfg)
        api_key = os.environ.get(cfg.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"openai_compat adapter '{cfg.name}' needs api_key_env "
                f"(env var '{cfg.api_key_env or 'OPENAI_API_KEY'}' is empty)"
            )
        self.client = client or OpenAI(api_key=api_key, base_url=cfg.base_url)
        self.model = cfg.model or "gpt-4o-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def respond(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        core = example["instruction_core"]
        system_msg = _SYSTEM_TEMPLATE.format(
            role=core.get("role", ""),
            task=core.get("task", ""),
            opening_line=core.get("opening_line", ""),
            required_information=_fmt_bullets(core.get("required_information", [])),
            constraints=_fmt_bullets(core.get("constraints", {}).get("items", [])),
            knowledge_points=_fmt_bullets(core.get("knowledge_points", [])),
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        for turn in history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["text"]})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
