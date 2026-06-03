"""Adapter that wraps the existing ``APIAgent`` (Kimi/DeepSeek)."""
from __future__ import annotations

from typing import Any

from agent.adapters.base import AgentAdapter
from agent.api_agent import APIAgent
from config.eval_config import AgentCfg


class BuiltinLLMAdapter(AgentAdapter):
    """Forwards to ``agent.api_agent.APIAgent`` using the configured model."""

    def __init__(self, cfg: AgentCfg, inner: APIAgent | None = None) -> None:
        super().__init__(cfg)
        self.inner = inner or APIAgent(model=cfg.model)

    def respond(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        return self.inner.respond(example, scenario, history)
