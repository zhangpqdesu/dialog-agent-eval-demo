"""Factory for constructing the right adapter from an ``AgentCfg``."""
from __future__ import annotations

from typing import Any

from agent.adapters.base import AgentAdapter
from config.eval_config import AgentCfg


def make_adapter(cfg: AgentCfg, **kwargs: Any) -> AgentAdapter:
    """Dispatch on ``cfg.type`` to instantiate the matching adapter."""
    if cfg.type == "builtin_llm":
        from agent.adapters.builtin_llm import BuiltinLLMAdapter
        return BuiltinLLMAdapter(cfg, **kwargs)
    if cfg.type == "http":
        from agent.adapters.http_agent import HTTPAdapter
        return HTTPAdapter(cfg, **kwargs)
    if cfg.type == "openai_compat":
        from agent.adapters.openai_compat import OpenAICompatAdapter
        return OpenAICompatAdapter(cfg, **kwargs)
    if cfg.type == "offline_log":
        from agent.adapters.offline_log import OfflineLogAdapter
        return OfflineLogAdapter(cfg, **kwargs)
    raise ValueError(f"Unknown adapter type: {cfg.type}")
