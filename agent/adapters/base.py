"""Abstract base for all agent-under-test adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from config.eval_config import AgentCfg


class AgentAdapter(ABC):
    """All adapters expose the same ``respond`` contract.

    Implementations may be stateful (e.g. an HTTP session) but ``respond``
    must be safe to call across many scenarios.
    """

    def __init__(self, cfg: AgentCfg) -> None:
        self.cfg = cfg
        self.name = cfg.name

    @abstractmethod
    def respond(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        """Return the agent's next utterance.

        ``example`` is the full structured instruction record.
        ``scenario`` is the persona/scenario the user simulator is using.
        ``history`` is the conversation so far; each turn is
        ``{"role": "user"|"agent", "text": str}``.
        """
