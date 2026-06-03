"""HTTP adapter — POSTs the conversation context to an external endpoint.

Wire format (request)::

    {
      "instruction": <example.instruction_core>,
      "scenario": {"scenario_id": "...", "profile_id": "..."},
      "history": [{"role": "user"|"agent", "text": "..."}, ...]
    }

Wire format (response)::

    {"reply": "<next agent utterance>"}
"""
from __future__ import annotations

from typing import Any

import requests

from agent.adapters.base import AgentAdapter
from config.eval_config import AgentCfg


class HTTPAdapter(AgentAdapter):
    def __init__(self, cfg: AgentCfg, timeout: int = 30) -> None:
        super().__init__(cfg)
        if not cfg.endpoint:
            raise ValueError(f"http adapter '{cfg.name}' requires endpoint")
        self.endpoint = cfg.endpoint
        self.timeout = timeout

    def respond(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        payload = {
            "instruction": example["instruction_core"],
            "scenario": {
                "scenario_id": scenario.get("scenario_id") or scenario.get("id", ""),
                "profile_id": scenario.get("profile_id", "custom"),
            },
            "history": history,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("reply", "")).strip()
