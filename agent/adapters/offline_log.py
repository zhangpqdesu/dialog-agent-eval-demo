"""Offline-log adapter — replays a pre-recorded transcript.

Useful for scoring real production conversations without re-running any
agent. The log file format::

    {
      "conversations": [
        {
          "instruction_id": "1",
          "scenario_id": "1_cooperative_user",
          "turns": [
            {"role": "user", "text": "..."},
            {"role": "agent", "text": "..."},
            ...
          ]
        },
        ...
      ]
    }

The adapter exposes ``respond`` like any other adapter but returns the
next pre-recorded agent turn in order. When ``UserSimulator`` is paired
with this adapter, set ``user_mode='offline'`` so the user side replays
its own turns from the same log (handled by the runner).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.adapters.base import AgentAdapter
from config.eval_config import AgentCfg


class OfflineLogAdapter(AgentAdapter):
    def __init__(self, cfg: AgentCfg) -> None:
        super().__init__(cfg)
        if not cfg.log_path:
            raise ValueError(f"offline_log adapter '{cfg.name}' requires log_path")
        self.log_path = Path(cfg.log_path)
        payload = json.loads(self.log_path.read_text(encoding="utf-8"))
        # Index by (instruction_id, scenario_id) -> queue of agent turns
        self._queues: dict[tuple[str, str], list[str]] = {}
        for conv in payload.get("conversations", []):
            key = (str(conv["instruction_id"]), str(conv["scenario_id"]))
            agent_turns = [t["text"] for t in conv["turns"] if t["role"] == "agent"]
            self._queues[key] = list(agent_turns)
        self._cursor: dict[tuple[str, str], int] = {k: 0 for k in self._queues}

    def has_pair(self, instruction_id: str, scenario_id: str) -> bool:
        return (instruction_id, scenario_id) in self._queues

    def user_turns(self, instruction_id: str, scenario_id: str) -> list[str]:
        """Companion accessor for the runner to drive a 'replay user'."""
        payload = json.loads(self.log_path.read_text(encoding="utf-8"))
        for conv in payload.get("conversations", []):
            if (str(conv["instruction_id"]) == instruction_id and
                    str(conv["scenario_id"]) == scenario_id):
                return [t["text"] for t in conv["turns"] if t["role"] == "user"]
        return []

    def respond(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        instruction_id = str(example["instruction_id"])
        scenario_id = str(scenario.get("scenario_id") or scenario.get("id", ""))
        key = (instruction_id, scenario_id)
        if key not in self._queues:
            raise KeyError(
                f"offline_log adapter '{self.name}' has no recording for "
                f"instruction_id={instruction_id} scenario_id={scenario_id}"
            )
        queue = self._queues[key]
        cur = self._cursor[key]
        if cur >= len(queue):
            return ""  # transcript exhausted; runner will terminate the loop
        self._cursor[key] = cur + 1
        return queue[cur]
