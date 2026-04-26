#!/usr/bin/env python3
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PERSONAS_FILE = ROOT / "data" / "personas.json"


class PersonaStore:
    def __init__(self) -> None:
        self._path = PERSONAS_FILE
        self._data: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = payload.get("personas", [])
        else:
            self._data = []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps({"personas": self._data}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def list(self) -> list[dict[str, Any]]:
        return list(self._data)

    def get(self, persona_id: str) -> dict[str, Any] | None:
        return next((p for p in self._data if p["id"] == persona_id), None)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        persona = {
            "id": payload.get("id") or str(uuid.uuid4())[:8],
            "name": payload["name"],
            "instruction_id": payload.get("instruction_id", "1"),
            "goal": payload.get("goal", ""),
            "style": payload.get("style", []),
            "initial_utterance": payload.get("initial_utterance", "你好，请说。"),
            "max_turns": int(payload.get("max_turns", 6)),
            "profile_id": payload.get("profile_id", "custom"),
        }
        self._data.append(persona)
        self._save()
        return persona

    def update(self, persona_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        for i, p in enumerate(self._data):
            if p["id"] == persona_id:
                self._data[i] = {**p, **payload, "id": persona_id}
                self._save()
                return self._data[i]
        return None

    def delete(self, persona_id: str) -> bool:
        before = len(self._data)
        self._data = [p for p in self._data if p["id"] != persona_id]
        if len(self._data) < before:
            self._save()
            return True
        return False
