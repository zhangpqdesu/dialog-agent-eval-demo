#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class KimiClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        self.base_url = (base_url or os.environ.get("KIMI_BASE_URL") or "https://api.moonshot.cn/v1").rstrip("/")
        self.model = model or os.environ.get("KIMI_MODEL") or "kimi-k2.6"
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Missing KIMI_API_KEY or MOONSHOT_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 1 if self.model == "kimi-k2.6" else temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.model_dump()

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1,
        max_tokens: int = 1024,
    ) -> str:
        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        message = response["choices"][0]["message"]
        text = message.get("content") or ""
        if text.strip():
            return text

        retry_messages = list(messages) + [
            {
                "role": "user",
                "content": "请不要输出思考过程，只输出最终回答内容。",
            }
        ]
        retry_response = self.chat(
            retry_messages,
            temperature=temperature,
            max_tokens=max(max_tokens, 1024),
        )
        retry_text = retry_response["choices"][0]["message"].get("content") or ""
        if retry_text.strip():
            return retry_text

        reasoning_text = message.get("reasoning_content") or ""
        raise ValueError(
            f"Could not extract text from model output. content={text!r} reasoning={reasoning_text!r}"
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        response = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        message = response["choices"][0]["message"]
        text = message.get("content") or ""
        if text.strip():
            return extract_json_object(text)

        retry_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "请不要输出思考过程，不要解释。"
                    "现在只输出一个合法 JSON 对象，且必须可被 json.loads 直接解析。"
                ),
            }
        ]
        retry_response = self.chat(
            retry_messages,
            temperature=temperature,
            max_tokens=max(max_tokens, 2048),
        )
        retry_text = retry_response["choices"][0]["message"].get("content") or ""
        if retry_text.strip():
            return extract_json_object(retry_text)

        reasoning_text = message.get("reasoning_content") or ""
        raise ValueError(
            f"Could not extract JSON from model output. content={text!r} reasoning={reasoning_text!r}"
        )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError(f"Could not extract JSON from model output: {text}")
