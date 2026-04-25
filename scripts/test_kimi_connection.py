#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.kimi_client import KimiClient, load_dotenv


def main() -> None:
    load_dotenv(ROOT / ".env")
    client = KimiClient()
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，请做简短回答。",
            },
            {
                "role": "user",
                "content": "请回复：Kimi connection ok",
            },
        ],
        temperature=1,
        max_tokens=64,
    )
    content = response["choices"][0]["message"]["content"]
    print(
        json.dumps(
            {
                "model": client.model,
                "base_url": client.base_url,
                "content": content,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
