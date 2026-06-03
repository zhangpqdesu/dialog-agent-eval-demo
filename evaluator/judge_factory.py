"""LLM judge factory for the layered scorer.

Returns a ``complete_json(messages) -> dict`` callable, or ``None`` if the
configured provider has no usable credentials. The callable always asks
for ``temperature=0`` so judging is as deterministic as the upstream API
allows (improves the "可复现" axis of the spec).

Why a factory at all?  The CLI / dashboard need a single switch to
choose Kimi vs DeepSeek per ``ScoringCfg.judge_provider`` without
hard-coding LLM clients into the scoring layer (the scorer stays
provider-agnostic, taking only a callable).

When the factory returns ``None``:
  * L2 FlowCoverage falls back to its keyword-only verifier.
  * L3 SemanticJudge is skipped entirely by LayeredScorer, and the
    final ScoringResult is marked ``needs_human_review=True`` with a
    capped confidence so callers don't mistake a missing judge for a
    high-confidence pass.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from config import ScoringCfg


JudgeCallable = Callable[[list[dict[str, str]]], dict[str, Any]]


def _kimi_callable(model: str) -> JudgeCallable | None:
    if not (os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")):
        return None
    from llm.kimi_client import KimiClient  # local import keeps cold-start light

    client = KimiClient(model=model)

    def _call(messages: list[dict[str, str]]) -> dict[str, Any]:
        # Temperature 0 (kimi-k2.6 will still force 1 server-side; that's
        # documented in the client and unavoidable). max_tokens 1024 is
        # plenty for the judge JSON payloads we use.
        return client.complete_json(messages, temperature=0, max_tokens=1024)

    return _call


def _deepseek_callable(model: str) -> JudgeCallable | None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    from llm.deepseek_client import DeepSeekClient

    client = DeepSeekClient(model=model)

    def _call(messages: list[dict[str, str]]) -> dict[str, Any]:
        return client.complete_json(messages, temperature=0, max_tokens=1024)

    return _call


def make_judge(cfg: ScoringCfg, *, dotenv_path: str | Path | None = None) -> JudgeCallable | None:
    """Build the judge callable for ``cfg.judge_provider``.

    Loads the project ``.env`` first so credentials picked up only via
    that file are visible (mirrors the runner CLI's behaviour).

    Returns ``None`` when the provider has no usable credentials. Never
    raises for the missing-credentials case — the caller decides whether
    to keep going with degraded scoring or hard-fail.
    """
    if dotenv_path is not None:
        from llm.deepseek_client import load_dotenv

        load_dotenv(dotenv_path)
    provider = cfg.judge_provider
    model = cfg.judge_model
    if provider == "kimi":
        return _kimi_callable(model)
    if provider == "deepseek":
        return _deepseek_callable(model)
    raise ValueError(f"Unknown judge_provider: {provider}")


__all__ = ["JudgeCallable", "make_judge"]
