"""Agent adapters: a uniform interface for any agent under test.

Four implementations:

- ``BuiltinLLMAdapter`` — wraps the existing ``APIAgent`` (Kimi/DeepSeek).
- ``HTTPAdapter`` — POSTs JSON to an external endpoint.
- ``OpenAICompatAdapter`` — calls any OpenAI-compatible chat completion API.
- ``OfflineLogAdapter`` — replays pre-recorded agent turns from disk.
"""
from agent.adapters.base import AgentAdapter
from agent.adapters.factory import make_adapter

__all__ = ["AgentAdapter", "make_adapter"]
