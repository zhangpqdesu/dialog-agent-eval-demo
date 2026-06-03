from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.adapters import AgentAdapter, make_adapter
from agent.adapters.base import AgentAdapter as _Base
from agent.adapters.builtin_llm import BuiltinLLMAdapter
from config.eval_config import AgentCfg


def test_base_is_abstract():
    with pytest.raises(TypeError):
        _Base(AgentCfg(name="x", type="builtin_llm"))  # type: ignore[abstract]


def test_make_adapter_dispatches_builtin_llm():
    cfg = AgentCfg(name="x", type="builtin_llm", model="kimi-k2-turbo-preview")
    inner = MagicMock()
    adapter = make_adapter(cfg, inner=inner)
    assert isinstance(adapter, BuiltinLLMAdapter)
    assert adapter.name == "x"


def test_builtin_adapter_delegates_to_inner():
    inner = MagicMock()
    inner.respond.return_value = "你好"
    cfg = AgentCfg(name="x", type="builtin_llm", model="m")
    adapter = BuiltinLLMAdapter(cfg, inner=inner)
    example = {"instruction_core": {"task": "t"}, "instruction_id": "1"}
    scenario = {"scenario_id": "s", "profile_id": "p"}
    out = adapter.respond(example, scenario, [{"role": "user", "text": "hi"}])
    assert out == "你好"
    inner.respond.assert_called_once_with(example, scenario, [{"role": "user", "text": "hi"}])


def test_http_adapter_posts_and_returns_reply(monkeypatch):
    from agent.adapters.http_agent import HTTPAdapter
    cfg = AgentCfg(name="prod", type="http", endpoint="http://x/respond")
    adapter = HTTPAdapter(cfg)

    captured = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"reply": "ok"}
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResp()

    monkeypatch.setattr("agent.adapters.http_agent.requests.post", fake_post)
    out = adapter.respond(
        {"instruction_core": {"task": "t"}, "instruction_id": "1"},
        {"scenario_id": "s", "profile_id": "p"},
        [{"role": "user", "text": "hi"}],
    )
    assert out == "ok"
    assert captured["url"] == "http://x/respond"
    assert captured["json"]["history"] == [{"role": "user", "text": "hi"}]
    assert captured["json"]["scenario"] == {"scenario_id": "s", "profile_id": "p"}


def test_http_adapter_raises_on_non_200(monkeypatch):
    import requests
    from agent.adapters.http_agent import HTTPAdapter
    cfg = AgentCfg(name="prod", type="http", endpoint="http://x/respond")
    adapter = HTTPAdapter(cfg)

    class FakeResp:
        status_code = 500
        def raise_for_status(self):
            raise requests.HTTPError("500")

    monkeypatch.setattr(
        "agent.adapters.http_agent.requests.post",
        lambda url, json, timeout: FakeResp(),
    )
    with pytest.raises(requests.HTTPError):
        adapter.respond(
            {"instruction_core": {}, "instruction_id": "1"},
            {"scenario_id": "s", "profile_id": "p"},
            [],
        )


def test_http_adapter_requires_endpoint():
    cfg = AgentCfg(name="x", type="http")
    with pytest.raises(ValueError, match="endpoint"):
        from agent.adapters.http_agent import HTTPAdapter
        HTTPAdapter(cfg)


def test_offline_log_adapter_replays_in_order(tmp_path):
    from agent.adapters.offline_log import OfflineLogAdapter
    log = tmp_path / "log.json"
    log.write_text(
        '{"conversations":[{"instruction_id":"1","scenario_id":"sc1","turns":['
        '{"role":"user","text":"u1"},'
        '{"role":"agent","text":"a1"},'
        '{"role":"user","text":"u2"},'
        '{"role":"agent","text":"a2"}'
        ']}]}',
        encoding="utf-8",
    )
    cfg = AgentCfg(name="rec", type="offline_log", log_path=str(log))
    adapter = OfflineLogAdapter(cfg)
    example = {"instruction_id": "1", "instruction_core": {}}
    scenario = {"scenario_id": "sc1", "profile_id": "p"}
    assert adapter.respond(example, scenario, []) == "a1"
    assert adapter.respond(example, scenario, []) == "a2"
    assert adapter.respond(example, scenario, []) == ""  # exhausted
    assert adapter.user_turns("1", "sc1") == ["u1", "u2"]


def test_offline_log_adapter_unknown_pair_raises(tmp_path):
    from agent.adapters.offline_log import OfflineLogAdapter
    log = tmp_path / "log.json"
    log.write_text('{"conversations":[]}', encoding="utf-8")
    cfg = AgentCfg(name="rec", type="offline_log", log_path=str(log))
    adapter = OfflineLogAdapter(cfg)
    with pytest.raises(KeyError):
        adapter.respond({"instruction_id": "9", "instruction_core": {}},
                        {"scenario_id": "missing", "profile_id": "p"}, [])


def test_unknown_adapter_type_raises():
    # Bypass Pydantic validation by constructing AgentCfg with valid type then mutating
    cfg = AgentCfg(name="x", type="builtin_llm")
    cfg.type = "nonsense"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown adapter type"):
        make_adapter(cfg)
