"""LLM-based instruction extractor.

Replaces the 1000-line regex/heuristic extractor with a single LLM call
that takes raw instruction text and produces the structured
``instruction_core`` block defined by ``dialog_instruction_eval.schema.json``.

A jsonschema validator gates the output; on validation failure the error
message is fed back to the LLM for up to ``max_repair_rounds`` retries.

Usage::

    from scripts.extract_instruction_schema_llm import extract_instruction
    record = extract_instruction(raw_markdown, instruction_id="3")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data" / "schemas" / "dialog_instruction_eval.schema.json"


_PROMPT_TEMPLATE = """你是一个对话任务指令的结构化抽取器。你的工作是把下面的"原始任务指令文本"\
抽取为符合给定 JSON Schema 的单条 `eval_instruction` 对象。

要求：
- 严格遵守 JSON Schema 的字段、类型、required 列表。
- `instruction_core` 必须完整：role / task / opening_line / required_information / call_flow / knowledge_points / constraints。
- `constraints` 中：尽量从原文识别出 `max_chars_per_turn`、`forbidden_expressions`、`out_of_scope_reply`、`has_end_call_condition`。识别不到的字段，按 Schema 给出合理默认值（数字给 null、列表给空数组、布尔给 false）。
- `success_criteria` 至少要覆盖：`goal_delivery`、`opening_line_used`、每个 `call_flow.steps[i]` 对应一个 `flow_step_{{i+1}}`、`faq_consistency`、`turn_length_limit`。
- `failure_conditions` 至少要覆盖：`miss_primary_task`、`miss_required_flow`、`exceed_turn_length`、`use_forbidden_expression`、`bad_out_of_scope_handling`、`continue_when_user_driving`、`miss_end_call_condition`。
- `user_simulator_profiles` 至少 4 个：cooperative_user、busy_user、skeptical_user、off_topic_user，并按指令领域加 1-2 个特化画像。
- 只输出一个 JSON 对象，不要任何解释或 markdown 代码围栏。

输入参数：
  instruction_id = {instruction_id}
  domain_hint    = {domain_hint}

JSON Schema（节选——你必须严格遵守）：
{schema_excerpt}

原始任务指令文本：
\"\"\"
{raw_text}
\"\"\"
"""


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_excerpt(schema: dict[str, Any]) -> str:
    """Strip the wrapper and only show the `eval_instruction` definition."""
    return json.dumps(schema["$defs"]["eval_instruction"], ensure_ascii=False, indent=2)


def _validate(record: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid).

    Validates the record against the eval_instruction subschema while keeping
    the full schema document available so $ref pointers resolve.
    """
    schema = _load_schema()
    subschema = dict(schema["$defs"]["eval_instruction"])
    subschema["$defs"] = schema["$defs"]
    validator = jsonschema.Draft202012Validator(subschema)
    return [
        f"{'.'.join(str(p) for p in err.absolute_path)}: {err.message}"
        for err in validator.iter_errors(record)
    ]


def extract_instruction(
    raw_text: str,
    instruction_id: str,
    domain_hint: str = "unknown",
    llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
    max_repair_rounds: int = 2,
) -> dict[str, Any]:
    """Extract one structured eval_instruction record from raw text.

    ``llm_complete_json`` is a callable that takes OpenAI-style messages and
    returns a JSON object. Defaulting to ``None`` lets callers inject any
    client (Kimi / DeepSeek / mock). When ``None``, uses ``KimiClient``.

    Returns the validated record. Raises ``ValueError`` if the model still
    produces invalid JSON after ``max_repair_rounds`` repair attempts.
    """
    if llm_complete_json is None:
        from llm.kimi_client import KimiClient
        client = KimiClient()
        def _call(messages: list[dict[str, str]]) -> dict[str, Any]:
            return client.complete_json(messages, temperature=0.2, max_tokens=4096)
        llm_complete_json = _call

    schema = _load_schema()
    excerpt = _schema_excerpt(schema)
    prompt = _PROMPT_TEMPLATE.format(
        instruction_id=instruction_id,
        domain_hint=domain_hint,
        schema_excerpt=excerpt,
        raw_text=raw_text,
    )
    messages = [
        {"role": "system", "content": "你只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]
    record = llm_complete_json(messages)

    for round_idx in range(max_repair_rounds):
        errors = _validate(record)
        if not errors:
            return record
        repair_msg = (
            "你上一次的 JSON 未通过 JSON Schema 校验。错误：\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n请输出修复后的完整 JSON 对象。"
        )
        messages = messages + [
            {"role": "assistant", "content": json.dumps(record, ensure_ascii=False)},
            {"role": "user", "content": repair_msg},
        ]
        record = llm_complete_json(messages)

    errors = _validate(record)
    if errors:
        raise ValueError(
            f"Failed to produce schema-valid record after "
            f"{max_repair_rounds} repair rounds. Last errors: {errors}"
        )
    return record


def extract_from_file(
    raw_path: str | Path,
    instruction_id: str,
    domain_hint: str = "unknown",
    **kwargs: Any,
) -> dict[str, Any]:
    raw_text = Path(raw_path).read_text(encoding="utf-8")
    record = extract_instruction(raw_text, instruction_id, domain_hint, **kwargs)
    record.setdefault("instruction_id", instruction_id)
    record.setdefault("raw_markdown", raw_text)
    return record


if __name__ == "__main__":  # pragma: no cover
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    iid = sys.argv[2] if len(sys.argv) > 2 else "X"
    print(json.dumps(extract_from_file(raw, iid), ensure_ascii=False, indent=2))
