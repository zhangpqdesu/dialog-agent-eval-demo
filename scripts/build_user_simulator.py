#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA_PATH = ROOT / "data" / "processed" / "dialog_instruction_eval_examples.json"
SCHEMA_PATH = ROOT / "data" / "schemas" / "user_simulator.schema.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "user_simulator_scenarios.json"


def ensure_dirs() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def initial_user_utterance(profile_id: str, domain: str) -> str:
    mapping = {
        "cooperative_user": {
            "rider_operations": "是我，你说。",
            "merchant_education_saas": "我是负责人，你说。",
            "default": "你说，我在听。",
        },
        "silent_user": {
            "rider_operations": "嗯。",
            "merchant_education_saas": "嗯，你说。",
            "default": "嗯。",
        },
        "busy_user": {
            "rider_operations": "我现在忙，你快点说。",
            "merchant_education_saas": "我这边正忙，你简短说。",
            "default": "我有点忙，你快说。",
        },
        "skeptical_user": {
            "rider_operations": "什么意思？你先说清楚。",
            "merchant_education_saas": "具体改了什么？你说清楚。",
            "default": "你先把情况说清楚。",
        },
        "off_topic_user": {
            "rider_operations": "先别说这个，我问你别的。",
            "merchant_education_saas": "我先问个别的事。",
            "default": "我先问你个别的问题。",
        },
        "random_question_user": {
            "rider_operations": "你先说，不过我想到什么就问什么。",
            "merchant_education_saas": "你先说，我可能会随时打断问。",
            "default": "你先说，我有别的问题也会问。",
        },
        "driving_user": {
            "default": "我在开车，晚点再说。",
        },
        "refusing_user": {
            "rider_operations": "我今天不想跑，别劝了。",
            "default": "这事我不想做。",
        },
    }
    domain_mapping = mapping.get(profile_id, {})
    return domain_mapping.get(domain, domain_mapping.get("default", "你说。"))


def state_templates(profile_id: str, domain: str) -> list[dict[str, Any]]:
    if profile_id == "cooperative_user":
        return [
            {
                "state_id": "listen_and_confirm",
                "intent": "listen",
                "default_response": "行，我知道了，还有吗？",
                "expected_agent_signals": ["合同", "低延迟", "标准直播", "配送", "发课"],
                "fallback_response": "你先说重点。",
                "transition_to": "ask_one_question",
                "terminal": False,
            },
            {
                "state_id": "ask_one_question",
                "intent": "clarify",
                "default_response": "那具体我接下来怎么做？",
                "expected_agent_signals": ["可以", "怎么", "流程", "注意"],
                "fallback_response": "那我接下来要做什么？",
                "transition_to": "close_call",
                "terminal": False,
            },
            {
                "state_id": "close_call",
                "intent": "close",
                "default_response": "行，那先这样。",
                "expected_agent_signals": ["再见", "先这样", "有问题"],
                "fallback_response": "我知道了。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    if profile_id == "silent_user":
        return [
            {
                "state_id": "minimal_ack_1",
                "intent": "minimal_ack",
                "default_response": "嗯。",
                "expected_agent_signals": ["合同", "低延迟", "直播", "配送"],
                "fallback_response": "你继续。",
                "transition_to": "minimal_ack_2",
                "terminal": False,
            },
            {
                "state_id": "minimal_ack_2",
                "intent": "minimal_ack",
                "default_response": "好。",
                "expected_agent_signals": ["要求", "规则", "注意", "怎么做"],
                "fallback_response": "还有吗？",
                "transition_to": "close_call",
                "terminal": False,
            },
            {
                "state_id": "close_call",
                "intent": "close",
                "default_response": "那先这样。",
                "expected_agent_signals": ["先这样", "有问题", "再联系"],
                "fallback_response": "我知道了。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    if profile_id == "busy_user":
        return [
            {
                "state_id": "interrupt_early",
                "intent": "interrupt",
                "default_response": "我很忙，你一句话说重点。",
                "expected_agent_signals": ["1分钟", "简短", "重点", "马上说完"],
                "fallback_response": "你太长了，快点。",
                "transition_to": "push_for_summary",
                "terminal": False,
            },
            {
                "state_id": "push_for_summary",
                "intent": "compress",
                "default_response": "好，你继续，尽量短点。",
                "expected_agent_signals": ["重点", "简单说", "一句话"],
                "fallback_response": "还是太绕了。",
                "transition_to": "close_quickly",
                "terminal": False,
            },
            {
                "state_id": "close_quickly",
                "intent": "close",
                "default_response": "行，我知道了，先挂了。",
                "expected_agent_signals": ["先这样", "稍后", "再联系"],
                "fallback_response": "先这样吧。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    if profile_id == "skeptical_user":
        return [
            {
                "state_id": "question_rules",
                "intent": "challenge",
                "default_response": "为什么要这样？依据是什么？",
                "expected_agent_signals": ["合同", "升级", "规则", "低延迟"],
                "fallback_response": "你说得太笼统了。",
                "transition_to": "ask_detail",
                "terminal": False,
            },
            {
                "state_id": "ask_detail",
                "intent": "faq",
                "default_response": "那价格/要求具体怎么算？",
                "expected_agent_signals": ["价格", "要求", "知识", "规则"],
                "fallback_response": "那具体标准是什么？",
                "transition_to": "close_after_answer",
                "terminal": False,
            },
            {
                "state_id": "close_after_answer",
                "intent": "close",
                "default_response": "行，我大概明白了。",
                "expected_agent_signals": ["解释", "就是", "因此"],
                "fallback_response": "先这样吧。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    if profile_id == "off_topic_user":
        return [
            {
                "state_id": "ask_out_of_scope",
                "intent": "out_of_scope",
                "default_response": (
                    "那你顺便告诉我别的规则怎么处理？"
                    if domain == "rider_operations"
                    else "那你顺便说下别的产品问题。"
                ),
                "expected_agent_signals": ["你好", "通知", "升级", "合同"],
                "fallback_response": "我问的不是这个。",
                "transition_to": "return_to_main_topic",
                "terminal": False,
            },
            {
                "state_id": "return_to_main_topic",
                "intent": "back_to_topic",
                "default_response": "行，你先说回这次通知。",
                "expected_agent_signals": ["确认后再回电", "现在能回答的先回答", "先说回"],
                "fallback_response": "那你先说这次正事。",
                "transition_to": "close_call",
                "terminal": False,
            },
            {
                "state_id": "close_call",
                "intent": "close",
                "default_response": "知道了，先这样。",
                "expected_agent_signals": ["再见", "有问题", "联系"],
                "fallback_response": "先挂了。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    if profile_id == "random_question_user":
        return [
            {
                "state_id": "interrupt_with_random_question",
                "intent": "random_interrupt",
                "default_response": (
                    "你先等等，那这个跟我之前那个问题有关系吗？"
                    if domain == "rider_operations"
                    else "你先等等，那别的功能会不会也一起变？"
                ),
                "expected_agent_signals": ["合同", "升级", "通知", "低延迟"],
                "fallback_response": "你先说这次重点吧。",
                "transition_to": "half_related_question",
                "terminal": False,
            },
            {
                "state_id": "half_related_question",
                "intent": "half_related_question",
                "default_response": (
                    "那如果我做不到，会有什么影响？"
                    if domain == "rider_operations"
                    else "那如果前端没显示，我要怎么处理？"
                ),
                "expected_agent_signals": ["影响", "处理", "规则", "显示"],
                "fallback_response": "那具体怎么办？",
                "transition_to": "close_call",
                "terminal": False,
            },
            {
                "state_id": "close_call",
                "intent": "close",
                "default_response": "行，我差不多知道了。",
                "expected_agent_signals": ["先这样", "有问题", "联系"],
                "fallback_response": "先这样吧。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    if profile_id == "driving_user":
        return [
            {
                "state_id": "declare_driving",
                "intent": "unsafe_to_continue",
                "default_response": "我在开车，晚点再打。",
                "expected_agent_signals": ["稍后再打", "不打扰", "晚点联系"],
                "fallback_response": "我真在开车，先挂了。",
                "transition_to": None,
                "terminal": True,
            }
        ]

    if profile_id == "refusing_user":
        return [
            {
                "state_id": "refuse_once",
                "intent": "refuse_task",
                "default_response": "我今天真不想跑。",
                "expected_agent_signals": ["开始配送", "合同", "完成配送"],
                "fallback_response": "反正我跑不了。",
                "transition_to": "refuse_twice",
                "terminal": False,
            },
            {
                "state_id": "refuse_twice",
                "intent": "refuse_task_again",
                "default_response": "我这边确实做不到，你别劝了。",
                "expected_agent_signals": ["尽量", "挽留", "鼓励", "名额"],
                "fallback_response": "我还是不行。",
                "transition_to": "accept_close",
                "terminal": False,
            },
            {
                "state_id": "accept_close",
                "intent": "close_after_comfort",
                "default_response": "行，那先这样。",
                "expected_agent_signals": ["注意安全", "理解", "安慰", "先挂"],
                "fallback_response": "我先挂了。",
                "transition_to": None,
                "terminal": True,
            },
        ]

    return [
        {
            "state_id": "generic",
            "intent": "generic",
            "default_response": "你继续说。",
            "expected_agent_signals": [],
            "fallback_response": "你说重点。",
            "transition_to": None,
            "terminal": True,
        }
    ]


def build_scenario(example: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    domain = example["domain"]
    scenario_id = f"{example['instruction_id']}_{profile['profile_id']}"
    states = state_templates(profile["profile_id"], domain)
    return {
        "scenario_id": scenario_id,
        "instruction_id": example["instruction_id"],
        "domain": domain,
        "profile_id": profile["profile_id"],
        "persona": profile["persona"],
        "goal": profile["goal"],
        "style": profile["style"],
        "trigger_scenarios": profile["trigger_scenarios"],
        "must_cover": profile["must_cover"],
        "initial_user_utterance": initial_user_utterance(profile["profile_id"], domain),
        "initial_state_id": states[0]["state_id"],
        "max_turns": max(3, len(states) + 1),
        "states": states,
        "instruction_summary": {
            "task": example["instruction_core"]["task"],
            "opening_line": example["instruction_core"]["opening_line"],
            "required_information": example["instruction_core"]["required_information"][:5],
        },
    }


def build_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/schemas/user_simulator.schema.json",
        "title": "User Simulator Scenario Dataset",
        "description": "面向复杂外呼任务评测的用户模拟器场景数据集。",
        "type": "object",
        "required": [
            "dataset_name",
            "source_file",
            "schema_ref",
            "scenario_count",
            "scenarios",
        ],
        "properties": {
            "dataset_name": {"type": "string"},
            "source_file": {"type": "string"},
            "schema_ref": {"type": "string"},
            "scenario_count": {"type": "integer", "minimum": 0},
            "scenarios": {
                "type": "array",
                "items": {"$ref": "#/$defs/scenario"},
            },
        },
        "$defs": {
            "scenario": {
                "type": "object",
                "required": [
                    "scenario_id",
                    "instruction_id",
                    "domain",
                    "profile_id",
                    "persona",
                    "goal",
                    "style",
                    "trigger_scenarios",
                    "must_cover",
                    "initial_user_utterance",
                    "initial_state_id",
                    "max_turns",
                    "states",
                    "instruction_summary",
                ],
                "properties": {
                    "scenario_id": {"type": "string"},
                    "instruction_id": {"type": "string"},
                    "domain": {"type": "string"},
                    "profile_id": {"type": "string"},
                    "persona": {"type": "string"},
                    "goal": {"type": "string"},
                    "style": {"type": "array", "items": {"type": "string"}},
                    "trigger_scenarios": {"type": "array", "items": {"type": "string"}},
                    "must_cover": {"type": "array", "items": {"type": "string"}},
                    "initial_user_utterance": {"type": "string"},
                    "initial_state_id": {"type": "string"},
                    "max_turns": {"type": "integer", "minimum": 1},
                    "states": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/state"},
                    },
                    "instruction_summary": {
                        "type": "object",
                        "required": ["task", "opening_line", "required_information"],
                        "properties": {
                            "task": {"type": "string"},
                            "opening_line": {"type": "string"},
                            "required_information": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            "state": {
                "type": "object",
                "required": [
                    "state_id",
                    "intent",
                    "default_response",
                    "expected_agent_signals",
                    "fallback_response",
                    "transition_to",
                    "terminal",
                ],
                "properties": {
                    "state_id": {"type": "string"},
                    "intent": {"type": "string"},
                    "default_response": {"type": "string"},
                    "expected_agent_signals": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "fallback_response": {"type": "string"},
                    "transition_to": {"type": ["string", "null"]},
                    "terminal": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


def build_dataset() -> dict[str, Any]:
    payload = json.loads(EVAL_DATA_PATH.read_text(encoding="utf-8"))
    scenarios = []
    for example in payload["examples"]:
        for profile in example["user_simulator_profiles"]:
            scenarios.append(build_scenario(example, profile))

    return {
        "dataset_name": f"{payload['dataset_name']}（用户模拟器版）",
        "source_file": str(EVAL_DATA_PATH),
        "schema_ref": str(SCHEMA_PATH),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def main() -> None:
    ensure_dirs()
    schema = build_schema()
    dataset = build_dataset()
    SCHEMA_PATH.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote schema to: {SCHEMA_PATH}")
    print(f"Wrote scenarios to: {OUTPUT_PATH}")
    print(f"Generated scenarios: {dataset['scenario_count']}")


if __name__ == "__main__":
    main()
