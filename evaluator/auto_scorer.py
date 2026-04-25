#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def flatten_turn_text(turns: list[dict[str, Any]], role: str) -> str:
    return "\n".join(turn["text"] for turn in turns if turn["role"] == role)


def count_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword and keyword in text)


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def extract_keywords_from_text(text: str) -> list[str]:
    seeds = [
        "合同",
        "配送",
        "飞毛腿",
        "低延迟",
        "标准直播",
        "发布页",
        "价格",
        "费用",
        "负责人",
        "企业微信",
        "安全",
        "退出",
        "奖励",
        "控制台",
        "开车",
        "稍后再打",
    ]
    return [item for item in seeds if item in text]


class AutoScorer:
    def __init__(self, eval_examples: dict[str, Any]) -> None:
        self.examples = {
            example["instruction_id"]: example
            for example in eval_examples["examples"]
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "AutoScorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload)

    def score_conversation(
        self,
        instruction_id: str,
        scenario: dict[str, Any],
        conversation: list[dict[str, Any]],
    ) -> dict[str, Any]:
        example = self.examples[instruction_id]
        agent_text = flatten_turn_text(conversation, "agent")
        user_text = flatten_turn_text(conversation, "user")
        constraints = example["instruction_core"]["constraints"]

        success_results = self._score_success_criteria(example, agent_text, conversation)
        failure_results = self._score_failure_conditions(
            example, scenario, conversation, agent_text, user_text
        )
        category_scores = self._aggregate_category_scores(success_results)
        overall_score = round(sum(category_scores.values()) / max(len(category_scores), 1), 2)

        return {
            "instruction_id": instruction_id,
            "scenario_id": scenario["scenario_id"],
            "profile_id": scenario["profile_id"],
            "overall_score": overall_score,
            "category_scores": category_scores,
            "success_results": success_results,
            "failure_results": failure_results,
            "summary": {
                "agent_turn_count": sum(1 for turn in conversation if turn["role"] == "agent"),
                "user_turn_count": sum(1 for turn in conversation if turn["role"] == "user"),
                "max_chars_per_turn": constraints.get("max_chars_per_turn"),
                "violations": [item["condition_id"] for item in failure_results if item["triggered"]],
            },
        }

    def _score_success_criteria(
        self,
        example: dict[str, Any],
        agent_text: str,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results = []
        step_texts = [
            step["instruction"]
            for step in example["instruction_core"]["call_flow"]["steps"]
        ]

        for criterion in example["success_criteria"]:
            desc = criterion["description"]
            hit_count = count_hits(agent_text, extract_keywords_from_text(desc))
            passed = False
            score = 0.0

            if criterion["criterion_id"] == "goal_delivery":
                passed = hit_count >= 2
                score = 1.0 if passed else 0.0
            elif criterion["criterion_id"] == "opening_line_used":
                opening_keywords = extract_keywords_from_text(
                    example["instruction_core"]["opening_line"]
                )
                first_agent = next((turn["text"] for turn in conversation if turn["role"] == "agent"), "")
                opening_hit = count_hits(first_agent, opening_keywords)
                passed = opening_hit >= 1
                score = 1.0 if passed else 0.0
            elif criterion["criterion_id"].startswith("flow_step_"):
                source_text = desc
                step_hit = count_hits(agent_text, extract_keywords_from_text(source_text))
                passed = step_hit >= 1
                score = 1.0 if passed else 0.0
            elif criterion["criterion_id"] == "faq_consistency":
                faq_hits = 0
                for point in example["instruction_core"]["knowledge_points"]:
                    faq_hits += min(1, count_hits(agent_text, extract_keywords_from_text(point)))
                passed = faq_hits >= 1
                score = min(1.0, faq_hits / max(len(example["instruction_core"]["knowledge_points"]), 1) * 2)
            elif criterion["criterion_id"] == "turn_length_limit":
                max_chars = example["instruction_core"]["constraints"].get("max_chars_per_turn", 9999)
                over_limit = [
                    turn for turn in conversation
                    if turn["role"] == "agent" and len(turn["text"]) > max_chars
                ]
                passed = not over_limit
                score = 1.0 if passed else 0.0
            else:
                passed = hit_count >= 1
                score = 1.0 if passed else 0.0

            evidence = []
            if passed:
                evidence.append("matched_expected_signal")
            elif step_texts:
                evidence.append("missing_required_signal")

            results.append(
                {
                    "criterion_id": criterion["criterion_id"],
                    "category": criterion["category"],
                    "priority": criterion["priority"],
                    "passed": passed,
                    "score": round(score, 2),
                    "evidence": evidence,
                }
            )
        return results

    def _score_failure_conditions(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        conversation: list[dict[str, Any]],
        agent_text: str,
        user_text: str,
    ) -> list[dict[str, Any]]:
        results = []
        constraints = example["instruction_core"]["constraints"]

        for condition in example["failure_conditions"]:
            triggered = False

            if condition["condition_id"] == "miss_primary_task":
                goal_keywords = extract_keywords_from_text(example["instruction_core"]["task"])
                triggered = count_hits(agent_text, goal_keywords) < 2
            elif condition["condition_id"] == "miss_required_flow":
                step_hits = 0
                for step in example["instruction_core"]["call_flow"]["steps"]:
                    step_hits += min(1, count_hits(agent_text, extract_keywords_from_text(step["instruction"])))
                triggered = step_hits < max(1, len(example["instruction_core"]["call_flow"]["steps"]) // 2)
            elif condition["condition_id"] == "exceed_turn_length":
                max_chars = constraints.get("max_chars_per_turn", 9999)
                triggered = any(
                    len(turn["text"]) > max_chars
                    for turn in conversation
                    if turn["role"] == "agent"
                )
            elif condition["condition_id"] == "use_forbidden_expression":
                forbidden = constraints.get("forbidden_expressions", [])
                triggered = contains_any(agent_text, forbidden)
            elif condition["condition_id"] == "bad_out_of_scope_handling":
                if scenario["profile_id"] == "off_topic_user":
                    required = constraints.get("out_of_scope_reply", "")
                    required_keywords = extract_keywords_from_text(required)
                    triggered = required_keywords and count_hits(agent_text, required_keywords) == 0
            elif condition["condition_id"] == "continue_when_user_driving":
                if scenario["profile_id"] == "driving_user":
                    triggered = "稍后再打" not in agent_text and "晚点" not in agent_text
            elif condition["condition_id"] == "miss_end_call_condition":
                if scenario["profile_id"] in {"driving_user", "refusing_user"}:
                    last_agent = ""
                    for turn in reversed(conversation):
                        if turn["role"] == "agent":
                            last_agent = turn["text"]
                            break
                    triggered = not contains_any(last_agent, ["先这样", "稍后", "再联系", "安全", "挂"])

            results.append(
                {
                    "condition_id": condition["condition_id"],
                    "severity": condition["severity"],
                    "triggered": triggered,
                }
            )
        return results

    def _aggregate_category_scores(self, success_results: list[dict[str, Any]]) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        for item in success_results:
            bucket.setdefault(item["category"], []).append(item["score"])
        return {
            key: round(sum(values) / len(values) * 100, 2)
            for key, values in bucket.items()
        }


def demo() -> None:
    root = Path(__file__).resolve().parents[1]
    scorer = AutoScorer.from_file(root / "data" / "processed" / "dialog_instruction_eval_examples.json")
    sample_conversation = [
        {"role": "user", "text": "是我，你说。"},
        {"role": "agent", "text": "你好，飞毛腿合同今天生效了，你今天能开始配送吗？"},
        {"role": "user", "text": "行，你继续。"},
        {"role": "agent", "text": "单日和多日合同都有要求，注意安全，有问题再联系。"},
    ]
    scenario = {"scenario_id": "demo", "profile_id": "cooperative_user"}
    report = scorer.score_conversation("1", scenario, sample_conversation)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
