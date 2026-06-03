"""L3 — SemanticJudge: multi-dimension LLM scoring with sample voting.

Five dimensions on a 1-5 scale, each judged independently ``N`` times.
For each dimension we report:
  - score: median of the N samples
  - confidence: 1 - normalized stdev of the samples
  - rationale: from the median-closest sample
  - evidence: turn indices the judges referenced

This is where "可靠性" / "可解释性" live: the multi-sample distribution
becomes a confidence signal, and every dimension carries an explicit
rationale + evidence pointer.
"""
from __future__ import annotations

import json
import statistics
from typing import Any, Callable

from evaluator.scoring.evidence import Evidence, Finding, excerpt_of


DIMENSIONS: dict[str, str] = {
    "naturalness": "表达自然度：agent 的语气、节奏是否像真人外呼，是否啰嗦、生硬或机械。",
    "accuracy": "信息事实准确度：agent 提供的事实/规则是否与给定 knowledge_points 一致，是否编造或混淆。",
    "empathy_handling": "情绪与拒绝处理：当用户表达忙碌、拒绝、催促或情绪时，agent 是否得体回应。",
    "goal_pursuit": "目标推进：agent 是否在多轮中有效推进 task 目标而不绕弯。",
    "constraint_awareness": "约束敏感度：agent 对长度、语气、范围限制等显隐性约束的遵守。",
}


def _format_conversation(conversation: list[dict[str, Any]]) -> str:
    lines = []
    for i, turn in enumerate(conversation):
        lines.append(f"[{i}] ({turn['role']}) {turn['text']}")
    return "\n".join(lines)


_JUDGE_PROMPT = """你是对话指令遵循评测员。请按 1-5 分对 agent 在如下维度给分：

维度: {dimension_name}
维度说明: {dimension_desc}

评分指南:
  5 = 出色; 4 = 良好; 3 = 合格但有瑕疵; 2 = 明显问题; 1 = 严重失败

任务说明:
{task_block}

完整对话:
{conversation_block}

只输出 JSON:
{{
  "score": <1..5 整数>,
  "rationale": "一句话理由",
  "evidence_turn_indices": [<int 索引>, ...]
}}
"""


def _task_block(example: dict[str, Any]) -> str:
    core = example["instruction_core"]
    parts = [
        f"role: {core.get('role', '')}",
        f"task: {core.get('task', '')}",
        f"max_chars_per_turn: {core.get('constraints', {}).get('max_chars_per_turn', '不限')}",
    ]
    kps = core.get("knowledge_points", [])
    if kps:
        parts.append("knowledge_points:")
        for kp in kps[:5]:
            parts.append(f"  - {kp}")
    return "\n".join(parts)


def _single_judgement(
    dim_name: str,
    example: dict[str, Any],
    conversation: list[dict[str, Any]],
    llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]],
) -> dict[str, Any]:
    prompt = _JUDGE_PROMPT.format(
        dimension_name=dim_name,
        dimension_desc=DIMENSIONS[dim_name],
        task_block=_task_block(example),
        conversation_block=_format_conversation(conversation),
    )
    try:
        result = llm_complete_json([
            {"role": "system", "content": "你只输出 JSON。"},
            {"role": "user", "content": prompt},
        ])
    except Exception as exc:  # noqa: BLE001
        return {"score": 3, "rationale": f"LLM 调用失败: {exc}", "evidence_turn_indices": []}
    score = result.get("score")
    if not isinstance(score, (int, float)) or not (1 <= score <= 5):
        score = 3
    return {
        "score": int(score),
        "rationale": str(result.get("rationale", "")),
        "evidence_turn_indices": [int(i) for i in result.get("evidence_turn_indices", [])
                                  if isinstance(i, (int, float))
                                  and 0 <= int(i) < len(conversation)],
    }


def _aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [s["score"] for s in samples]
    if not scores:
        return {"score": 0.0, "confidence": 0.0, "rationale": "无样本",
                "evidence_turn_indices": []}
    median = statistics.median(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    confidence = max(0.0, 1.0 - stdev / 2.0)  # stdev=0 -> 1.0; stdev=2 -> 0.0
    # pick the sample closest to the median for rationale + evidence
    chosen = min(samples, key=lambda s: abs(s["score"] - median))
    return {
        "score": float(median),
        "confidence": round(confidence, 4),
        "rationale": chosen["rationale"],
        "evidence_turn_indices": chosen["evidence_turn_indices"],
        "samples": scores,
    }


class SemanticJudge:
    def __init__(
        self,
        llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]],
        n_samples: int = 3,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self.llm = llm_complete_json
        self.n = n_samples

    def evaluate(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        conversation: list[dict[str, Any]],
    ) -> list[Finding]:
        return run_semantic_judge(example, conversation, self.llm, self.n)


def run_semantic_judge(
    example: dict[str, Any],
    conversation: list[dict[str, Any]],
    llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]],
    n_samples: int = 3,
) -> list[Finding]:
    findings: list[Finding] = []
    for dim_name in DIMENSIONS:
        samples = [_single_judgement(dim_name, example, conversation, llm_complete_json)
                   for _ in range(n_samples)]
        agg = _aggregate_samples(samples)
        median = agg["score"]
        normalized = (median - 1) / 4 if median > 0 else 0.0  # 1-5 -> 0-1
        ev = [
            Evidence(
                turn_index=i, role=conversation[i]["role"],
                excerpt=excerpt_of(conversation[i]["text"]),
                reason="judge 引用",
            )
            for i in agg["evidence_turn_indices"]
        ]
        status = ("pass" if median >= 4 else
                  "partial" if median >= 3 else "fail")
        findings.append(Finding(
            finding_id=f"L3.{dim_name}",
            layer="L3", category="semantic", status=status,
            score=normalized, confidence=agg["confidence"],
            rationale=agg["rationale"],
            evidence=ev,
            extra={
                "raw_score_1to5": median,
                "samples": agg["samples"],
                "dimension_desc": DIMENSIONS[dim_name],
            },
        ))
    return findings
