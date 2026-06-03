"""L2 — FlowCoverage: rule-based candidate finding + LLM semantic verify.

For each step in ``call_flow.steps``, this layer:

1. Picks candidate agent turns by keyword/character overlap with the step
   instruction.
2. Asks an LLM judge: "Did the agent actually accomplish this step in
   these turns?" -> returns pass | partial | fail + a short rationale.

The candidate-picking is deterministic and fast; the LLM only sees the
candidates plus the step text, keeping cost low and judgements grounded.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from evaluator.scoring.evidence import Evidence, Finding, excerpt_of


def _tokenize_step(text: str) -> list[str]:
    """Extract content tokens from a step instruction for fuzzy matching.

    Strategy: drop markdown noise + punctuation, then take overlapping
    Chinese bigrams plus any contiguous alpha-numeric run. Bigrams are
    cheap and survive Chinese punctuation-heavy step text.
    """
    import re
    cleaned = re.sub(r"[\*\(\)（）【】《》\"'`]+", "", text)
    cleaned = re.sub(r"[，。、：；！？\s,.:;!?]+", " ", cleaned)
    tokens: list[str] = []
    for chunk in cleaned.split():
        if not chunk:
            continue
        # Latin / numeric run preserved whole
        if re.fullmatch(r"[A-Za-z0-9]+", chunk):
            tokens.append(chunk)
            continue
        # Chinese / mixed: emit overlapping bigrams of CJK characters
        cjk = re.sub(r"[^一-鿿]+", "", chunk)
        for i in range(len(cjk) - 1):
            tokens.append(cjk[i : i + 2])
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out[:20]


def _pick_candidates(
    step_text: str,
    conversation: list[dict[str, Any]],
    max_candidates: int = 3,
) -> list[tuple[int, dict[str, Any], int]]:
    """Return [(turn_index, turn, hit_count), ...] sorted by hit_count desc."""
    tokens = _tokenize_step(step_text)
    if not tokens:
        return []
    scored = []
    for idx, turn in enumerate(conversation):
        if turn["role"] != "agent":
            continue
        hits = sum(1 for tok in tokens if tok in turn["text"])
        if hits > 0:
            scored.append((idx, turn, hits))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:max_candidates]


_VERIFY_PROMPT = """你是对话流程评测员。判断 agent 是否完成了下面这个流程步骤。

步骤要求：{step_text}

候选 agent 回合（按相关度排序）：
{candidate_block}

只输出 JSON：
{{
  "verdict": "pass" | "partial" | "fail",
  "rationale": "一句话理由",
  "best_turn_index": <int 或 null>
}}
"""


def _verify_step_with_llm(
    step_text: str,
    candidates: list[tuple[int, dict[str, Any], int]],
    llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        return {"verdict": "fail", "rationale": "无候选回合", "best_turn_index": None}
    block = "\n".join(
        f"[turn {idx}] {excerpt_of(turn['text'], 200)}" for idx, turn, _ in candidates
    )
    prompt = _VERIFY_PROMPT.format(step_text=step_text, candidate_block=block)
    try:
        result = llm_complete_json([
            {"role": "system", "content": "你只输出 JSON。"},
            {"role": "user", "content": prompt},
        ])
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "partial", "rationale": f"LLM 调用失败: {exc}",
                "best_turn_index": candidates[0][0]}
    if not isinstance(result, dict):
        result = {}
    verdict = result.get("verdict", "fail")
    if verdict not in {"pass", "partial", "fail"}:
        verdict = "fail"
    return {
        "verdict": verdict,
        "rationale": str(result.get("rationale", "")),
        "best_turn_index": result.get("best_turn_index"),
    }


def _verify_step_keyword_only(
    step_text: str,
    candidates: list[tuple[int, dict[str, Any], int]],
) -> dict[str, Any]:
    """Fallback verifier when no LLM is configured: pass if any candidate."""
    if not candidates:
        return {"verdict": "fail", "rationale": "无关键词命中", "best_turn_index": None}
    top_idx, _, hits = candidates[0]
    tokens_total = max(len(_tokenize_step(step_text)), 1)
    ratio = hits / tokens_total
    verdict = "pass" if ratio >= 0.25 else ("partial" if ratio >= 0.1 else "fail")
    return {
        "verdict": verdict,
        "rationale": f"关键词覆盖率 {ratio:.0%} ({hits}/{tokens_total})",
        "best_turn_index": top_idx,
    }


class FlowCoverage:
    def __init__(
        self,
        llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm_complete_json

    def evaluate(
        self,
        example: dict[str, Any],
        scenario: dict[str, Any],
        conversation: list[dict[str, Any]],
    ) -> list[Finding]:
        return run_flow_coverage(example, conversation, llm_complete_json=self.llm)


def run_flow_coverage(
    example: dict[str, Any],
    conversation: list[dict[str, Any]],
    llm_complete_json: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    steps = example["instruction_core"].get("call_flow", {}).get("steps", [])
    for step in steps:
        candidates = _pick_candidates(step["instruction"], conversation)
        if llm_complete_json is not None:
            verdict_info = _verify_step_with_llm(step["instruction"], candidates, llm_complete_json)
        else:
            verdict_info = _verify_step_keyword_only(step["instruction"], candidates)

        verdict = verdict_info["verdict"]
        score = {"pass": 1.0, "partial": 0.5, "fail": 0.0}[verdict]
        ev: list[Evidence] = []
        for idx, turn, _hits in candidates:
            ev.append(Evidence(
                turn_index=idx, role="agent",
                excerpt=excerpt_of(turn["text"]),
                reason=("最佳匹配" if idx == verdict_info.get("best_turn_index")
                        else "候选匹配"),
            ))
        findings.append(Finding(
            finding_id=f"L2.{step['step_id']}",
            layer="L2", category="flow_coverage", status=verdict,
            score=score, confidence=1.0 if llm_complete_json else 0.7,
            rationale=verdict_info["rationale"],
            evidence=ev,
            extra={"step_id": step["step_id"], "step_text": step["instruction"]},
        ))
    return findings
