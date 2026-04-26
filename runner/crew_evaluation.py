#!/usr/bin/env python3
"""
CrewAI 三角色评测编排
- DialogAgent:    被测外呼 Agent
- UserSimulator:  LLM 用户模拟器
- Evaluator:      规则评分 + LLM 语义判断
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from agent.api_agent import APIAgent
from evaluator.auto_scorer import AutoScorer
from evaluator.llm_judge import LLMJudge
from llm.deepseek_client import DeepSeekClient, load_dotenv
from simulator.llm_user_simulator import LLMUserSimulator
from simulator.user_simulator import UserSimulator


# ---------------------------------------------------------------------------
# 全局状态（在 Tool 闭包中共享）
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _build_llm(model: str | None = None) -> Any:
    """返回 CrewAI 兼容的 LLM 配置（使用 DeepSeek OpenAI 兼容接口）。"""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    resolved_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    # CrewAI 支持 openai/ 前缀指定 provider
    return {
        "model": f"openai/{resolved_model}",
        "api_key": api_key,
        "base_url": base_url,
    }


# ---------------------------------------------------------------------------
# Tools（供 CrewAI Agent 调用）
# ---------------------------------------------------------------------------

@tool("dialog_agent_respond")
def dialog_agent_respond_tool(dummy: str = "") -> str:  # noqa: ARG001
    """外呼 Agent 根据当前对话历史生成下一轮回复。"""
    api_agent: APIAgent = _state["api_agent"]
    example: dict = _state["example"]
    scenario: dict = _state["scenario"]
    conversation: list = _state["conversation"]
    reply = api_agent.respond(example, scenario, conversation)
    _state["last_agent_reply"] = reply
    return reply


@tool("user_simulator_reply")
def user_simulator_reply_tool(dummy: str = "") -> str:  # noqa: ARG001
    """用户模拟器根据当前对话历史生成下一轮用户回复，并判断是否结束。"""
    llm_user: LLMUserSimulator = _state["llm_user"]
    example: dict = _state["example"]
    scenario: dict = _state["scenario"]
    conversation: list = _state["conversation"]
    result = llm_user.reply(scenario, example, conversation)
    _state["last_user_result"] = result
    return json.dumps(result, ensure_ascii=False)


@tool("run_auto_scorer")
def run_auto_scorer_tool(dummy: str = "") -> str:  # noqa: ARG001
    """对完整对话记录执行规则评分，返回结构化评测报告。"""
    scorer: AutoScorer = _state["scorer"]
    instruction_id: str = _state["instruction_id"]
    scenario: dict = _state["scenario"]
    conversation: list = _state["conversation"]
    report = scorer.score_conversation(instruction_id, scenario, conversation)
    _state["rule_report"] = report
    return json.dumps(report, ensure_ascii=False, indent=2)


@tool("run_llm_judge")
def run_llm_judge_tool(dummy: str = "") -> str:  # noqa: ARG001
    """LLM 评测专家对对话进行语义层面的判断，结合规则报告输出综合结论。"""
    llm_judge: LLMJudge = _state["llm_judge"]
    example: dict = _state["example"]
    scenario: dict = _state["scenario"]
    conversation: list = _state["conversation"]
    rule_report: dict = _state.get("rule_report", {})
    report = llm_judge.judge(example, scenario, conversation, rule_report)
    _state["llm_report"] = report
    return json.dumps(report, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 核心编排：run_crew_evaluation
# ---------------------------------------------------------------------------

def run_crew_evaluation(
    scenario_id: str,
    output_path: str | Path | None = None,
    judge_mode: str = "hybrid",
    agent_model: str | None = None,
    user_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """
    用 CrewAI 编排三角色完成一次完整评测。

    Args:
        scenario_id:  场景 ID，如 "1_cooperative_user"
        output_path:  JSON 结果输出路径（可选）
        judge_mode:   "rule" | "api" | "hybrid"
        agent_model:  DeepSeek 模型名（覆盖环境变量）
        user_model:   DeepSeek 模型名（覆盖环境变量）
        judge_model:  DeepSeek 模型名（覆盖环境变量）

    Returns:
        包含 scenario / conversation / report 的完整结果字典
    """
    load_dotenv(ROOT / ".env")

    resolved_agent_model = agent_model or os.environ.get("DEEPSEEK_AGENT_MODEL", "deepseek-chat")
    resolved_user_model = user_model or os.environ.get("DEEPSEEK_USER_MODEL", "deepseek-chat")
    resolved_judge_model = judge_model or os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")

    # 加载数据
    scenario_path = ROOT / "data" / "processed" / "user_simulator_scenarios.json"
    eval_data_path = ROOT / "data" / "processed" / "dialog_instruction_eval_examples.json"
    eval_examples = json.loads(eval_data_path.read_text(encoding="utf-8"))
    example_index = {ex["instruction_id"]: ex for ex in eval_examples["examples"]}
    simulator = UserSimulator.from_file(scenario_path)

    scenario = simulator.scenarios[scenario_id]
    instruction_id = scenario["instruction_id"]
    example = example_index[instruction_id]

    # 初始化组件
    api_agent = APIAgent(model=resolved_agent_model)
    llm_user = LLMUserSimulator(model=resolved_user_model)
    scorer = AutoScorer(eval_examples)
    llm_judge = LLMJudge(DeepSeekClient(model=resolved_judge_model)) if judge_mode in {"api", "hybrid"} else None

    # 初始化共享状态
    _state.clear()
    _state.update({
        "api_agent": api_agent,
        "llm_user": llm_user,
        "scorer": scorer,
        "llm_judge": llm_judge,
        "example": example,
        "scenario": scenario,
        "instruction_id": instruction_id,
        "conversation": [],
        "rule_report": {},
        "llm_report": None,
    })

    # 第一轮：用户开场白
    initial_user = llm_user.start(scenario)
    _state["conversation"].append({"role": "user", "text": initial_user})

    # 对话循环（最多 max_turns 轮）
    max_turns = scenario.get("max_turns", 6)
    for _ in range(max_turns):
        # Agent 回复
        agent_reply = api_agent.respond(example, scenario, _state["conversation"])
        _state["conversation"].append({"role": "agent", "text": agent_reply})

        # 用户回复
        result = llm_user.reply(scenario, example, _state["conversation"])
        user_reply = result.get("user_reply", "")
        if user_reply:
            _state["conversation"].append({"role": "user", "text": user_reply})

        if result.get("finished"):
            break

    # CrewAI：评测阶段（规则 + LLM）
    llm_cfg = _build_llm(resolved_judge_model)

    evaluator_agent = Agent(
        role="外呼任务评测专家",
        goal="对给定对话进行规则评分和语义判断，输出完整评测报告",
        backstory=(
            "你是一位专注于外呼任务质量评测的专家，"
            "擅长从流程完整性、约束遵守、FAQ 覆盖等维度进行量化评估。"
        ),
        tools=[run_auto_scorer_tool, run_llm_judge_tool] if judge_mode in {"api", "hybrid"} else [run_auto_scorer_tool],
        verbose=False,
        llm=llm_cfg,
    )

    if judge_mode in {"api", "hybrid"}:
        eval_task = Task(
            description=(
                "请依次执行以下步骤：\n"
                "1. 调用 run_auto_scorer 工具执行规则评分\n"
                "2. 调用 run_llm_judge 工具执行语义评判\n"
                "完成后输出：'评测完成'"
            ),
            expected_output="评测完成",
            agent=evaluator_agent,
        )
    else:
        eval_task = Task(
            description="调用 run_auto_scorer 工具执行规则评分，完成后输出：'评测完成'",
            expected_output="评测完成",
            agent=evaluator_agent,
        )

    crew = Crew(
        agents=[evaluator_agent],
        tasks=[eval_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()

    rule_report = _state.get("rule_report", {})
    llm_report = _state.get("llm_report")

    output: dict[str, Any] = {
        "scenario": {
            "scenario_id": scenario["scenario_id"],
            "instruction_id": instruction_id,
            "profile_id": scenario["profile_id"],
            "persona": scenario["persona"],
        },
        "modes": {
            "agent_model": resolved_agent_model,
            "user_model": resolved_user_model,
            "judge_mode": judge_mode,
            "judge_model": resolved_judge_model,
        },
        "conversation": _state["conversation"],
        "report": {
            "rule_report": rule_report,
            "llm_report": llm_report,
        },
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return output


if __name__ == "__main__":
    result = run_crew_evaluation(
        "1_cooperative_user",
        output_path=ROOT / "outputs" / "crew_demo.json",
        judge_mode="hybrid",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
