#!/usr/bin/env python3
"""
统一入口：运行评测并生成前端报告数据

用法：
    python run.py                          # 跑全部场景，hybrid 评判
    python run.py --scenario 1_cooperative_user   # 跑单个场景
    python run.py --judge rule             # 只用规则评分
    python run.py --judge api              # 只用 LLM 评分
    python run.py --judge hybrid           # 规则 + LLM（默认）
    python run.py --mode legacy            # 使用旧版 run_evaluation（不用 CrewAI）
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.deepseek_client import load_dotenv


def run_all(judge_mode: str, use_crew: bool) -> None:
    load_dotenv(ROOT / ".env")
    scenario_path = ROOT / "data" / "processed" / "user_simulator_scenarios.json"
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]

    output_dir = ROOT / "outputs" / "batch_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detailed_results = []
    all_conversations = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        print(f"[评测] {scenario_id} ...", flush=True)

        try:
            if use_crew:
                from runner.crew_evaluation import run_crew_evaluation
                result = run_crew_evaluation(
                    scenario_id,
                    output_path=output_dir / f"{scenario_id}.json",
                    judge_mode=judge_mode,
                )
            else:
                from runner.run_evaluation import run_one_evaluation
                result = run_one_evaluation(
                    scenario_id,
                    output_path=output_dir / f"{scenario_id}.json",
                    agent_mode="api",
                    user_mode="api",
                    judge_mode=judge_mode,
                )
        except Exception as exc:
            print(f"  [错误] {scenario_id}: {exc}", flush=True)
            continue

        rule_report = result["report"]["rule_report"]
        llm_report = result["report"].get("llm_report")
        final_score = rule_report["overall_score"]
        if llm_report and "overall_score" in llm_report:
            final_score = round((rule_report["overall_score"] + llm_report["overall_score"]) / 2, 2)

        row = {
            "scenario_id": scenario_id,
            "instruction_id": result["scenario"]["instruction_id"],
            "profile_id": result["scenario"]["profile_id"],
            "persona": result["scenario"]["persona"],
            "overall_score": final_score,
            "rule_score": rule_report["overall_score"],
            "llm_score": llm_report.get("overall_score") if llm_report else None,
            "agent_turn_count": rule_report["summary"]["agent_turn_count"],
            "user_turn_count": rule_report["summary"]["user_turn_count"],
            "violations": rule_report["summary"]["violations"],
            "category_scores": rule_report.get("category_scores", {}),
            "llm_summary": llm_report.get("summary", "") if llm_report else "",
            "strengths": llm_report.get("strengths", []) if llm_report else [],
            "weaknesses": llm_report.get("weaknesses", []) if llm_report else [],
        }
        summary_rows.append(row)
        detailed_results.append(result)
        all_conversations.append({
            "scenario_id": scenario_id,
            "persona": result["scenario"]["persona"],
            "score": final_score,
            "conversation": result["conversation"],
        })
        print(f"  完成: score={final_score}", flush=True)

    scores = [row["overall_score"] for row in summary_rows]
    aggregate = {
        "scenario_count": len(summary_rows),
        "average_score": round(statistics.mean(scores), 2) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "failed_scenarios": [row["scenario_id"] for row in summary_rows if row["violations"]],
        "judge_mode": judge_mode,
        "use_crew": use_crew,
    }

    summary = {"aggregate": aggregate, "rows": summary_rows}

    # 保存到 outputs/
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "all_results.json").write_text(
        json.dumps(detailed_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "conversations.json").write_text(
        json.dumps(all_conversations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 同步到前端
    frontend_data = ROOT / "frontend" / "data"
    frontend_data.mkdir(parents=True, exist_ok=True)
    (frontend_data / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (frontend_data / "conversations.json").write_text(
        json.dumps(all_conversations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n评测完成：{len(summary_rows)} 个场景")
    print(f"平均分：{aggregate['average_score']}")
    print(f"输出目录：{output_dir}")
    print(f"前端数据：{frontend_data}")


def run_single(scenario_id: str, judge_mode: str, use_crew: bool) -> None:
    load_dotenv(ROOT / ".env")
    output_dir = ROOT / "outputs" / "single"
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_crew:
        from runner.crew_evaluation import run_crew_evaluation
        result = run_crew_evaluation(
            scenario_id,
            output_path=output_dir / f"{scenario_id}.json",
            judge_mode=judge_mode,
        )
    else:
        from runner.run_evaluation import run_one_evaluation
        result = run_one_evaluation(
            scenario_id,
            output_path=output_dir / f"{scenario_id}.json",
            agent_mode="api",
            user_mode="api",
            judge_mode=judge_mode,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="对话 Agent 评测系统")
    parser.add_argument("--scenario", default="all", help="场景 ID 或 'all'（默认全部）")
    parser.add_argument(
        "--judge",
        default="hybrid",
        choices=["rule", "api", "hybrid"],
        help="评判模式：rule=纯规则 / api=纯LLM / hybrid=混合（默认）",
    )
    parser.add_argument(
        "--mode",
        default="crew",
        choices=["crew", "legacy"],
        help="编排模式：crew=CrewAI（默认）/ legacy=旧版",
    )
    args = parser.parse_args()

    use_crew = args.mode == "crew"

    if args.scenario == "all":
        run_all(judge_mode=args.judge, use_crew=use_crew)
    else:
        run_single(scenario_id=args.scenario, judge_mode=args.judge, use_crew=use_crew)


if __name__ == "__main__":
    main()
