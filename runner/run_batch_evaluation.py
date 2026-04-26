#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.deepseek_client import load_dotenv
from runner.run_evaluation import run_one_evaluation


def main() -> None:
    load_dotenv(ROOT / ".env")
    scenario_path = ROOT / "data" / "processed" / "user_simulator_scenarios.json"
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]

    output_dir = ROOT / "outputs" / "batch_eval_14"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detailed_results = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        result = run_one_evaluation(
            scenario_id,
            output_path=output_dir / f"{scenario_id}.json",
            agent_mode="api",
            user_mode="api",
            judge_mode="rule",
        )
        rule_report = result["report"]["rule_report"]
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "instruction_id": result["scenario"]["instruction_id"],
                "profile_id": result["scenario"]["profile_id"],
                "persona": result["scenario"]["persona"],
                "overall_score": rule_report["overall_score"],
                "agent_turn_count": rule_report["summary"]["agent_turn_count"],
                "user_turn_count": rule_report["summary"]["user_turn_count"],
                "violations": rule_report["summary"]["violations"],
            }
        )
        detailed_results.append(result)
        print(f"finished {scenario_id}: score={rule_report['overall_score']}")

    scores = [row["overall_score"] for row in summary_rows]
    aggregate = {
        "scenario_count": len(summary_rows),
        "average_score": round(statistics.mean(scores), 2) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "failed_scenarios": [row["scenario_id"] for row in summary_rows if row["violations"]],
    }

    summary = {
        "aggregate": aggregate,
        "rows": summary_rows,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "all_results.json").write_text(
        json.dumps(detailed_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_dir={output_dir}")


if __name__ == "__main__":
    main()
