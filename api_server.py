#!/usr/bin/env python3
"""
FastAPI 后端服务
运行：python api_server.py  或  uvicorn api_server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from llm.deepseek_client import load_dotenv
from store.persona_store import PersonaStore

load_dotenv(ROOT / ".env")

app = FastAPI(title="对话 Agent 评测系统 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

persona_store = PersonaStore()
_executor = ThreadPoolExecutor(max_workers=4)

# 运行中和已完成的实验结果
_runs: dict[str, dict[str, Any]] = {}
_run_queues: dict[str, asyncio.Queue] = {}


# ── Pydantic Models ──

class PersonaCreate(BaseModel):
    name: str
    instruction_id: str = "1"
    goal: str = ""
    style: list[str] = []
    initial_utterance: str = "你好，请说。"
    max_turns: int = 6
    profile_id: str = "custom"

class RunRequest(BaseModel):
    instruction_id: str
    persona_ids: list[str]
    judge_mode: str = "hybrid"


# ── 工具函数 ──

def _load_eval_examples() -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "dialog_instruction_eval_examples.json"
    return json.loads(path.read_text(encoding="utf-8"))

def _load_results() -> list[dict[str, Any]]:
    out_dir = ROOT / "outputs" / "batch_eval"
    results = []
    if out_dir.exists():
        for f in sorted(out_dir.glob("*.json")):
            if f.name in ("summary.json", "all_results.json", "conversations.json"):
                continue
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    return results


# ── API 端点 ──

@app.get("/api/instructions")
def get_instructions():
    d = _load_eval_examples()
    return {"instructions": [
        {
            "instruction_id": ex["instruction_id"],
            "domain": ex.get("domain", ""),
            "role": ex["instruction_core"]["role"],
            "task": ex["instruction_core"]["task"],
            "opening_line": ex["instruction_core"]["opening_line"],
            "step_count": len(ex["instruction_core"]["call_flow"]["steps"]),
            "constraints": ex["instruction_core"]["constraints"],
        }
        for ex in d["examples"]
    ]}


@app.get("/api/personas")
def get_personas():
    return {"personas": persona_store.list()}


@app.post("/api/personas", status_code=201)
def create_persona(body: PersonaCreate):
    persona = persona_store.create(body.model_dump())
    return persona


@app.put("/api/personas/{persona_id}")
def update_persona(persona_id: str, body: PersonaCreate):
    updated = persona_store.update(persona_id, body.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Persona not found")
    return updated


@app.delete("/api/personas/{persona_id}", status_code=204)
def delete_persona(persona_id: str):
    if not persona_store.delete(persona_id):
        raise HTTPException(status_code=404, detail="Persona not found")


@app.get("/api/results")
def get_results():
    results = _load_results()
    rows = []
    for r in results:
        rr = r.get("report", {}).get("rule_report", {})
        lr = r.get("report", {}).get("llm_report") or {}
        rule_score = rr.get("overall_score", 0)
        llm_score = lr.get("overall_score")
        final = round((rule_score + llm_score) / 2, 2) if llm_score else rule_score
        rows.append({
            "scenario_id": r["scenario"]["scenario_id"],
            "instruction_id": r["scenario"]["instruction_id"],
            "persona": r["scenario"]["persona"],
            "profile_id": r["scenario"]["profile_id"],
            "overall_score": final,
            "rule_score": rule_score,
            "llm_score": llm_score,
            "agent_turn_count": rr.get("summary", {}).get("agent_turn_count", 0),
            "violations": rr.get("summary", {}).get("violations", []),
            "category_scores": rr.get("category_scores", {}),
            "llm_summary": lr.get("summary", ""),
        })
    scores = [r["overall_score"] for r in rows]
    import statistics
    return {
        "aggregate": {
            "scenario_count": len(rows),
            "average_score": round(statistics.mean(scores), 2) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "failed_scenarios": [r["scenario_id"] for r in rows if r["violations"]],
        },
        "rows": rows,
    }


@app.post("/api/run")
async def start_run(body: RunRequest):
    run_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()
    _run_queues[run_id] = queue
    _runs[run_id] = {"status": "running", "results": []}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _execute_run,
        run_id,
        body.instruction_id,
        body.persona_ids,
        body.judge_mode,
        loop,
        queue,
    )
    return {"run_id": run_id}


def _execute_run(
    run_id: str,
    instruction_id: str,
    persona_ids: list[str],
    judge_mode: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> None:
    """在线程池中执行评测，通过 queue 把事件推给 SSE。"""
    from agent.api_agent import APIAgent
    from evaluator.auto_scorer import AutoScorer
    from evaluator.llm_judge import LLMJudge
    from llm.deepseek_client import DeepSeekClient

    def emit(event: str, data: Any) -> None:
        asyncio.run_coroutine_threadsafe(
            queue.put({"event": event, "data": json.dumps(data, ensure_ascii=False)}),
            loop,
        )

    try:
        eval_examples = _load_eval_examples()
        example_index = {ex["instruction_id"]: ex for ex in eval_examples["examples"]}
        example = example_index.get(instruction_id)
        if not example:
            emit("error", {"message": f"instruction_id {instruction_id} not found"})
            return

        scorer = AutoScorer(eval_examples)
        api_agent = APIAgent()
        llm_judge = LLMJudge() if judge_mode in {"api", "hybrid"} else None

        from simulator.llm_user_simulator import LLMUserSimulator
        llm_user = LLMUserSimulator()

        personas = [persona_store.get(pid) for pid in persona_ids]
        personas = [p for p in personas if p]

        for persona in personas:
            emit("persona_start", {
                "persona_id": persona["id"],
                "persona_name": persona["name"],
            })

            conversation: list[dict[str, str]] = []
            initial = llm_user.start(persona)
            conversation.append({"role": "user", "text": initial})
            emit("turn", {"role": "user", "text": initial, "turn_index": 0})

            max_turns = persona.get("max_turns", 6)
            for turn_idx in range(max_turns):
                # Agent 回复
                agent_reply = api_agent.respond(example, persona, conversation)
                conversation.append({"role": "agent", "text": agent_reply})
                emit("turn", {"role": "agent", "text": agent_reply, "turn_index": turn_idx * 2 + 1})

                # 用户回复
                result = llm_user.reply(persona, example, conversation)
                user_reply = result.get("user_reply", "")
                if user_reply:
                    conversation.append({"role": "user", "text": user_reply})
                    emit("turn", {"role": "user", "text": user_reply, "turn_index": turn_idx * 2 + 2})

                if result.get("finished"):
                    break

            # 规则评分
            rule_report = scorer.score_conversation(instruction_id, persona, conversation)
            for item in rule_report.get("success_results", []):
                emit("score_item", {
                    "type": "success",
                    "criterion_id": item["criterion_id"],
                    "passed": item["passed"],
                    "score": item["score"],
                    "category": item.get("category", ""),
                })
            for item in rule_report.get("failure_results", []):
                emit("score_item", {
                    "type": "failure",
                    "criterion_id": item["condition_id"],
                    "triggered": item["triggered"],
                    "severity": item.get("severity", ""),
                })

            llm_report = None
            if llm_judge:
                llm_report = llm_judge.judge(example, persona, conversation, rule_report)

            rule_score = rule_report.get("overall_score", 0)
            llm_score = llm_report.get("overall_score") if llm_report else None
            final_score = round((rule_score + (llm_score or rule_score)) / 2, 2) if llm_score else rule_score

            # 保存结果
            out_dir = ROOT / "outputs" / "batch_eval"
            out_dir.mkdir(parents=True, exist_ok=True)
            result_data = {
                "scenario": {
                    "scenario_id": persona["id"],
                    "instruction_id": instruction_id,
                    "profile_id": persona.get("profile_id", "custom"),
                    "persona": persona["name"],
                },
                "conversation": conversation,
                "report": {"rule_report": rule_report, "llm_report": llm_report},
            }
            (out_dir / f"{persona['id']}.json").write_text(
                json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            emit("persona_done", {
                "persona_id": persona["id"],
                "overall_score": final_score,
                "rule_score": rule_score,
                "llm_score": llm_score,
                "violations": rule_report.get("summary", {}).get("violations", []),
                "category_scores": rule_report.get("category_scores", {}),
                "llm_summary": llm_report.get("summary", "") if llm_report else "",
                "strengths": llm_report.get("strengths", []) if llm_report else [],
                "weaknesses": llm_report.get("weaknesses", []) if llm_report else [],
            })

        emit("all_done", {"run_id": run_id, "persona_count": len(personas)})

    except Exception as exc:
        emit("error", {"message": str(exc)})
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    queue = _run_queues.get(run_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Run not found")

    async def generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return EventSourceResponse(generator())


# ── 前端静态文件 ──
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
