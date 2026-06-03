#!/usr/bin/env python3
"""
FastAPI 后端服务
运行：python api_server.py  或  uvicorn api_server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
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

from config import loads_eval_config
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

# Layered runs (new /runs family). Keyed by server-generated run_id.
# Each value: {state, run_name, started_at, finished_at, error,
#              output_dir, summary, aggregate}
_layered_runs: dict[str, dict[str, Any]] = {}


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


# ── /runs API (layered runner) ──

class RunSubmitRequest(BaseModel):
    """Inline YAML config submission for the layered runner.

    We accept the config body as text rather than a server-side path so
    callers can't trick the server into reading arbitrary files. The
    ``run_name`` field is optional; when omitted we use whatever the
    YAML already declares.
    """
    yaml_config: str
    scenario: str | None = None


def _execute_layered_run(run_id: str, yaml_text: str, scenario: str | None) -> None:
    """Background worker that drives ``runner.cli.run`` for one submission."""
    # Local import keeps the FastAPI module importable even if the runner
    # has heavy/optional dependencies.
    from runner.cli import run as cli_run

    record = _layered_runs[run_id]
    try:
        cfg = loads_eval_config(yaml_text)
        # Force a unique run_name so concurrent submissions don't share an
        # output directory if a YAML hard-codes the same value.
        original_name = cfg.run_name
        cfg.run_name = f"{original_name}__{run_id}"
        record["run_name"] = cfg.run_name
        record["state"] = "running"
        record["output_dir"] = str(cfg.resolved_output_dir())

        summary = cli_run(cfg, scenario_filter=scenario)
        record["summary"] = summary

        # Aggregate is written by the runner itself — load it back so the
        # /report endpoint can serve it without re-running aggregation.
        agg_path = cfg.resolved_output_dir() / "aggregate.json"
        if agg_path.exists():
            record["aggregate"] = json.loads(agg_path.read_text(encoding="utf-8"))

        record["state"] = "succeeded"
    except Exception as exc:  # noqa: BLE001 — surface the failure to the API caller
        record["state"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        # Stash a short traceback excerpt for debuggability without
        # leaking server internals beyond the existing API surface.
        record["traceback"] = traceback.format_exc(limit=5)
    finally:
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


def _summarise_layered_run(run_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Project an internal record to the public listing/detail shape."""
    summary = record.get("summary") or {}
    return {
        "run_id": run_id,
        "state": record.get("state"),
        "run_name": record.get("run_name"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "error": record.get("error"),
        "output_dir": record.get("output_dir"),
        "result_count": len(summary.get("results") or []),
        "judge_available": summary.get("judge_available"),
    }


@app.post("/runs", status_code=202)
def submit_layered_run(body: RunSubmitRequest):
    """Kick off a layered evaluation run from inline YAML config."""
    # Validate early so the API caller gets a 400 instead of a background
    # failure recorded on a run id they never see.
    try:
        loads_eval_config(body.yaml_config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid yaml_config: {exc}") from exc

    run_id = uuid.uuid4().hex[:12]
    _layered_runs[run_id] = {
        "state": "queued",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": None,
        "error": None,
        "run_name": None,
        "output_dir": None,
        "summary": None,
        "aggregate": None,
    }
    _executor.submit(_execute_layered_run, run_id, body.yaml_config, body.scenario)
    return {"run_id": run_id, "state": "queued"}


@app.get("/runs")
def list_layered_runs():
    return {
        "runs": [
            _summarise_layered_run(rid, rec)
            for rid, rec in sorted(_layered_runs.items(), key=lambda kv: kv[1].get("started_at") or "")
        ],
    }


@app.get("/runs/{run_id}")
def get_layered_run(run_id: str):
    rec = _layered_runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    out = _summarise_layered_run(run_id, rec)
    # When the run is done, surface the summary inline so the UI can show
    # the per-scenario rows without a second round-trip.
    if rec.get("summary"):
        out["summary"] = rec["summary"]
    return out


@app.get("/runs/{run_id}/report")
def get_layered_run_report(run_id: str):
    rec = _layered_runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    if rec.get("state") != "succeeded":
        # Tell the caller why the aggregate isn't available — much more
        # useful than a generic 404 for a run that legitimately exists.
        raise HTTPException(
            status_code=409,
            detail={"state": rec.get("state"), "error": rec.get("error")},
        )
    return {
        "run_id": run_id,
        "run_name": rec.get("run_name"),
        "aggregate": rec.get("aggregate"),
        "summary": rec.get("summary"),
    }


# ── 前端静态文件 ──
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
