from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


AgentType = Literal["builtin_llm", "http", "openai_compat", "offline_log"]


class InstructionsCfg(BaseModel):
    source: str
    filter_ids: list[str] | None = None


class PersonasCfg(BaseModel):
    source: str
    include_red_team: bool = False
    filter_profile_ids: list[str] | None = None


class AgentCfg(BaseModel):
    name: str
    type: AgentType
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    log_path: str | None = None


class ScoringCfg(BaseModel):
    l3_samples: int = 3
    judge_model: str = "kimi-k2-turbo-preview"
    confidence_threshold: float = 0.6
    l1_penalty_per_violation: float = 10.0
    l2_weight: float = 40.0
    l3_weight: float = 60.0
    judge_provider: Literal["kimi", "deepseek"] = "kimi"


class OutputCfg(BaseModel):
    dir: str = "outputs/{run_name}"
    formats: list[Literal["json", "html"]] = Field(default_factory=lambda: ["json"])


class EvalConfig(BaseModel):
    run_name: str
    instructions: InstructionsCfg
    personas: PersonasCfg
    agents_under_test: list[AgentCfg]
    scoring: ScoringCfg = Field(default_factory=ScoringCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)
    seed: int = 42
    max_concurrency: int = 2

    @field_validator("agents_under_test")
    @classmethod
    def _at_least_one_agent(cls, v: list[AgentCfg]) -> list[AgentCfg]:
        if not v:
            raise ValueError("agents_under_test must contain at least one entry")
        names = [a.name for a in v]
        if len(names) != len(set(names)):
            raise ValueError("agent names must be unique")
        return v

    def resolved_output_dir(self) -> Path:
        return Path(self.output.dir.format(run_name=self.run_name))


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    try:
        return EvalConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid eval config {path}: {exc}") from exc
