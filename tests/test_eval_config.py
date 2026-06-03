from pathlib import Path
import pytest
from config.eval_config import EvalConfig, load_eval_config


FIXTURE = Path(__file__).parent / "fixtures" / "eval_config.minimal.yaml"


def test_load_minimal_yaml_returns_eval_config():
    cfg = load_eval_config(FIXTURE)
    assert isinstance(cfg, EvalConfig)
    assert cfg.run_name == "test_run"
    assert cfg.agents_under_test[0].name == "kimi-k2"
    assert cfg.agents_under_test[0].type == "builtin_llm"
    assert cfg.scoring.l3_samples == 3


def test_output_dir_substitutes_run_name():
    cfg = load_eval_config(FIXTURE)
    assert cfg.resolved_output_dir() == Path("outputs/test_run")


def test_unknown_agent_type_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(FIXTURE.read_text(encoding="utf-8").replace("builtin_llm", "fictional"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_config(bad)


def test_filter_ids_optional():
    cfg = load_eval_config(FIXTURE)
    assert cfg.instructions.filter_ids is None


def test_personas_include_red_team_default_false():
    cfg = load_eval_config(FIXTURE)
    assert cfg.personas.include_red_team is False
