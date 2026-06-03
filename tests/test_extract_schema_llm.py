"""Tests for the LLM-based instruction extractor.

We test the plumbing (schema validation, repair loop) by injecting a
mock LLM. No real LLM calls happen here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_instruction_schema_llm import extract_instruction, _validate


def _make_valid_record() -> dict:
    """Borrow the existing seed example as a known-good record."""
    seed = json.loads(
        Path("data/processed/dialog_instruction_eval_examples.json").read_text(encoding="utf-8")
    )
    return seed["examples"][0]


def test_validate_known_good_record_has_no_errors():
    record = _make_valid_record()
    assert _validate(record) == []


def test_validate_catches_missing_required_field():
    record = _make_valid_record()
    record.pop("success_criteria")
    errors = _validate(record)
    assert errors, "expected at least one validation error"
    assert any("success_criteria" in e for e in errors)


def test_extract_returns_valid_record_on_first_try():
    good = _make_valid_record()

    def fake_llm(messages):
        return good

    out = extract_instruction("raw", instruction_id="1", llm_complete_json=fake_llm)
    assert out["instruction_id"] == good["instruction_id"]


def test_extract_repairs_after_validation_error():
    good = _make_valid_record()
    bad = {k: v for k, v in good.items() if k != "success_criteria"}
    call_count = {"n": 0}

    def fake_llm(messages):
        call_count["n"] += 1
        return bad if call_count["n"] == 1 else good

    out = extract_instruction(
        "raw",
        instruction_id="1",
        llm_complete_json=fake_llm,
        max_repair_rounds=2,
    )
    assert call_count["n"] == 2
    assert _validate(out) == []


def test_extract_raises_after_max_repair_rounds():
    bad = {"instruction_id": "1"}  # grossly incomplete

    def fake_llm(messages):
        return bad

    with pytest.raises(ValueError, match="repair rounds"):
        extract_instruction(
            "raw",
            instruction_id="1",
            llm_complete_json=fake_llm,
            max_repair_rounds=1,
        )
