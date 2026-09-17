"""Behavioral contracts for the issue-1 zero-CoT teacher definition."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from timeline_self_distillation import run_opd_micro as opd
from timeline_self_distillation.run_teacher_pilot import BOX_OPEN, QUERY


class _Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in ids)


class _Processor:
    tokenizer = _Tokenizer()


class _Model:
    def __init__(self):
        self.model = SimpleNamespace(rope_deltas=torch.tensor([7]))

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return SimpleNamespace(past_key_values=("c0",))


def test_zero_cot_teacher_jumps_from_c0_directly_to_bbox_prompt(monkeypatch):
    def advance(_model, cache, token_ids):
        return (*cache, *token_ids), None

    monkeypatch.setattr(opd, "_load_rgb_image", lambda _path: object())
    monkeypatch.setattr(opd, "_render_and_process", lambda *_args: ("chat template\n<think>\n", {}))
    monkeypatch.setattr(opd, "_advance", advance)
    monkeypatch.setattr(opd, "fork", lambda cache: tuple(cache))
    row = {
        "sample_id": "row-0",
        "image_path": "unused.jpg",
        "expression": "the right-hand white horse",
        "entity": "white horse",
        "reasoning_ids": [9001, 9002, 9003],
        "first_span_offset": 2,
        "scrubbed_reasoning_ids": [8001, 8002, 8003],
    }

    state = opd.build_states(_Model(), _Processor(), [row], "zero_cot")[0]

    bbox_ids = _Processor.tokenizer.encode(BOX_OPEN, add_special_tokens=False)
    repeated_query_ids = _Processor.tokenizer.encode(
        QUERY.format(entity=row["entity"], question=row["expression"]), add_special_tokens=False
    )
    assert state["teacher"] == ("c0", *bbox_ids[:-1])
    assert state["student"] == (
        "c0",
        *row["reasoning_ids"],
        *repeated_query_ids,
        *bbox_ids[:-1],
    )
    assert state["last_opening_id"] == bbox_ids[-1]
    assert state["teacher_conditioning"] == {
        "kind": "zero_cot_direct_bbox",
        "reasoning_tokens": 0,
        "uses_entity_bridge": False,
        "repeats_question": False,
        "suffix": BOX_OPEN,
    }


def test_zero_cot_rejects_c0_that_is_not_at_empty_thinking_state(monkeypatch):
    monkeypatch.setattr(opd, "_load_rgb_image", lambda _path: object())
    monkeypatch.setattr(opd, "_render_and_process", lambda *_args: ("assistant without think opener", {}))
    monkeypatch.setattr(opd, "fork", lambda cache: tuple(cache))
    row = {
        "sample_id": "row-0",
        "image_path": "unused.jpg",
        "expression": "white horse",
        "entity": "white horse",
        "reasoning_ids": [1],
        "first_span_offset": 1,
        "scrubbed_reasoning_ids": [1],
    }

    with pytest.raises(RuntimeError, match="empty assistant thinking state"):
        opd.build_states(_Model(), _Processor(), [row], "zero_cot")


def test_cli_requires_explicit_teacher_after_negative_zero_cot_pilot():
    with pytest.raises(SystemExit):
        opd.parse_args(["--pilot-records", "pilot.jsonl", "--output-dir", "out"])

    args = opd.parse_args(
        ["--pilot-records", "pilot.jsonl", "--output-dir", "out", "--teacher", "zero_cot"]
    )
    assert args.teacher == "zero_cot"
    assert args.pilot_records == Path("pilot.jsonl")
    assert args.output_dir == Path("out")
    assert set(opd.TEACHER_CHOICES) == {
        "zero_cot",
        "early_step0",
        "early_span1",
        "late_scrub",
        "late_entity",
    }


def test_zero_cot_protocol_receipt_is_explicit_about_actual_conditioning():
    assert opd.teacher_protocol_receipt("zero_cot") == {
        "kind": "zero_cot_direct_bbox",
        "cache_origin": "c0_empty_assistant_thinking_state",
        "reasoning": "none",
        "uses_entity_bridge": False,
        "repeats_question": False,
        "suffix": BOX_OPEN,
        "limitation": "question-conditioned through c0; not text-free pure perception",
    }


def test_bbox_suffix_must_round_trip_exactly_through_tokenizer():
    class _DriftingTokenizer(_Tokenizer):
        def decode(self, ids, **_kwargs):
            return super().decode(ids) + " "

    with pytest.raises(RuntimeError, match="exact tokenizer round-trip"):
        opd._encode_exact_suffix(_DriftingTokenizer(), BOX_OPEN, "teacher suffix")


def test_legacy_span1_remains_an_explicit_entity_query_control(monkeypatch):
    def advance(_model, cache, token_ids):
        return (*cache, *token_ids), None

    monkeypatch.setattr(opd, "_load_rgb_image", lambda _path: object())
    monkeypatch.setattr(opd, "_render_and_process", lambda *_args: ("chat template\n<think>\n", {}))
    monkeypatch.setattr(opd, "_advance", advance)
    monkeypatch.setattr(opd, "fork", lambda cache: tuple(cache))
    row = {
        "sample_id": "row-0",
        "image_path": "unused.jpg",
        "expression": "the right-hand white horse",
        "entity": "white horse",
        "reasoning_ids": [9001, 9002, 9003],
        "first_span_offset": 2,
        "scrubbed_reasoning_ids": [8001, 8002, 8003],
    }

    state = opd.build_states(_Model(), _Processor(), [row], "early_span1")[0]

    legacy_suffix = QUERY.format(entity=row["entity"], question=row["expression"]) + BOX_OPEN
    suffix_ids = _Processor.tokenizer.encode(legacy_suffix, add_special_tokens=False)
    assert state["teacher"] == ("c0", 9001, 9002, *suffix_ids[:-1])
    assert state["teacher_conditioning"]["kind"] == "legacy_span1_entity_query"


def test_teacher_evaluation_free_generates_from_teacher_cache(monkeypatch, tmp_path):
    seen_roles = []

    def sample(_model, _tokenizer, _grammar, _state, seed, *, cache_role="student"):
        seen_roles.append(cache_role)
        return {
            "ids": [1],
            "supports": [],
            "bbox": [10, 20, 30, 40],
            "parse_valid": True,
            "completed": True,
            "seed": seed,
        }

    monkeypatch.setattr(opd, "sample_bbox", sample)
    adapter = SimpleNamespace(enabled=True)
    states = [{"row": {"sample_id": "row-0", "ground_truth_bbox": [10, 20, 30, 40]}}]

    summary = opd.evaluate_teacher(None, None, adapter, None, states, tmp_path, draws=2)

    assert seen_roles == ["teacher", "teacher"]
    assert adapter.enabled is False
    assert summary == {"condition": "teacher", "mean_iou": 1.0, "acc_05": 1, "valid": 2, "n": 2}


def test_n50_terminal_direct_uses_saved_reasoning_without_entity_bridge(monkeypatch):
    def advance(_model, cache, token_ids):
        return (*cache, *token_ids), None

    monkeypatch.setattr(opd, "_load_rgb_image", lambda _path: object())
    monkeypatch.setattr(opd, "_render_and_process", lambda *_args: ("chat template\n<think>\n", {}))
    monkeypatch.setattr(opd, "_advance", advance)
    monkeypatch.setattr(opd, "fork", lambda cache: tuple(cache))
    row = {
        "sample_id": "row-49",
        "image_path": "unused.jpg",
        "expression": "target object",
        "ground_truth_bbox": [10, 20, 30, 40],
        "reasoning_token_ids": [7001, 7002, 7003],
    }

    state = opd.build_states(
        _Model(), _Processor(), [row], "zero_cot", student_conditioning="terminal_direct"
    )[0]

    bbox_ids = _Processor.tokenizer.encode(BOX_OPEN, add_special_tokens=False)
    assert state["teacher"] == ("c0", *bbox_ids[:-1])
    assert state["student"] == ("c0", 7001, 7002, 7003, *bbox_ids[:-1])
    assert state["student_conditioning"] == {
        "kind": "terminal_direct_bbox",
        "cache_origin": "c0_plus_full_saved_reasoning",
        "uses_entity_bridge": False,
        "repeats_question": False,
        "suffix": BOX_OPEN,
    }


def test_load_records_keeps_exactly_the_50_successful_trajectories(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    rows = [
        {"sample_id": "row-0", "status": "ok"},
        {"sample_id": "row-1", "status": "error"},
        {"sample_id": "legacy-without-status"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    records, receipt = opd.load_records(path, expected_samples=2)

    assert [row["sample_id"] for row in records] == ["row-0", "legacy-without-status"]
    assert receipt == {"source_rows": 3, "accepted_rows": 2, "excluded_non_ok_rows": 1}


def test_eval_seed_uses_stable_source_index_across_filtered_row_gaps():
    assert opd.eval_seed({"source_index": 54}, fallback_index=49, draw=3) == opd.SEED + 9_054_003
    assert opd.eval_seed({}, fallback_index=4, draw=2) == opd.SEED + 9_004_002


def test_frozen_record_rejects_rendered_prompt_drift(monkeypatch):
    monkeypatch.setattr(opd, "_load_rgb_image", lambda _path: object())
    monkeypatch.setattr(opd, "_render_and_process", lambda *_args: ("new prompt\n<think>\n", {}))
    row = {
        "sample_id": "row-0",
        "image_path": "unused.jpg",
        "expression": "target",
        "rendered_prompt": "saved prompt\n<think>\n",
        "reasoning_token_ids": [1],
    }

    with pytest.raises(RuntimeError, match="rendered prompt drift"):
        opd.build_states(
            _Model(), _Processor(), [row], "zero_cot", student_conditioning="terminal_direct"
        )
