from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from reasoning_checkpoints.rollout_artifact import (
    build_replay_tensors,
    load_and_validate_replay_tensors,
    write_rollout_artifact,
)


def _processed_prompt() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "pixel_values": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
    }


def _trajectory() -> dict:
    return {
        "sample_id": "row-7",
        "prompt_input_ids": [10, 11, 12],
        "reasoning_token_ids": [20, 21],
        "reasoning_close_token_ids": [30],
        "bbox_prefix_token_ids": [40],
        "bbox_output_ids": [41, 42],
    }


def _checkpoints() -> list[dict]:
    return [
        {
            "checkpoint_index": 0,
            "checkpoint_kind": "reasoning_start",
            "token_offset": 0,
            "reasoning_progress": 0.0,
            "prefix_token_ids": [],
        },
        {
            "checkpoint_index": 1,
            "checkpoint_kind": "reasoning_end",
            "token_offset": 2,
            "reasoning_progress": 1.0,
            "prefix_token_ids": [20, 21],
        },
    ]


def test_rollout_artifact_round_trip_is_tensor_exact_and_compact(tmp_path) -> None:
    trajectory = _trajectory()
    tensors = build_replay_tensors(
        _processed_prompt(),
        prompt_input_ids=trajectory["prompt_input_ids"],
        reasoning_token_ids=trajectory["reasoning_token_ids"],
    )
    artifact = write_rollout_artifact(
        tmp_path / "row-7",
        trajectory=trajectory,
        checkpoints=_checkpoints(),
        replay_tensors=tensors,
        image=Image.new("RGB", (8, 6), color=(10, 20, 30)),
    )

    loaded = load_and_validate_replay_tensors(
        artifact,
        expected_input_ids=[10, 11, 12, 20, 21],
    )
    for name, expected in tensors.items():
        assert torch.equal(loaded[name], expected)

    manifest = json.loads((tmp_path / "row-7" / "manifest.json").read_text())
    assert manifest["attention_saved_during_vllm_rollout"] is False
    assert manifest["checkpoint_schedule"] == [
        {
            "checkpoint_index": 0,
            "checkpoint_kind": "reasoning_start",
            "token_offset": 0,
            "reasoning_progress": 0.0,
        },
        {
            "checkpoint_index": 1,
            "checkpoint_kind": "reasoning_end",
            "token_offset": 2,
            "reasoning_progress": 1.0,
        },
    ]
    saved_trajectory = json.loads((tmp_path / "row-7" / "trajectory.json").read_text())
    assert saved_trajectory["replay_artifact"] == artifact


def test_rollout_artifact_refuses_retokenized_or_overwritten_inputs(tmp_path) -> None:
    trajectory = _trajectory()
    tensors = build_replay_tensors(
        _processed_prompt(),
        prompt_input_ids=trajectory["prompt_input_ids"],
        reasoning_token_ids=trajectory["reasoning_token_ids"],
    )
    artifact_dir = tmp_path / "row-7"
    artifact = write_rollout_artifact(
        artifact_dir,
        trajectory=trajectory,
        checkpoints=_checkpoints(),
        replay_tensors=tensors,
        image=Image.new("RGB", (8, 6)),
    )

    with pytest.raises(ValueError, match="differ from prompt"):
        load_and_validate_replay_tensors(
            artifact,
            expected_input_ids=[10, 11, 12, 20, 99],
        )
    with pytest.raises(FileExistsError, match="already exists"):
        write_rollout_artifact(
            artifact_dir,
            trajectory=trajectory,
            checkpoints=_checkpoints(),
            replay_tensors=tensors,
            image=Image.new("RGB", (8, 6)),
        )


def test_build_replay_tensors_rejects_prompt_drift() -> None:
    with pytest.raises(ValueError, match="differ from the rollout"):
        build_replay_tensors(
            _processed_prompt(),
            prompt_input_ids=[10, 99, 12],
            reasoning_token_ids=[20, 21],
        )
