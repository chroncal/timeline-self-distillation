"""Artifact-closed capture for natural-reasoning multimodal rollouts.

The rollout backend does not expose full decoder attentions.  This module
therefore persists the exact tensors needed for a later teacher-forced eager
forward, together with the unmodified token trajectory and checkpoint schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 2
REPLAY_TENSOR_FILENAME = "model_inputs.safetensors"
MANIFEST_FILENAME = "manifest.json"
TRAJECTORY_FILENAME = "trajectory.json"
CHECKPOINTS_FILENAME = "checkpoints.jsonl"
IMAGE_FILENAME = "original_image.png"
REQUIRED_REPLAY_TENSORS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    }
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def token_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in token_ids], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_replay_tensors(
    processed_prompt: Mapping[str, Any],
    *,
    prompt_input_ids: Sequence[int],
    reasoning_token_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Build the exact unpadded prompt+reasoning inputs used by IVA replay."""

    prompt_ids = [int(value) for value in prompt_input_ids]
    reasoning_ids = [int(value) for value in reasoning_token_ids]
    processed_ids = processed_prompt.get("input_ids")
    if processed_ids is None:
        raise ValueError("processed prompt is missing input_ids")
    processed_ids = torch.as_tensor(processed_ids).detach().cpu()
    if processed_ids.ndim != 2 or processed_ids.shape[0] != 1:
        raise ValueError("processed prompt input_ids must have shape (1, prompt_length)")
    if processed_ids[0].tolist() != prompt_ids:
        raise ValueError("processed prompt input_ids differ from the rollout prompt IDs")

    prompt_types = processed_prompt.get("mm_token_type_ids")
    if prompt_types is None:
        raise ValueError("processed prompt is missing Qwen3.5 mm_token_type_ids")
    prompt_types = torch.as_tensor(prompt_types).detach().cpu()
    if prompt_types.shape != processed_ids.shape:
        raise ValueError("mm_token_type_ids shape differs from prompt input_ids")

    pixel_values = processed_prompt.get("pixel_values")
    image_grid_thw = processed_prompt.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("processed prompt is missing pixel_values or image_grid_thw")

    input_ids = torch.tensor([prompt_ids + reasoning_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    suffix_types = torch.zeros((1, len(reasoning_ids)), dtype=prompt_types.dtype)
    mm_token_type_ids = torch.cat((prompt_types, suffix_types), dim=1)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": torch.as_tensor(pixel_values).detach().cpu().contiguous(),
        "image_grid_thw": torch.as_tensor(image_grid_thw).detach().cpu().contiguous(),
    }
    for name, value in tensors.items():
        if value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense strided tensor")
    return tensors


def compact_checkpoint_schedule(checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the lossless checkpoint boundary contract without O(n^2) prefixes."""

    return [
        {
            "checkpoint_index": int(row["checkpoint_index"]),
            "checkpoint_kind": str(row["checkpoint_kind"]),
            "token_offset": int(row["token_offset"]),
            "reasoning_progress": float(row["reasoning_progress"]),
        }
        for row in checkpoints
    ]


def write_rollout_artifact(
    artifact_dir: Path,
    *,
    trajectory: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    replay_tensors: Mapping[str, torch.Tensor],
    image: Image.Image,
) -> dict[str, Any]:
    """Atomically write one self-contained rollout capture directory."""

    artifact_dir = artifact_dir.resolve()
    if artifact_dir.exists():
        raise FileExistsError(f"rollout artifact already exists: {artifact_dir}")
    missing = REQUIRED_REPLAY_TENSORS - set(replay_tensors)
    if missing:
        raise ValueError(f"replay tensors are missing: {sorted(missing)}")

    prompt_ids = [int(value) for value in trajectory["prompt_input_ids"]]
    reasoning_ids = [int(value) for value in trajectory["reasoning_token_ids"]]
    expected_input_ids = prompt_ids + reasoning_ids
    if replay_tensors["input_ids"].shape != (1, len(expected_input_ids)):
        raise ValueError("saved replay input_ids have the wrong shape")
    if replay_tensors["input_ids"][0].tolist() != expected_input_ids:
        raise ValueError("saved replay input_ids do not equal prompt + original reasoning IDs")

    artifact_reference = {
        "schema_version": SCHEMA_VERSION,
        "artifact_dir": str(artifact_dir),
        "manifest_path": str(artifact_dir / MANIFEST_FILENAME),
        "model_inputs_path": str(artifact_dir / REPLAY_TENSOR_FILENAME),
        "trajectory_path": str(artifact_dir / TRAJECTORY_FILENAME),
        "checkpoints_path": str(artifact_dir / CHECKPOINTS_FILENAME),
        "original_image_path": str(artifact_dir / IMAGE_FILENAME),
    }
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_dir.parent / f".{artifact_dir.name}.tmp-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        tensor_path = temporary / REPLAY_TENSOR_FILENAME
        dense_tensors = {
            name: value.detach().cpu().contiguous() for name, value in replay_tensors.items()
        }
        save_file(dense_tensors, tensor_path)

        image_path = temporary / IMAGE_FILENAME
        image.convert("RGB").save(image_path, format="PNG")

        trajectory_path = temporary / TRAJECTORY_FILENAME
        checkpoint_path = temporary / CHECKPOINTS_FILENAME
        trajectory_payload = dict(trajectory)
        trajectory_payload["replay_artifact"] = artifact_reference
        _write_json(trajectory_path, trajectory_payload)
        _write_jsonl(checkpoint_path, checkpoints)

        tensor_contract = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_sha256(value),
            }
            for name, value in dense_tensors.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "natural_reasoning_multimodal_replay",
            "sample_id": str(trajectory["sample_id"]),
            "generation_called": True,
            "training_called": False,
            "attention_saved_during_vllm_rollout": False,
            "attention_extraction_requires_teacher_forced_eager_replay": True,
            "prompt_token_count": len(prompt_ids),
            "reasoning_token_count": len(reasoning_ids),
            "prompt_input_ids_sha256": token_sha256(prompt_ids),
            "reasoning_token_ids_sha256": token_sha256(reasoning_ids),
            "checkpoint_schedule": compact_checkpoint_schedule(checkpoints),
            "tensor_contract": tensor_contract,
            "files": {
                "model_inputs": REPLAY_TENSOR_FILENAME,
                "trajectory": TRAJECTORY_FILENAME,
                "checkpoints": CHECKPOINTS_FILENAME,
                "original_image": IMAGE_FILENAME,
            },
            "original_image": {
                "mode": "RGB",
                "size_wh": list(image.size),
                "png_sha256": file_sha256(image_path),
            },
        }
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        os.replace(temporary, artifact_dir)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise

    return artifact_reference


def load_and_validate_replay_tensors(
    artifact: Mapping[str, Any],
    *,
    expected_input_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Load a captured replay sidecar and verify every recorded tensor digest."""

    manifest_path = Path(str(artifact["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported rollout artifact schema: {manifest.get('schema_version')}")
    tensor_path = Path(str(artifact["model_inputs_path"]))
    tensors = load_file(tensor_path, device="cpu")
    missing = REQUIRED_REPLAY_TENSORS - set(tensors)
    if missing:
        raise ValueError(f"captured replay tensors are missing: {sorted(missing)}")
    contract = manifest.get("tensor_contract", {})
    for name in REQUIRED_REPLAY_TENSORS:
        expected = contract.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(f"manifest has no tensor contract for {name}")
        value = tensors[name]
        if list(value.shape) != list(expected.get("shape", [])):
            raise ValueError(f"captured {name} shape differs from its manifest")
        if str(value.dtype) != expected.get("dtype"):
            raise ValueError(f"captured {name} dtype differs from its manifest")
        if tensor_sha256(value) != expected.get("sha256"):
            raise ValueError(f"captured {name} digest differs from its manifest")
    expected_ids = [int(value) for value in expected_input_ids]
    if tensors["input_ids"].shape != (1, len(expected_ids)):
        raise ValueError("captured input_ids length differs from the token trajectory")
    if tensors["input_ids"][0].tolist() != expected_ids:
        raise ValueError("captured input_ids differ from prompt + original reasoning IDs")
    return tensors
