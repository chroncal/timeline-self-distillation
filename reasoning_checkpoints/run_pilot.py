#!/usr/bin/env python3
"""Export token-faithful reasoning checkpoints and replay them with Qwen3.5.

This utility deliberately does not change the prompt or decoding protocol.  It
runs the current two-request Student trajectory (natural reasoning, then the
grammar-constrained bbox tail), saves every generated token id, and derives
checkpoints only by slicing the saved reasoning ids.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

# Several reused verl/data modules import torch indirectly.  Select the GPU
# before those imports; changing CUDA_VISIBLE_DEVICES after torch import can
# silently bind replay to physical GPU 0 instead of the requested device.
if "--device" in sys.argv:
    _device_argument_index = sys.argv.index("--device") + 1
    if _device_argument_index >= len(sys.argv):
        raise ValueError("--device requires a value")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_device_argument_index]
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from reasoning_checkpoints.extractor import decode_ids, extract_reasoning_checkpoints, split_reasoning_close
from reasoning_checkpoints.rollout_artifact import (
    SCHEMA_VERSION as ROLLOUT_ARTIFACT_SCHEMA_VERSION,
)
from reasoning_checkpoints.rollout_artifact import (
    build_replay_tensors,
    load_and_validate_replay_tensors,
    write_rollout_artifact,
)
from scripts.routed_grounding.run_diagnostics import (
    BBOX_TAIL_REGEX,
    DEFAULT_SEED,
    DEFAULT_STUDENT_MODEL,
    STUDENT_INSTRUCTION,
    VLLM_ENGINE_KWARGS,
    _load_rgb_image,
    read_routing_records,
    target_bbox_from_record,
)
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIMIT = 50
DEFAULT_DEVICE = "2"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / f"reasoning_checkpoints/pilot_seed{DEFAULT_SEED}_n{DEFAULT_LIMIT}_rollout_artifact_v2"
)
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data/refcocog_umd_pilot/train.parquet"
MIN_PIXELS = 3136
MAX_PIXELS = 262144
BBOX_PREFIX_TEXT = '<answer>{"bbox":['
BBOX_TAIL_MAX_TOKENS = 48
RESPONSE_LENGTH = 4096
HIT_IOU = 0.5


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _configure_device(device: str) -> None:
    if "torch" in globals() or "vllm" in globals():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set before importing GPU libraries")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _processor(model_path: Path) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    return processor


def _render_and_process(processor: Any, expression: str, image: Any) -> tuple[str, dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": STUDENT_INSTRUCTION.format(expression=expression)},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    processed = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
        padding=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return rendered, dict(processed)


def _first_completion(outputs: Any) -> tuple[Any, Any]:
    request = outputs[0]
    return request, request.outputs[0]


def _sampling_params(*, seed: int, max_tokens: int, reasoning: bool) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    kwargs: dict[str, Any] = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
    }
    if reasoning:
        kwargs.update(stop=["</think>"], include_stop_str_in_output=True)
    else:
        kwargs["structured_outputs"] = StructuredOutputsParams(regex=BBOX_TAIL_REGEX)
    return SamplingParams(**kwargs)


def _protocol_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    """Drop terminal EOS/padding for parsing while retaining raw ids on disk."""

    ids = [int(value) for value in token_ids]
    terminal_ids = {
        int(value)
        for value in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None))
        if value is not None
    }
    while ids and ids[-1] in terminal_ids:
        ids.pop()
    return ids


def _create_engine(model_path: Path) -> Any:
    from vllm import LLM

    kwargs = dict(VLLM_ENGINE_KWARGS)
    kwargs["gpu_memory_utilization"] = 0.82
    kwargs["limit_mm_per_prompt"] = {"image": 1}
    return LLM(model=str(model_path), **kwargs)


def _unload_engine(engine: Any) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        shutdown = getattr(getattr(getattr(engine, "llm_engine", None), "engine_core", None), "shutdown", None)
    if callable(shutdown):
        shutdown()
    del engine
    gc.collect()
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _generate_one(
    engine: Any,
    processor: Any,
    *,
    source_index: int,
    record: Mapping[str, Any],
    image: Any,
    seed: int,
    model_path: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rendered_prompt, processed = _render_and_process(processor, str(record["expression"]), image)
    prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
    mm_kwargs = {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
    common_request = {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": {"image": image},
        "mm_processor_kwargs": mm_kwargs,
    }
    first_outputs = engine.generate(
        common_request,
        sampling_params=_sampling_params(seed=seed, max_tokens=RESPONSE_LENGTH, reasoning=True),
        use_tqdm=False,
    )
    first_request, first_completion = _first_completion(first_outputs)
    reasoning_generation_ids = [int(value) for value in first_completion.token_ids]
    reasoning_ids, reasoning_close_ids = split_reasoning_close(processor.tokenizer, reasoning_generation_ids)

    bbox_prefix_ids = [int(value) for value in processor.tokenizer.encode(BBOX_PREFIX_TEXT, add_special_tokens=False)]
    remaining = RESPONSE_LENGTH - len(reasoning_generation_ids) - len(bbox_prefix_ids)
    if remaining <= 0:
        raise RuntimeError("reasoning left no token budget for the bbox phase")
    bbox_prompt_ids = prompt_ids + reasoning_generation_ids + bbox_prefix_ids
    second_outputs = engine.generate(
        {
            "prompt_token_ids": bbox_prompt_ids,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": mm_kwargs,
        },
        sampling_params=_sampling_params(
            seed=seed,
            max_tokens=min(BBOX_TAIL_MAX_TOKENS, remaining),
            reasoning=False,
        ),
        use_tqdm=False,
    )
    second_request, second_completion = _first_completion(second_outputs)
    bbox_tail_ids = [int(value) for value in second_completion.token_ids]
    response_ids = reasoning_generation_ids + bbox_prefix_ids + bbox_tail_ids
    protocol_response = "<think>" + decode_ids(processor.tokenizer, _protocol_ids(processor.tokenizer, response_ids))
    parsed = parse_response(protocol_response)
    predicted_bbox = None if parsed.bbox is None else list(parsed.bbox)
    gt_bbox = target_bbox_from_record(record)
    bbox_iou = None if predicted_bbox is None else xyxy_iou(predicted_bbox, gt_bbox)
    sample_id = f"row-{source_index}"
    checkpoints = extract_reasoning_checkpoints(
        processor.tokenizer,
        reasoning_ids,
        sample_id=sample_id,
        final_bbox=predicted_bbox,
        bbox_iou=bbox_iou,
        bbox_hit_gt=None if bbox_iou is None else bbox_iou >= HIT_IOU,
    )
    trajectory = {
        "schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "source_index": source_index,
        "status": "ok",
        "seed": int(seed),
        "response_length": RESPONSE_LENGTH,
        "model_path": str(model_path),
        "tokenizer_class": type(processor.tokenizer).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_processor_kwargs": mm_kwargs,
        "image_path": str(record["image_path"]),
        "expression": str(record["expression"]),
        "rendered_prompt": rendered_prompt,
        "prompt_input_ids": prompt_ids,
        "reasoning_generation_ids": reasoning_generation_ids,
        "reasoning_token_ids": reasoning_ids,
        "reasoning_close_token_ids": reasoning_close_ids,
        "bbox_prefix_token_ids": bbox_prefix_ids,
        "bbox_output_ids": bbox_tail_ids,
        "response_output_ids": response_ids,
        "full_trajectory_input_ids": prompt_ids + response_ids,
        "full_original_reasoning": decode_ids(processor.tokenizer, reasoning_ids),
        "protocol_response": protocol_response,
        "parse": parsed.to_dict(),
        "original_final_bbox": predicted_bbox,
        "ground_truth_bbox": gt_bbox,
        "original_bbox_iou": bbox_iou,
        "original_bbox_hit_gt": None if bbox_iou is None else bbox_iou >= HIT_IOU,
        "checkpoint_count": len(checkpoints),
        "vllm_prompt_token_count_phase1": len(first_request.prompt_token_ids),
        "vllm_prompt_token_count_phase2": len(second_request.prompt_token_ids),
        "reasoning_stop_reason": first_completion.stop_reason,
        "bbox_stop_reason": second_completion.stop_reason,
    }
    replay_tensors = build_replay_tensors(
        processed,
        prompt_input_ids=prompt_ids,
        reasoning_token_ids=reasoning_ids,
    )
    trajectory["replay_artifact"] = write_rollout_artifact(
        artifact_root / sample_id,
        trajectory=trajectory,
        checkpoints=checkpoints,
        replay_tensors=replay_tensors,
        image=image,
    )
    return trajectory, checkpoints


def generate_pilot(
    *,
    data_path: Path,
    output_dir: Path,
    model_path: Path,
    limit: int,
    seed: int,
) -> None:
    if not 50 <= limit <= 100:
        raise ValueError("pilot limit must be in the requested 50..100 range")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Scan at most 100 input samples and stop as soon as ``limit`` complete
    # trajectories exist.  Long generations that never close </think> remain
    # explicit errors and are never counted as pilot trajectories.
    records = read_routing_records(data_path, limit=100)
    processor = _processor(model_path)
    engine = _create_engine(model_path)
    trajectories_path = output_dir / "trajectories.jsonl"
    checkpoints_path = output_dir / "checkpoints.jsonl"
    existing = _read_jsonl(trajectories_path) if trajectories_path.exists() else []
    trajectories: list[dict[str, Any]] = [item for item in existing if item.get("status") == "ok"]
    errors: list[dict[str, Any]] = [item for item in existing if item.get("status") != "ok"]
    checkpoints: list[dict[str, Any]] = _read_jsonl(checkpoints_path) if checkpoints_path.exists() else []
    completed_ids = {str(item["sample_id"]) for item in existing}
    try:
        for source_index, record in records:
            if len(trajectories) >= limit:
                break
            if f"row-{source_index}" in completed_ids:
                continue
            try:
                image = _load_rgb_image(str(record["image_path"]))
                trajectory, sample_checkpoints = _generate_one(
                    engine,
                    processor,
                    source_index=source_index,
                    record=record,
                    image=image,
                    seed=seed,
                    model_path=model_path,
                    artifact_root=output_dir / "artifacts",
                )
                trajectories.append(trajectory)
                checkpoints.extend(sample_checkpoints)
            except Exception as error:
                errors.append(
                    {
                        "sample_id": f"row-{source_index}",
                        "source_index": source_index,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    finally:
        _unload_engine(engine)
    if len(trajectories) < limit:
        raise RuntimeError(f"only {len(trajectories)} complete trajectories found in the first 100 samples")
    _write_jsonl(trajectories_path, trajectories + errors)
    _write_jsonl(checkpoints_path, checkpoints)
    _write_reports(output_dir, trajectories, checkpoints, errors)


def _write_reports(
    output_dir: Path,
    trajectories: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> None:
    counts = [int(item["checkpoint_count"]) for item in trajectories]
    lengths = [len(item["reasoning_token_ids"]) for item in trajectories]
    distribution = dict(sorted(Counter(counts).items()))
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for checkpoint in checkpoints:
        by_sample.setdefault(str(checkpoint["sample_id"]), []).append(checkpoint)
    segment_char_lengths: list[int] = []
    segment_token_lengths: list[int] = []
    for sample_checkpoints in by_sample.values():
        previous_prefix = ""
        previous_offset = 0
        for checkpoint in sorted(sample_checkpoints, key=lambda item: int(item["checkpoint_index"]))[1:]:
            prefix = str(checkpoint["prefix_text"])
            segment_char_lengths.append(len(prefix[len(previous_prefix) :].strip()))
            current_offset = int(checkpoint["token_offset"])
            segment_token_lengths.append(current_offset - previous_offset)
            previous_prefix = prefix
            previous_offset = current_offset
    low_checkpoint_long = [
        str(item["sample_id"])
        for item in trajectories
        if int(item["checkpoint_count"]) <= 2 and len(item["reasoning_token_ids"]) >= 128
    ]
    summary = {
        "schema_version": 1,
        "requested_samples": len(trajectories) + len(errors),
        "successful_samples": len(trajectories),
        "failed_samples": len(errors),
        "total_checkpoints": len(checkpoints),
        "checkpoint_count_distribution": distribution,
        "checkpoint_count_min": min(counts) if counts else None,
        "checkpoint_count_mean": sum(counts) / len(counts) if counts else None,
        "checkpoint_count_max": max(counts) if counts else None,
        "checkpoint_segment_char_length_min": min(segment_char_lengths) if segment_char_lengths else None,
        "checkpoint_segment_char_length_median": median(segment_char_lengths) if segment_char_lengths else None,
        "checkpoint_segment_char_length_max": max(segment_char_lengths) if segment_char_lengths else None,
        "checkpoint_segments_at_most_2_chars": sum(length <= 2 for length in segment_char_lengths),
        "checkpoint_segments_at_most_8_chars": sum(length <= 8 for length in segment_char_lengths),
        "checkpoint_segment_token_length_min": min(segment_token_lengths) if segment_token_lengths else None,
        "checkpoint_segment_token_length_median": median(segment_token_lengths) if segment_token_lengths else None,
        "checkpoint_segment_token_length_max": max(segment_token_lengths) if segment_token_lengths else None,
        "checkpoint_segments_at_least_64_tokens": sum(length >= 64 for length in segment_token_lengths),
        "checkpoint_segments_at_least_128_tokens": sum(length >= 128 for length in segment_token_lengths),
        "reasoning_token_length_min": min(lengths) if lengths else None,
        "reasoning_token_length_mean": sum(lengths) / len(lengths) if lengths else None,
        "reasoning_token_length_max": max(lengths) if lengths else None,
        "response_length_cap_distribution": dict(
            sorted(Counter(int(item["response_length"]) for item in trajectories).items())
        ),
        "long_reasoning_with_only_1_or_2_checkpoints": low_checkpoint_long,
        "hit_gt_count": sum(item.get("original_bbox_hit_gt") is True for item in trajectories),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Pilot checkpoint text", ""]
    for trajectory in trajectories:
        sample_id = str(trajectory["sample_id"])
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"reasoning_tokens={len(trajectory['reasoning_token_ids'])}; "
                f"checkpoints={trajectory['checkpoint_count']}; "
                f"IoU={trajectory['original_bbox_iou']}; hit_gt={trajectory['original_bbox_hit_gt']}",
                "",
            ]
        )
        for checkpoint in by_sample.get(sample_id, []):
            prefix = str(checkpoint["prefix_text"])
            tail = prefix[-240:].replace("\n", "\\n")
            lines.append(
                f"- cp={checkpoint['checkpoint_index']} kind={checkpoint['checkpoint_kind']} "
                f"offset={checkpoint['token_offset']} progress={checkpoint['reasoning_progress']:.4f}: `{tail}`"
            )
        lines.append("")
    (output_dir / "checkpoint_texts.md").write_text("\n".join(lines), encoding="utf-8")
    case_ids = set(low_checkpoint_long)
    case_ids.update(str(item["sample_id"]) for item in trajectories[:8])
    cases = ["# Human-readable cases", ""]
    for item in trajectories:
        if str(item["sample_id"]) not in case_ids:
            continue
        cases.extend(
            [
                f"## {item['sample_id']}",
                "",
                f"Expression: {item['expression']}",
                "",
                f"Pred bbox / GT / IoU / hit: {item['original_final_bbox']} / {item['ground_truth_bbox']} / "
                f"{item['original_bbox_iou']} / {item['original_bbox_hit_gt']}",
                "",
                "```text",
                str(item["full_original_reasoning"]),
                "```",
                "",
            ]
        )
    (output_dir / "cases.md").write_text("\n".join(cases), encoding="utf-8")


def refresh_saved_parser_views(*, output_dir: Path, model_path: Path) -> None:
    """Recompute text parsing/metrics from saved raw ids without generation."""

    processor = _processor(model_path)
    trajectories_path = output_dir / "trajectories.jsonl"
    checkpoints_path = output_dir / "checkpoints.jsonl"
    rows = _read_jsonl(trajectories_path)
    trajectories = [item for item in rows if item.get("status") == "ok"]
    errors = [item for item in rows if item.get("status") != "ok"]
    checkpoints: list[dict[str, Any]] = []
    for trajectory in trajectories:
        response_ids = [int(value) for value in trajectory["response_output_ids"]]
        protocol_response = "<think>" + decode_ids(
            processor.tokenizer, _protocol_ids(processor.tokenizer, response_ids)
        )
        parsed = parse_response(protocol_response)
        predicted_bbox = None if parsed.bbox is None else list(parsed.bbox)
        gt_bbox = [float(value) for value in trajectory["ground_truth_bbox"]]
        bbox_iou = None if predicted_bbox is None else xyxy_iou(predicted_bbox, gt_bbox)
        hit = None if bbox_iou is None else bbox_iou >= HIT_IOU
        trajectory.update(
            protocol_response=protocol_response,
            parse=parsed.to_dict(),
            original_final_bbox=predicted_bbox,
            original_bbox_iou=bbox_iou,
            original_bbox_hit_gt=hit,
        )
        sample_checkpoints = extract_reasoning_checkpoints(
            processor.tokenizer,
            trajectory["reasoning_token_ids"],
            sample_id=str(trajectory["sample_id"]),
            final_bbox=predicted_bbox,
            bbox_iou=bbox_iou,
            bbox_hit_gt=hit,
        )
        trajectory["checkpoint_count"] = len(sample_checkpoints)
        checkpoints.extend(sample_checkpoints)
    _write_jsonl(trajectories_path, trajectories + errors)
    _write_jsonl(checkpoints_path, checkpoints)
    _write_reports(output_dir, trajectories, checkpoints, errors)


def _tensor_error(left: Any, right: Any) -> dict[str, float]:
    difference = (left.float() - right.float()).abs()
    return {"max_abs_error": float(difference.max().item()), "mean_abs_error": float(difference.mean().item())}


def replay_sanity_check(*, output_dir: Path, model_path: Path, limit: int | None = None) -> None:
    import torch
    from transformers import AutoModelForImageTextToText

    trajectories = [item for item in _read_jsonl(output_dir / "trajectories.jsonl") if item.get("status") == "ok"]
    if limit is not None:
        trajectories = trajectories[:limit]
    checkpoints = _read_jsonl(output_dir / "checkpoints.jsonl")
    checkpoint_map: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in checkpoints:
        checkpoint_map.setdefault(str(checkpoint["sample_id"]), []).append(checkpoint)
    processor = None
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        # Replay is a numerical audit rather than rollout generation.  FP32
        # avoids conflating token/cache mismatches with the different BF16
        # reduction order of Qwen3.5 GDN prefill and recurrent decode kernels.
        dtype=torch.float32,
    ).eval().cuda()
    length_sorted = sorted(trajectories, key=lambda item: len(item["reasoning_token_ids"]))
    quantile_indices = {round((len(length_sorted) - 1) * fraction / 4) for fraction in range(5)}
    kv_sample_ids = {str(length_sorted[index]["sample_id"]) for index in quantile_indices}
    results: list[dict[str, Any]] = []
    for trajectory in trajectories:
        sample_id = str(trajectory["sample_id"])
        stored_prompt_ids = [int(value) for value in trajectory["prompt_input_ids"]]
        reasoning_ids = [int(value) for value in trajectory["reasoning_token_ids"]]
        replay_artifact = trajectory.get("replay_artifact")
        if replay_artifact is not None:
            captured = load_and_validate_replay_tensors(
                replay_artifact,
                expected_input_ids=stored_prompt_ids + reasoning_ids,
            )
            prefix_ids = captured["input_ids"].cuda()
            attention_prefix = captured["attention_mask"].cuda()
            prompt_mm_types = captured["mm_token_type_ids"][:, : len(stored_prompt_ids)].cuda()
            model_inputs = {
                "pixel_values": captured["pixel_values"].cuda(),
                "image_grid_thw": captured["image_grid_thw"].cuda(),
            }
            replay_input_source = "captured_rollout_tensors"
        else:
            if processor is None:
                processor = _processor(model_path)
            image = _load_rgb_image(str(trajectory["image_path"]))
            rendered, processed = _render_and_process(processor, str(trajectory["expression"]), image)
            prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
            if rendered != trajectory["rendered_prompt"] or prompt_ids != stored_prompt_ids:
                raise RuntimeError(f"{sample_id}: replay prompt/image tokenization differs from saved rollout")
            prefix_ids = torch.tensor(
                [stored_prompt_ids + reasoning_ids], dtype=torch.long, device="cuda"
            )
            attention_prefix = torch.ones_like(prefix_ids)
            prompt_mm_types = processed["mm_token_type_ids"].cuda()
            model_inputs = {
                key: value.cuda() if hasattr(value, "cuda") else value
                for key, value in processed.items()
                if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}
            }
            replay_input_source = "legacy_processor_reconstruction"
        full_ids = torch.tensor([trajectory["full_trajectory_input_ids"]], dtype=torch.long, device="cuda")
        sample_checkpoints = sorted(checkpoint_map[sample_id], key=lambda item: int(item["checkpoint_index"]))
        absolute_positions = torch.tensor(
            [len(stored_prompt_ids) + int(item["token_offset"]) - 1 for item in sample_checkpoints],
            dtype=torch.long,
            device="cuda",
        )
        if int(absolute_positions.min()) < 0:
            raise RuntimeError(f"{sample_id}: empty prompt cannot anchor reasoning_start")
        attention_full = torch.ones_like(full_ids)
        full_mm_types = torch.cat(
            [
                prompt_mm_types,
                torch.zeros(
                    (1, full_ids.shape[1] - prompt_mm_types.shape[1]),
                    dtype=prompt_mm_types.dtype,
                    device="cuda",
                ),
            ],
            dim=1,
        )
        prefix_mm_types = full_mm_types[:, : prefix_ids.shape[1]]
        full_position_ids, _ = model.model.get_rope_index(
            full_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_full,
            mm_token_type_ids=full_mm_types,
        )
        prefix_position_ids, _ = model.model.get_rope_index(
            prefix_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_prefix,
            mm_token_type_ids=prefix_mm_types,
        )
        with torch.inference_mode():
            full = model(
                input_ids=full_ids,
                attention_mask=attention_full,
                position_ids=full_position_ids,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=absolute_positions,
                **model_inputs,
            )
            prefix = model(
                input_ids=prefix_ids,
                attention_mask=attention_prefix,
                position_ids=prefix_position_ids,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=absolute_positions,
                **model_inputs,
            )
            full_hidden = full.hidden_states[-1][0, absolute_positions]
            prefix_hidden = prefix.hidden_states[-1][0, absolute_positions]
            hidden_error = _tensor_error(full_hidden, prefix_hidden)
            logits_error = _tensor_error(full.logits[0], prefix.logits[0])

            cache_hidden_error = None
            cache_logits_error = None
            cached_prompt = None
            cached_reasoning = None
            if sample_id in kv_sample_ids:
                prompt_tensor = torch.tensor([stored_prompt_ids], dtype=torch.long, device="cuda")
                prompt_attention = torch.ones_like(prompt_tensor)
                cached_prompt = model(
                    input_ids=prompt_tensor,
                    attention_mask=prompt_attention,
                    position_ids=prefix_position_ids[:, :, : len(stored_prompt_ids)],
                    cache_position=torch.arange(len(stored_prompt_ids), dtype=torch.long, device="cuda"),
                    output_hidden_states=not reasoning_ids,
                    use_cache=True,
                    logits_to_keep=1,
                    **model_inputs,
                )
                cached_state = cached_prompt.past_key_values
                if not reasoning_ids:
                    cached_reasoning = cached_prompt
                for reasoning_index, token_id in enumerate(reasoning_ids):
                    absolute_index = len(stored_prompt_ids) + reasoning_index
                    cached_reasoning = model(
                        input_ids=torch.tensor([[token_id]], dtype=torch.long, device="cuda"),
                        attention_mask=torch.ones((1, absolute_index + 1), dtype=torch.long, device="cuda"),
                        position_ids=prefix_position_ids[:, :, absolute_index : absolute_index + 1],
                        cache_position=torch.tensor([absolute_index], dtype=torch.long, device="cuda"),
                        past_key_values=cached_state,
                        output_hidden_states=reasoning_index == len(reasoning_ids) - 1,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cached_state = cached_reasoning.past_key_values
                if cached_reasoning is None:
                    raise RuntimeError(f"{sample_id}: KV replay did not produce a reasoning-end state")
                end_hidden_direct = prefix.hidden_states[-1][0, -1]
                end_logits_direct = prefix.logits[0, -1]
                cache_hidden_error = _tensor_error(end_hidden_direct, cached_reasoning.hidden_states[-1][0, -1])
                cache_logits_error = _tensor_error(end_logits_direct, cached_reasoning.logits[0, -1])
        results.append(
            {
                "sample_id": sample_id,
                "checkpoint_count": len(sample_checkpoints),
                "compared_checkpoint_indices": [int(item["checkpoint_index"]) for item in sample_checkpoints],
                "reasoning_end_token_offset": len(reasoning_ids),
                "bbox_start_absolute_token_offset": len(stored_prompt_ids) + len(reasoning_ids),
                "prompt_ids_exact_match": True,
                "replay_input_source": replay_input_source,
                "prefix_ids_are_saved_raw_slice": all(
                    item["prefix_token_ids"] == reasoning_ids[: int(item["token_offset"])]
                    for item in sample_checkpoints
                ),
                "padding_used": False,
                "position_ids_mode": "explicit Qwen M-RoPE from identical unpadded ids/image_grid_thw",
                "image_inputs_replayed": sorted(model_inputs),
                "kv_cache_checked": sample_id in kv_sample_ids,
                "full_vs_prefix_hidden": hidden_error,
                "full_vs_prefix_logits": logits_error,
                "direct_vs_kv_cache_reasoning_end_hidden": cache_hidden_error,
                "direct_vs_kv_cache_reasoning_end_logits": cache_logits_error,
            }
        )
        del full, prefix, cached_prompt, cached_reasoning
        torch.cuda.empty_cache()
    _write_jsonl(output_dir / "sanity_check.jsonl", results)
    metrics = [
        "full_vs_prefix_hidden",
        "full_vs_prefix_logits",
        "direct_vs_kv_cache_reasoning_end_hidden",
        "direct_vs_kv_cache_reasoning_end_logits",
    ]
    aggregate: dict[str, Any] = {}
    for metric in metrics:
        metric_values = [item[metric] for item in results if item[metric] is not None]
        aggregate[metric] = {
            "sample_count": len(metric_values),
            "max_abs_error": max(item["max_abs_error"] for item in metric_values),
            "mean_abs_error": sum(item["mean_abs_error"] for item in metric_values) / len(metric_values),
        }
    tolerance = {"hidden_max_abs": 0.02, "logits_max_abs": 0.05}
    passed = bool(results) and (
        aggregate["full_vs_prefix_hidden"]["max_abs_error"] <= tolerance["hidden_max_abs"]
        and aggregate["full_vs_prefix_logits"]["max_abs_error"] <= tolerance["logits_max_abs"]
        and aggregate["direct_vs_kv_cache_reasoning_end_hidden"]["max_abs_error"] <= tolerance["hidden_max_abs"]
        and aggregate["direct_vs_kv_cache_reasoning_end_logits"]["max_abs_error"] <= tolerance["logits_max_abs"]
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "sample_count": len(results),
        "checkpoint_count": sum(item["checkpoint_count"] for item in results),
        "kv_cache_sample_ids": sorted(kv_sample_ids),
        "comparison": "full trajectory vs raw reasoning-end prefix; causal positions cover every saved checkpoint",
        "aggregate_errors": aggregate,
        "tolerance": tolerance,
        "padding_used": False,
        "tokenizer_reconstruction_used_for_reasoning_prefix": False,
        "replay_input_source_distribution": dict(
            sorted(Counter(item["replay_input_source"] for item in results).items())
        ),
        "kv_cache_checked_at": "reasoning_end immediately before </think>/<answer>/bbox",
        "kv_cache_mode": "prompt prefill then one saved reasoning token per decode call",
    }
    (output_dir / "sanity_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Prefix replay sanity check",
        "",
        f"Result: **{'PASS' if passed else 'FAIL'}**",
        "",
        f"Samples/checkpoints: {len(results)} / {report['checkpoint_count']}",
        "",
        "No padding was used. Schema-v2 rows loaded the tensors captured during rollout; legacy rows reran the "
        "processor only to assert exact prompt equality. Every reasoning prefix was a direct slice of saved rollout "
        "ids. All checkpoint positions were compared under the causal mask, and reasoning_end was also replayed "
        "through the model KV cache.",
        "",
        "| comparison | max abs | mean abs |",
        "|---|---:|---:|",
    ]
    for metric in metrics:
        values = aggregate[metric]
        markdown.append(f"| {metric} | {values['max_abs_error']:.8g} | {values['mean_abs_error']:.8g} |")
    (output_dir / "sanity_check_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"prefix replay sanity check failed; see {output_dir / 'sanity_check_report.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "report", "replay", "all"), default="all")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--replay-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_device(str(args.device))
    output_dir = args.output_dir.resolve()
    if args.phase in {"generate", "all"}:
        generate_pilot(
            data_path=args.data_path.resolve(),
            output_dir=output_dir,
            model_path=args.model.resolve(),
            limit=int(args.limit),
            seed=int(args.seed),
        )
    if args.phase in {"report", "all"}:
        refresh_saved_parser_views(output_dir=output_dir, model_path=args.model.resolve())
    if args.phase in {"replay", "all"}:
        replay_sanity_check(
            output_dir=output_dir,
            model_path=args.model.resolve(),
            limit=args.replay_limit,
        )


if __name__ == "__main__":
    main()
