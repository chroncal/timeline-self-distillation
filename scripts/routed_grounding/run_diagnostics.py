#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the frozen 32-example routed-grounding pretraining audit.

This is deliberately an *audit* entry point rather than a training entry
point.  It first obtains a two-stage Student answer for every selected pilot
example, records the router decision (and, when warranted, four exact-prefix
probes), and only then releases the Student before loading the frozen Teacher.

The script keeps vLLM, torch, pandas, and Pillow imports lazy.  Besides making
the command's GPU ownership explicit, that keeps the receipt helpers usable in
CPU-only tests without accidentally constructing a model engine.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.routed_grounding.prepare_verl_data import GROUNDING_INSTRUCTION
from verl.experimental.routed_grounding.router import (
    match_prediction,
    parse_response,
    route_response,
    validate_prediction_bbox,
)
from verl.experimental.routed_grounding.teacher import (
    build_localization_teacher_prompt,
    build_localization_teacher_target,
    crop_gt_bbox,
    gt_overlay,
    parse_teacher_target,
    verify_teacher_target,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data/refcocog_umd_pilot/smoke_train.parquet"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/diagnostics"
DEFAULT_STUDENT_MODEL = Path("/mnt/sda/sujingyang/models/Qwen3.5-0.8B")
DEFAULT_TEACHER_MODEL = Path("/mnt/sda/sujingyang/models/Qwen3.5-4B")

DEFAULT_LIMIT = 32
DEFAULT_DEVICE = "5"
DEFAULT_SEED = 260600564
MAX_RESPONSE_TOKENS = 2048
BBOX_TAIL_MAX_TOKENS = 48
TEACHER_BBOX_MAX_TOKENS = 48
TEACHER_CORRECTION_MAX_TOKENS = 64
EXPECTED_PROBE_COUNT = 4
VLLM_ENGINE_KWARGS = {
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "enforce_eager": True,
    "max_model_len": 18433,
    "max_num_batched_tokens": 24576,
    "max_num_seqs": 8,
    "enable_prefix_caching": True,
    "limit_mm_per_prompt": {"image": 2},
}

# vLLM structured outputs constrain only the continuation after this exact
# token prefix.  Keep this local instead of importing agent_loop.py, whose
# trainer dependencies are intentionally unnecessary for a stand-alone audit.
BBOX_PREFIX_TEXT = '</think><answer>{"bbox":['
_NUMBER_REGEX = r"(?:1000(?:\.0+)?|(?:0|[1-9][0-9]{0,2})(?:\.[0-9]+)?)"
BBOX_TAIL_REGEX = (
    rf"\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}"
    rf"\s*,\s*{_NUMBER_REGEX}\s*\]\}}\s*</answer>"
)
BBOX_JSON_REGEX = (
    rf"\s*\{{\s*\"bbox\"\s*:\s*\[\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}"
    rf"\s*,\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}\s*\]\s*\}}\s*"
)
REFERENT_TAIL_REGEX = (
    rf"\s*[^\n<>.!?]{{1,240}}[.!?]\s*</think>\s*<answer>\s*"
    rf"\{{\s*\"bbox\"\s*:\s*\[\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}"
    rf"\s*,\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}\s*\]\s*\}}\s*</answer>\s*"
)

# The image is supplied by vLLM's supported ``image_pil`` content part, so the
# text must be the frozen training instruction after removing only its literal
# ``<image>\n`` placeholder.  Importing the preparation constant prevents a
# diagnostic/training prompt drift.
if not GROUNDING_INSTRUCTION.startswith("<image>\n"):
    raise RuntimeError("frozen grounding instruction must begin with its image placeholder")
STUDENT_INSTRUCTION = GROUNDING_INSTRUCTION.removeprefix("<image>\n")
TEACHER_IMAGE_COORDINATE_INSTRUCTION = (
    "Image 1 is the original image. Image 2 is a 1.5x crop centered on the ground-truth "
    "region and is provided only for visual detail. Every bbox you return must use Image 1's "
    "original [0,1000] xyxy coordinates, never crop-relative coordinates."
)


def _jsonable(value: Any) -> Any:
    """Convert lightweight array/scalar wrappers into JSON-safe receipt values."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _write_jsonl_line(handle: Any, record: Mapping[str, Any]) -> None:
    """Append one durable-enough audit receipt without buffering another row."""

    handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL in {path} at line {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            records.append(value)
    return records


def _sample_id(source_index: int) -> str:
    return f"row-{source_index}"


def _completed_sample_ids(
    records: Iterable[Mapping[str, Any]], *, record_type: str, successful_only: bool = False
) -> set[str]:
    return {
        str(record["sample_id"])
        for record in records
        if record.get("record_type") == record_type
        and record.get("sample_id") is not None
        and (not successful_only or record.get("status") == "ok")
    }


def _archive_failed_teacher_attempt(teacher_path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve a failed Teacher attempt, then leave only valid receipts to resume.

    A retry must not turn one sample into two final audit records.  Keep the
    immutable failed attempt beside the canonical receipt file and reconstruct
    that canonical file from already-successful rows before retrying errors.
    """

    has_failed_audit = any(
        record.get("record_type") == "teacher_audit" and record.get("status") == "error" for record in records
    )
    if not has_failed_audit:
        return records

    attempt_number = 1
    while True:
        attempt_path = teacher_path.with_name(f"{teacher_path.stem}_attempt{attempt_number}{teacher_path.suffix}")
        if not attempt_path.exists():
            break
        attempt_number += 1
    teacher_path.replace(attempt_path)
    retained_records = [
        record
        for record in records
        if not (record.get("record_type") == "teacher_audit" and record.get("status") == "error")
    ]
    with teacher_path.open("w", encoding="utf-8") as teacher_handle:
        for record in retained_records:
            _write_jsonl_line(teacher_handle, record)
    return retained_records


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _routing_record_from_parquet_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read the manifest record solely from the frozen parquet routing receipt."""

    extra_info = _as_mapping(row.get("extra_info"), name="extra_info")
    encoded = extra_info.get("routing_record_json")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("extra_info.routing_record_json is required")
    try:
        record = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("extra_info.routing_record_json is invalid JSON") from error
    if not isinstance(record, dict):
        raise ValueError("extra_info.routing_record_json must contain an object")
    required = ("expression", "image_path", "image_width", "image_height", "instances", "target_ann_id", "target_bbox")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"routing record is missing required fields: {', '.join(missing)}")
    if not isinstance(record["instances"], list) or not record["instances"]:
        raise ValueError("routing record instances must be a non-empty list")
    return record


def read_routing_records(data_path: Path, *, limit: int) -> list[tuple[int, dict[str, Any]]]:
    """Load up to ``limit`` manifest records from the frozen smoke parquet."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - production dependency check
        raise RuntimeError("run_diagnostics.py requires pandas with parquet support") from error
    frame = pd.read_parquet(data_path)
    if len(frame) < limit:
        raise ValueError(f"requested limit={limit}, but {data_path} contains only {len(frame)} records")
    result: list[tuple[int, dict[str, Any]]] = []
    for source_index, (_, row) in enumerate(frame.iloc[:limit].iterrows()):
        result.append((source_index, _routing_record_from_parquet_row(row.to_dict())))
    return result


def normalize_xywh(bbox: Sequence[Any], image_width: Any, image_height: Any) -> list[float]:
    """Normalize one COCO xywh bbox to the source image's [0,1000] space."""

    if len(bbox) != 4:
        raise ValueError("target_bbox must contain four xywh coordinates")
    width, height = float(image_width), float(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, box_width, box_height = (float(value) for value in bbox)
    return [
        1000.0 * x / width,
        1000.0 * y / height,
        1000.0 * (x + box_width) / width,
        1000.0 * (y + box_height) / height,
    ]


def target_bbox_from_record(record: Mapping[str, Any]) -> list[float]:
    return normalize_xywh(record["target_bbox"], record["image_width"], record["image_height"])


def build_student_messages(record: Mapping[str, Any], image: Any) -> list[dict[str, Any]]:
    """Create the one-image Qwen chat request used for Student reasoning."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": STUDENT_INSTRUCTION.format(expression=str(record["expression"]))},
            ],
        }
    ]


def build_teacher_messages(original_image: Any, crop_image: Any, instruction: str) -> list[dict[str, Any]]:
    """Create a two-image Teacher request in fixed original-then-crop order."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": original_image},
                {"type": "image_pil", "image_pil": crop_image},
                {"type": "text", "text": f"{TEACHER_IMAGE_COORDINATE_INSTRUCTION}\n\n{instruction}"},
            ],
        }
    ]


def _make_structured_outputs(regex: str) -> Any:
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:  # pragma: no cover - production dependency check
        raise RuntimeError("Routed Grounding diagnostics requires vLLM 0.24 StructuredOutputsParams") from error
    return StructuredOutputsParams(regex=regex)


def _sampling_params(**kwargs: Any) -> Any:
    try:
        from vllm import SamplingParams
    except ImportError as error:  # pragma: no cover - production dependency check
        raise RuntimeError("Routed Grounding diagnostics requires vLLM 0.24") from error
    return SamplingParams(**kwargs)


def student_reasoning_sampling_params(*, seed: int, max_tokens: int) -> Any:
    return _sampling_params(
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        seed=int(seed),
        max_tokens=int(max_tokens),
        stop=["</think>"],
        include_stop_str_in_output=False,
    )


def student_bbox_sampling_params(*, seed: int, max_tokens: int) -> Any:
    return _sampling_params(
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        seed=int(seed),
        max_tokens=int(max_tokens),
        structured_outputs=_make_structured_outputs(BBOX_TAIL_REGEX),
    )


def teacher_bbox_sampling_params() -> Any:
    return _sampling_params(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=0,
        max_tokens=TEACHER_BBOX_MAX_TOKENS,
        structured_outputs=_make_structured_outputs(BBOX_JSON_REGEX),
    )


def teacher_referent_sampling_params() -> Any:
    return _sampling_params(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=0,
        max_tokens=TEACHER_CORRECTION_MAX_TOKENS,
        structured_outputs=_make_structured_outputs(REFERENT_TAIL_REGEX),
    )


def _first_completion(request_outputs: Any) -> tuple[Any, Any]:
    try:
        request = request_outputs[0]
        completion = request.outputs[0]
    except (IndexError, AttributeError, TypeError) as error:
        raise RuntimeError("vLLM returned no completion") from error
    return request, completion


def _completion_receipt(request: Any, completion: Any) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in getattr(completion, "token_ids", ())]
    prompt_ids = getattr(request, "prompt_token_ids", None)
    return {
        "request_id": str(getattr(request, "request_id", "")),
        "prompt_token_count": None if prompt_ids is None else len(prompt_ids),
        "raw_text": str(getattr(completion, "text", "")),
        "token_count": len(token_ids),
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": getattr(completion, "stop_reason", None),
    }


def _encode_no_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def build_bbox_continuation_prompt_ids(
    prompt_token_ids: Sequence[Any], reasoning_token_ids: Sequence[Any], tokenizer: Any
) -> list[int]:
    """Reuse phase-one tokens and append the exact canonical bbox prefix.

    The Student reasoning is never decode/re-encoded here.  That makes all
    probe requests byte/token-identical up through ``<answer>{\"bbox\":[``.
    """

    return (
        [int(token_id) for token_id in prompt_token_ids]
        + [int(token_id) for token_id in reasoning_token_ids]
        + _encode_no_special_tokens(tokenizer, BBOX_PREFIX_TEXT)
    )


def _canonical_student_prefix(reasoning: str) -> str:
    return f'<think>\n{reasoning}</think><answer>{{"bbox":['


def assemble_student_response(reasoning: str, bbox_tail: str) -> str:
    return _canonical_student_prefix(reasoning) + bbox_tail


def _sampling_receipt(*, seed: int, max_tokens: int, structured_regex: str | None) -> dict[str, Any]:
    return {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "structured_regex": structured_regex,
    }


def _invalid_student_result(reasoning: str, *, reason: str, phase_one: Mapping[str, Any]) -> dict[str, Any]:
    response = f"<think>\n{reasoning}"
    parsed = parse_response(response)
    return {
        "response": response,
        "bbox": None,
        "parse": parsed.to_dict(),
        "reasoning": reasoning,
        "bbox_prefix": None,
        "bbox_continuation_prompt_ids": None,
        "phase_one": dict(phase_one),
        "phase_two": {"skipped": True, "reason": reason},
    }


def run_student_two_stage(
    engine: Any,
    *,
    record: Mapping[str, Any],
    image: Any,
    seed: int = DEFAULT_SEED,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
) -> dict[str, Any]:
    """Produce one Student answer with separate natural-thinking and bbox phases."""

    messages = build_student_messages(record, image)
    phase_one_outputs = engine.chat(
        messages,
        sampling_params=student_reasoning_sampling_params(seed=seed, max_tokens=max_response_tokens),
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": True},
    )
    phase_one_request, phase_one_completion = _first_completion(phase_one_outputs)
    reasoning_token_ids = [int(token_id) for token_id in getattr(phase_one_completion, "token_ids", ())]
    reasoning = str(getattr(phase_one_completion, "text", ""))
    phase_one = _completion_receipt(phase_one_request, phase_one_completion)
    phase_one["sampling"] = _sampling_receipt(seed=seed, max_tokens=max_response_tokens, structured_regex=None)
    phase_one["chat_template_kwargs"] = {"enable_thinking": True}

    prompt_token_ids = getattr(phase_one_request, "prompt_token_ids", None)
    if prompt_token_ids is None:
        return _invalid_student_result(
            reasoning, reason="phase_one_did_not_return_prompt_token_ids", phase_one=phase_one
        )
    if str(getattr(phase_one_completion, "stop_reason", "")) != "</think>":
        return _invalid_student_result(
            reasoning,
            reason="reasoning_did_not_stop_at_</think>",
            phase_one=phase_one,
        )
    remaining_tokens = int(max_response_tokens) - len(reasoning_token_ids)
    if remaining_tokens <= 0:
        return _invalid_student_result(reasoning, reason="reasoning_exhausted_max_response_tokens", phase_one=phase_one)

    tokenizer = engine.get_tokenizer()
    continuation_prompt_ids = build_bbox_continuation_prompt_ids(prompt_token_ids, reasoning_token_ids, tokenizer)
    bbox_max_tokens = min(BBOX_TAIL_MAX_TOKENS, remaining_tokens)
    phase_two_outputs = engine.generate(
        {
            "prompt_token_ids": continuation_prompt_ids,
            "multi_modal_data": {"image": image},
        },
        sampling_params=student_bbox_sampling_params(seed=seed, max_tokens=bbox_max_tokens),
        use_tqdm=False,
    )
    phase_two_request, phase_two_completion = _first_completion(phase_two_outputs)
    bbox_tail = str(getattr(phase_two_completion, "text", ""))
    response = assemble_student_response(reasoning, bbox_tail)
    parsed = parse_response(response)
    phase_two = _completion_receipt(phase_two_request, phase_two_completion)
    phase_two["sampling"] = _sampling_receipt(seed=seed, max_tokens=bbox_max_tokens, structured_regex=BBOX_TAIL_REGEX)
    return {
        "response": response,
        "bbox": None if parsed.bbox is None else list(parsed.bbox),
        "parse": parsed.to_dict(),
        "reasoning": reasoning,
        "bbox_prefix": parsed.bbox_prefix,
        "bbox_continuation_prompt_ids": continuation_prompt_ids,
        "phase_one": phase_one,
        "phase_two": phase_two,
    }


def probe_seeds(seed: int) -> tuple[int, int, int, int]:
    """Return four independent, reproducible seeds distinct from the main rollout."""

    return tuple(int(seed) + offset for offset in range(1, EXPECTED_PROBE_COUNT + 1))  # type: ignore[return-value]


def run_bbox_probes(
    engine: Any,
    *,
    continuation_prompt_ids: Sequence[Any],
    image: Any,
    reasoning: str,
    seed: int,
    remaining_tokens: int,
) -> list[dict[str, Any]]:
    """Sample exactly four tail continuations from one immutable bbox prefix."""

    if remaining_tokens <= 0:
        raise ValueError("probes require remaining response tokens")
    bbox_max_tokens = min(BBOX_TAIL_MAX_TOKENS, int(remaining_tokens))
    prefix = _canonical_student_prefix(reasoning)
    probes: list[dict[str, Any]] = []
    fixed_prompt_ids = [int(token_id) for token_id in continuation_prompt_ids]
    for probe_index, probe_seed in enumerate(probe_seeds(seed)):
        outputs = engine.generate(
            {
                "prompt_token_ids": fixed_prompt_ids,
                "multi_modal_data": {"image": image},
            },
            sampling_params=student_bbox_sampling_params(seed=probe_seed, max_tokens=bbox_max_tokens),
            use_tqdm=False,
        )
        request, completion = _first_completion(outputs)
        raw_tail = str(getattr(completion, "text", ""))
        response = prefix + raw_tail
        parsed = parse_response(response)
        receipt = _completion_receipt(request, completion)
        receipt.update(
            {
                "probe_index": probe_index,
                "seed": probe_seed,
                "raw_tail": raw_tail,
                "response": response,
                "bbox": None if parsed.bbox is None else list(parsed.bbox),
                "parse": parsed.to_dict(),
                "prefix_token_count": len(fixed_prompt_ids),
                "sampling": _sampling_receipt(
                    seed=probe_seed, max_tokens=bbox_max_tokens, structured_regex=BBOX_TAIL_REGEX
                ),
            }
        )
        probes.append(receipt)
    return probes


def _student_route_receipt(
    *,
    source_index: int,
    record: Mapping[str, Any],
    student: Mapping[str, Any],
    route: Any,
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    route_dict = route.to_dict()
    for probe, match in zip(probes, route.probe_matches, strict=False):
        probe["match"] = match.to_dict()  # type: ignore[index]
    return {
        "schema_version": 1,
        "record_type": "student_route",
        "sample_id": _sample_id(source_index),
        "source_index": source_index,
        "status": "ok",
        "seed": DEFAULT_SEED,
        "routing_record": dict(record),
        "target_bbox": target_bbox_from_record(record),
        "student": dict(student),
        "route": route_dict,
        "probes": list(probes),
    }


def audit_student_example(engine: Any, *, source_index: int, record: Mapping[str, Any], image: Any) -> dict[str, Any]:
    """Run Student routing plus conditional exact-prefix probes for one record."""

    try:
        student = run_student_two_stage(engine, record=record, image=image)
        response = str(student["response"])
        parsed = parse_response(response)
        instances = record["instances"]
        target_ann_id = record["target_ann_id"]
        preliminary = route_response(response, instances, target_ann_id)
        probes: list[dict[str, Any]] = []

        # Only a valid non-target annotation/background/ambiguity gets four
        # probes.  Correct and target-low-IoU localization responses spend no
        # extra Student calls, and malformed outputs remain uncertain.
        original_match = preliminary.original_match
        needs_probes = (
            parsed.format_valid
            and parsed.bbox is not None
            and original_match is not None
            and original_match.status in {"matched", "background", "ambiguous"}
            and not (original_match.status == "matched" and original_match.ann_id == target_ann_id)
        )
        routed = preliminary
        if needs_probes:
            continuation_prompt_ids = student.get("bbox_continuation_prompt_ids")
            reasoning = student.get("reasoning")
            if not isinstance(continuation_prompt_ids, list) or not isinstance(reasoning, str):
                raise RuntimeError("valid Student response is missing its exact bbox continuation prefix")
            remaining_tokens = MAX_RESPONSE_TOKENS - int(student["phase_one"]["token_count"])
            probes = run_bbox_probes(
                engine,
                continuation_prompt_ids=continuation_prompt_ids,
                image=image,
                reasoning=reasoning,
                seed=DEFAULT_SEED,
                remaining_tokens=remaining_tokens,
            )
            routed = route_response(response, instances, target_ann_id, [probe["response"] for probe in probes])
        return _student_route_receipt(
            source_index=source_index,
            record=record,
            student=student,
            route=routed,
            probes=probes,
        )
    except Exception as error:
        return {
            "schema_version": 1,
            "record_type": "student_route",
            "sample_id": _sample_id(source_index),
            "source_index": source_index,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "routing_record": dict(record),
            "target_bbox": target_bbox_from_record(record),
            "student": None,
            "route": {"route": "uncertain", "error": "student_audit_error"},
            "probes": [],
        }


def _load_rgb_image(image_path: str | Path) -> Any:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - production dependency check
        raise RuntimeError("run_diagnostics.py requires Pillow for image loading") from error
    with Image.open(image_path) as opened:
        return opened.convert("RGB")


def _parse_bbox_json(raw_text: str) -> tuple[float, float, float, float]:
    try:
        value = json.loads(raw_text.strip())
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Teacher bbox is not JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"bbox"}:
        raise ValueError('Teacher bbox must be exactly {"bbox": [...]}')
    return validate_prediction_bbox(value["bbox"])


def _teacher_chat(engine: Any, messages: list[dict[str, Any]], sampling_params: Any) -> tuple[str, dict[str, Any]]:
    outputs = engine.chat(
        messages,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    request, completion = _first_completion(outputs)
    receipt = _completion_receipt(request, completion)
    receipt["chat_template_kwargs"] = {"enable_thinking": False}
    return str(getattr(completion, "text", "")), receipt


def _hf_template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert vLLM's ``image_pil`` chat parts to HF template image markers.

    The standalone vLLM chat API consumes PIL images through ``image_pil``.
    In contrast, the Qwen HF template used for a token-ID continuation only
    accepts ``type: image`` placeholders; the actual two PIL images are then
    supplied to ``engine.generate`` as multimodal data below.
    """

    template_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, str | bytes):
            raise ValueError("Teacher chat message content must be a sequence of parts")
        template_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise ValueError("Teacher chat message part must be a mapping")
            if part.get("type") == "image_pil":
                template_content.append({"type": "image"})
            else:
                template_content.append(dict(part))
        template_message = dict(message)
        template_message["content"] = template_content
        template_messages.append(template_message)
    return template_messages


def _chat_template_token_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Render the two-image Qwen prompt through its thinking-enabled template."""

    rendered = tokenizer.apply_chat_template(
        _hf_template_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if isinstance(rendered, Mapping):
        token_ids = rendered.get("input_ids")
    else:
        token_ids = getattr(rendered, "input_ids", rendered)
    if token_ids is None:
        raise ValueError("Teacher tokenizer did not return input_ids")
    if token_ids and isinstance(token_ids[0], Sequence) and not isinstance(token_ids[0], str | bytes):
        if len(token_ids) != 1:
            raise ValueError("Teacher chat template unexpectedly returned a batch")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _teacher_referent_continuation(
    engine: Any,
    *,
    messages: list[dict[str, Any]],
    original_image: Any,
    crop_image: Any,
    protocol_reasoning_prefix: str,
) -> tuple[str, dict[str, Any]]:
    """Make one structured Teacher call from the original reasoning prefix.

    Qwen's chat template supplies ``<think>\n``.  The suffix from the actual
    Student protocol prefix is appended after that token sequence, so the
    single Teacher completion begins precisely where the correction sentence
    belongs.  The structured tail itself owns ``</think><answer>...``.
    """

    template_prefix = "<think>\n"
    if not protocol_reasoning_prefix.startswith(template_prefix):
        raise ValueError("Student reasoning prefix does not begin with Qwen's <think> newline")
    reasoning_suffix = protocol_reasoning_prefix[len(template_prefix) :]
    tokenizer = engine.get_tokenizer()
    prompt_token_ids = _chat_template_token_ids(tokenizer, messages)
    continuation_prompt_ids = prompt_token_ids + _encode_no_special_tokens(tokenizer, reasoning_suffix)
    outputs = engine.generate(
        {
            "prompt_token_ids": continuation_prompt_ids,
            "multi_modal_data": {"image": [original_image, crop_image]},
        },
        sampling_params=teacher_referent_sampling_params(),
        use_tqdm=False,
    )
    request, completion = _first_completion(outputs)
    receipt = _completion_receipt(request, completion)
    receipt.update(
        {
            "kind": "referent_correction_bbox",
            "prompt_token_count": len(continuation_prompt_ids),
            "structured_regex": REFERENT_TAIL_REGEX,
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_prefix_token_count": len(continuation_prompt_ids) - len(prompt_token_ids),
        }
    )
    return str(getattr(completion, "text", "")), receipt


def _teacher_verification(
    *,
    mode: str,
    teacher_target: str,
    target_bbox: Sequence[Any],
    student_bbox: Sequence[Any] | None,
    instances: Sequence[Any],
    target_ann_id: Any,
) -> dict[str, Any]:
    result = verify_teacher_target(
        mode=mode,
        teacher_output=teacher_target,
        target_bbox=target_bbox,
        student_bbox=student_bbox,
        instances=instances,
        target_ann_id=target_ann_id,
    )
    return result.to_dict()


def _teacher_error_verification(
    *,
    mode: str,
    error: Exception,
    target_bbox: Sequence[Any],
    student_bbox: Sequence[Any] | None,
    instances: Sequence[Any],
    target_ann_id: Any,
) -> dict[str, Any]:
    # Still invoke the common verifier with every required grounding argument;
    # its malformed-target result is retained alongside the more specific audit
    # reason below.
    verification = _teacher_verification(
        mode=mode,
        teacher_target="",
        target_bbox=target_bbox,
        student_bbox=student_bbox,
        instances=instances,
        target_ann_id=target_ann_id,
    )
    verification["reason"] = f"{type(error).__name__}: {error}"
    verification["accepted"] = False
    return verification


def _teacher_bbox_match(bbox: Sequence[Any] | None, instances: Sequence[Any]) -> dict[str, Any] | None:
    if bbox is None:
        return None
    try:
        return match_prediction(bbox, instances).to_dict()
    except (TypeError, ValueError) as error:
        return {"status": "invalid", "error": str(error)}


def _teacher_base_receipt(
    route_receipt: Mapping[str, Any], *, mode: str, crop_mapping: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "teacher_audit",
        "sample_id": route_receipt["sample_id"],
        "source_index": route_receipt["source_index"],
        "status": "ok",
        "route": route_receipt["route"]["route"],
        "mode": mode,
        "crop": dict(crop_mapping),
    }


def audit_teacher_example(
    engine: Any,
    *,
    route_receipt: Mapping[str, Any],
    image: Any,
    use_gt_overlay: bool = False,
) -> dict[str, Any]:
    """Call the frozen Teacher for one repair route and verify its target."""

    record = _as_mapping(route_receipt["routing_record"], name="route receipt routing_record")
    route_name = str(_as_mapping(route_receipt["route"], name="route receipt route")["route"])
    student = _as_mapping(route_receipt["student"], name="route receipt student")
    student_response = str(student["response"])
    parsed = parse_response(student_response)
    if not parsed.format_valid or parsed.bbox is None or parsed.reasoning is None:
        raise ValueError("Teacher route requires a valid original Student response")
    target_bbox = target_bbox_from_record(record)
    instances = record["instances"]
    target_ann_id = record["target_ann_id"]

    teacher_original = gt_overlay(image, target_bbox) if use_gt_overlay else image
    crop_result = crop_gt_bbox(image, target_bbox, expansion=1.5)
    crop_mapping = crop_result.mapping.to_dict()

    if route_name in {"localization", "localization_recoverable"}:
        instruction = build_localization_teacher_prompt(parsed.reasoning, query=str(record["expression"]))
        messages = build_teacher_messages(teacher_original, crop_result.crop, instruction)
        raw_text, bbox_call = _teacher_chat(engine, messages, teacher_bbox_sampling_params())
        bbox_call["kind"] = "localization_bbox"
        bbox: tuple[float, float, float, float] | None = None
        teacher_target = ""
        try:
            bbox = _parse_bbox_json(raw_text)
            teacher_target = build_localization_teacher_target(parsed.reasoning, bbox)
            verification = _teacher_verification(
                mode="localization",
                teacher_target=teacher_target,
                target_bbox=target_bbox,
                student_bbox=parsed.bbox,
                instances=instances,
                target_ann_id=target_ann_id,
            )
        except (TypeError, ValueError) as error:
            verification = _teacher_error_verification(
                mode="localization",
                error=error,
                target_bbox=target_bbox,
                student_bbox=parsed.bbox,
                instances=instances,
                target_ann_id=target_ann_id,
            )
        receipt = _teacher_base_receipt(route_receipt, mode="localization", crop_mapping=crop_mapping)
        receipt.update(
            {
                "raw_text": raw_text,
                "bbox": None if bbox is None else list(bbox),
                "iou": verification.get("iou"),
                "match": _teacher_bbox_match(bbox, instances),
                "accepted": bool(verification["accepted"]),
                "reason": verification["reason"],
                "verification": verification,
                "teacher_target": teacher_target or None,
                "teacher_calls": [bbox_call],
            }
        )
        return receipt

    if route_name != "referent_unrecoverable":
        raise ValueError(f"route {route_name!r} does not call the Teacher")

    if parsed.reasoning_prefix is None:
        raise ValueError("referent route is missing the original reasoning prefix")
    continuation_instruction = (
        "Continue the assistant reasoning already present in the prompt. Emit exactly one short correction "
        "sentence, then </think> and exactly one canonical bbox JSON answer. Do not repeat the original "
        "reasoning and do not add any other tags.\n"
        f"Referring expression: {record['expression']}\n"
        "The original reasoning prefix ends immediately before this completion."
    )
    continuation_messages = build_teacher_messages(teacher_original, crop_result.crop, continuation_instruction)
    raw_text, referent_call = _teacher_referent_continuation(
        engine,
        messages=continuation_messages,
        original_image=teacher_original,
        crop_image=crop_result.crop,
        protocol_reasoning_prefix=parsed.reasoning_prefix,
    )
    bbox: tuple[float, float, float, float] | None = None
    teacher_target = ""
    correction: str | None = None
    try:
        teacher_target = parsed.reasoning_prefix + raw_text
        parsed_teacher_target = parse_teacher_target(
            teacher_target,
            mode="referent",
            original_response=parsed.reasoning_prefix,
        )
        if parsed_teacher_target is None:
            raise ValueError("Teacher correction+bbox continuation violates the referent target contract")
        bbox = parsed_teacher_target.bbox
        correction = parsed_teacher_target.correction_sentence
        verification = _teacher_verification(
            mode="referent",
            teacher_target=teacher_target,
            target_bbox=target_bbox,
            student_bbox=None,
            instances=instances,
            target_ann_id=target_ann_id,
        )
    except (TypeError, ValueError) as error:
        verification = _teacher_error_verification(
            mode="referent",
            error=error,
            target_bbox=target_bbox,
            student_bbox=None,
            instances=instances,
            target_ann_id=target_ann_id,
        )
    receipt = _teacher_base_receipt(route_receipt, mode="referent", crop_mapping=crop_mapping)
    receipt.update(
        {
            "raw_text": raw_text,
            "correction": correction,
            "bbox": None if bbox is None else list(bbox),
            "iou": verification.get("iou"),
            "match": _teacher_bbox_match(bbox, instances),
            "accepted": bool(verification["accepted"]),
            "reason": verification["reason"],
            "verification": verification,
            "teacher_target": teacher_target or None,
            "teacher_calls": [referent_call],
        }
    )
    return receipt


def _route_requires_teacher(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("record_type") != "student_route" or receipt.get("status") != "ok":
        return False
    route = receipt.get("route")
    return isinstance(route, Mapping) and route.get("route") in {
        "localization",
        "localization_recoverable",
        "referent_unrecoverable",
    }


def _configure_cuda_device(device: str) -> None:
    if not device:
        raise ValueError("device must be a CUDA-visible device index or mask")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def create_vllm_engine(model_path: Path) -> Any:
    """Create one inference-only vLLM 0.24 engine for a local Qwen3.5 model."""

    try:
        import vllm
        from vllm import LLM
    except ImportError as error:  # pragma: no cover - production dependency check
        raise RuntimeError("run_diagnostics.py requires local vLLM 0.24") from error
    if not str(vllm.__version__).startswith("0.24."):
        raise RuntimeError(f"expected vLLM 0.24.x, found {vllm.__version__}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    return LLM(model=str(model_path), **VLLM_ENGINE_KWARGS)


def clear_gpu_cache() -> None:
    """Release cached CUDA allocations after one model is deleted."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in production
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def unload_engine(engine: Any) -> None:
    """Drop one vLLM engine, collect it, then clear CUDA cache before the next model."""

    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        llm_engine = getattr(engine, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del engine
    gc.collect()
    clear_gpu_cache()


def _no_teacher_route_summary(route_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    for receipt in route_records:
        route = receipt.get("route")
        if isinstance(route, Mapping):
            label = str(route.get("route", "unknown"))
            route_counts[label] = route_counts.get(label, 0) + 1
    return {
        "schema_version": 1,
        "record_type": "summary",
        "status": "no_teacher_routes",
        "teacher_loaded": False,
        "teacher_model": str(DEFAULT_TEACHER_MODEL),
        "requested_teacher_audits": 0,
        "route_counts": route_counts,
        "reason": "No completed Student route required a Teacher call; no Teacher receipt was fabricated.",
    }


def run_audit(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int = DEFAULT_LIMIT,
    device: str = DEFAULT_DEVICE,
    student_model: Path = DEFAULT_STUDENT_MODEL,
    teacher_model: Path = DEFAULT_TEACHER_MODEL,
    resume: bool = True,
    smoke_gt_overlay: bool = False,
) -> None:
    """Run/resume the complete Student-then-Teacher diagnostics audit."""

    if smoke_gt_overlay and not 0 < limit <= DEFAULT_LIMIT:
        raise ValueError(f"--smoke-gt-overlay requires 0 < --limit <= {DEFAULT_LIMIT}")
    records = read_routing_records(data_path, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    routes_path = output_dir / "diagnostic_routes.jsonl"
    teacher_path = output_dir / "teacher_audit.jsonl"
    if not resume:
        for path in (routes_path, teacher_path):
            if path.exists():
                path.unlink()

    existing_routes = _read_jsonl(routes_path)
    completed_routes = _completed_sample_ids(existing_routes, record_type="student_route")
    pending_student = [(index, record) for index, record in records if _sample_id(index) not in completed_routes]

    if pending_student:
        _configure_cuda_device(device)
        student_engine = create_vllm_engine(student_model)
        try:
            with routes_path.open("a", encoding="utf-8") as route_handle:
                for source_index, record in pending_student:
                    try:
                        image = _load_rgb_image(record["image_path"])
                        receipt = audit_student_example(
                            student_engine,
                            source_index=source_index,
                            record=record,
                            image=image,
                        )
                    except Exception as error:
                        receipt = {
                            "schema_version": 1,
                            "record_type": "student_route",
                            "sample_id": _sample_id(source_index),
                            "source_index": source_index,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                            "routing_record": record,
                            "target_bbox": target_bbox_from_record(record),
                            "student": None,
                            "route": {"route": "uncertain", "error": "image_or_student_error"},
                            "probes": [],
                        }
                    _write_jsonl_line(route_handle, receipt)
        finally:
            try:
                unload_engine(student_engine)
            finally:
                # ``del`` inside unload_engine only drops its local reference.
                # Clear this caller-owned reference before constructing Teacher.
                student_engine = None
    else:
        # A resume that has no Student work may still need Teacher-only work;
        # make that transition start from an empty CUDA cache as well.
        clear_gpu_cache()

    selected_sample_ids = {_sample_id(source_index) for source_index, _record in records}
    route_records = [
        record
        for record in _read_jsonl(routes_path)
        if record.get("record_type") == "student_route" and str(record.get("sample_id")) in selected_sample_ids
    ]
    existing_teacher = _read_jsonl(teacher_path)
    if resume:
        existing_teacher = _archive_failed_teacher_attempt(teacher_path, existing_teacher)
    completed_teacher = _completed_sample_ids(
        existing_teacher,
        record_type="teacher_audit",
        successful_only=True,
    )
    requested_teacher = [record for record in route_records if _route_requires_teacher(record)]
    pending_teacher = [record for record in requested_teacher if str(record["sample_id"]) not in completed_teacher]

    if not requested_teacher:
        has_summary = any(record.get("status") == "no_teacher_routes" for record in existing_teacher)
        if not has_summary:
            with teacher_path.open("a", encoding="utf-8") as teacher_handle:
                _write_jsonl_line(teacher_handle, _no_teacher_route_summary(route_records))
        return
    if not pending_teacher:
        return

    # This occurs only after the Student engine has been deleted and its CUDA
    # allocations collected above.  No teacher call is made for any other
    # route, preserving the routed protocol rather than manufacturing labels.
    _configure_cuda_device(device)
    teacher_engine = create_vllm_engine(teacher_model)
    try:
        with teacher_path.open("a", encoding="utf-8") as teacher_handle:
            for route_receipt in pending_teacher:
                try:
                    record = _as_mapping(route_receipt["routing_record"], name="route receipt routing_record")
                    image = _load_rgb_image(str(record["image_path"]))
                    teacher_receipt = audit_teacher_example(
                        teacher_engine,
                        route_receipt=route_receipt,
                        image=image,
                        use_gt_overlay=smoke_gt_overlay,
                    )
                except Exception as error:
                    teacher_receipt = {
                        "schema_version": 1,
                        "record_type": "teacher_audit",
                        "sample_id": route_receipt["sample_id"],
                        "source_index": route_receipt["source_index"],
                        "status": "error",
                        "route": _as_mapping(route_receipt["route"], name="route receipt route").get("route"),
                        "raw_text": None,
                        "bbox": None,
                        "iou": None,
                        "match": None,
                        "accepted": False,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                _write_jsonl_line(teacher_handle, teacher_receipt)
    finally:
        try:
            unload_engine(teacher_engine)
        finally:
            teacher_engine = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA_VISIBLE_DEVICES value (default: GPU 5)")
    parser.add_argument("--student-model", type=Path, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="replace existing diagnostics JSONL files instead of resuming completed sample receipts",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--smoke-gt-overlay",
        action="store_true",
        help=f"draw the GT box for an engineering smoke run with 0 < --limit <= {DEFAULT_LIMIT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        limit=args.limit,
        device=str(args.device),
        student_model=args.student_model.resolve(),
        teacher_model=args.teacher_model.resolve(),
        resume=bool(args.resume),
        smoke_gt_overlay=bool(args.smoke_gt_overlay),
    )


if __name__ == "__main__":
    # Keep process-spawn model imports behind this guard.  In particular, a
    # vLLM worker must never re-execute an interactive/stdin invocation.
    main()
