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

"""Routed Grounding Repair AgentLoop.

The loop deliberately keeps the normal Student rollout as the first request.
Only after that request has been parsed and routed can it create a repair
trajectory.  A repair is *not* an imagined Student rollout: its preserved
Student prefix has ``response_mask=0`` and only the accepted Teacher target
suffix is labelled.

``AgentLoopWorker`` normally only gives an AgentLoop the Student server.  The
small manager/worker pair at the end of this module passes the already-created
Teacher clients through the existing ``teacher_client`` argument.  This keeps
teacher serving on verl's teacher resource pool and does not alter FSDP,
rollout-model, or model-head code.

The teacher and Student must use compatible token/vision protocols.  Teacher
generation is token-in/token-out, so the loop renders the Teacher request with
the configured Student processor and sends its ids to the Teacher server.  If
that contract cannot be met, the loop records the error and returns the
original Standard-OPD rollout rather than manufacturing a repair target.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import hydra
import ray
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    ToolListWrap,
    _agent_loop_registry,
    register,
)
from verl.experimental.routed_grounding.router import (
    EXPECTED_PROBE_COUNT,
    ParseResult,
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
    render_bbox_json,
    verify_teacher_target,
)
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_attr
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ``structured_outputs`` constrains only the continuation from the already
# present ``<answer>{\"bbox\":[`` prefix.  Natural reasoning (the Student
# rollout and the Teacher correction sentence) intentionally has no constraint.
# Keep every constrained coordinate within the closed normalized range.  The
# 1000 branch permits only a fractional zero suffix (``1000.0``), not 1000.5.
_NUMBER_REGEX = r"(?:1000(?:\.0+)?|(?:0|[1-9][0-9]{0,2})(?:\.[0-9]+)?)"
BBOX_TAIL_REGEX = (
    rf"\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}"
    rf"\s*,\s*{_NUMBER_REGEX}\s*\]\}}\s*</answer>"
)
BBOX_JSON_REGEX = (
    rf"\s*\{{\s*\"bbox\"\s*:\s*\[\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}"
    rf"\s*,\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}\s*\]\s*\}}\s*"
)
# One generated correction sentence followed by the fixed completion tags and
# canonical bbox JSON.  This is used only for a Teacher continuation: natural
# Student reasoning remains unconstrained.
REFERENT_TAIL_REGEX = (
    rf"\s*[^\n<>.!?]{{1,240}}[.!?]\s*</think>\s*<answer>\s*"
    rf"\{{\s*\"bbox\"\s*:\s*\[\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}"
    rf"\s*,\s*{_NUMBER_REGEX}\s*,\s*{_NUMBER_REGEX}\s*\]\s*\}}\s*</answer>\s*"
)
STUDENT_BBOX_PREFIX = '<answer>{"bbox":['
# The opt-in forced-close path must leave room for a complete bbox phase even
# when reasoning reaches its hard generation limit.  This is deliberately a
# protocol constant: the routed-grounding pilot fixes that phase at 48 tokens,
# independent of a caller accidentally raising ``student_bbox_max_tokens``.
_STUDENT_BBOX_TOKEN_RESERVE = 48
REPAIR_ARMS = frozenset({"standard_opd", "generic_sopd", "routed_repair"})

STANDARD_REPAIR_KIND = 0
LOCALIZATION_REPAIR_KIND = 1
REFERENT_REPAIR_KIND = 2


def _jsonable(value: Any) -> Any:
    """Convert receipts to values safe for ``extra_fields`` / JSON dumping."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode without tokenizer cleanup, preserving literal protocol markers."""

    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        # Lightweight test tokenizers and older HF tokenizers do not expose the
        # cleanup keyword.
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode a Teacher *target* suffix with the Student tokenizer.

    This is intentionally not used to reconstruct a Student prefix.  Prefixes
    always reuse Student-generated token ids; only a new Teacher target suffix
    must be tokenized for Student teacher-forcing.
    """

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def _protocol_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    """Drop only terminal EOS/padding while preserving internal special tokens."""

    protocol_ids = list(token_ids)
    terminal_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None))
        if token_id is not None
    }
    while protocol_ids and protocol_ids[-1] in terminal_ids:
        protocol_ids.pop()
    return protocol_ids


def _decode_protocol_generation(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return _decode(tokenizer, _protocol_token_ids(tokenizer, token_ids))


def _student_protocol_response(
    tokenizer: Any, prompt_ids: Sequence[int], response_ids: Sequence[int]
) -> tuple[str, str, str]:
    """Return token text plus a parser-only thinking prefix when Qwen owns it.

    Qwen3.5 with ``enable_thinking=True`` renders ``<think>\n`` into the
    generation prompt.  The server completion therefore starts *inside* the
    thinking block.  Keeping the opening tag out of ``response_ids`` is
    essential: adding it again would duplicate prompt tokens and make the
    rollout unreplayable.  Routing, however, deliberately uses the canonical
    full response grammar, so it sees a virtual prefix only when the decoded
    prompt ends at an open think tag.
    """

    response_text = _decode(tokenizer, response_ids)
    # vLLM may include the model's terminal EOS token in ``token_ids`` even
    # though it is not user-visible response text.  Preserve it in the raw
    # token receipt, but exclude trailing EOS/padding from strict protocol
    # parsing.  Internal special tokens remain visible and still fail closed.
    protocol_text = _decode_protocol_generation(tokenizer, response_ids)
    if protocol_text.startswith("<think>"):
        return response_text, protocol_text, ""
    prompt_text = _decode(tokenizer, prompt_ids)
    if prompt_text.rstrip().endswith("<think>"):
        implicit_prefix = "<think>"
        return response_text, implicit_prefix + protocol_text, implicit_prefix
    return response_text, protocol_text, ""


def _token_prefix_from_protocol_prefix(protocol_prefix: str, implicit_prefix: str) -> str:
    """Remove a Qwen prompt-owned virtual prefix before locating response ids."""

    if not implicit_prefix:
        return protocol_prefix
    if not protocol_prefix.startswith(implicit_prefix):
        raise ValueError("protocol prefix does not begin with the prompt-owned think tag")
    return protocol_prefix[len(implicit_prefix) :]


def _forced_suffix_token_ids(generated_ids: Sequence[int], forced_ids: Sequence[int]) -> list[int]:
    """Return only the missing suffix of a deterministic token sequence.

    A length-limited generation can end after a tokenizer has already emitted
    the first token(s) of ``</think>``.  Appending the whole marker would then
    produce a duplicated, invalid delimiter.  Matching token ids keeps the
    inserted continuation exact without decoding and re-encoding Student
    output.
    """

    forced = list(forced_ids)
    if not forced:
        raise ValueError("tokenizer produced no tokens for deterministic reasoning close")
    generated = list(generated_ids)
    max_prefix = min(len(generated), len(forced) - 1)
    for prefix_len in range(max_prefix, 0, -1):
        if generated[-prefix_len:] == forced[:prefix_len]:
            return forced[prefix_len:]
    return forced


def _generation_budget_exhausted(output: TokenOutput, generated_token_count: int, max_tokens: int) -> bool:
    """Recognize vLLM length termination while remaining mock-backend friendly."""

    if generated_token_count >= max_tokens:
        return True
    stop_reason = str(output.stop_reason or "").strip().lower()
    return stop_reason in {"length", "max_tokens", "max_token", "token_limit", "length_limit"}


def _ends_with_terminal_token(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    terminal_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None))
        if token_id is not None
    }
    return bool(token_ids) and int(token_ids[-1]) in terminal_ids


def find_exact_prefix_token_count(tokenizer: Any, token_ids: Sequence[int], prefix: str) -> int | None:
    """Locate a text boundary without re-encoding generated Student tokens.

    A missing exact token boundary is unsafe for continuation, so callers must
    fall back to Standard OPD rather than approximate a prefix with a decoded
    and re-encoded string.
    """

    for count in range(len(token_ids) + 1):
        if _decode(tokenizer, token_ids[:count]) == prefix:
            return count
    return None


def _structured_outputs_regex(regex: str) -> Any:
    """Build the vLLM 0.24 per-request structured-output value.

    ``SamplingParams`` expects ``StructuredOutputsParams``, not the old
    ``guided_*`` options.  The import is intentionally lazy so CPU-only parser
    tests do not need the vLLM wheel; a real repair fails closed when vLLM is
    absent from the agent-worker environment.
    """

    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:  # pragma: no cover - exercised by production env check
        raise RuntimeError("Routed Grounding Repair requires vLLM 0.24 StructuredOutputsParams") from error
    return StructuredOutputsParams(regex=regex)


def bbox_sampling_params(
    rollout_sampling_params: Mapping[str, Any],
    *,
    max_tokens: int,
    seed: int | None,
    regex: str,
    structured_outputs_factory: Callable[[str], Any] = _structured_outputs_regex,
) -> dict[str, Any]:
    """Copy Student decoding settings and constrain just a bbox generation."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    result = dict(rollout_sampling_params)
    # The four settings are copied explicitly.  Keeping the rest is useful for
    # backend-specific stable options, while these are the scientific decoding
    # controls that probes are required to share with the Student rollout.
    for key in ("temperature", "top_p", "top_k"):
        if key not in rollout_sampling_params:
            raise ValueError(f"rollout sampling params are missing {key!r}")
        result[key] = rollout_sampling_params[key]
    result["max_tokens"] = int(max_tokens)
    result.pop("max_new_tokens", None)
    # A reasoning-stage stop sequence must not leak into a bbox-tail request.
    for key in ("stop", "stop_token_ids", "include_stop_str_in_output"):
        result.pop(key, None)
    result["logprobs"] = False
    result["structured_outputs"] = structured_outputs_factory(regex)
    if seed is not None:
        result["seed"] = int(seed)
    return result


def _parse_bbox_json(text: str) -> tuple[float, float, float, float]:
    """Accept only the canonical Teacher bbox-only response."""

    try:
        value = json.loads(text.strip())
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"teacher bbox is not JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"bbox"}:
        raise ValueError('teacher bbox must be exactly {"bbox": [...]}')
    return validate_prediction_bbox(value["bbox"])


def _bbox_tail_from_canonical_json(bbox: Sequence[float]) -> str:
    encoded = render_bbox_json(bbox)
    opener = '{"bbox":['
    if not encoded.startswith(opener):  # defensive: keep target construction exact
        raise RuntimeError("canonical bbox renderer changed its expected prefix")
    return encoded[len(opener) :] + "</answer>"


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is required and must be a mapping")
    return value


@dataclass(frozen=True)
class RoutedGroundingInputs:
    """Validated annotation fields needed by routing and Teacher verification."""

    target_bbox: tuple[float, float, float, float]
    target_ann_id: Any
    instances: tuple[Mapping[str, Any], ...]
    query: str | None


def extract_routed_grounding_inputs(kwargs: Mapping[str, Any]) -> RoutedGroundingInputs:
    """Read the frozen manifest fields forwarded through ``extra_info``.

    ``prepare_verl_data.py`` retains both the normalized target box and the
    original COCO-style instance list in ``extra_info``.  The reward-model
    ground truth is an allowed fallback for the normalized target only.
    """

    extra_info = _as_mapping(kwargs.get("extra_info"), name="extra_info")
    routing_record: Mapping[str, Any] = {}
    routing_record_json = extra_info.get("routing_record_json")
    if routing_record_json is not None:
        if not isinstance(routing_record_json, str):
            raise ValueError("extra_info.routing_record_json must be a JSON string")
        try:
            routing_record = _as_mapping(json.loads(routing_record_json), name="extra_info.routing_record_json")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid extra_info.routing_record_json: {error}") from error
    reward_model = kwargs.get("reward_model")
    target_bbox = extra_info.get("target_bbox_normalized")
    if target_bbox is None:
        target_bbox = routing_record.get("target_bbox_normalized")
    if target_bbox is None and routing_record.get("target_bbox") is not None:
        raw_target = routing_record["target_bbox"]
        width = routing_record.get("image_width", extra_info.get("image_width"))
        height = routing_record.get("image_height", extra_info.get("image_height"))
        try:
            x, y, width_box, height_box = (float(value) for value in raw_target)
            width = float(width)
            height = float(height)
        except (TypeError, ValueError) as error:
            raise ValueError("routing_record target_bbox/image dimensions are invalid") from error
        if width <= 0 or height <= 0:
            raise ValueError("routing_record image dimensions must be positive")
        target_bbox = [
            1000.0 * x / width,
            1000.0 * y / height,
            1000.0 * (x + width_box) / width,
            1000.0 * (y + height_box) / height,
        ]
    if target_bbox is None and isinstance(reward_model, Mapping):
        target_bbox = reward_model.get("ground_truth")
    if target_bbox is None:
        raise ValueError("target_bbox_normalized (or reward_model.ground_truth) is required")
    target_ann_id = routing_record.get("target_ann_id", extra_info.get("target_ann_id"))
    if target_ann_id is None:
        raise ValueError("extra_info.target_ann_id is required")
    raw_instances = routing_record.get("instances", extra_info.get("instances"))
    if not isinstance(raw_instances, Sequence) or isinstance(raw_instances, str | bytes):
        raise ValueError("extra_info.instances is required")
    instances = tuple(_as_mapping(instance, name="extra_info.instances item") for instance in raw_instances)
    if not instances:
        raise ValueError("extra_info.instances must not be empty")
    query = routing_record.get("expression", extra_info.get("expression"))
    return RoutedGroundingInputs(
        target_bbox=validate_prediction_bbox(target_bbox),
        target_ann_id=target_ann_id,
        instances=instances,
        query=str(query) if query is not None else None,
    )


@dataclass
class _StudentTrajectory:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    routed_experts: Any
    raw_response: str
    protocol_response: str
    implicit_protocol_prefix: str
    phase_receipts: list[dict[str, Any]]
    metrics: dict[str, Any]
    extra_fields: dict[str, Any]


@register("routed_grounding")
class RoutedGroundingAgentLoop(AgentLoopBase):
    """Routed Grounding Repair implementation for the frozen RefCOCOg pilot."""

    def __init__(
        self,
        *args: Any,
        teacher_client: Mapping[str, Any] | None = None,
        teacher_client_key: str | None = None,
        probe_seed_base: int = 3407,
        probe_max_tokens: int = 48,
        student_bbox_max_tokens: int = 48,
        teacher_bbox_max_tokens: int = 48,
        teacher_correction_max_tokens: int = 64,
        teacher_temperature: float = 0.0,
        teacher_top_p: float = 1.0,
        teacher_top_k: int = 0,
        teacher_view: str = "gt_crop",
        arm: str = "routed_repair",
        standard_opd_probe_diagnostics: bool = False,
        student_force_reasoning_close: bool = False,
        structured_outputs_factory: Callable[[str], Any] = _structured_outputs_regex,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_client = dict(teacher_client or {})
        self.teacher_client_key = teacher_client_key
        self.probe_seed_base = int(probe_seed_base)
        self.probe_max_tokens = int(probe_max_tokens)
        self.student_bbox_max_tokens = int(student_bbox_max_tokens)
        self.teacher_bbox_max_tokens = int(teacher_bbox_max_tokens)
        self.teacher_correction_max_tokens = int(teacher_correction_max_tokens)
        self.teacher_temperature = float(teacher_temperature)
        self.teacher_top_p = float(teacher_top_p)
        self.teacher_top_k = int(teacher_top_k)
        self.teacher_view = str(teacher_view)
        self.arm = str(arm)
        self.standard_opd_probe_diagnostics = bool(standard_opd_probe_diagnostics)
        self.student_force_reasoning_close = bool(student_force_reasoning_close)
        self.structured_outputs_factory = structured_outputs_factory
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        if (
            min(
                self.probe_max_tokens,
                self.student_bbox_max_tokens,
                self.teacher_bbox_max_tokens,
                self.teacher_correction_max_tokens,
            )
            <= 0
        ):
            raise ValueError("Routed Grounding Repair generation budgets must be positive")
        if self.arm not in REPAIR_ARMS:
            raise ValueError(f"arm must be one of {sorted(REPAIR_ARMS)}, got {self.arm!r}")
        if self.teacher_view not in {"gt_crop", "gt_overlay"}:
            raise ValueError("teacher_view must be 'gt_crop' or 'gt_overlay'")

    def _teacher(self) -> Any:
        if not self.teacher_client:
            raise RuntimeError("no Teacher client was injected into RoutedGroundingAgentLoop")
        if self.teacher_client_key is not None:
            try:
                return self.teacher_client[self.teacher_client_key]
            except KeyError as error:
                raise RuntimeError(
                    f"teacher_client_key={self.teacher_client_key!r} is not configured; "
                    f"available={sorted(self.teacher_client)}"
                ) from error
        if len(self.teacher_client) != 1:
            raise RuntimeError(
                "multiple Teacher clients are configured; set teacher_client_key explicitly "
                f"(available={sorted(self.teacher_client)})"
            )
        return next(iter(self.teacher_client.values()))

    async def _student_rollout(
        self,
        sampling_params: Mapping[str, Any],
        *,
        prompt_ids: list[int],
        images: list[Any] | None,
        mm_processor_kwargs: dict[str, Any],
        priority: int,
    ) -> _StudentTrajectory:
        """Generate reasoning then a grammar-constrained bbox tail.

        The answer opener is a deterministic continuation token span, not a
        sampled action.  It is retained in the response stream with a zero
        mask so ``prompt + response`` exactly recreates the second vLLM request.
        When explicitly enabled, the first request reserves room for the close
        marker, answer opener, and the fixed 48-token bbox phase.  A
        length-exhausted reasoning request may then receive the missing suffix
        of ``</think>`` as a synthetic, zero-mask continuation.  The default
        path keeps the legacy staged budgets and only accepts a natural close.
        """

        metrics: dict[str, Any] = {}
        bbox_prefix_ids = _encode(self.tokenizer, STUDENT_BBOX_PREFIX)
        reasoning_close_ids = _encode(self.tokenizer, "</think>")
        force_reasoning_close = bool(getattr(self, "student_force_reasoning_close", False))
        if force_reasoning_close:
            reasoning_budget = (
                self.response_length - len(reasoning_close_ids) - len(bbox_prefix_ids) - _STUDENT_BBOX_TOKEN_RESERVE
            )
        else:
            reasoning_budget = self.response_length
        if force_reasoning_close and reasoning_budget <= 0:
            raise ValueError(
                "routed grounding response_length cannot reserve </think>, the bbox prefix, and 48 bbox tokens"
            )
        reasoning_params = dict(sampling_params)
        reasoning_params.pop("structured_outputs", None)
        reasoning_params.pop("max_new_tokens", None)
        reasoning_params["max_tokens"] = reasoning_budget
        reasoning_params["stop"] = ["</think>"]
        reasoning_params["include_stop_str_in_output"] = True
        reasoning_request_id = (
            f"routed-grounding-reasoning-{priority}"
            if getattr(self.rollout_config, "full_determinism", False)
            else uuid4().hex
        )
        with simple_timer("generate_sequences", metrics):
            reasoning_output: TokenOutput = await self.server_manager.generate(
                request_id=reasoning_request_id,
                prompt_ids=prompt_ids,
                sampling_params=reasoning_params,
                image_data=images,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        reasoning_merge, reasoning_mask, _reasoning_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            reasoning_output.token_ids,
            [],
            [] if reasoning_output.log_probs else None,
            assistant_logprobs=reasoning_output.log_probs if reasoning_output.log_probs else None,
        )
        reasoning_ids = reasoning_merge.token_ids[-len(reasoning_mask) :] if reasoning_mask else []
        merged_prompt_ids = reasoning_merge.token_ids[: len(reasoning_merge.token_ids) - len(reasoning_mask)]
        phase_receipts: list[dict[str, Any]] = [
            {
                "kind": "student_reasoning",
                "request_id": reasoning_request_id,
                "prompt_token_count": len(prompt_ids),
                "sampling": _jsonable(reasoning_params),
                "structured_regex": None,
                "token_ids": list(reasoning_output.token_ids),
                "token_count": len(reasoning_output.token_ids),
                "logprobs": _jsonable(reasoning_output.log_probs),
                "raw_output": _decode(self.tokenizer, reasoning_ids),
                "stop_reason": reasoning_output.stop_reason,
                "budget_mode": "forced_close" if force_reasoning_close else "legacy_staged",
                "total_response_budget": self.response_length,
                "reasoning_token_budget": reasoning_budget,
                "reserved_close_token_count": len(reasoning_close_ids) if force_reasoning_close else 0,
                "reserved_bbox_prefix_token_count": len(bbox_prefix_ids) if force_reasoning_close else 0,
                "reserved_bbox_token_count": _STUDENT_BBOX_TOKEN_RESERVE if force_reasoning_close else 0,
                "close_status": "invalid",
                "forced_close": False,
                "forced_close_token_count": 0,
                "forced_close_token_ids": [],
                "server_extra_fields": _jsonable(reasoning_output.extra_fields),
            }
        ]
        reasoning_raw, reasoning_protocol, implicit_prefix = _student_protocol_response(
            self.tokenizer, merged_prompt_ids, reasoning_ids
        )
        close_count = reasoning_protocol.count("</think>")
        if close_count == 1 and reasoning_protocol.rstrip().endswith("</think>"):
            phase_receipts[0]["close_status"] = "natural"
        elif close_count > 1:
            phase_receipts[0]["close_status"] = "invalid"
            phase_receipts[0]["phase_error"] = "multiple reasoning close markers"
        elif close_count == 1:
            phase_receipts[0]["close_status"] = "invalid"
            phase_receipts[0]["phase_error"] = "reasoning close marker is not terminal"
        elif _ends_with_terminal_token(self.tokenizer, reasoning_output.token_ids):
            phase_receipts[0]["close_status"] = "invalid"
            phase_receipts[0]["phase_error"] = "EOS/padding appeared before reasoning close marker"
        elif force_reasoning_close and _generation_budget_exhausted(
            reasoning_output, len(reasoning_ids), reasoning_budget
        ):
            synthetic_close_ids = _forced_suffix_token_ids(reasoning_ids, reasoning_close_ids)
            if len(reasoning_ids) + len(synthetic_close_ids) > self.response_length:
                phase_receipts[0]["close_status"] = "invalid"
                phase_receipts[0]["phase_error"] = "forced reasoning close exceeds response budget"
            else:
                reasoning_ids = reasoning_ids + synthetic_close_ids
                reasoning_mask = reasoning_mask + [0] * len(synthetic_close_ids)
                reasoning_raw, reasoning_protocol, implicit_prefix = _student_protocol_response(
                    self.tokenizer, merged_prompt_ids, reasoning_ids
                )
                # The synthetic suffix has no Student sampling distribution.
                # Keep it in the exact continuation prompt while masking it out
                # of every downstream loss and avoiding fabricated logprobs.
                phase_receipts[0].update(
                    {
                        "close_status": "forced",
                        "forced_close": True,
                        "forced_close_token_count": len(synthetic_close_ids),
                        "forced_close_token_ids": list(synthetic_close_ids),
                    }
                )
        else:
            phase_receipts[0]["close_status"] = "invalid"
            phase_receipts[0]["phase_error"] = (
                "reasoning phase did not terminate at </think> before its budget was exhausted"
            )

        phase_receipts[0]["raw_output"] = _decode(self.tokenizer, reasoning_ids)
        phase_receipts[0]["protocol_output"] = reasoning_protocol

        # There is no safe bbox phase without a closed reasoning block.  Return
        # this partial real rollout; routing will mark it invalid and retain
        # Standard OPD rather than synthesize the missing delimiter.
        if phase_receipts[0]["close_status"] == "invalid":
            phase_receipts[0]["raw_output"] = _decode(self.tokenizer, reasoning_ids)
            return _StudentTrajectory(
                prompt_ids=merged_prompt_ids,
                response_ids=reasoning_ids[: self.response_length],
                response_mask=reasoning_mask[: self.response_length],
                # The constrained bbox phase does not expose comparable rollout
                # logprobs.  Omit them for every trajectory so mixed staged and
                # fail-closed samples remain batchable; RayPPO recomputes the
                # Student old logprobs before the update.
                response_logprobs=None,
                routed_experts=(
                    reasoning_output.routed_experts[: len(merged_prompt_ids) + self.response_length]
                    if reasoning_output.routed_experts is not None
                    else None
                ),
                raw_response=reasoning_raw,
                protocol_response=reasoning_protocol,
                implicit_protocol_prefix=implicit_prefix,
                phase_receipts=phase_receipts,
                metrics=metrics,
                extra_fields=dict(reasoning_output.extra_fields),
            )

        remaining = self.response_length - len(reasoning_ids) - len(bbox_prefix_ids)
        if remaining <= 0:
            phase_receipts[0]["phase_error"] = "no response budget remains for bbox phase"
            return _StudentTrajectory(
                prompt_ids=merged_prompt_ids,
                response_ids=reasoning_ids[: self.response_length],
                response_mask=reasoning_mask[: self.response_length],
                response_logprobs=None,
                routed_experts=(
                    reasoning_output.routed_experts[: len(merged_prompt_ids) + self.response_length]
                    if reasoning_output.routed_experts is not None
                    else None
                ),
                raw_response=reasoning_raw,
                protocol_response=reasoning_protocol,
                implicit_protocol_prefix=implicit_prefix,
                phase_receipts=phase_receipts,
                metrics=metrics,
                extra_fields=dict(reasoning_output.extra_fields),
            )

        bbox_max_tokens = min(self.student_bbox_max_tokens, remaining)
        if force_reasoning_close:
            bbox_max_tokens = min(bbox_max_tokens, _STUDENT_BBOX_TOKEN_RESERVE)
        bbox_params = bbox_sampling_params(
            sampling_params,
            max_tokens=bbox_max_tokens,
            seed=None,
            regex=BBOX_TAIL_REGEX,
            structured_outputs_factory=self.structured_outputs_factory,
        )
        bbox_prompt_ids = merged_prompt_ids + reasoning_ids + bbox_prefix_ids
        bbox_request_id = (
            f"routed-grounding-bbox-{priority}"
            if getattr(self.rollout_config, "full_determinism", False)
            else uuid4().hex
        )
        with simple_timer("generate_bbox", metrics):
            bbox_output: TokenOutput = await self.server_manager.generate(
                request_id=bbox_request_id,
                prompt_ids=bbox_prompt_ids,
                sampling_params=bbox_params,
                image_data=images,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        bbox_merge, bbox_mask, _bbox_logprobs = await self.ct_merge_assistant_token(
            bbox_prompt_ids,
            bbox_output.token_ids,
            [],
            [] if bbox_output.log_probs else None,
            assistant_logprobs=bbox_output.log_probs if bbox_output.log_probs else None,
        )
        bbox_ids = bbox_merge.token_ids[-len(bbox_mask) :] if bbox_mask else []
        phase_receipts.append(
            {
                "kind": "student_bbox",
                "request_id": bbox_request_id,
                "prompt_token_count": len(bbox_prompt_ids),
                "coordinate_prefix": STUDENT_BBOX_PREFIX,
                "coordinate_prefix_token_count": len(bbox_prefix_ids),
                "sampling": _jsonable(bbox_params),
                "total_response_budget": self.response_length,
                "remaining_response_budget": remaining,
                "bbox_token_budget": bbox_max_tokens,
                "structured_regex": BBOX_TAIL_REGEX,
                "token_ids": list(bbox_output.token_ids),
                "token_count": len(bbox_output.token_ids),
                "logprobs": _jsonable(bbox_output.log_probs),
                "raw_output": _decode(self.tokenizer, bbox_ids),
                "stop_reason": bbox_output.stop_reason,
                "server_extra_fields": _jsonable(bbox_output.extra_fields),
            }
        )
        response_ids = reasoning_ids + bbox_prefix_ids + bbox_ids
        response_mask = reasoning_mask + [0] * len(bbox_prefix_ids) + bbox_mask
        raw_response, protocol_response, implicit_protocol_prefix = _student_protocol_response(
            self.tokenizer, merged_prompt_ids, response_ids
        )
        metrics["num_preempted"] = sum(output.num_preempted or 0 for output in (reasoning_output, bbox_output))
        return _StudentTrajectory(
            prompt_ids=merged_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            routed_experts=(
                reasoning_output.routed_experts[: len(merged_prompt_ids) + self.response_length]
                if reasoning_output.routed_experts is not None
                else None
            ),
            raw_response=raw_response,
            protocol_response=protocol_response,
            implicit_protocol_prefix=implicit_protocol_prefix,
            phase_receipts=phase_receipts,
            metrics=metrics,
            extra_fields=dict(reasoning_output.extra_fields),
        )

    async def _teacher_request(
        self,
        *,
        instruction: str,
        continuation_prefix: str = "",
        original_image: Any,
        teacher_view_image: Any,
        sampling_params: Mapping[str, Any],
        max_tokens: int,
        structured_regex: str | None,
        receipt_kind: str,
        priority: int,
    ) -> tuple[str, dict[str, Any]]:
        """Run one Teacher request with original image plus the selected view."""

        # The two placeholders are essential: their order is the same as the
        # two image objects passed to vLLM, original first and crop second.
        messages = [{"role": "user", "content": f"<image><image>\n{instruction}"}]
        teacher_images = [original_image, teacher_view_image]
        self._assert_mm_supported(True)
        prompt_ids = await self.ct_build_initial_tokens(messages, images=teacher_images)
        continuation_ids = _encode(self.tokenizer, continuation_prefix) if continuation_prefix else []
        # This append is intentional: the Teacher server sees the exact token
        # continuation used by the Student, not merely a textual quote of it in
        # the instruction.  Qwen prompt-owned ``<think>`` is omitted from this
        # span by the caller, avoiding a duplicate opening tag.
        prompt_ids = prompt_ids + continuation_ids
        teacher_params = dict(sampling_params)
        teacher_params["max_tokens"] = int(max_tokens)
        teacher_params.pop("max_new_tokens", None)
        teacher_params["logprobs"] = False
        teacher_params["temperature"] = self.teacher_temperature
        teacher_params["top_p"] = self.teacher_top_p
        teacher_params["top_k"] = self.teacher_top_k
        teacher_params.pop("structured_outputs", None)
        if structured_regex is not None:
            teacher_params["structured_outputs"] = self.structured_outputs_factory(structured_regex)
        request_id = uuid4().hex
        output: TokenOutput = await self._teacher().generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=teacher_params,
            image_data=teacher_images,
            mm_processor_kwargs=self._get_mm_processor_kwargs(),
            priority=priority,
        )
        raw_token_output = _decode(self.tokenizer, output.token_ids)
        protocol_output = _decode_protocol_generation(self.tokenizer, output.token_ids)
        receipt = {
            "kind": receipt_kind,
            "request_id": request_id,
            "prompt_token_count": len(prompt_ids),
            "image_count": len(teacher_images),
            "images": ["original", self.teacher_view],
            "continuation_prefix": continuation_prefix,
            "continuation_token_count": len(continuation_ids),
            "sampling": {
                key: _jsonable(teacher_params.get(key))
                for key in ("temperature", "top_p", "top_k", "max_tokens", "seed")
                if key in teacher_params
            },
            "structured_regex": structured_regex,
            "raw_output": raw_token_output,
            "protocol_output": protocol_output,
            "token_count": len(output.token_ids),
            "stop_reason": output.stop_reason,
            "server_extra_fields": _jsonable(output.extra_fields),
        }
        return protocol_output, receipt

    async def _probe_responses(
        self,
        *,
        student: _StudentTrajectory,
        parsed: ParseResult,
        sampling_params: Mapping[str, Any],
        images: list[Any] | None,
        mm_processor_kwargs: dict[str, Any],
        priority: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if parsed.bbox_prefix is None:
            raise ValueError("cannot probe without a bbox prefix")
        token_bbox_prefix = _token_prefix_from_protocol_prefix(parsed.bbox_prefix, student.implicit_protocol_prefix)
        boundary = find_exact_prefix_token_count(self.tokenizer, student.response_ids, token_bbox_prefix)
        if boundary is None:
            raise ValueError("bbox prefix does not end on a Student token boundary")
        prompt_ids = student.prompt_ids + student.response_ids[:boundary]
        responses: list[str] = []
        receipts: list[dict[str, Any]] = []
        for probe_index in range(EXPECTED_PROBE_COUNT):
            seed = self.probe_seed_base + probe_index
            params = bbox_sampling_params(
                sampling_params,
                max_tokens=self.probe_max_tokens,
                seed=seed,
                regex=BBOX_TAIL_REGEX,
                structured_outputs_factory=self.structured_outputs_factory,
            )
            request_id = uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=params,
                image_data=images,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
            raw_continuation = _decode(self.tokenizer, output.token_ids)
            continuation = _decode_protocol_generation(self.tokenizer, output.token_ids)
            full_response = parsed.bbox_prefix + continuation
            responses.append(full_response)
            receipts.append(
                {
                    "probe_index": probe_index,
                    "seed": seed,
                    "request_id": request_id,
                    "prefix_token_count": boundary,
                    "sampling": {
                        "temperature": params["temperature"],
                        "top_p": params["top_p"],
                        "top_k": params["top_k"],
                        "max_tokens": params["max_tokens"],
                    },
                    "structured_regex": BBOX_TAIL_REGEX,
                    "raw_continuation": raw_continuation,
                    "protocol_continuation": continuation,
                    "full_response": full_response,
                    "parse": _jsonable(parse_response(full_response)),
                    "server_extra_fields": _jsonable(output.extra_fields),
                }
            )
        return responses, receipts

    def _standard_output(
        self,
        student: _StudentTrajectory,
        *,
        multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any],
        receipt: dict[str, Any],
    ) -> AgentLoopOutput:
        extra_fields = dict(student.extra_fields)
        extra_fields.update(
            {
                "repair_kind": STANDARD_REPAIR_KIND,
                "routed_grounding_receipt": _jsonable(receipt),
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return AgentLoopOutput(
            prompt_ids=student.prompt_ids,
            response_ids=student.response_ids,
            response_mask=student.response_mask,
            response_logprobs=student.response_logprobs,
            routed_experts=student.routed_experts,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=student.metrics,
            extra_fields=extra_fields,
        )

    def _repair_output(
        self,
        student: _StudentTrajectory,
        *,
        prefix_text: str,
        suffix_text: str,
        repair_kind: int,
        multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any],
        receipt: dict[str, Any],
    ) -> AgentLoopOutput | None:
        boundary = find_exact_prefix_token_count(self.tokenizer, student.response_ids, prefix_text)
        if boundary is None:
            receipt["repair_rejected"] = "Student repair prefix does not end on a token boundary"
            return None
        suffix_ids = _encode(self.tokenizer, suffix_text)
        response_ids = student.response_ids[:boundary] + suffix_ids
        if not suffix_ids:
            receipt["repair_rejected"] = "Teacher suffix tokenization was empty"
            return None
        if len(response_ids) > self.response_length:
            receipt["repair_rejected"] = (
                f"repair response length {len(response_ids)} exceeds rollout.response_length={self.response_length}"
            )
            return None
        # Preserve real rollout logprobs only for the unlabelled Student prefix.
        # The Teacher suffix was never sampled by the Student, so zero is a
        # transparent placeholder; it is excluded from policy losses by the
        # repair path's teacher-forcing objective.
        response_logprobs = None
        if student.response_logprobs is not None:
            response_logprobs = student.response_logprobs[:boundary] + [0.0] * len(suffix_ids)
        extra_fields = dict(student.extra_fields)
        extra_fields.update(
            {
                "repair_kind": int(repair_kind),
                "routed_grounding_receipt": _jsonable(receipt),
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return AgentLoopOutput(
            prompt_ids=student.prompt_ids,
            response_ids=response_ids,
            response_mask=[0] * boundary + [1] * len(suffix_ids),
            response_logprobs=response_logprobs,
            # Existing records cover only the genuine Student request.  The
            # base AgentLoop alignment zero-pads the Teacher suffix and replay
            # masks it, rather than inventing MoE routing decisions.
            routed_experts=student.routed_experts,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=student.metrics,
            extra_fields=extra_fields,
        )

    async def _teacher_bbox(
        self,
        *,
        reasoning: str,
        inputs: RoutedGroundingInputs,
        original_image: Any,
        teacher_view_image: Any,
        sampling_params: Mapping[str, Any],
        priority: int,
        receipts: list[dict[str, Any]],
    ) -> tuple[float, float, float, float]:
        instruction = build_localization_teacher_prompt(reasoning, query=inputs.query)
        raw, receipt = await self._teacher_request(
            instruction=instruction,
            original_image=original_image,
            teacher_view_image=teacher_view_image,
            sampling_params=sampling_params,
            max_tokens=self.teacher_bbox_max_tokens,
            structured_regex=BBOX_JSON_REGEX,
            receipt_kind="localization_bbox",
            priority=priority,
        )
        receipts.append(receipt)
        bbox = _parse_bbox_json(raw)
        receipt["parsed_bbox"] = list(bbox)
        return bbox

    async def _teacher_correction_bbox_continuation(
        self,
        *,
        protocol_reasoning_prefix: str,
        token_reasoning_prefix: str,
        receipt_kind: str,
        inputs: RoutedGroundingInputs,
        original_image: Any,
        teacher_view_image: Any,
        sampling_params: Mapping[str, Any],
        priority: int,
        receipts: list[dict[str, Any]],
    ) -> tuple[str, tuple[float, float, float, float]]:
        instruction = (
            "Continue the assistant reasoning already present in the prompt. "
            "Emit exactly one short correction sentence, then </think> and one canonical bbox JSON answer. "
            "Do not repeat the reasoning prefix and do not add any other tags.\n"
            f"Referring expression: {inputs.query or ''}\n"
            "The token prompt already ends at the original reasoning prefix."
        )
        raw, receipt = await self._teacher_request(
            instruction=instruction,
            continuation_prefix=token_reasoning_prefix,
            original_image=original_image,
            teacher_view_image=teacher_view_image,
            sampling_params=sampling_params,
            max_tokens=self.teacher_correction_max_tokens,
            structured_regex=REFERENT_TAIL_REGEX,
            receipt_kind=receipt_kind,
            priority=priority,
        )
        receipts.append(receipt)
        teacher_target = protocol_reasoning_prefix + raw
        parsed = parse_teacher_target(
            teacher_target,
            mode="referent",
            original_response=protocol_reasoning_prefix,
        )
        if parsed is None:
            raise ValueError("Teacher correction+bbox continuation violates the referent target contract")
        receipt["parsed_teacher_target"] = _jsonable(parsed)
        return teacher_target, parsed.bbox

    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, validate: bool = False, **kwargs: Any
    ) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        multi_modal = await self.process_multi_modal_info(messages)
        images = multi_modal.get("images")
        self._assert_mm_supported(bool(images))
        if not images:
            raise ValueError("Routed Grounding Repair requires one original image")
        original_image = images[0]
        # Student trajectories carry exactly the original source image.  The GT
        # crop exists only in Teacher requests and is never put in this field.
        student_multi_modal = {"images": [original_image]}
        mm_processor_kwargs = self._get_mm_processor_kwargs()
        prompt_ids = await self.ct_build_initial_tokens(messages, images=[original_image])
        student = await self._student_rollout(
            sampling_params,
            prompt_ids=prompt_ids,
            images=[original_image],
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
        )
        parsed = parse_response(student.protocol_response)
        receipt: dict[str, Any] = {
            "arm": self.arm,
            "student_parse": _jsonable(parsed),
            "student_token_response": student.raw_response,
            "student_protocol_response": student.protocol_response,
            "student_protocol_prefix": student.implicit_protocol_prefix,
            "student_phase_receipts": student.phase_receipts,
            "student_total_response_budget": self.response_length,
            "student_force_reasoning_close_enabled": bool(getattr(self, "student_force_reasoning_close", False)),
            "student_close_status": student.phase_receipts[0].get("close_status", "invalid")
            if student.phase_receipts
            else "invalid",
            "student_forced_reasoning_close": bool(
                student.phase_receipts and student.phase_receipts[0].get("close_status") == "forced"
            ),
            "student_response_token_count": len(student.response_ids),
            "probe_receipts": [],
            "teacher_receipts": [],
            "student_image_count": 1,
            "repair_kind": STANDARD_REPAIR_KIND,
        }
        try:
            if self.arm == "standard_opd" and not self.standard_opd_probe_diagnostics and not validate:
                receipt["route"] = {"route": "standard_opd", "error": None}
                return self._standard_output(
                    student,
                    multi_modal_data=student_multi_modal,
                    mm_processor_kwargs=mm_processor_kwargs,
                    receipt=receipt,
                )
            inputs = extract_routed_grounding_inputs(kwargs)
            if not parsed.format_valid or not parsed.response_prefixes.valid:
                receipt["route"] = {
                    "route": "uncertain",
                    "error": parsed.error or parsed.response_prefixes.error or "invalid Student response",
                }
                return self._standard_output(
                    student,
                    multi_modal_data=student_multi_modal,
                    mm_processor_kwargs=mm_processor_kwargs,
                    receipt=receipt,
                )

            # Direct target matches do not spend probe calls.  Every other
            # valid response receives exactly four independently seeded probes.
            initial_route = route_response(student.protocol_response, inputs.instances, inputs.target_ann_id)
            if initial_route.route in {"correct", "localization"}:
                routed = initial_route
            else:
                probes, probe_receipts = await self._probe_responses(
                    student=student,
                    parsed=parsed,
                    sampling_params=sampling_params,
                    images=[original_image],
                    mm_processor_kwargs=mm_processor_kwargs,
                    priority=priority,
                )
                receipt["probe_receipts"] = probe_receipts
                routed = route_response(student.protocol_response, inputs.instances, inputs.target_ann_id, probes)
            receipt["route"] = _jsonable(routed)
            # Validation reports must score the unmodified Student response.
            # Routing/probe receipts remain available for diagnostics, but no
            # Teacher request or target substitution is allowed in this path.
            if validate:
                receipt["validate_teacher_bypassed"] = True
                return self._standard_output(
                    student,
                    multi_modal_data=student_multi_modal,
                    mm_processor_kwargs=mm_processor_kwargs,
                    receipt=receipt,
                )
            if self.arm == "standard_opd" or routed.route in {"correct", "uncertain"}:
                return self._standard_output(
                    student,
                    multi_modal_data=student_multi_modal,
                    mm_processor_kwargs=mm_processor_kwargs,
                    receipt=receipt,
                )

            if self.teacher_view == "gt_crop":
                crop_result = crop_gt_bbox(original_image, inputs.target_bbox)
                teacher_view_image = crop_result.crop
                receipt["teacher_view"] = {"kind": "gt_crop", "mapping": crop_result.mapping.to_dict()}
            else:
                teacher_view_image = gt_overlay(original_image, inputs.target_bbox)
                receipt["teacher_view"] = {"kind": "gt_overlay"}
            teacher_receipts: list[dict[str, Any]] = receipt["teacher_receipts"]
            localization_route = routed.route in {"localization", "localization_recoverable"}
            if self.arm == "routed_repair" and localization_route:
                if routed.parsed.bbox_prefix is None or routed.parsed.reasoning is None or routed.parsed.bbox is None:
                    raise ValueError("localization route is missing a parsed bbox/reasoning prefix")
                teacher_bbox = await self._teacher_bbox(
                    reasoning=routed.parsed.reasoning,
                    inputs=inputs,
                    original_image=original_image,
                    teacher_view_image=teacher_view_image,
                    sampling_params=sampling_params,
                    priority=priority,
                    receipts=teacher_receipts,
                )
                teacher_target = build_localization_teacher_target(routed.parsed.reasoning, teacher_bbox)
                verification = verify_teacher_target(
                    mode="localization",
                    teacher_output=teacher_target,
                    target_bbox=inputs.target_bbox,
                    student_bbox=routed.parsed.bbox,
                    instances=inputs.instances,
                    target_ann_id=inputs.target_ann_id,
                )
                receipt["teacher_verification"] = _jsonable(verification)
                if not verification.accepted:
                    receipt["teacher_rejected"] = verification.reason
                    return self._standard_output(
                        student,
                        multi_modal_data=student_multi_modal,
                        mm_processor_kwargs=mm_processor_kwargs,
                        receipt=receipt,
                    )
                suffix = _bbox_tail_from_canonical_json(teacher_bbox)
                receipt["repair_kind"] = LOCALIZATION_REPAIR_KIND
                repaired = self._repair_output(
                    student,
                    prefix_text=_token_prefix_from_protocol_prefix(
                        routed.parsed.bbox_prefix, student.implicit_protocol_prefix
                    ),
                    suffix_text=suffix,
                    repair_kind=LOCALIZATION_REPAIR_KIND,
                    multi_modal_data=student_multi_modal,
                    mm_processor_kwargs=mm_processor_kwargs,
                    receipt=receipt,
                )
            else:
                if routed.parsed.reasoning_prefix is None:
                    raise ValueError("repair route is missing a reasoning prefix")
                token_reasoning_prefix = _token_prefix_from_protocol_prefix(
                    routed.parsed.reasoning_prefix, student.implicit_protocol_prefix
                )
                teacher_target, _teacher_bbox_value = await self._teacher_correction_bbox_continuation(
                    protocol_reasoning_prefix=routed.parsed.reasoning_prefix,
                    token_reasoning_prefix=token_reasoning_prefix,
                    receipt_kind=(
                        "generic_correction_bbox" if self.arm == "generic_sopd" else "referent_correction_bbox"
                    ),
                    inputs=inputs,
                    original_image=original_image,
                    teacher_view_image=teacher_view_image,
                    sampling_params=sampling_params,
                    priority=priority,
                    receipts=teacher_receipts,
                )
                verification = verify_teacher_target(
                    mode="referent",
                    teacher_output=teacher_target,
                    target_bbox=inputs.target_bbox,
                    instances=inputs.instances,
                    target_ann_id=inputs.target_ann_id,
                )
                receipt["teacher_verification"] = _jsonable(verification)
                if not verification.accepted:
                    receipt["teacher_rejected"] = verification.reason
                    return self._standard_output(
                        student,
                        multi_modal_data=student_multi_modal,
                        mm_processor_kwargs=mm_processor_kwargs,
                        receipt=receipt,
                    )
                receipt["repair_kind"] = REFERENT_REPAIR_KIND
                repaired = self._repair_output(
                    student,
                    prefix_text=token_reasoning_prefix,
                    suffix_text=teacher_target[len(routed.parsed.reasoning_prefix) :],
                    repair_kind=REFERENT_REPAIR_KIND,
                    multi_modal_data=student_multi_modal,
                    mm_processor_kwargs=mm_processor_kwargs,
                    receipt=receipt,
                )
            if repaired is not None:
                return repaired
            return self._standard_output(
                student, multi_modal_data=student_multi_modal, mm_processor_kwargs=mm_processor_kwargs, receipt=receipt
            )
        except Exception as error:
            logger.warning("Routed Grounding Repair failed closed: %s", error)
            receipt["orchestration_error"] = f"{type(error).__name__}: {error}"
            return self._standard_output(
                student, multi_modal_data=student_multi_modal, mm_processor_kwargs=mm_processor_kwargs, receipt=receipt
            )


class RoutedGroundingAgentLoopWorker(AgentLoopWorker):
    """Inject existing Teacher clients when hydra constructs an AgentLoop."""

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs: Any,
    ) -> Any:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )
            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name],
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                # This is the sole custom integration hook.  The client dict is
                # already created by MultiTeacherModelManager and is passed by
                # the upstream Trainer to AgentLoopManager.
                teacher_client=self.teacher_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                hf_model_type=self.hf_model_type,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output: AgentLoopOutput = await agent_loop.run(
                sampling_params,
                validate=trajectory["validate"],
                **kwargs,
            )
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

    def _postprocess(self, inputs: list[Any], input_non_tensor_batch: dict | None = None, validate: bool = False):
        """Move the per-sample repair kind into the tensor batch for the loss."""

        output = super()._postprocess(inputs, input_non_tensor_batch=input_non_tensor_batch, validate=validate)
        values = output.non_tensor_batch.pop("repair_kind", None)
        if values is None:
            kinds = [STANDARD_REPAIR_KIND] * len(inputs)
        else:
            kinds = []
            for value in values:
                if hasattr(value, "item") and callable(value.item):
                    value = value.item()
                kinds.append(int(value) if value is not None else STANDARD_REPAIR_KIND)
        output.batch["repair_kind"] = torch.tensor(kinds, dtype=torch.long)
        return output


def aggregate_routed_grounding_metrics(receipts: Sequence[Any]) -> dict[str, float]:
    """Aggregate auditable per-sample receipts without changing the PPO batch."""

    records = [item for item in receipts if isinstance(item, Mapping)]
    total = len(records)
    if total == 0:
        return {}
    routes: dict[str, int] = {}
    probe_calls: list[int] = []
    teacher_calls: list[int] = []
    teacher_accepted = 0
    teacher_attempted = 0
    invalid = 0
    response_lengths: list[int] = []
    student_image_counts: list[int] = []
    teacher_image_counts: list[int] = []
    for receipt in records:
        route = receipt.get("route", {})
        route_name = route.get("route", "missing") if isinstance(route, Mapping) else "missing"
        routes[str(route_name)] = routes.get(str(route_name), 0) + 1
        probes = receipt.get("probe_receipts", ())
        teacher = receipt.get("teacher_receipts", ())
        probe_calls.append(len(probes) if isinstance(probes, Sequence) else 0)
        teacher_calls.append(len(teacher) if isinstance(teacher, Sequence) else 0)
        teacher_attempted += int(isinstance(teacher, Sequence) and len(teacher) > 0)
        verification = receipt.get("teacher_verification", {})
        if isinstance(verification, Mapping) and bool(verification.get("accepted")):
            teacher_accepted += 1
        student_parse = receipt.get("student_parse", {})
        if not isinstance(student_parse, Mapping) or not bool(student_parse.get("format_valid")):
            invalid += 1
        response_lengths.append(int(receipt.get("student_response_token_count", 0)))
        student_image_counts.append(int(receipt.get("student_image_count", 0)))
        if isinstance(teacher, Sequence):
            teacher_image_counts.extend(
                int(item.get("image_count", 0)) for item in teacher if isinstance(item, Mapping)
            )
    metrics: dict[str, float] = {
        "routed_grounding/samples": float(total),
        "routed_grounding/probe_calls/mean": sum(probe_calls) / total,
        "routed_grounding/teacher_calls/mean": sum(teacher_calls) / total,
        "routed_grounding/teacher_accepted/count": float(teacher_accepted),
        "routed_grounding/teacher_accepted/proportion": teacher_accepted / total,
        "routed_grounding/teacher_acceptance_rate": (
            teacher_accepted / teacher_attempted if teacher_attempted else 0.0
        ),
        "routed_grounding/invalid/proportion": invalid / total,
        "routed_grounding/response_length/mean": sum(response_lengths) / total,
        "routed_grounding/student_image_count/mean": sum(student_image_counts) / total,
        "routed_grounding/teacher_image_count/mean": (
            sum(teacher_image_counts) / len(teacher_image_counts) if teacher_image_counts else 0.0
        ),
    }
    for route_name, count in routes.items():
        metrics[f"routed_grounding/route/{route_name}/count"] = float(count)
        metrics[f"routed_grounding/route/{route_name}/proportion"] = count / total
    return metrics


class RoutedGroundingAgentLoopManager(AgentLoopManager):
    """Use :class:`RoutedGroundingAgentLoopWorker` without changing verl core."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.agent_loop_workers_class = ray.remote(RoutedGroundingAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts: Any) -> Any:
        output = await super().generate_sequences(prompts)
        output.meta_info["routed_grounding_metrics"] = aggregate_routed_grounding_metrics(
            output.non_tensor_batch.get("routed_grounding_receipt", ())
        )
        return output


__all__ = [
    "BBOX_JSON_REGEX",
    "BBOX_TAIL_REGEX",
    "LOCALIZATION_REPAIR_KIND",
    "REFERENT_REPAIR_KIND",
    "STANDARD_REPAIR_KIND",
    "RoutedGroundingAgentLoop",
    "RoutedGroundingAgentLoopManager",
    "RoutedGroundingAgentLoopWorker",
    "aggregate_routed_grounding_metrics",
    "bbox_sampling_params",
    "extract_routed_grounding_inputs",
    "find_exact_prefix_token_count",
]
