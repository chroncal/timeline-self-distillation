"""Pure contracts for error-span OPD annotation.

The worker in :mod:`verl.experimental.error_span_opd.worker` owns model and
server calls.  This module keeps the parts that determine offsets, grouping,
and tensor placement independent of vLLM, Ray, and image processing.  In
particular, prompt log-probabilities are kept at their native (unrenormalized)
top-k values; they are only sliced and moved into the padded student layout.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch


TOPK = 30
HIT_IOU = 0.5
SELECTED_SPAN_SCOPE = "selected_span"
ERROR_SUFFIX_SCOPE = "error_suffix"
SUPERVISION_SCOPES = frozenset({SELECTED_SPAN_SCOPE, ERROR_SUFFIX_SCOPE})


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def validate_span_offsets(reasoning_length: int, before: int, after: int) -> tuple[int, int]:
    """Validate a half-open reasoning span ``[before, after)``.

    Empty spans are rejected because the training loss and the teacher top-k
    receipt both need at least one target row.  ``after == reasoning_length``
    is valid; the end boundary itself is handled by callers when selecting an
    *intermediate* intervention span.
    """

    length = _integer(reasoning_length, name="reasoning_length")
    start = _integer(before, name="before")
    end = _integer(after, name="after")
    if length < 0:
        raise ValueError("reasoning_length must be non-negative")
    if not (0 <= start < end <= length):
        raise ValueError(
            f"span must satisfy 0 <= before < after <= reasoning_length, got "
            f"before={start}, after={end}, reasoning_length={length}"
        )
    return start, end


def span_mask(
    reasoning_length: int,
    before: int,
    after: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Return a one-dimensional mask with ones exactly on ``[before, after)``."""

    start, end = validate_span_offsets(reasoning_length, before, after)
    mask = torch.zeros(length := int(reasoning_length), dtype=dtype, device=device)
    mask[start:end] = 1
    return mask


def resolve_supervision_offsets(
    scope: str,
    *,
    response_length: int,
    selected_before: int,
    selected_after: int,
) -> tuple[int, int]:
    """Map one diagnostic span to the token interval used for distillation.

    ``selected_span`` preserves the historical local-OPD behavior.  The
    ``error_suffix`` mode mirrors ROSD's structural masking rule: localization
    supplies the start offset, and every remaining token on the same student
    response is eligible through the recorded response end.
    """

    if scope not in SUPERVISION_SCOPES:
        raise ValueError(
            f"supervision_scope must be one of {sorted(SUPERVISION_SCOPES)}, got {scope!r}"
        )
    start, selected_end = validate_span_offsets(
        response_length, selected_before, selected_after
    )
    if scope == SELECTED_SPAN_SCOPE:
        return start, selected_end
    return start, int(response_length)


def span_prediction_positions(prompt_length: int, before: int, after: int) -> torch.Tensor:
    """Return causal rows predicting a response-relative span.

    For a prompt followed by response tokens, the row predicting response
    token ``i`` is ``prompt_length + i - 1``.  This helper makes the ``-1``
    explicit and is shared by the student and Teacher-vLLM receipts.
    """

    prompt = _integer(prompt_length, name="prompt_length")
    if prompt <= 0:
        raise ValueError("prompt_length must be positive for a causal prediction")
    start = _integer(before, name="before")
    end = _integer(after, name="after")
    if not (0 <= start < end):
        raise ValueError(f"span must satisfy 0 <= before < after, got before={start}, after={end}")
    return torch.arange(prompt + start - 1, prompt + after - 1, dtype=torch.long)


def _as_rank2(value: Any, *, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, top_k] (or [1, rows, top_k])")
    return tensor


def extract_prompt_span_topk(
    prompt_ids: Sequence[int] | torch.Tensor,
    prompt_logprobs: Any,
    *,
    teacher_prompt_length: int,
    before: int,
    after: int,
    top_k: int = TOPK,
    mass_tolerance: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice vLLM prompt log-probs for a response-relative span.

    verl's vLLM adapter drops the first prompt position and appends one dummy
    row, so the row predicting response-relative token ``i`` is at
    ``teacher_prompt_length - 1 + i``.  The returned top-k values are copied
    exactly and are deliberately *not* renormalized.
    """

    teacher_length = _integer(teacher_prompt_length, name="teacher_prompt_length")
    if teacher_length <= 0:
        raise ValueError("teacher_prompt_length must be positive")
    k = _integer(top_k, name="top_k")
    if k <= 0:
        raise ValueError("top_k must be positive")
    ids = _as_rank2(prompt_ids, name="prompt_ids", dtype=torch.long)
    logs = _as_rank2(prompt_logprobs, name="prompt_logprobs")
    if ids.shape != logs.shape:
        raise ValueError(f"prompt_ids and prompt_logprobs shapes differ: {ids.shape} vs {logs.shape}")
    start, end = validate_span_offsets(max(0, ids.shape[0] - teacher_length + 1), before, after)
    row_start = teacher_length - 1 + start
    row_end = teacher_length - 1 + end
    if row_start < 0 or row_end > logs.shape[0]:
        raise ValueError(
            "prompt log-probability rows do not contain the requested span: "
            f"rows={logs.shape[0]}, teacher_prompt_length={teacher_length}, before={start}, after={end}"
        )
    if logs.shape[1] < k:
        raise ValueError(f"prompt log-probs contain {logs.shape[1]} columns, fewer than requested top_k={k}")
    sliced_ids = ids[row_start:row_end, :k].clone()
    sliced_logs = logs[row_start:row_end, :k].clone()
    validate_topk_mass(sliced_logs, tolerance=mass_tolerance)
    return sliced_ids, sliced_logs


def validate_topk_mass(log_probs: Any, *, tolerance: float = 1e-4) -> torch.Tensor:
    """Check raw top-k probability mass without applying a top-k softmax."""

    if tolerance < 0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be finite and non-negative")
    values = log_probs if isinstance(log_probs, torch.Tensor) else torch.as_tensor(log_probs)
    if values.ndim < 1:
        raise ValueError("log_probs must have at least one dimension")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("top-k log-probabilities contain NaN or Inf")
    mass = values.float().exp().sum(dim=-1)
    if bool((mass > 1.0 + float(tolerance)).any().item()):
        raise ValueError("top-k probability mass exceeds one; values must not be renormalized")
    return mass


def map_span_topk_to_student_padded(
    topk_ids: Any,
    topk_logprobs: Any,
    *,
    prompt_width: int,
    response_width: int,
    before: int,
    after: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place span top-k rows into a full padded ``[prompt + response, K]`` layout.

    The first two returned tensors are suitable for the native verl top-k
    distillation path after adding a batch dimension.  Non-span rows contain a
    harmless pad token and zero log-probability; ``mask`` identifies the only
    rows that may contribute to the error-span loss.
    """

    prompt = _integer(prompt_width, name="prompt_width")
    response = _integer(response_width, name="response_width")
    if prompt < 0 or response < 0:
        raise ValueError("prompt_width and response_width must be non-negative")
    start, end = validate_span_offsets(response, before, after)
    ids = _as_rank2(topk_ids, name="topk_ids", dtype=torch.long)
    logs = _as_rank2(topk_logprobs, name="topk_logprobs")
    if ids.shape != logs.shape:
        raise ValueError(f"topk_ids and topk_logprobs shapes differ: {ids.shape} vs {logs.shape}")
    if ids.shape[0] != end - start:
        raise ValueError(f"top-k rows={ids.shape[0]} do not match span length={end - start}")
    validate_topk_mass(logs)
    total = prompt + response
    padded_ids = torch.full(
        (total, ids.shape[1]), int(pad_token_id), dtype=torch.long, device=ids.device
    )
    padded_logs = torch.zeros((total, logs.shape[1]), dtype=logs.dtype, device=logs.device)
    padded_mask = torch.zeros(total, dtype=torch.bool, device=ids.device)
    if prompt == 0:
        raise ValueError("a causal prediction requires a nonempty prompt")
    target_slice = slice(prompt + start - 1, prompt + end - 1)
    padded_ids[target_slice] = ids
    padded_logs[target_slice] = logs.to(device=padded_logs.device)
    padded_mask[target_slice] = True
    return padded_ids, padded_logs, padded_mask


def group_by_sample(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by ``sample_id`` while preserving input order."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each record must be a mapping")
        if "sample_id" not in record:
            raise ValueError("record is missing sample_id")
        grouped[str(record["sample_id"])].append(dict(record))
    return dict(grouped)


def _record_order(record: Mapping[str, Any], fallback: int) -> tuple[int, int]:
    for key in ("rollout_index", "attempt_index", "draw_index", "index", "order"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value), fallback
    return fallback, fallback


def is_success_reference(record: Mapping[str, Any], *, hit_iou: float = HIT_IOU) -> bool:
    """Return whether one rollout record is eligible as a successful reference."""

    closed = record.get("reasoning_naturally_closed", record.get("naturally_closed"))
    if closed is not True:
        return False
    parsed = record.get("parse")
    parse_valid = record.get("parse_valid")
    if parse_valid is None and isinstance(parsed, Mapping):
        parse_valid = parsed.get("parse_valid")
    if parse_valid is not True:
        return False
    iou = record.get("iou", record.get("bbox_iou", record.get("original_bbox_iou")))
    try:
        iou_value = float(iou)
    except (TypeError, ValueError):
        return False
    return math.isfinite(iou_value) and iou_value >= float(hit_iou)


def first_success_reference(
    records: Sequence[Mapping[str, Any]], *, hit_iou: float = HIT_IOU
) -> dict[str, Any] | None:
    """Choose the first eligible success in rollout order, never by IoU."""

    indexed = [(index, dict(record)) for index, record in enumerate(records)]
    ordered = [
        record
        for _, record in sorted(indexed, key=lambda item: _record_order(item[1], item[0]))
    ]
    for record in ordered:
        if is_success_reference(record, hit_iou=hit_iou):
            return record
    return None


def group_first_success(
    records: Sequence[Mapping[str, Any]], *, hit_iou: float = HIT_IOU
) -> dict[str, dict[str, Any] | None]:
    """Return one first-success record (or ``None``) for every sample."""

    return {
        sample_id: first_success_reference(group, hit_iou=hit_iou)
        for sample_id, group in group_by_sample(records).items()
    }


def select_maximum_decline(boundary_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select the largest adjacent advantage decline with earliest tie-break.

    ``boundary_scores`` must be ordered by increasing offset and contain one
    score for the start sentinel, all natural boundaries, and the end
    sentinel.  The final reasoning-end boundary is intentionally ineligible;
    this leaves an intermediate natural span for intervention.  No positive
    threshold is applied, so the least-bad decline is still auditable.
    """

    if len(boundary_scores) < 2:
        return None
    ordered = sorted(
        (dict(record) for record in boundary_scores),
        key=lambda record: (int(record["after_offset"]), int(record.get("checkpoint_index", 0))),
    )
    reasoning_length = max(int(record["after_offset"]) for record in ordered)
    candidates: list[dict[str, Any]] = []
    for before, after in zip(ordered, ordered[1:]):
        if str(after.get("checkpoint_kind")) != "natural_boundary":
            continue
        after_offset = int(after["after_offset"])
        if after_offset >= reasoning_length:
            continue
        before_advantage = float(before["correct_advantage"])
        after_advantage = float(after["correct_advantage"])
        if not math.isfinite(before_advantage) or not math.isfinite(after_advantage):
            raise ValueError("boundary score advantages must be finite")
        candidates.append(
            {
                "before_offset": int(before["after_offset"]),
                "after_offset": after_offset,
                "score": before_advantage - after_advantage,
                "positive_decline": before_advantage - after_advantage > 0.0,
                "before_advantage": before_advantage,
                "after_advantage": after_advantage,
                "checkpoint_index": int(after.get("checkpoint_index", 0)),
                "rule": "maximum own-margin decline among strict intermediate natural boundaries; earliest exact tie",
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (-float(item["score"]), int(item["after_offset"])))


__all__ = [
    "ERROR_SUFFIX_SCOPE",
    "HIT_IOU",
    "SELECTED_SPAN_SCOPE",
    "SUPERVISION_SCOPES",
    "TOPK",
    "extract_prompt_span_topk",
    "first_success_reference",
    "group_by_sample",
    "group_first_success",
    "is_success_reference",
    "map_span_topk_to_student_padded",
    "resolve_supervision_offsets",
    "select_maximum_decline",
    "span_mask",
    "span_prediction_positions",
    "validate_span_offsets",
    "validate_topk_mass",
]
