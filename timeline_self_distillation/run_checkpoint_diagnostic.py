"""Read-only diagnostics for the two five-image terminal-LoRA checkpoints.

The diagnostic deliberately keeps the old experiment's protocol fixed.  It
replays the *actual* twenty step-0 reverse rollouts, obtains a grammar support
at every prefix, and scores those same rows with the frozen early-span-1
teacher, the base model, and both saved adapters.  It also scores one
canonical rounded-GT sequence and runs sampler/cache/gradient parity checks.

This is an audit of a seen-image exploratory run, not a training entry point:
no optimizer is constructed, no parameter outside the two checkpoint tensors
is writable, and rounded GT is used only as a teacher-forced scoring sequence,
never for free generation or an update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bind the selected physical device before importing torch/model modules.  The
# module remains importable for CPU unit tests; model loading happens in run().
if "--device" in sys.argv:
    _device_position = sys.argv.index("--device") + 1
    if _device_position >= len(sys.argv):
        raise ValueError("--device requires a value")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_device_position]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import xgrammar as xgr  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402

from live_kv_probe_prototype.run_hf_fork import (  # noqa: E402
    _advance,
    _append_jsonl,
    _git_receipt,
    _write_json,
)
from reasoning_checkpoints.extractor import decode_ids  # noqa: E402
from reasoning_checkpoints.run_pilot import (  # noqa: E402
    _load_rgb_image,
    _processor,
    _render_and_process,
)
from timeline_self_distillation.run_opd_micro import (  # noqa: E402
    build_states as reference_build_states,
)
from timeline_self_distillation.run_opd_micro import (  # noqa: E402
    sample_bbox as reference_sample_bbox,
)
from timeline_self_distillation.run_teacher_pilot import (  # noqa: E402
    BOX_OPEN,
    BOX_REGEX,
    DEFAULT_SAMPLE_IDS,
    MODEL,
    QUERY,
    SEED,
    fork,
)
from timeline_self_distillation.terminal_adapter import (  # noqa: E402
    install_terminal_query_lora,
    terminal_parameter_whitelist,
)
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou  # noqa: E402

__all__ = [
    "BOX_REGEX",
    "build_grammar_supports",
    "canonical_bbox_tail",
    "digit_token_positions",
    "load_adapter_checkpoint",
    "reference_build_states",
    "reference_sample_bbox",
    "sample_bbox",
    "score_fixed_sequence",
    "token_distribution_metrics",
]

# Public sampler alias for read-only checkpoint interventions.  It is the
# exact sampler used by run_opd_micro, including its grammar and seed path.
sample_bbox = reference_sample_bbox

_ROW_ID_RE = re.compile(r"^row-(?P<index>[0-9]+)$")
_DRAWS_PER_SAMPLE = 4
_EXPECTED_TOTAL_DRAW_COUNT = 20
_CHECKPOINT_NAME = "final_adapter.pt"
_DEPENDENCY_FILES = (
    "timeline_self_distillation/run_checkpoint_diagnostic.py",
    "timeline_self_distillation/run_opd_micro.py",
    "timeline_self_distillation/run_teacher_pilot.py",
    "timeline_self_distillation/terminal_adapter.py",
    "live_kv_probe_prototype/run_hf_fork.py",
    "reasoning_checkpoints/extractor.py",
    "reasoning_checkpoints/run_pilot.py",
    "verl/experimental/routed_grounding/router.py",
)


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not math.isfinite(converted):
        raise FloatingPointError(f"explicit nonfinite diagnostic value: {value!r}")
    return converted


def _sample_index(sample_id: str) -> int:
    match = _ROW_ID_RE.fullmatch(str(sample_id))
    if match is None:
        raise ValueError(f"sample_id must use the fixed row-N form, got {sample_id!r}")
    return int(match.group("index"))


def _is_digit_piece(piece: str) -> bool:
    """Return true only for a token whose complete decoded piece is digits.

    In particular, ``"1,"`` and ``" </"`` are not numeric rows.  This is
    important for tokenizers whose vocabulary mixes digits and punctuation.
    This intentionally follows the old runner's exact ``piece.isdigit()``
    mask; a token containing whitespace or punctuation is not a numeric row.
    """

    return str(piece).isdigit()


def digit_token_positions(tokenizer: Any, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Describe target rows whose decoded token is a complete numeric piece."""

    positions: list[dict[str, Any]] = []
    for position, token_id in enumerate(token_ids):
        piece = decode_ids(tokenizer, [int(token_id)])
        if _is_digit_piece(piece):
            positions.append(
                {
                    "position": int(position),
                    "token_id": int(token_id),
                    "text": str(piece),
                }
            )
    return positions


def rounded_bbox(values: Sequence[float]) -> list[int]:
    """Round four GT coordinates and clamp each one to the grammar domain."""

    if len(values) != 4:
        raise ValueError(f"ground_truth_bbox must contain four values, got {values!r}")
    rounded: list[int] = []
    for value in values:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"ground_truth_bbox contains a nonfinite value: {value!r}")
        rounded.append(min(1000, max(0, int(round(converted)))))
    return rounded


def canonical_bbox_tail(values: Sequence[float]) -> tuple[list[int], str]:
    """Return a BOX_REGEX-valid integer tail, without ``BOX_OPEN``.

    The returned text is exactly the continuation after the already consumed
    ``BOX_OPEN``.  It is checked against the same regex used by the sampler so
    GT scoring cannot silently introduce a different serialization.
    """

    integer_bbox = rounded_bbox(values)
    tail = ",".join(str(value) for value in integer_bbox) + "]}</answer>"
    if re.fullmatch(BOX_REGEX, tail) is None:
        raise ValueError(f"rounded GT tail does not satisfy BOX_REGEX: {tail!r}")
    return integer_bbox, tail


def token_distribution_metrics(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    chosen_index: int,
) -> dict[str, torch.Tensor]:
    """Compute finite-support KL/entropy/NLL in FP32.

    ``student_log_probs`` and ``teacher_log_probs`` are already normalized on
    the *same grammar support*.  The implementation guards products such as
    ``0 * inf`` so an impossible token cannot turn a valid row into NaN.
    Reverse KL is ``KL(student || teacher)`` and forward KL is
    ``KL(teacher || student)``.
    """

    if student_log_probs.ndim != 1 or teacher_log_probs.ndim != 1:
        raise ValueError("log-probability rows must be one-dimensional")
    if student_log_probs.shape != teacher_log_probs.shape or not student_log_probs.numel():
        raise ValueError("student/teacher rows must have the same non-empty support")
    if not 0 <= int(chosen_index) < student_log_probs.numel():
        raise IndexError(f"chosen index {chosen_index} is outside the support")

    student_log_probs = student_log_probs.float()
    teacher_log_probs = teacher_log_probs.float()
    if not torch.isfinite(student_log_probs).all() or not torch.isfinite(teacher_log_probs).all():
        raise FloatingPointError("finite-support log-probability rows must be finite")
    p = student_log_probs.exp()
    q = teacher_log_probs.exp()
    log_ratio_pq = student_log_probs - teacher_log_probs
    log_ratio_qp = -log_ratio_pq

    reverse_kl = (p * log_ratio_pq).sum()
    forward_kl = (q * log_ratio_qp).sum()
    teacher_entropy = (q * (-teacher_log_probs)).sum()
    chosen_nll = -student_log_probs.float()[int(chosen_index)]
    result = {
        "reverse_kl": reverse_kl,
        "forward_kl": forward_kl,
        "teacher_entropy": teacher_entropy,
        "chosen_token_nll": chosen_nll,
    }
    if not all(torch.isfinite(value).item() for value in result.values()):
        raise FloatingPointError("nonfinite finite-support diagnostic metric")
    return result


def build_grammar_supports(
    grammar: Any,
    vocab_size: int,
    token_ids: Sequence[int],
    *,
    device: torch.device | str = "cpu",
) -> list[list[int]]:
    """Replay a complete fixed token sequence through xgrammar.

    A support is recorded *before* accepting each target token.  No model
    forward is needed: applying the grammar bitmask to a zero FP32 row gives
    the legal vocabulary exactly.  Thus support construction cannot change a
    cache or consume the final opening token; model scoring consumes that
    opening exactly once in :func:`score_fixed_sequence`.
    """

    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        raise ValueError("token_ids must be non-empty")
    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, int(vocab_size))
    scores = torch.zeros((1, int(vocab_size)), dtype=torch.float32, device=device)
    supports: list[list[int]] = []
    for position, token_id in enumerate(ids):
        scores.zero_()
        xgr.reset_token_bitmask(bitmask)
        if matcher.fill_next_token_bitmask(bitmask):
            xgr.apply_token_bitmask_inplace(scores, bitmask.to(scores.device), vocab_size=int(vocab_size))
        support = torch.isfinite(scores[0]).nonzero(as_tuple=False).flatten().tolist()
        if not support:
            raise RuntimeError(f"grammar produced an empty support at prefix position {position}")
        if token_id not in support:
            raise RuntimeError(f"fixed token {token_id} is outside grammar support at prefix position {position}")
        if not matcher.accept_token(token_id):
            raise RuntimeError(f"grammar rejected fixed token {token_id} at position {position}")
        supports.append([int(value) for value in support])
        if matcher.is_completed() and position != len(ids) - 1:
            raise RuntimeError("fixed sequence has tokens after grammar completion")
    if not matcher.is_completed():
        raise RuntimeError("fixed sequence did not complete BOX_REGEX")
    return supports


def _restore_rope_deltas(model: Any, value: torch.Tensor) -> None:
    model.model.rope_deltas = value.clone()


def score_fixed_sequence(
    model: Any,
    adapter: Any,
    state: Mapping[str, Any],
    token_ids: Sequence[int],
    supports: Sequence[Sequence[int]],
    *,
    adapter_enabled: bool,
    grad_enabled: bool = False,
) -> dict[str, Any]:
    """Score fixed next-token rows from one detached student/teacher cache.

    The first forward consumes ``state['last_opening_id']``; each following
    forward consumes the preceding target token.  No extra opening or target
    token is inserted.  Only support logits are softmaxed in FP32, while the
    returned chosen index makes numeric-row NLL extraction unambiguous.

    This helper is intentionally importable: callers may load a checkpoint,
    alter only a whitelist tensor (for example scale ``up.weight``), and
    re-score without any training side effect.
    """

    ids = [int(token_id) for token_id in token_ids]
    if len(ids) != len(supports) or not ids:
        raise ValueError("token_ids and supports must be equally sized and non-empty")
    adapter.enabled = bool(adapter_enabled)
    cache = fork(state["student"])
    _restore_rope_deltas(model, state["rope_deltas"])
    current_input = int(state["last_opening_id"])
    log_probs: list[torch.Tensor] = []
    chosen_indices: list[int] = []
    context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with context:
        for position, (token_id, support_values) in enumerate(zip(ids, supports, strict=True)):
            cache, logits = _advance(model, cache, [current_input])
            support = torch.as_tensor(
                [int(value) for value in support_values],
                dtype=torch.long,
                device=logits.device,
            )
            if support.ndim != 1 or support.numel() == 0:
                raise ValueError(f"empty support at position {position}")
            if token_id not in support_values:
                raise ValueError(f"target token {token_id} missing from support at position {position}")
            selected = logits[0].float().index_select(0, support)
            if not torch.isfinite(selected).all():
                raise FloatingPointError(f"nonfinite support logits at position {position}")
            row_log_probs = torch.log_softmax(selected, dim=-1)
            if not torch.isfinite(row_log_probs).all():
                raise FloatingPointError(f"nonfinite support log-probs at position {position}")
            log_probs.append(row_log_probs if grad_enabled else row_log_probs.detach())
            chosen_indices.append([int(value) for value in support_values].index(token_id))
            current_input = token_id
    return {
        "log_probs": log_probs,
        "chosen_indices": chosen_indices,
        "token_ids": ids,
        "supports": [[int(value) for value in support] for support in supports],
    }


def load_adapter_checkpoint(model: Any, adapter: Any, path: Path) -> dict[str, torch.Tensor]:
    """Copy exactly the terminal whitelist from a weights-only checkpoint.

    Shape and dtype are checked before copying.  No optimizer, scheduler, or
    non-whitelist model state is loaded.  The returned CPU tensors are useful
    for restoring the zero-increment control without touching the backbone.
    """

    whitelist = terminal_parameter_whitelist(model, adapter)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"checkpoint {path} must contain a mapping")
    expected = set(whitelist)
    observed = set(str(key) for key in loaded)
    if observed != expected:
        raise ValueError(f"checkpoint whitelist mismatch: expected {sorted(expected)}, got {sorted(observed)}")
    copied: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, parameter in whitelist.items():
            value = loaded[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"checkpoint entry {name} is not a tensor")
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(
                    f"checkpoint entry {name} has shape/dtype {tuple(value.shape)}/{value.dtype}; "
                    f"expected {tuple(parameter.shape)}/{parameter.dtype}"
                )
            parameter.copy_(value.to(device=parameter.device))
            copied[name] = value.detach().clone()
    return copied


def _copy_whitelist_state(model: Any, adapter: Any, state: Mapping[str, torch.Tensor]) -> None:
    whitelist = terminal_parameter_whitelist(model, adapter)
    if set(state) != set(whitelist):
        raise ValueError("state does not exactly match terminal whitelist")
    with torch.no_grad():
        for name, parameter in whitelist.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"state entry {name} shape/dtype drifted")
            parameter.copy_(value.to(device=parameter.device))


def _tensor_digest(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().cpu()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _cache_tensors(cache: Any) -> list[tuple[str, torch.Tensor]]:
    tensors: list[tuple[str, torch.Tensor]] = []
    for layer_index, layer in enumerate(cache.layers):
        for name in sorted(vars(layer)):
            value = getattr(layer, name)
            if isinstance(value, torch.Tensor):
                tensors.append((f"layers[{layer_index}].{name}", value))
    return tensors


def snapshot_cache(cache: Any) -> list[dict[str, Any]]:
    """Capture tensor version/content digests for a master cache."""

    return [
        {
            "path": path,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "version": int(value._version),
            "sha256": _tensor_digest(value),
        }
        for path, value in _cache_tensors(cache)
    ]


def compare_cache_snapshot(cache: Any, expected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actual = snapshot_cache(cache)
    same = actual == [dict(item) for item in expected]
    version_same = len(actual) == len(expected) and all(
        int(a["version"]) == int(e["version"]) for a, e in zip(actual, expected, strict=True)
    )
    content_same = len(actual) == len(expected) and all(
        str(a["sha256"]) == str(e["sha256"]) for a, e in zip(actual, expected, strict=True)
    )
    return {
        "unchanged": bool(same),
        "tensor_count": len(actual),
        "version_unchanged": bool(version_same),
        "content_unchanged": bool(content_same),
    }


def _build_diagnostic_states(model: Any, processor: Any, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build C0, early-span-1 teacher, and student branches with one prefill.

    This mirrors ``run_opd_micro.build_states`` exactly where it matters, but
    retains C0 and its digest so the audit can prove the master cache was never
    mutated.  The old helper is imported above for protocol/source parity; a
    second call would prefill each image twice and is intentionally avoided.
    """

    states: list[dict[str, Any]] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for row in records:
            sample_id = str(row["sample_id"])
            reasoning_ids = [int(value) for value in row["reasoning_ids"]]
            if decode_ids(processor.tokenizer, reasoning_ids) != str(row["reasoning_text"]):
                raise RuntimeError(f"{sample_id}: frozen reasoning IDs/text are not an exact pair")
            _, processed = _render_and_process(
                processor,
                str(row["expression"]),
                _load_rgb_image(str(row["image_path"])),
            )
            prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
            if "prompt_token_count" in row and len(prompt_ids) != int(row["prompt_token_count"]):
                raise RuntimeError(f"{sample_id}: prompt token count drifted from pilot")
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in processed.items()
            }
            print(f"PREFILL {sample_id} (one image prefill)", flush=True)
            prefill = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            c0 = prefill.past_key_values
            rope_deltas = model.model.rope_deltas.detach().clone()
            c0_snapshot = snapshot_cache(c0)
            cT, _ = _advance(model, fork(c0), reasoning_ids)
            teacher_cache, _ = _advance(
                model,
                fork(c0),
                reasoning_ids[: int(row["first_span_offset"])],
            )
            suffix = QUERY.format(entity=str(row["entity"]), question=str(row["expression"])) + BOX_OPEN
            suffix_ids = [int(value) for value in processor.tokenizer.encode(suffix, add_special_tokens=False)]
            if not suffix_ids or decode_ids(processor.tokenizer, suffix_ids) != suffix:
                raise RuntimeError(f"{sample_id}: fixed QUERY+BOX_OPEN does not round-trip")
            student_cache, _ = _advance(model, fork(cT), suffix_ids[:-1])
            teacher_cache, _ = _advance(model, fork(teacher_cache), suffix_ids[:-1])
            states.append(
                {
                    "row": dict(row),
                    "c0": c0,
                    "master_cache_snapshot": c0_snapshot,
                    "student": student_cache,
                    "teacher": teacher_cache,
                    "last_opening_id": int(suffix_ids[-1]),
                    "rope_deltas": rope_deltas,
                    "suffix_ids": suffix_ids,
                }
            )
    return states


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_pilot_records(path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    expected = list(DEFAULT_SAMPLE_IDS)
    observed = [str(row.get("sample_id")) for row in records]
    if observed != expected:
        raise ValueError(f"pilot records must contain fixed five IDs in order {expected}, got {observed}")
    required = {"image_path", "expression", "entity", "reasoning_ids", "reasoning_text", "ground_truth_bbox"}
    for row in records:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"pilot row {row.get('sample_id')} missing fields: {missing}")
        _sample_index(str(row["sample_id"]))
    return records


def read_eval_step(path: Path, step: int, sample_ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Read exactly four saved rows per image (twenty rows over five images)."""

    rows = [row for row in _read_jsonl(path) if int(row.get("step", -1)) == int(step)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in sample_ids:
            raise ValueError(f"unexpected sample_id {sample_id!r} in {path}")
        ids = row.get("ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path}: {sample_id} step {step} has no actual bbox IDs")
        grouped[sample_id].append(row)
    missing = [sample_id for sample_id in sample_ids if sample_id not in grouped]
    if missing:
        raise ValueError(f"{path}: missing step {step} rows for {missing}")
    result: dict[str, list[dict[str, Any]]] = {}
    for sample_id in sample_ids:
        entries = grouped[sample_id]
        if len(entries) != _DRAWS_PER_SAMPLE:
            raise ValueError(
                f"{path}: expected {_DRAWS_PER_SAMPLE} rows for {sample_id} step {step}, got {len(entries)}"
            )
        entries = sorted(entries, key=lambda row: int(row["draw"]) if "draw" in row else int(row["seed"]))
        seeds = [int(row["seed"]) for row in entries]
        if len(set(seeds)) != _DRAWS_PER_SAMPLE:
            raise ValueError(f"{path}: duplicate seeds for {sample_id} step {step}")
        result[sample_id] = entries
    observed_total = sum(len(entries) for entries in result.values())
    if observed_total != _EXPECTED_TOTAL_DRAW_COUNT:
        raise ValueError(
            f"{path}: expected {_EXPECTED_TOTAL_DRAW_COUNT} total rows for step {step}, got {observed_total}"
        )
    return result


def _numeric_bundle(
    tokenizer: Any,
    token_ids: Sequence[int],
    student_score: Mapping[str, Any],
    teacher_score: Mapping[str, Any],
) -> dict[str, Any]:
    numeric = digit_token_positions(tokenizer, token_ids)
    reverse: list[float] = []
    forward: list[float] = []
    entropy: list[float] = []
    nll: list[float] = []
    for item in numeric:
        position = int(item["position"])
        values = token_distribution_metrics(
            student_score["log_probs"][position],
            teacher_score["log_probs"][position],
            int(student_score["chosen_indices"][position]),
        )
        reverse.append(_finite_float(values["reverse_kl"].detach().cpu().item()))
        forward.append(_finite_float(values["forward_kl"].detach().cpu().item()))
        entropy.append(_finite_float(values["teacher_entropy"].detach().cpu().item()))
        nll.append(_finite_float(values["chosen_token_nll"].detach().cpu().item()))
    return {
        "numeric_token_count": len(numeric),
        "token_positions": [int(item["position"]) for item in numeric],
        "token_ids": [int(item["token_id"]) for item in numeric],
        "token_texts": [str(item["text"]) for item in numeric],
        "reverse_kl": reverse,
        "forward_kl": forward,
        "teacher_entropy": entropy,
        "chosen_token_nll": nll,
    }


def _teacher_bundle(tokenizer: Any, token_ids: Sequence[int], teacher_score: Mapping[str, Any]) -> dict[str, Any]:
    numeric = digit_token_positions(tokenizer, token_ids)
    entropy: list[float] = []
    nll: list[float] = []
    for item in numeric:
        position = int(item["position"])
        logq = teacher_score["log_probs"][position]
        chosen = int(teacher_score["chosen_indices"][position])
        if not torch.isfinite(logq).all():
            raise FloatingPointError(f"nonfinite teacher support log-probs at position {position}")
        q = logq.float().exp()
        entropy_value = (q * (-logq.float())).sum()
        entropy.append(_finite_float(entropy_value.detach().cpu().item()))
        nll.append(_finite_float((-logq.float()[chosen]).detach().cpu().item()))
    return {
        "numeric_token_count": len(numeric),
        "token_positions": [int(item["position"]) for item in numeric],
        "token_ids": [int(item["token_id"]) for item in numeric],
        "token_texts": [str(item["text"]) for item in numeric],
        "teacher_entropy": entropy,
        "chosen_token_nll": nll,
    }


def _cpu_log_probs(score: Mapping[str, Any]) -> list[list[float]]:
    return [[_finite_float(value) for value in row.detach().float().cpu().tolist()] for row in score["log_probs"]]


def _score_first_numeric(
    model: Any,
    adapter: Any,
    state: Mapping[str, Any],
    token_ids: Sequence[int],
    supports: Sequence[Sequence[int]],
    tokenizer: Any,
    *,
    adapter_enabled: bool,
    grad_enabled: bool,
) -> dict[str, Any]:
    numeric = digit_token_positions(tokenizer, token_ids)
    if not numeric:
        return {"position": None, "log_probs": []}
    target_position = int(numeric[0]["position"])
    adapter.enabled = bool(adapter_enabled)
    cache = fork(state["student"])
    _restore_rope_deltas(model, state["rope_deltas"])
    current_input = int(state["last_opening_id"])
    found: torch.Tensor | None = None
    context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with context:
        for position, (token_id, support_values) in enumerate(zip(token_ids, supports, strict=True)):
            cache, logits = _advance(model, cache, [int(current_input)])
            if position == target_position:
                support = torch.as_tensor(support_values, dtype=torch.long, device=logits.device)
                selected = logits[0].float().index_select(0, support)
                if not torch.isfinite(selected).all():
                    raise FloatingPointError("nonfinite first numeric support logits")
                found = torch.log_softmax(selected, dim=-1).detach().cpu()
                break
            current_input = int(token_id)
    if found is None:
        raise RuntimeError("failed to capture first numeric row")
    return {"position": target_position, "log_probs": [_finite_float(value) for value in found.tolist()]}


def _safe_iou(prediction: Mapping[str, Any], ground_truth: Sequence[float]) -> float:
    if not bool(prediction.get("parse_valid")) or prediction.get("bbox") is None:
        return 0.0
    try:
        value = float(xyxy_iou(prediction["bbox"], ground_truth))
    except (TypeError, ValueError, RuntimeError, IndexError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _compare_rollout(
    expected: Mapping[str, Any], actual: Mapping[str, Any], ground_truth: Sequence[float]
) -> dict[str, Any]:
    expected_ids = [int(value) for value in expected["ids"]]
    actual_ids = [int(value) for value in actual["ids"]]
    expected_iou = _finite_float(expected.get("iou"), 0.0)
    actual_iou = _safe_iou(actual, ground_truth)
    return {
        "seed": int(expected["seed"]),
        "ids_match": actual_ids == expected_ids,
        "expected_ids": expected_ids,
        "actual_ids": actual_ids,
        "iou_match": math.isclose(actual_iou, expected_iou, rel_tol=0.0, abs_tol=1e-6),
        "expected_iou": expected_iou,
        "actual_iou": actual_iou,
        "iou_abs_diff": abs(actual_iou - expected_iou),
        "parse_valid": bool(actual.get("parse_valid")),
    }


def _parity_rollouts(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    grammar: Any,
    states: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, list[dict[str, Any]]],
    *,
    adapter_enabled: bool,
    label: str,
) -> dict[str, Any]:
    adapter.enabled = bool(adapter_enabled)
    entries: list[dict[str, Any]] = []
    for row in records:
        state = states[str(row["sample_id"])]
        for old in expected[str(row["sample_id"])]:
            prediction = reference_sample_bbox(
                model,
                tokenizer,
                grammar,
                state,
                int(old["seed"]),
            )
            compared = _compare_rollout(old, prediction, row["ground_truth_bbox"])
            compared.update(sample_id=str(row["sample_id"]), label=label)
            entries.append(compared)
    return {
        "label": label,
        "count": len(entries),
        "ids_exact_count": sum(bool(item["ids_match"]) for item in entries),
        "iou_match_count": sum(bool(item["iou_match"]) for item in entries),
        "all_ids_match": all(bool(item["ids_match"]) for item in entries),
        "all_iou_match": all(bool(item["iou_match"]) for item in entries),
        "entries": entries,
    }


def _aggregate_bundles(records: Sequence[Mapping[str, Any]], conditions: Sequence[str]) -> dict[str, Any]:
    per_sample: dict[str, Any] = {}
    overall: dict[str, Any] = {}
    for condition in conditions:
        values: dict[str, list[float]] = defaultdict(list)
        for record in records:
            bundle = record["metrics"][condition]
            for metric in ("reverse_kl", "forward_kl", "teacher_entropy", "chosen_token_nll"):
                values[metric].extend(_finite_float(value) for value in bundle.get(metric, []))
        overall[condition] = {
            "numeric_token_count": len(values["chosen_token_nll"]),
            **{f"{metric}_mean": _mean(values[metric]) for metric in values},
        }
    sample_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        sample_groups[str(record["sample_id"])].append(record)
    for sample_id, sample_records in sample_groups.items():
        per_sample[sample_id] = {}
        for condition in conditions:
            values: dict[str, list[float]] = defaultdict(list)
            for record in sample_records:
                bundle = record["metrics"][condition]
                for metric in ("reverse_kl", "forward_kl", "teacher_entropy", "chosen_token_nll"):
                    values[metric].extend(_finite_float(value) for value in bundle.get(metric, []))
            per_sample[sample_id][condition] = {
                "numeric_token_count": len(values["chosen_token_nll"]),
                **{f"{metric}_mean": _mean(values[metric]) for metric in values},
            }
    return {"overall": overall, "per_sample": per_sample}


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "xgrammar"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _state_metric(
    model: Any,
    adapter: Any,
    state: Mapping[str, Any],
    token_ids: Sequence[int],
    supports: Sequence[Sequence[int]],
    tokenizer: Any,
    teacher_score: Mapping[str, Any],
    *,
    adapter_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    student_score = score_fixed_sequence(
        model,
        adapter,
        state,
        token_ids,
        supports,
        adapter_enabled=adapter_enabled,
    )
    return _numeric_bundle(tokenizer, token_ids, student_score, teacher_score), student_score


def _run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device).lower() == "cpu":
        raise ValueError("the real checkpoint diagnostic requires a CUDA device; CPU scope is unit tests only")
    started = time.perf_counter()
    records = read_pilot_records(args.pilot_records)
    sample_ids = [str(row["sample_id"]) for row in records]
    reverse_eval = args.reverse_dir / "eval.jsonl"
    forward_eval = args.forward_dir / "eval.jsonl"
    reverse_checkpoint = args.reverse_dir / _CHECKPOINT_NAME
    forward_checkpoint = args.forward_dir / _CHECKPOINT_NAME
    for path in (reverse_eval, forward_eval, reverse_checkpoint, forward_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    reverse_step0 = read_eval_step(reverse_eval, 0, sample_ids)
    reverse_step20 = read_eval_step(reverse_eval, 20, sample_ids)
    forward_step20 = read_eval_step(forward_eval, 20, sample_ids)

    args.output_dir.mkdir(parents=False, exist_ok=False)
    command = [sys.executable, *sys.argv]
    receipts = {
        str(ROOT / relative): _file_receipt(ROOT / relative)
        for relative in _DEPENDENCY_FILES
        if (ROOT / relative).is_file()
    }
    input_receipts = {
        "pilot_records": _file_receipt(args.pilot_records),
        "reverse_eval": _file_receipt(reverse_eval),
        "forward_eval": _file_receipt(forward_eval),
        "reverse_checkpoint": _file_receipt(reverse_checkpoint),
        "forward_checkpoint": _file_receipt(forward_checkpoint),
    }
    common = {
        "kind": "read_only_checkpoint_negative_result_diagnostic",
        "sample_ids": sample_ids,
        "sample_count": len(records),
        "teacher": "early_span1",
        "model": str(MODEL),
        "device": str(args.device),
        "seed": int(SEED),
        "draw_policy": "actual saved seeds and IDs only; no resampling seed is invented",
        "old_reverse_step0_draws": {
            "per_sample": _DRAWS_PER_SAMPLE,
            "total": _EXPECTED_TOTAL_DRAW_COUNT,
        },
        "fixed_prefix": "reverse eval.jsonl step=0 actual IDs; grammar support before each target token",
        "score_rows": "only target tokens whose complete decoded piece is numeric",
        "divergence": {
            "reverse_kl": "KL(student p || frozen early teacher q)",
            "forward_kl": "KL(frozen early teacher q || student p)",
            "temperature": 1.0,
            "softmax_dtype": "float32",
        },
        "query_suffix": "QUERY.format(entity, expression) + BOX_OPEN, identical for all branches",
        "opening_protocol": "last BOX_OPEN token is consumed exactly once by each scorer",
        "cache_protocol": "one image prefill per sample; all later work uses COW forks; master C0 retained",
        "ground_truth_policy": (
            "rounded GT is teacher-forced only for canonical NLL scoring; it is never a free-generation input "
            "and never used for an update"
        ),
        "invalid_box_policy": "parse/numeric invalid rollout receives IoU=0 and Acc05=false",
        "training": {"optimizer": False, "parameter_update": False, "backbone_frozen": True},
        "scope": "five fixed seen images; checkpoint/mechanism audit, not a generalization claim",
    }
    _write_json(args.output_dir / "protocol.json", common)
    _write_json(
        args.output_dir / "manifest.json",
        {
            **common,
            "command": command,
            "git": _git_receipt(),
            "versions": _package_versions(),
            "entry_and_dependency_hashes": receipts,
            "input_and_checkpoint_hashes": input_receipts,
        },
    )

    torch.manual_seed(int(SEED))
    torch.cuda.manual_seed_all(int(SEED))
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(MODEL)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
    )
    adapter = install_terminal_query_lora(model, rank=8)
    whitelist = terminal_parameter_whitelist(model, adapter)
    zero_state = {name: parameter.detach().cpu().clone() for name, parameter in whitelist.items()}
    adapter.enabled = False
    frozen_versions_before = {
        name: int(parameter._version) for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    states_list = _build_diagnostic_states(model, processor, records)
    states = {str(state["row"]["sample_id"]): state for state in states_list}
    vocab_size = int(model.config.text_config.vocab_size)
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=vocab_size)
    ).compile_regex(BOX_REGEX)

    fixed_rows: list[dict[str, Any]] = []
    internal_scores: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        state = states[sample_id]
        for draw, old in enumerate(reverse_step0[sample_id]):
            token_ids = [int(value) for value in old["ids"]]
            supports = build_grammar_supports(
                grammar,
                vocab_size,
                token_ids,
                device=next(model.parameters()).device,
            )
            numeric = digit_token_positions(processor.tokenizer, token_ids)
            adapter.enabled = False
            teacher_score = score_fixed_sequence(
                model,
                adapter,
                {**state, "student": state["teacher"]},
                token_ids,
                supports,
                adapter_enabled=False,
            )
            base_bundle, base_score = _state_metric(
                model,
                adapter,
                state,
                token_ids,
                supports,
                processor.tokenizer,
                teacher_score,
                adapter_enabled=False,
            )
            load_adapter_checkpoint(model, adapter, reverse_checkpoint)
            reverse_bundle, reverse_score = _state_metric(
                model,
                adapter,
                state,
                token_ids,
                supports,
                processor.tokenizer,
                teacher_score,
                adapter_enabled=True,
            )
            load_adapter_checkpoint(model, adapter, forward_checkpoint)
            forward_bundle, forward_score = _state_metric(
                model,
                adapter,
                state,
                token_ids,
                supports,
                processor.tokenizer,
                teacher_score,
                adapter_enabled=True,
            )
            teacher_bundle = _teacher_bundle(processor.tokenizer, token_ids, teacher_score)
            parsed = parse_response("<think>probe" + BOX_OPEN + decode_ids(processor.tokenizer, token_ids))
            bbox = None if parsed.bbox is None else [float(value) for value in parsed.bbox]
            parse_valid = bool(parsed.parse_valid and bbox is not None and all(math.isfinite(value) for value in bbox))
            fixed_record = {
                "sample_id": sample_id,
                "source_index": _sample_index(sample_id),
                "draw": int(draw),
                "seed": int(old["seed"]),
                "step": 0,
                "ids": token_ids,
                "token_text": decode_ids(processor.tokenizer, token_ids),
                "token_pieces": [decode_ids(processor.tokenizer, [value]) for value in token_ids],
                "supports": supports,
                "numeric_target_positions": [int(item["position"]) for item in numeric],
                "numeric_target_ids": [int(item["token_id"]) for item in numeric],
                "bbox": bbox if parse_valid else None,
                "parse_valid": parse_valid,
                "iou": _safe_iou({"parse_valid": parse_valid, "bbox": bbox}, row["ground_truth_bbox"]),
                "metrics": {
                    "teacher_early_span1": teacher_bundle,
                    "base": base_bundle,
                    "reverse_checkpoint": reverse_bundle,
                    "forward_checkpoint": forward_bundle,
                },
            }
            fixed_rows.append(fixed_record)
            internal_scores[(sample_id, int(draw))] = {
                "base": _cpu_log_probs(base_score),
                "supports": supports,
                "ids": token_ids,
            }
            _append_jsonl(args.output_dir / "fixed_prefix.jsonl", fixed_record)

    # Canonical rounded-GT scoring.  Grammar supports are built once per row;
    # all four branches consume the same legal target sequence.
    gt_rows: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row["sample_id"])
        state = states[sample_id]
        integer_bbox, tail = canonical_bbox_tail(row["ground_truth_bbox"])
        target_ids = [int(value) for value in processor.tokenizer.encode(tail, add_special_tokens=False)]
        if decode_ids(processor.tokenizer, target_ids) != tail:
            raise RuntimeError(f"{sample_id}: canonical GT tail does not round-trip")
        target_supports = build_grammar_supports(
            grammar,
            vocab_size,
            target_ids,
            device=next(model.parameters()).device,
        )
        adapter.enabled = False
        teacher_score = score_fixed_sequence(
            model,
            adapter,
            {**state, "student": state["teacher"]},
            target_ids,
            target_supports,
            adapter_enabled=False,
        )
        teacher_nll = _teacher_bundle(processor.tokenizer, target_ids, teacher_score)
        base_bundle, _ = _state_metric(
            model,
            adapter,
            state,
            target_ids,
            target_supports,
            processor.tokenizer,
            teacher_score,
            adapter_enabled=False,
        )
        load_adapter_checkpoint(model, adapter, reverse_checkpoint)
        reverse_bundle, _ = _state_metric(
            model, adapter, state, target_ids, target_supports, processor.tokenizer, teacher_score, adapter_enabled=True
        )
        load_adapter_checkpoint(model, adapter, forward_checkpoint)
        forward_bundle, _ = _state_metric(
            model, adapter, state, target_ids, target_supports, processor.tokenizer, teacher_score, adapter_enabled=True
        )
        gt_record = {
            "sample_id": sample_id,
            "source_index": _sample_index(sample_id),
            "rounded_ground_truth_bbox": integer_bbox,
            "target_tail": tail,
            "target_ids": target_ids,
            "target_pieces": [decode_ids(processor.tokenizer, [value]) for value in target_ids],
            "numeric_target_positions": [
                int(item["position"]) for item in digit_token_positions(processor.tokenizer, target_ids)
            ],
            "conditions": {
                "teacher_early_span1": teacher_nll,
                "base": base_bundle,
                "reverse_checkpoint": reverse_bundle,
                "forward_checkpoint": forward_bundle,
            },
            "ground_truth_only": True,
            "parameter_update": False,
        }
        gt_rows.append(gt_record)
        _append_jsonl(args.output_dir / "gt_nll.jsonl", gt_record)

    parity: dict[str, Any] = {}

    # Same late student cache, scored twice with adapter off: the two rows are
    # mathematically identical distributions, so both KL directions must be 0.
    self_entries: list[dict[str, Any]] = []
    zero_entries: list[dict[str, Any]] = []
    first_fixed = {str(row["sample_id"]): reverse_step0[str(row["sample_id"])][0] for row in records}
    for row in records:
        sample_id = str(row["sample_id"])
        state = states[sample_id]
        old = first_fixed[sample_id]
        token_ids = [int(value) for value in old["ids"]]
        supports = internal_scores[(sample_id, 0)]["supports"]
        adapter.enabled = False
        off_a = score_fixed_sequence(model, adapter, state, token_ids, supports, adapter_enabled=False)
        off_b = score_fixed_sequence(model, adapter, state, token_ids, supports, adapter_enabled=False)
        numeric = digit_token_positions(processor.tokenizer, token_ids)
        self_kl: list[float] = []
        for item in numeric:
            position = int(item["position"])
            metrics = token_distribution_metrics(
                off_a["log_probs"][position],
                off_b["log_probs"][position],
                int(off_a["chosen_indices"][position]),
            )
            self_kl.append(_finite_float(metrics["reverse_kl"].detach().cpu().item()))
        load_zero = lambda: _copy_whitelist_state(model, adapter, zero_state)
        load_zero()
        adapter.enabled = True
        zero_on = score_fixed_sequence(model, adapter, state, token_ids, supports, adapter_enabled=True)
        max_diff = 0.0
        zero_kl: list[float] = []
        for off_row, on_row in zip(off_a["log_probs"], zero_on["log_probs"], strict=True):
            max_diff = max(max_diff, _finite_float((off_row - on_row).abs().max().detach().cpu().item()))
        for item in numeric:
            position = int(item["position"])
            metrics = token_distribution_metrics(
                zero_on["log_probs"][position],
                off_a["log_probs"][position],
                int(zero_on["chosen_indices"][position]),
            )
            zero_kl.append(_finite_float(metrics["reverse_kl"].detach().cpu().item()))
        self_entries.append(
            {
                "sample_id": sample_id,
                "numeric_rows": len(numeric),
                "max_off_repeat_reverse_kl": max(self_kl, default=0.0),
                "off_repeat_forward_reverse_kl": max(self_kl, default=0.0),
            }
        )
        zero_entries.append(
            {
                "sample_id": sample_id,
                "numeric_rows": len(numeric),
                "max_zero_enabled_off_logprob_abs_diff": max_diff,
                "max_zero_enabled_reverse_kl": max(zero_kl, default=0.0),
            }
        )
    parity["late_self_teacher"] = {
        "cache_source": "same state.student master cache, independent COW forks",
        "entries": self_entries,
        "max_reverse_kl": max((entry["max_off_repeat_reverse_kl"] for entry in self_entries), default=0.0),
        "all_zero_within_1e-6": all(entry["max_off_repeat_reverse_kl"] <= 1e-6 for entry in self_entries),
    }
    parity["zero_initialized_adapter"] = {
        "entries": zero_entries,
        "all_equal_within_1e-6": all(
            entry["max_zero_enabled_off_logprob_abs_diff"] <= 1e-6 and entry["max_zero_enabled_reverse_kl"] <= 1e-6
            for entry in zero_entries
        ),
    }

    # Exact old-sampler replay controls.  The old step-0 rows are the fixed
    # student prefixes above; step-20 rows are checkpoint-specific targets.
    _copy_whitelist_state(model, adapter, zero_state)
    parity["base_resample_vs_reverse_step0"] = _parity_rollouts(
        model,
        processor.tokenizer,
        adapter,
        grammar,
        states,
        records,
        reverse_step0,
        adapter_enabled=False,
        label="base_adapter_off_vs_old_reverse_step0",
    )
    load_adapter_checkpoint(model, adapter, reverse_checkpoint)
    parity["reverse_checkpoint_resample_vs_reverse_step20"] = _parity_rollouts(
        model,
        processor.tokenizer,
        adapter,
        grammar,
        states,
        records,
        reverse_step20,
        adapter_enabled=True,
        label="reverse_checkpoint_vs_old_reverse_step20",
    )
    load_adapter_checkpoint(model, adapter, forward_checkpoint)
    parity["forward_checkpoint_resample_vs_forward_step20"] = _parity_rollouts(
        model,
        processor.tokenizer,
        adapter,
        grammar,
        states,
        records,
        forward_step20,
        adapter_enabled=True,
        label="forward_checkpoint_vs_old_forward_step20",
    )

    # A checkpoint's disabled wrapper must restore the base output on the same
    # fixed first draw.  Compare every support row, not only numeric rows.
    base_off_entries: list[dict[str, Any]] = []
    for checkpoint_name, checkpoint_path in (
        ("reverse_checkpoint", reverse_checkpoint),
        ("forward_checkpoint", forward_checkpoint),
    ):
        load_adapter_checkpoint(model, adapter, checkpoint_path)
        adapter.enabled = False
        checkpoint_diffs: list[float] = []
        for row in records:
            sample_id = str(row["sample_id"])
            old = first_fixed[sample_id]
            info = internal_scores[(sample_id, 0)]
            checkpoint_score = score_fixed_sequence(
                model,
                adapter,
                states[sample_id],
                info["ids"],
                info["supports"],
                adapter_enabled=False,
            )
            base_cpu = info["base"]
            diffs = [
                _finite_float((torch.as_tensor(base_row) - checkpoint_row.detach().float().cpu()).abs().max().item())
                for base_row, checkpoint_row in zip(base_cpu, checkpoint_score["log_probs"], strict=True)
            ]
            checkpoint_diffs.extend(diffs)
        base_off_entries.append(
            {
                "checkpoint": checkpoint_name,
                "rows": len(checkpoint_diffs),
                "max_abs_support_logprob_diff": max(checkpoint_diffs, default=0.0),
                "restores_base_within_1e-6": max(checkpoint_diffs, default=0.0) <= 1e-6,
            }
        )
    parity["checkpoint_adapter_off_restores_base"] = {
        "entries": base_off_entries,
        "all_equal_within_1e-6": all(entry["restores_base_within_1e-6"] for entry in base_off_entries),
    }

    # BF16 grad-enabled/no-grad replay parity, one first numeric row per image
    # and condition.  No backward or optimizer step is performed.
    grad_entries: list[dict[str, Any]] = []
    for condition, checkpoint_path, enabled in (
        # Keep the zero-increment adapter enabled: this exercises the same
        # q-LoRA forward graph as a checkpoint, while remaining bit-identical
        # to base by construction.
        ("base", None, True),
        ("reverse_checkpoint", reverse_checkpoint, True),
        ("forward_checkpoint", forward_checkpoint, True),
    ):
        if checkpoint_path is None:
            _copy_whitelist_state(model, adapter, zero_state)
        else:
            load_adapter_checkpoint(model, adapter, checkpoint_path)
        for row in records:
            sample_id = str(row["sample_id"])
            info = internal_scores[(sample_id, 0)]
            no_grad = _score_first_numeric(
                model,
                adapter,
                states[sample_id],
                info["ids"],
                info["supports"],
                processor.tokenizer,
                adapter_enabled=enabled,
                grad_enabled=False,
            )
            grad = _score_first_numeric(
                model,
                adapter,
                states[sample_id],
                info["ids"],
                info["supports"],
                processor.tokenizer,
                adapter_enabled=enabled,
                grad_enabled=True,
            )
            no_tensor = torch.as_tensor(no_grad["log_probs"], dtype=torch.float32)
            grad_tensor = torch.as_tensor(grad["log_probs"], dtype=torch.float32)
            diff = _finite_float((no_tensor - grad_tensor).abs().max().item())
            agreement = float((no_tensor.argmax() == grad_tensor.argmax()).item()) if no_tensor.numel() else 0.0
            grad_entries.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "position": no_grad["position"],
                    "support_size": len(info["supports"][int(no_grad["position"])])
                    if no_grad["position"] is not None
                    else 0,
                    "max_abs_support_logprob_diff": diff,
                    "argmax_agreement": agreement,
                }
            )
            for parameter in adapter.parameters():
                parameter.grad = None
    parity["grad_enabled_vs_no_grad_first_numeric"] = {
        "coverage": (
            "forward logits only: first numeric target row of each fixed draw=0, all five images, "
            "base zero-increment/reverse/forward enabled q-LoRA; no backward/optimizer claim"
        ),
        "entries": grad_entries,
        "max_abs_support_logprob_diff": max(
            (entry["max_abs_support_logprob_diff"] for entry in grad_entries),
            default=0.0,
        ),
        "argmax_agreement_rate": _mean([entry["argmax_agreement"] for entry in grad_entries]),
        "all_argmax_agree": all(bool(entry["argmax_agreement"]) for entry in grad_entries),
    }

    frozen_versions_after = {
        name: int(parameter._version) for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    parity["frozen_model_parameter_versions"] = {
        "count": len(frozen_versions_before),
        "unchanged": frozen_versions_before == frozen_versions_after,
        "changed_names": sorted(
            name
            for name in set(frozen_versions_before) | set(frozen_versions_after)
            if frozen_versions_before.get(name) != frozen_versions_after.get(name)
        ),
    }
    parity["master_cache"] = {
        str(state["row"]["sample_id"]): compare_cache_snapshot(state["c0"], state["master_cache_snapshot"])
        for state in states_list
    }
    parity["master_cache_all_unchanged"] = all(bool(value["unchanged"]) for value in parity["master_cache"].values())
    parity["diagnostic_is_read_only"] = True
    _write_json(args.output_dir / "parity.json", parity)

    fixed_aggregate = _aggregate_bundles(
        fixed_rows,
        ("base", "reverse_checkpoint", "forward_checkpoint"),
    )
    gt_aggregate: dict[str, Any] = {}
    for condition in ("teacher_early_span1", "base", "reverse_checkpoint", "forward_checkpoint"):
        nll_values = [
            _finite_float(value)
            for row in gt_rows
            for value in row["conditions"][condition].get("chosen_token_nll", [])
        ]
        gt_aggregate[condition] = {
            "numeric_token_count": len(nll_values),
            "chosen_token_nll_mean": _mean(nll_values),
        }
    per_sample = []
    for sample_id in sample_ids:
        per_sample.append(
            {
                "sample_id": sample_id,
                "fixed_prefix": fixed_aggregate["per_sample"].get(sample_id, {}),
                "gt_nll": {
                    condition: {
                        "chosen_token_nll_mean": _mean(
                            [
                                _finite_float(value)
                                for row in gt_rows
                                if row["sample_id"] == sample_id
                                for value in row["conditions"][condition].get("chosen_token_nll", [])
                            ]
                        )
                    }
                    for condition in gt_aggregate
                },
            }
        )
    summary = {
        **common,
        "fixed_prefix_metrics": fixed_aggregate,
        "gt_nll": {"overall": gt_aggregate, "scoring_only": True},
        "per_sample": per_sample,
        "parity": {
            "base_step0_ids_exact": parity["base_resample_vs_reverse_step0"]["all_ids_match"],
            "reverse_step20_ids_exact": parity["reverse_checkpoint_resample_vs_reverse_step20"]["all_ids_match"],
            "forward_step20_ids_exact": parity["forward_checkpoint_resample_vs_forward_step20"]["all_ids_match"],
            "master_cache_unchanged": parity["master_cache_all_unchanged"],
            "frozen_model_parameter_versions_unchanged": parity["frozen_model_parameter_versions"]["unchanged"],
            "checkpoint_off_restores_base": parity["checkpoint_adapter_off_restores_base"]["all_equal_within_1e-6"],
            "grad_no_grad_argmax_agreement": parity["grad_enabled_vs_no_grad_first_numeric"]["argmax_agreement_rate"],
        },
        "training_performed": False,
        "limitation": (
            "All metrics use five seen images and fixed saved prefixes/draws; they diagnose the negative result "
            "and do not establish test-set transfer or a causal training improvement."
        ),
        "seconds": time.perf_counter() - started,
    }
    # Replace the deliberately simple timer expression with a stable elapsed
    # value without introducing any extra output or model work.
    summary["seconds"] = _finite_float(summary["seconds"])
    _write_json(args.output_dir / "summary.json", summary)
    adapter.enabled = False
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3")
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--reverse-dir", type=Path, required=True)
    parser.add_argument("--forward-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    _run_diagnostic(parse_args())
