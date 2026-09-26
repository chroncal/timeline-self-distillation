"""Native worker for token-faithful error-span OPD.

The ordinary routed-grounding agent loop owns the real Student rollout.  This
module adds the deliberately separate annotation pass used by
``ErrorSpanManager``: it scores fixed bbox tails with the existing Student
client and obtains top-k rows for one selected natural reasoning span from the
already configured Teacher client.  No model is loaded by the annotation
method and no continuation is sampled for either diagnostic forward.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from checkpoint_repair_upper_bound.run_contrastive_bbox import canonical_target
from reasoning_checkpoints.extractor import (
    decode_ids,
    exact_decoded_prefix_offset,
    extract_reasoning_checkpoints,
    split_reasoning_close,
)
from scripts.routed_grounding.run_diagnostics import STUDENT_INSTRUCTION
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.error_span_opd.core import (
    SELECTED_SPAN_SCOPE,
    SUPERVISION_SCOPES,
    TOPK,
    extract_prompt_span_topk,
    resolve_supervision_offsets,
    select_maximum_decline,
    validate_topk_mass,
)
from verl.experimental.routed_grounding.agent_loop import (
    STUDENT_BBOX_PREFIX,
    RoutedGroundingAgentLoop,
    RoutedGroundingAgentLoopWorker,
)


MIN_PIXELS = 3136
MAX_PIXELS = 262144
_DUMMY_SEED = 0
_DUMMY_MAX_TOKENS = 1
LOCAL_SPAN_METHOD = "local_span"
ROSD_METHOD = "rosd"
METHOD_VARIANTS = frozenset({LOCAL_SPAN_METHOD, ROSD_METHOD})

LEGACY_TEACHER_PROMPT_MODE = "legacy"
OVERLAY_VERIFIED_TARGET_CATEGORY_MODE = "overlay_verified_target_category"
_TEACHER_PROMPT_MODES = frozenset(
    {LEGACY_TEACHER_PROMPT_MODE, OVERLAY_VERIFIED_TARGET_CATEGORY_MODE}
)


class ContextOverflowError(ValueError):
    """A fixed token request would exceed the configured model context."""


class ROSDReflectionFormatError(ValueError):
    """All bounded attempts to obtain a parseable ROSD reflection failed."""

    def __init__(self, message: str, *, attempts: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.attempts = [dict(attempt) for attempt in attempts]


def _cfg_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _teacher_prompt_mode(config: Any) -> str:
    """Return the explicitly selected Teacher prompt contract.

    Historical error-span runs used a numeric GT rectangle in the Teacher
    text.  Keep that contract as the default so old receipts remain
    reproducible; the new single-group run opts into the verified source
    target-category contract explicitly.
    """

    mode = _cfg_get(config, "teacher_prompt_mode", None)
    if mode is None:
        # Accept the more general spelling for callers constructing configs
        # outside Hydra, while keeping one canonical config key.
        mode = _cfg_get(config, "teacher_input_mode", LEGACY_TEACHER_PROMPT_MODE)
    aliases = {
        "legacy_gt_bbox": LEGACY_TEACHER_PROMPT_MODE,
        "legacy_bbox": LEGACY_TEACHER_PROMPT_MODE,
    }
    normalized = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    if normalized not in _TEACHER_PROMPT_MODES:
        raise ValueError(
            "error_span_opd.teacher_prompt_mode must be one of "
            f"{sorted(_TEACHER_PROMPT_MODES)}, got {mode!r}"
        )
    return normalized


def _scalar_identity(value: Any, *, name: str) -> str:
    """Normalize a manifest identity while rejecting absent values."""

    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        raise ValueError(f"{name} is required for verified FineCops target loading")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{name} is required for verified FineCops target loading")
    return result


def _configured_finecops_annotation_path(config: Any, source_split: str) -> Path:
    """Resolve one raw FineCops positive annotation file from config."""

    source_split = str(source_split).strip().lower()
    if source_split not in {"train", "val"}:
        raise ValueError(
            "verified FineCops target loading supports source_split='train' or 'val', "
            f"got {source_split!r}"
        )

    paths = _cfg_get(config, "finecops_positive_annotation_paths", None)
    if paths is None:
        # These aliases make the loader usable from a small standalone config
        # without changing the canonical formal-run key above.
        paths = _cfg_get(config, "finecops_annotation_paths", None)
    path: Any = None
    if isinstance(paths, Mapping):
        path = paths.get(source_split)
    elif isinstance(paths, str):
        path = paths.format(split=source_split)
    if path is None:
        root = _cfg_get(config, "finecops_positive_annotation_root", None)
        if root is None:
            root = _cfg_get(config, "finecops_annotation_root", None)
        if root is not None:
            path = Path(str(root)) / f"expression_pos_{source_split}_set.json"
    if path is None:
        raise ValueError(
            "verified FineCops target loading requires "
            "error_span_opd.finecops_positive_annotation_paths (or annotation root)"
        )
    return Path(str(path)).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_finecops_positive_annotations(path: str) -> dict[str, dict[str, Any]]:
    """Read and index one immutable FineCops vanilla positive annotation file."""

    annotation_path = Path(path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"FineCops positive annotation file is missing: {annotation_path}")
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FineCops positive annotations {annotation_path}: {error}") from error
    if isinstance(raw, Mapping):
        rows = list(raw.values())
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError(f"FineCops positive annotations must be an object or list: {annotation_path}")
    indexed: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"FineCops annotation row {row_index} is not an object: {annotation_path}")
        annotation_id = row.get("id")
        if annotation_id is None:
            raise ValueError(f"FineCops annotation row {row_index} has no id: {annotation_path}")
        key = str(annotation_id)
        if key in indexed:
            raise ValueError(f"duplicate FineCops annotation id {key!r}: {annotation_path}")
        indexed[key] = dict(row)
    return indexed


def load_finecops_target_category(
    *,
    config: Any,
    sample_id: Any,
    source_split: Any,
    source_annotation_id: Any,
    target_ann_id: Any,
    image_id: Any,
    image_path: Any,
    expression: Any,
) -> dict[str, Any]:
    """Load a verified FineCops target category by sample identity.

    The vanilla positive annotation's relation ``phrase`` field is deliberately
    ignored because it can name a relation reference object.  FineCops encodes
    the target category as ``tuple[0][0]``.  Every identity field is checked
    before that category is returned, and any missing/inconsistent source fails
    loudly rather than falling back to the dataset expression.
    """

    sample = _scalar_identity(sample_id, name="sample_id")
    split = _scalar_identity(source_split, name="source_split").lower()
    source_id = _scalar_identity(source_annotation_id, name="source_annotation_id")
    target_id = _scalar_identity(target_ann_id, name="target_ann_id")
    expected_image_id = _scalar_identity(image_id, name="image_id")
    expected_expression = _scalar_identity(expression, name="expression")
    expected_image_path = _scalar_identity(image_path, name="image_path")

    if target_id != source_id:
        raise ValueError(
            "FineCops source_annotation_id and target_ann_id differ for "
            f"sample_id={sample!r}: {source_id!r} != {target_id!r}"
        )
    suffix = sample.rsplit("-", 1)[-1]
    if suffix != source_id:
        raise ValueError(
            f"FineCops sample_id={sample!r} does not end with source_annotation_id={source_id!r}"
        )
    if sample.startswith("finecops-ref-") and not sample.startswith(f"finecops-ref-{split}-"):
        raise ValueError(f"FineCops sample_id/source_split mismatch: {sample!r} vs {split!r}")
    image_stem = Path(expected_image_path).stem
    if image_stem != expected_image_id:
        raise ValueError(
            f"FineCops image path/id mismatch for sample_id={sample!r}: "
            f"path stem {image_stem!r} != image_id {expected_image_id!r}"
        )

    annotation_path = _configured_finecops_annotation_path(config, split)
    annotations = _load_finecops_positive_annotations(str(annotation_path))
    annotation = annotations.get(source_id)
    if annotation is None:
        raise ValueError(
            f"FineCops {split} positive annotation {source_id!r} is missing for sample_id={sample!r}"
        )
    if str(annotation.get("id")) != source_id:
        raise ValueError(f"FineCops annotation id mismatch for sample_id={sample!r}")
    if str(annotation.get("image_id")) != expected_image_id:
        raise ValueError(
            f"FineCops image identity mismatch for sample_id={sample!r}: "
            f"annotation image_id={annotation.get('image_id')!r}, expected {expected_image_id!r}"
        )
    if annotation.get("expression") != expected_expression:
        raise ValueError(f"FineCops expression mismatch for sample_id={sample!r}")

    objects = annotation.get("objects_id")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        raise ValueError(f"FineCops objects_id is missing for sample_id={sample!r}")
    instance_id = annotation.get("instance_id")
    if instance_id is None or str(instance_id) != str(objects[0]):
        raise ValueError(
            f"FineCops instance_id/objects_id[0] mismatch for sample_id={sample!r}: "
            f"{instance_id!r} != {objects[0]!r}"
        )
    relation_tuple = annotation.get("tuple")
    if (
        not isinstance(relation_tuple, Sequence)
        or isinstance(relation_tuple, (str, bytes))
        or not relation_tuple
        or not isinstance(relation_tuple[0], Sequence)
        or isinstance(relation_tuple[0], (str, bytes))
        or not relation_tuple[0]
    ):
        raise ValueError(f"FineCops tuple[0][0] is missing for sample_id={sample!r}")
    target_category = relation_tuple[0][0]
    if not isinstance(target_category, str) or not target_category.strip():
        raise ValueError(f"FineCops tuple[0][0] is empty for sample_id={sample!r}")

    return {
        "sample_id": sample,
        "source_split": split,
        "source_annotation_id": source_id,
        "target_ann_id": target_id,
        "image_id": expected_image_id,
        "image_path": expected_image_path,
        "expression": expected_expression,
        "target_category": target_category.strip(),
        "source_annotation_path": str(annotation_path),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ids_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(token_id) for token_id in token_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _request_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _backend_metadata(extra: Mapping[str, Any], *, excluded: set[str]) -> dict[str, Any]:
    """Keep backend receipts while excluding duplicated full prompt arrays."""

    return {str(key): _jsonable(value) for key, value in extra.items() if str(key) not in excluded}


def _protocol_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    ids = [int(token_id) for token_id in token_ids]
    terminal = {
        int(token_id)
        for token_id in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None))
        if token_id is not None
    }
    while ids and ids[-1] in terminal:
        ids.pop()
    return ids


def _rows(value: Any, *, name: str) -> list[list[Any]]:
    """Normalize native vLLM/SGLang arrays to rows without changing values."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list or tensor")
    # Some lightweight adapters retain a batch dimension.  The native client
    # returns [sequence, width], while [1, sequence, width] is equivalent.
    if len(value) == 1 and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        value = value[0]
    if value and not isinstance(value[0], (list, tuple)):
        value = [[item] for item in value]
    result: list[list[Any]] = []
    for index, row in enumerate(value):
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            raise ValueError(f"{name}[{index}] is not a row")
        result.append(row)
    return result


def _model_vocab_size(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("tokenizer.get_vocab() must return a nonempty mapping")
    ids = [int(value) for value in vocab.values()]
    if min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError("tokenizer vocabulary contains invalid or duplicate ids")
    return max(ids) + 1


def _tokenizer_alignment(student: Any, teacher: Any) -> dict[str, Any]:
    """Require the complete token-to-id map to match exactly."""

    student_vocab = {str(key): int(value) for key, value in student.get_vocab().items()}
    teacher_vocab = {str(key): int(value) for key, value in teacher.get_vocab().items()}
    if student_vocab != teacher_vocab:
        differing = sorted(set(student_vocab) ^ set(teacher_vocab))
        differing += sorted(
            key for key in set(student_vocab) & set(teacher_vocab) if student_vocab[key] != teacher_vocab[key]
        )
        raise ValueError(f"Student/Teacher tokenizer id maps differ; first differences={differing[:8]}")
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        left, right = getattr(student, attr, None), getattr(teacher, attr, None)
        if left != right:
            raise ValueError(f"Student/Teacher tokenizer {attr} differs: {left!r} != {right!r}")
    return {
        "exact_vocab_map": True,
        "vocab_size": len(student_vocab),
        "max_token_id": max(student_vocab.values()),
        "vocab_sha256": _request_hash(student_vocab),
        "student_tokenizer_class": type(student).__name__,
        "teacher_tokenizer_class": type(teacher).__name__,
    }


def _set_image_processor_size(processor: Any) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    current = getattr(image_processor, "size", None)
    size = dict(current) if isinstance(current, Mapping) else {}
    size.update(shortest_edge=MIN_PIXELS, longest_edge=MAX_PIXELS)
    image_processor.size = size


def _processor_input_ids(processed: Any) -> list[int]:
    try:
        value = processed["input_ids"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("processor output is missing input_ids") from error
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError(f"processor input_ids has unexpected batch shape {tuple(value.shape)}")
            return [int(item) for item in value[0].tolist()]
        if value.ndim == 1:
            return [int(item) for item in value.tolist()]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _processor_input_ids({"input_ids": value.tolist()})
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("processor input_ids has an unexpected batch size")
            return [int(item) for item in value[0]]
        return [int(item) for item in value]
    raise ValueError("processor input_ids has an unsupported type")


def _load_rgb_image(path: Any, image_patch_size: int = 14) -> Any:
    if not isinstance(path, (str, Path)):
        return path
    from qwen_vl_utils import fetch_image

    return fetch_image({"image": str(path)}, image_patch_size=int(image_patch_size))


@register("error_span_opd")
class ErrorSpanAgentLoop(RoutedGroundingAgentLoop):
    """Error-span arm using the inherited staged Student rollout."""

    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, validate: bool = False, **kwargs: Any
    ) -> Any:
        # Validation is evaluated by the outer manager using its original
        # ``validate`` metadata.  The Student trajectory itself is identical
        # to the native standard-OPD path; avoid diagnostic probes that are
        # discarded by validation and never produce a repair target.
        bypass_probes = validate and self.arm == "standard_opd" and not self.standard_opd_probe_diagnostics
        output = await super().run(
            sampling_params,
            priority=priority,
            validate=False if bypass_probes else validate,
            **kwargs,
        )
        if bypass_probes:
            receipt = output.extra_fields.get("routed_grounding_receipt")
            if isinstance(receipt, Mapping):
                receipt = dict(receipt)
                receipt["validation_probe_bypassed"] = True
                output.extra_fields["routed_grounding_receipt"] = receipt
        return output


class ErrorSpanWorker(RoutedGroundingAgentLoopWorker):
    """Annotate one failed Student trajectory with a selected Teacher span."""

    def __init__(self, config: Any, llm_client: Any, teacher_client: Any = None, reward_loop_worker_handles: Any = None):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        self.error_span_config = _cfg_get(config, "error_span_opd", {})
        self.method_variant = str(
            _cfg_get(self.error_span_config, "method_variant", LOCAL_SPAN_METHOD)
        )
        if self.method_variant not in METHOD_VARIANTS:
            raise ValueError(
                f"error_span_opd.method_variant must be one of {sorted(METHOD_VARIANTS)}, "
                f"got {self.method_variant!r}"
            )
        self.supervision_scope = str(
            _cfg_get(self.error_span_config, "supervision_scope", SELECTED_SPAN_SCOPE)
        )
        if self.supervision_scope not in SUPERVISION_SCOPES:
            raise ValueError(
                "error_span_opd.supervision_scope must be one of "
                f"{sorted(SUPERVISION_SCOPES)}, got {self.supervision_scope!r}"
            )
        self.teacher_prompt_mode = _teacher_prompt_mode(self.error_span_config)
        if self.teacher_prompt_mode == OVERLAY_VERIFIED_TARGET_CATEGORY_MODE:
            # Resolve the configured paths once so a typo fails during worker
            # construction, before the first training request.  The JSON file
            # itself is loaded lazily per split and cached for this process.
            self.finecops_positive_annotation_paths = {
                split: str(_configured_finecops_annotation_path(self.error_span_config, split))
                for split in ("train", "val")
            }
        self.scoring_concurrency = int(_cfg_get(self.error_span_config, "scoring_concurrency", 1))
        if self.scoring_concurrency <= 0:
            raise ValueError("error_span_opd.scoring_concurrency must be positive")
        self.topk = int(_cfg_get(self.error_span_config, "topk", TOPK))
        if self.topk != TOPK:
            raise ValueError(f"error-span OPD requires topk={TOPK}, got {self.topk}")
        configured_teacher_limit = int(_cfg_get(self.error_span_config, "teacher_context_limit", 0) or 0)
        distillation = _cfg_get(config, "distillation", {})
        teacher_models = _cfg_get(distillation, "teacher_models", {})
        teacher_model_cfg = _cfg_get(teacher_models, "teacher_model", {})
        teacher_inference = _cfg_get(teacher_model_cfg, "inference", {})
        distillation_teacher_limit = int(_cfg_get(teacher_inference, "max_model_len", 0) or 0)
        limits = [limit for limit in (configured_teacher_limit, distillation_teacher_limit) if limit > 0]
        self.teacher_context_limit = min(limits) if limits else 0
        self.student_context_limit = int(_cfg_get(self.rollout_config, "max_model_len", 0) or 0)
        teacher_model_path = self._teacher_model_path(config)
        if not teacher_model_path:
            raise ValueError("error_span_opd.teacher_model is required")
        self.teacher_model_path = str(teacher_model_path)
        self.student_model_path = str(
            _cfg_get(_cfg_get(config, "actor_rollout_ref", {}), "model", {}).get("path", "")
            if isinstance(_cfg_get(_cfg_get(config, "actor_rollout_ref", {}), "model", {}), Mapping)
            else _cfg_get(_cfg_get(_cfg_get(config, "actor_rollout_ref", {}), "model", {}), "path", "")
        )
        self.teacher_processor, self.teacher_tokenizer = self._load_teacher_tokenizer_processor(self.teacher_model_path)
        self.tokenizer_alignment = _tokenizer_alignment(self.tokenizer, self.teacher_tokenizer)
        self.student_vocab_size = _model_vocab_size(self.tokenizer)
        self.teacher_vocab_size = _model_vocab_size(self.teacher_tokenizer)
        self.reflection_max_tokens = int(
            _cfg_get(self.error_span_config, "reflection_max_tokens", 768)
        )
        if self.reflection_max_tokens <= 0:
            raise ValueError("error_span_opd.reflection_max_tokens must be positive")
        self.reflection_max_attempts = int(
            _cfg_get(self.error_span_config, "reflection_max_attempts", 2)
        )
        if self.reflection_max_attempts <= 0:
            raise ValueError("error_span_opd.reflection_max_attempts must be positive")
        self._scoring_semaphore = asyncio.Semaphore(self.scoring_concurrency)

    @staticmethod
    def _teacher_model_path(config: Any) -> Any:
        direct = _cfg_get(_cfg_get(config, "error_span_opd", {}), "teacher_model", None)
        if direct:
            return direct
        dist = _cfg_get(config, "distillation", {})
        models = _cfg_get(dist, "teacher_models", {})
        model = _cfg_get(models, "teacher_model", {})
        return _cfg_get(model, "model_path", _cfg_get(model, "path", None))

    @staticmethod
    def _load_teacher_tokenizer_processor(model_path: str) -> tuple[Any, Any]:
        if not model_path:
            raise ValueError("error_span_opd.teacher_model is required")
        from transformers import AutoProcessor, AutoTokenizer

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        _set_image_processor_size(processor)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        return processor, tokenizer

    async def _compute_teacher_logprobs(self, *args: Any, **kwargs: Any) -> None:
        """Disable native whole-trajectory distillation; annotation owns Teacher calls."""

        del args, kwargs
        return None

    @staticmethod
    def _stable_rollout_seed(config: Any, trajectory: Mapping[str, Any], kwargs: Mapping[str, Any]) -> int:
        rollout = _cfg_get(_cfg_get(config, "actor_rollout_ref", {}), "rollout", {})
        base_seed = int(_cfg_get(rollout, "seed", 0))
        sample_id = kwargs.get("sample_id")
        if sample_id is None:
            extra = kwargs.get("extra_info")
            sample_id = _cfg_get(extra, "sample_id", None)
            if sample_id is None and isinstance(extra, Mapping):
                sample_id = extra.get("sample_id")
        if sample_id is None:
            sample_id = trajectory.get("sample_index", "unknown")
        if hasattr(sample_id, "item") and callable(sample_id.item):
            sample_id = sample_id.item()
        canonical = json.dumps(
            [base_seed, int(trajectory.get("step", -1)), str(sample_id), int(trajectory.get("rollout_n", 0))],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest(), "big") % (2**31)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs: Any,
    ) -> Any:
        seed = self._stable_rollout_seed(self.config, trajectory, kwargs)
        params = dict(sampling_params)
        params["seed"] = seed
        # Agent-loop implementations do not otherwise receive the trainer
        # trajectory.  Keep these private fields separate from dataset
        # ``extra_info`` so recovery loops can enforce step/validation scope.
        kwargs = dict(kwargs)
        kwargs["error_span_step"] = int(trajectory["step"])
        kwargs["error_span_validate"] = bool(trajectory["validate"])
        output = await super()._run_agent_loop(
            params, trajectory, agent_name=agent_name, trace=trace, **kwargs
        )
        output.extra_fields["error_span_rollout_seed"] = seed
        receipt = output.extra_fields.get("routed_grounding_receipt")
        if isinstance(receipt, Mapping):
            receipt = dict(receipt)
            receipt["error_span_rollout_seed"] = seed
            output.extra_fields["routed_grounding_receipt"] = receipt
        return output

    def _mm_kwargs(self) -> dict[str, Any]:
        values = _cfg_get(_cfg_get(self.config, "data", {}), "mm_processor_kwargs", None)
        if values is None:
            values = getattr(self, "mm_processor_kwargs", None)
        result = dict(values or {})
        result["min_pixels"] = MIN_PIXELS
        result["max_pixels"] = MAX_PIXELS
        return result

    def _verified_target_category(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Load and verify the FineCops target category for one payload."""

        data_source = payload.get("data_source")
        if data_source is not None and str(data_source) != "finecops_ref":
            raise ValueError(
                "overlay_verified_target_category requires data_source='finecops_ref', "
                f"got {data_source!r}"
            )
        config = getattr(self, "error_span_config", {})
        configured_paths = getattr(self, "finecops_positive_annotation_paths", None)
        if configured_paths is not None:
            # The worker stores resolved paths after normal construction.  A
            # small mapping keeps this helper usable in CPU-only unit tests
            # that instantiate the worker with ``object.__new__``.
            config = {"finecops_positive_annotation_paths": configured_paths}
        return load_finecops_target_category(
            config=config,
            sample_id=payload.get("sample_id"),
            source_split=payload.get("source_split", payload.get("split")),
            source_annotation_id=payload.get("source_annotation_id"),
            target_ann_id=payload.get("target_ann_id"),
            image_id=payload.get("image_id"),
            image_path=payload.get("image_path"),
            expression=payload.get("expression"),
        )

    @staticmethod
    def _render_processor(
        processor: Any,
        messages: list[dict[str, Any]],
        images: list[Any],
        mm_kwargs: Mapping[str, Any],
        *,
        enable_thinking: bool = True,
    ) -> tuple[str, list[int]]:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        processed = processor(
            text=[rendered],
            images=images,
            return_tensors="pt",
            padding=False,
            **dict(mm_kwargs),
        )
        return str(rendered), _processor_input_ids(processed)

    def _student_prompt(self, expression: str, image: Any, expected_ids: Sequence[int]) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        # RLHFDataset replaces the leading ``<image>`` marker
                        # with an image part and retains the separator newline.
                        # Preserve it here so scoring sees the exact rollout
                        # prompt prefix and token positions.
                        "text": "\n" + STUDENT_INSTRUCTION.format(expression=expression),
                    },
                ],
            }
        ]
        rendered, ids = self._render_processor(self.processor, messages, [image], self._mm_kwargs())
        expected = [int(token_id) for token_id in expected_ids]
        if ids != expected:
            raise ValueError("rebuilt Student prompt token ids differ from the rollout prompt")
        return {
            "rendered": rendered,
            "prompt_ids": expected,
            "prompt_ids_sha256": _ids_hash(expected),
            "image_count": 1,
            "model_role": "current_policy",
        }

    def _teacher_prompt(
        self,
        *,
        expression: str,
        ground_truth_bbox: Sequence[Any],
        image: Any,
        overlay: Any,
        arm: str,
        reference_reasoning: str | None,
        target_category: str | None = None,
        target_category_record: Mapping[str, Any] | None = None,
        rosd_correct_solution: str | None = None,
        rosd_reflection: str | None = None,
    ) -> dict[str, Any]:
        instruction = STUDENT_INSTRUCTION.format(expression=expression)
        mode = getattr(self, "teacher_prompt_mode", None)
        if mode is None:
            mode = _teacher_prompt_mode(getattr(self, "error_span_config", {}))
        else:
            mode = _teacher_prompt_mode({"teacher_prompt_mode": mode})
        if mode == LEGACY_TEACHER_PROMPT_MODE:
            instruction += (
                "\n\nPrivileged evidence for the same grounding task: the first image is the original; "
                "the second highlights the true referent. Ground-truth rectangle in normalized [0,1000] xyxy: "
                + json.dumps(list(ground_truth_bbox), ensure_ascii=False)
                + ". Use this evidence to produce the correct grounding answer."
            )
        else:
            if not isinstance(target_category, str) or not target_category.strip():
                raise ValueError("verified Teacher prompt requires a nonempty target_category")
            instruction += (
                "\n\nPrivileged visual evidence for the same grounding task: the first image is the original; "
                "the second is the original image with the true referent highlighted. "
                "Verified target category from the FineCops positive source annotation: "
                + target_category.strip()
                + ". Use both images and this target category to produce the correct grounding answer."
            )
        if arm.upper() == "B" and reference_reasoning is not None:
            instruction += (
                "\n\nA successful reasoning attempt from the student on this exact image and question "
                "(final answer omitted):\n<successful_student_reasoning>\n"
                + reference_reasoning
                + "\n</successful_student_reasoning>"
            )
        if arm.upper() == "ROSD":
            if not isinstance(rosd_correct_solution, str) or not rosd_correct_solution.strip():
                raise ValueError("ROSD Teacher prompt requires a nonempty correct solution")
            if not isinstance(rosd_reflection, str) or not rosd_reflection.strip():
                raise ValueError("ROSD Teacher prompt requires a nonempty reflection")
            instruction += (
                "\n\nA successful student solution for the same image and query is provided as "
                "private Teacher context:\n<successful_student_solution>\n"
                + rosd_correct_solution.strip()
                + "\n</successful_student_solution>\n\n"
                "A reflector diagnosed the current student trajectory as follows:\n"
                "<reflection>\n"
                + rosd_reflection.strip()
                + "\n</reflection>\n\n"
                "Use the visual evidence, successful solution, and reflection when evaluating how the "
                "current student trajectory should proceed."
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        rendered, ids = self._render_processor(
            self.teacher_processor, messages, [image, overlay], self._mm_kwargs()
        )
        receipt = {
            "rendered": rendered,
            "prompt_ids": ids,
            "prompt_ids_sha256": _ids_hash(ids),
            "image_count": 2,
            "model_role": "frozen_teacher",
            "reference_in_prompt": bool(arm.upper() == "B" and reference_reasoning is not None),
            "rosd_correct_solution_in_prompt": bool(arm.upper() == "ROSD"),
            "rosd_reflection_in_prompt": bool(arm.upper() == "ROSD"),
            "teacher_prompt_mode": mode,
        }
        if mode == OVERLAY_VERIFIED_TARGET_CATEGORY_MODE:
            # Keep only provenance needed to audit deterministic source
            # resolution.  In particular, never copy a bbox field into this
            # Teacher prompt receipt.
            if target_category_record is None:
                raise ValueError("verified Teacher prompt requires target category provenance")
            receipt["target_category"] = target_category.strip()
            receipt["target_category_source"] = {
                key: target_category_record[key]
                for key in (
                    "source_annotation_path",
                    "source_split",
                    "source_annotation_id",
                    "target_ann_id",
                    "image_id",
                    "image_path",
                    "expression",
                )
                if key in target_category_record
            }
        return receipt

    def _split_response(self, response_ids: Sequence[int]) -> dict[str, Any]:
        ids = _protocol_ids(self.tokenizer, response_ids)
        decoded = decode_ids(self.tokenizer, ids)
        marker = "</think>"
        if decoded.count(marker) != 1:
            raise ValueError("Student response must contain exactly one </think>")
        marker_index = decoded.index(marker)
        close_text = decoded[: marker_index + len(marker)]
        close_end = exact_decoded_prefix_offset(self.tokenizer, ids, close_text)
        if close_end is None:
            raise ValueError("Student </think> does not end on an exact token boundary")
        reasoning_ids, close_ids = split_reasoning_close(self.tokenizer, ids[:close_end])
        answer_ids = ids[close_end:]
        bbox_prefix_count = exact_decoded_prefix_offset(self.tokenizer, answer_ids, STUDENT_BBOX_PREFIX)
        if bbox_prefix_count is None or bbox_prefix_count <= 0:
            raise ValueError("Student response does not use the canonical bbox answer prefix")
        bbox_prefix_ids = answer_ids[:bbox_prefix_count]
        if not answer_ids[bbox_prefix_count:]:
            raise ValueError("Student response has an empty bbox tail")
        if reasoning_ids + close_ids + answer_ids != ids:
            raise ValueError("Student response token split is not lossless")
        return {
            "protocol_ids": ids,
            "reasoning_ids": reasoning_ids,
            "reasoning_length": len(reasoning_ids),
            "reasoning_text": decode_ids(self.tokenizer, reasoning_ids),
            "close_ids": close_ids,
            "bbox_prefix_ids": bbox_prefix_ids,
            "bbox_prefix_text": decode_ids(self.tokenizer, bbox_prefix_ids),
            "answer_ids": answer_ids,
        }

    def _validate_reference(self, payload: Mapping[str, Any], arm: str) -> str | None:
        if arm.upper() != "B" or not bool(payload.get("reference_available", False)):
            return None
        reference = payload.get("reference_reasoning")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("arm B reference_available requires reference_reasoning")
        if any(marker in reference for marker in ("</think>", "<answer>", "</answer>")):
            raise ValueError("reference_reasoning must omit the final answer protocol")
        reference_ids = payload.get("reference_response_ids")
        if reference_ids is not None:
            ref_protocol = _protocol_ids(self.tokenizer, reference_ids)
            ref_text = decode_ids(self.tokenizer, ref_protocol)
            if ref_text.count("</think>") != 1:
                raise ValueError("reference_response_ids must contain one reasoning closure")
            ref_prefix = ref_text.split("</think>", 1)[0]
            if ref_prefix.startswith("<think>"):
                ref_prefix = ref_prefix[len("<think>"):]
            if ref_prefix != reference:
                raise ValueError("reference_reasoning differs from the supplied reference_response_ids prefix")
            if "<answer>" not in ref_text[ref_text.index("</think>") :]:
                raise ValueError("reference_response_ids must include the successful final answer")
        reference_sample_id = payload.get("reference_sample_id")
        if reference_sample_id is not None and str(reference_sample_id) != str(payload.get("sample_id")):
            raise ValueError("arm B reference must come from the same sample/question")
        return reference

    def _check_context(self, length: int, *, role: str, request_id: str) -> None:
        teacher_roles = {"frozen_teacher", "frozen_teacher_reflector"}
        limit = self.teacher_context_limit if role in teacher_roles else self.student_context_limit
        if limit and int(length) + _DUMMY_MAX_TOKENS > limit:
            raise ContextOverflowError(
                f"{role} request {request_id} has {length} prompt tokens and needs one dummy token, "
                f"exceeding max_model_len={limit}; refusing to truncate a causal prefix"
            )

    def _teacher_client(self) -> Any:
        clients = self.teacher_client
        if clients is None:
            raise RuntimeError("error-span OPD has no injected frozen Teacher client")
        if not isinstance(clients, Mapping):
            return clients
        if not clients:
            raise RuntimeError("error-span OPD has no injected frozen Teacher client")
        if len(clients) != 1:
            key = _cfg_get(self.error_span_config, "teacher_client_key", None)
            if key is None or key not in clients:
                raise RuntimeError(
                    "error-span OPD requires one routed Teacher client (or teacher_client_key); "
                    f"available={sorted(clients)}"
                )
            return clients[key]
        return next(iter(clients.values()))

    def _scoring_slot(self) -> asyncio.Semaphore:
        semaphore = getattr(self, "_scoring_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.scoring_concurrency)
            self._scoring_semaphore = semaphore
        return semaphore

    @staticmethod
    def _sampling_params(*, prompt_logprobs: int) -> dict[str, Any]:
        # Prompt log-probabilities are a forward pass.  Still provide a fixed
        # dummy generation budget and seed so backend scheduling cannot consume
        # an ambient RNG stream.
        return {
            "max_tokens": _DUMMY_MAX_TOKENS,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": _DUMMY_SEED,
            "prompt_logprobs": int(prompt_logprobs),
            "logprobs": False,
        }

    def _reflection_sampling_params(self) -> dict[str, Any]:
        return {
            "max_tokens": int(getattr(self, "reflection_max_tokens", 768)),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": _DUMMY_SEED,
            "logprobs": False,
        }

    @staticmethod
    def _tag_content(text: str, tag: str) -> str | None:
        import re

        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE
        )
        if match is None:
            return None
        return match.group(1).strip()

    def _parse_rosd_reflection(
        self, text: str, *, require_error_quote: bool
    ) -> tuple[str | None, str]:
        explanation = (
            self._tag_content(text, "explaination")
            or self._tag_content(text, "explanation")
            or self._tag_content(text, "reflection")
        )
        error_quote = self._tag_content(text, "error_quote")
        if explanation is None or not explanation.strip():
            raise ValueError("ROSD reflection lacks a nonempty explanation tag")
        if require_error_quote and (error_quote is None or not error_quote.strip()):
            raise ValueError("ROSD wrong-trajectory reflection lacks a nonempty error_quote")
        return error_quote or None, explanation.strip()

    def _locate_rosd_error_quote(
        self, response_ids: Sequence[int], error_quote: str | None
    ) -> tuple[int, bool]:
        if not error_quote:
            return 0, False
        ids = [int(token_id) for token_id in response_ids]
        response_text = decode_ids(self.tokenizer, ids)
        candidates = [error_quote]
        stripped = error_quote.strip()
        if stripped and stripped != error_quote:
            candidates.append(stripped)
        for candidate in candidates:
            start_char = response_text.find(candidate)
            if start_char < 0:
                continue
            prefix_text = response_text[:start_char]
            offset = exact_decoded_prefix_offset(self.tokenizer, ids, prefix_text)
            if offset is not None:
                return int(offset), True
        return 0, False

    def _rosd_reflection_prompt(
        self,
        *,
        expression: str,
        current_solution: str,
        correct_solution: str,
        sample_success: bool,
        image: Any,
        overlay: Any,
        mm_kwargs: Mapping[str, Any],
        format_repair_of: str | None = None,
    ) -> dict[str, Any]:
        if sample_success:
            task = (
                "The current student solution is correct. Explain the visual evidence and reasoning that make "
                "it correct in at most 120 words. Return exactly one nonempty "
                "<explaination>...</explaination> section and no answer to the original query."
            )
        else:
            task = (
                "Diagnose the current incorrect student solution. Copy only the shortest exact phrase or "
                "sentence that reveals its earliest error into <error_quote>; it must be verbatim, contain no "
                "line break, and be at most 30 words. Close </error_quote> immediately after that quote. Then "
                "explain the exact mistake, how to fix it, and the correct visual reasoning in at most 120 "
                "words in <explaination>. Do not copy the full solution. Return exactly these two sections and "
                "no answer:\n"
                "<error_quote>...</error_quote>\n<explaination>...</explaination>"
            )
        instruction = (
            "The first image is the original scene. The second image highlights the true referent.\n"
            f"[Query]\n{expression}\n\n"
            f"[Successful Student Solution]\n{correct_solution}\n\n"
            f"[Current Student Solution]\n{current_solution}\n\n"
            f"[Task]\n{task}"
        )
        if format_repair_of is not None:
            instruction += (
                "\n\n[Format Repair]\nYour previous response did not contain the required closed, nonempty "
                "sections. Reformat the diagnosis now. Do not continue the previous analysis and do not "
                "repeat the full student solution.\n[Malformed Previous Response]\n"
                + format_repair_of[:3000]
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        rendered, ids = self._render_processor(
            self.teacher_processor,
            messages,
            [image, overlay],
            mm_kwargs,
            enable_thinking=False,
        )
        return {
            "rendered": rendered,
            "prompt_ids": ids,
            "prompt_ids_sha256": _ids_hash(ids),
            "image_count": 2,
            "image_roles": ["original", "gt_overlay"],
            "sample_success": bool(sample_success),
        }

    async def _generate_rosd_reflection(
        self,
        *,
        sample_id: str,
        expression: str,
        current_solution: str,
        correct_solution: str,
        sample_success: bool,
        image: Any,
        overlay: Any,
        mm_kwargs: Mapping[str, Any],
    ) -> tuple[str | None, str, dict[str, Any]]:
        params = self._reflection_sampling_params()
        started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        malformed_previous: str | None = None
        last_error = "unknown reflection format error"
        max_attempts = int(getattr(self, "reflection_max_attempts", 2))
        for attempt_index in range(max_attempts):
            prompt = self._rosd_reflection_prompt(
                expression=expression,
                current_solution=current_solution,
                correct_solution=correct_solution,
                sample_success=sample_success,
                image=image,
                overlay=overlay,
                mm_kwargs=mm_kwargs,
                format_repair_of=malformed_previous,
            )
            request_id = "rosd-reflection-" + _request_hash(
                {
                    "sample_id": sample_id,
                    "sample_success": bool(sample_success),
                    "attempt": attempt_index + 1,
                }
            )[:32]
            self._check_context(
                len(prompt["prompt_ids"]) + int(params["max_tokens"]),
                role="frozen_teacher_reflector",
                request_id=request_id,
            )
            attempt_started = time.perf_counter()
            async with self._scoring_slot():
                output = await self._teacher_client().generate(
                    request_id=request_id,
                    prompt_ids=prompt["prompt_ids"],
                    sampling_params=dict(params),
                    image_data=[image, overlay],
                    mm_processor_kwargs=dict(mm_kwargs),
                    priority=0,
                )
            output_ids = [int(token_id) for token_id in getattr(output, "token_ids", [])]
            raw_text = decode_ids(self.teacher_tokenizer, output_ids)
            attempt_receipt = {
                "attempt": attempt_index + 1,
                "request_id": request_id,
                "prompt": prompt,
                "output_token_ids": output_ids,
                "output_token_count": len(output_ids),
                "raw_output": raw_text,
                "timing": {"seconds": float(time.perf_counter() - attempt_started)},
            }
            try:
                error_quote, explanation = self._parse_rosd_reflection(
                    raw_text, require_error_quote=not sample_success
                )
            except ValueError as error:
                last_error = str(error)
                attempt_receipt["parse_error"] = last_error
                attempts.append(attempt_receipt)
                malformed_previous = raw_text
                continue
            attempts.append(attempt_receipt)
            receipt = {
                "kind": "rosd_reflection_generation",
                "model_role": "frozen_teacher_reflector",
                "model": self.teacher_model_path,
                "request_id": request_id,
                "prompt": prompt,
                "sampling_params": _jsonable(params),
                "output_token_ids": output_ids,
                "output_token_count": len(output_ids),
                "raw_output": raw_text,
                "error_quote": error_quote,
                "explaination": explanation,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "timing": {"seconds": float(time.perf_counter() - started)},
            }
            return error_quote, explanation, receipt
        raise ROSDReflectionFormatError(
            f"{last_error} after {max_attempts} bounded attempts",
            attempts=attempts,
        )

    @staticmethod
    def _output_extra(output: Any) -> Mapping[str, Any]:
        extra = getattr(output, "extra_fields", None)
        if not isinstance(extra, Mapping):
            raise ValueError("LLM backend output is missing extra_fields")
        return extra

    def _score_rows(
        self,
        *,
        output: Any,
        sequence_ids: Sequence[int],
        prompt_length: int,
        target_length: int,
    ) -> tuple[list[float], list[list[int]], list[list[float]]]:
        extra = self._output_extra(output)
        if "prompt_ids" not in extra or "prompt_logprobs" not in extra:
            raise ValueError("Student score response lacks native prompt_ids/prompt_logprobs rows")
        row_ids = _rows(extra["prompt_ids"], name="prompt_ids")
        row_lps = _rows(extra["prompt_logprobs"], name="prompt_logprobs")
        if len(row_ids) != len(sequence_ids) or len(row_lps) != len(sequence_ids):
            raise ValueError(
                "Student score prompt rows must include one native row per sequence token plus the dummy row: "
                f"got ids={len(row_ids)}, logprobs={len(row_lps)}, sequence={len(sequence_ids)}"
            )
        start = int(prompt_length) - 1
        end = start + int(target_length)
        if start < 0 or end > len(sequence_ids) - 1:
            raise ValueError("Student score target rows fall outside the native causal rows")
        normalized_ids: list[list[int]] = []
        normalized_lps: list[list[float]] = []
        for index, (ids, lps) in enumerate(zip(row_ids, row_lps, strict=True)):
            if len(ids) != 1 or len(lps) != 1:
                raise ValueError("prompt_logprobs=0 must return exactly one id/logprob per row")
            if ids[0] is None or lps[0] is None:
                raise ValueError(f"Student score row {index} contains a missing native value")
            token_id = int(ids[0])
            logprob = float(lps[0])
            # Multimodal vLLM can report -inf for non-target image/special
            # rows.  They are retained in the raw receipt for alignment but
            # never enter the score.  Only rows that predict the fixed target
            # require finite, non-positive log-probabilities.
            if start <= index < end and (not math.isfinite(logprob) or logprob > 1e-4):
                raise ValueError(f"Student score row {index} has invalid logprob {logprob!r}")
            normalized_ids.append([token_id])
            normalized_lps.append([logprob])
            if index < len(sequence_ids) - 1 and token_id != int(sequence_ids[index + 1]):
                raise ValueError(
                    "Student score backend changed the requested token history at row "
                    f"{index}: {token_id} != {sequence_ids[index + 1]}"
                )
        target_lps = [normalized_lps[index][0] for index in range(start, end)]
        if not all(math.isfinite(value) for value in target_lps):
            raise ValueError("Student target logprob contains NaN or Inf")
        return target_lps, normalized_ids, normalized_lps

    async def _score_student_target(
        self,
        *,
        sample_id: str,
        boundary: Mapping[str, Any],
        condition: str,
        score_prompt_ids: Sequence[int],
        target: Mapping[str, Any],
        image: Any,
        mm_kwargs: Mapping[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        target_ids = [int(token_id) for token_id in target["ids"]]
        sequence_ids = [int(token_id) for token_id in score_prompt_ids] + target_ids
        request_id = "error-span-score-" + _request_hash(
            {"sample_id": sample_id, "boundary": int(boundary["token_offset"]), "condition": condition}
        )[:32]
        self._check_context(len(sequence_ids), role="current_policy", request_id=request_id)
        params = self._sampling_params(prompt_logprobs=0)
        request_started = time.perf_counter()
        async with self._scoring_slot():
            output = await self.llm_client.generate(
                request_id=request_id,
                prompt_ids=sequence_ids,
                sampling_params=dict(params),
                image_data=[image],
                mm_processor_kwargs=dict(mm_kwargs),
                priority=0,
            )
        target_lps, row_ids, row_lps = self._score_rows(
            output=output,
            sequence_ids=sequence_ids,
            prompt_length=len(score_prompt_ids),
            target_length=len(target_ids),
        )
        receipt = {
            "kind": "student_fixed_target_score",
            "condition": condition,
            "model_role": "current_policy",
            "model": self.student_model_path,
            "request_id": request_id,
            "request_hash": _request_hash(
                {"prompt_ids": sequence_ids, "sampling_params": params, "image_count": 1}
            ),
            "prompt_token_count": len(score_prompt_ids),
            "prompt_ids_sha256": _ids_hash(score_prompt_ids),
            "sequence_token_count": len(sequence_ids),
            "sequence_ids_sha256": _ids_hash(sequence_ids),
            "target_ids": target_ids,
            "target_ids_sha256": _ids_hash(target_ids),
            "target_logprobs": target_lps,
            "mean_logprob": float(sum(target_lps) / len(target_lps)),
            "predictor_positions": list(range(len(score_prompt_ids) - 1, len(sequence_ids) - 1)),
            "sampling_params": _jsonable(params),
            "image_count": 1,
            "image_roles": ["original"],
            "future_trajectory_tokens_in_prompt": False,
            "raw_prompt_rows_sha256": _request_hash({"ids": row_ids, "logprobs": row_lps}),
            "server_extra_fields": _backend_metadata(
                self._output_extra(output), excluded={"prompt_ids", "prompt_logprobs"}
            ),
            "timing": {"seconds": float(time.perf_counter() - request_started)},
            "finite": True,
        }
        return receipt["mean_logprob"], receipt

    def _topk_rows(
        self, *, output: Any, sequence_length: int
    ) -> tuple[list[list[int]], list[list[float]], Mapping[str, Any]]:
        extra = self._output_extra(output)
        if "prompt_ids" not in extra or "prompt_logprobs" not in extra:
            raise ValueError("Teacher response lacks native prompt_ids/prompt_logprobs rows")
        raw_ids = _rows(extra["prompt_ids"], name="teacher prompt_ids")
        raw_lps = _rows(extra["prompt_logprobs"], name="teacher prompt_logprobs")
        if len(raw_ids) != sequence_length or len(raw_lps) != sequence_length:
            raise ValueError(
                "Teacher top-k rows must include one native row per sequence token plus the dummy row: "
                f"got ids={len(raw_ids)}, logprobs={len(raw_lps)}, sequence={sequence_length}"
            )
        ids_rows: list[list[int]] = []
        lp_rows: list[list[float]] = []
        for row_index, (ids, lps) in enumerate(zip(raw_ids, raw_lps, strict=True)):
            if len(ids) != len(lps) or len(ids) < self.topk:
                raise ValueError(
                    f"Teacher row {row_index} has {len(ids)} candidates; expected at least {self.topk}"
                )
            pairs: list[tuple[int, float, int]] = []
            for rank, (token_id, logprob) in enumerate(zip(ids, lps, strict=True)):
                if token_id is None or logprob is None:
                    raise ValueError(f"Teacher row {row_index} contains a missing top-k value")
                token_id = int(token_id)
                logprob = float(logprob)
                if token_id < 0 or token_id >= self.teacher_vocab_size or not math.isfinite(logprob):
                    raise ValueError(f"Teacher row {row_index} contains an invalid id/logprob")
                pairs.append((token_id, logprob, rank))
            # verl appends one all-zero dummy row for the final sequence token;
            # no causal target is read from it.  Preserve it for the native
            # offset helper, but do not mistake its repeated pad ids for a
            # ranked vocabulary row.
            if row_index == sequence_length - 1:
                ids_rows.append([item[0] for item in pairs[: self.topk]])
                lp_rows.append([item[1] for item in pairs[: self.topk]])
                continue
            # A raw vLLM response may contain the sampled token in addition to
            # the requested top-k.  Sorting by native logprob and retaining the
            # first K recovers the true ranked top-k without renormalization.
            pairs.sort(key=lambda item: (-item[1], item[2]))
            selected = pairs[: self.topk]
            selected_ids = [item[0] for item in selected]
            selected_lps = [item[1] for item in selected]
            if len(set(selected_ids)) != self.topk:
                raise ValueError(f"Teacher row {row_index} contains duplicate top-k token ids")
            ids_rows.append(selected_ids)
            lp_rows.append(selected_lps)
        if sequence_length > 1:
            validate_topk_mass(torch.tensor(lp_rows[:-1], dtype=torch.float32))
        return ids_rows, lp_rows, extra

    async def _teacher_topk(
        self,
        *,
        sample_id: str,
        boundary: Mapping[str, Any],
        teacher_prompt: Mapping[str, Any],
        trajectory_ids: Sequence[int],
        before: int,
        after: int,
        image: Any,
        overlay: Any,
        mm_kwargs: Mapping[str, Any],
    ) -> tuple[list[list[int]], list[list[float]], dict[str, Any]]:
        prompt_ids = [int(token_id) for token_id in teacher_prompt["prompt_ids"]]
        causal_prefix = [int(token_id) for token_id in trajectory_ids[:after]]
        sequence_ids = prompt_ids + causal_prefix
        request_id = "error-span-teacher-" + _request_hash(
            {"sample_id": sample_id, "before": before, "after": after}
        )[:32]
        self._check_context(len(sequence_ids), role="frozen_teacher", request_id=request_id)
        params = self._sampling_params(prompt_logprobs=self.topk)
        request_started = time.perf_counter()
        async with self._scoring_slot():
            output = await self._teacher_client().generate(
                request_id=request_id,
                prompt_ids=sequence_ids,
                sampling_params=dict(params),
                image_data=[image, overlay],
                mm_processor_kwargs=dict(mm_kwargs),
                priority=0,
            )
        topk_ids, topk_lps, extra = self._topk_rows(output=output, sequence_length=len(sequence_ids))
        selected_ids, selected_lps = extract_prompt_span_topk(
            topk_ids,
            topk_lps,
            teacher_prompt_length=len(prompt_ids),
            before=before,
            after=after,
            top_k=self.topk,
        )
        selected_ids_list = [[int(item) for item in row] for row in selected_ids.tolist()]
        selected_lps_list = [[float(item) for item in row] for row in selected_lps.tolist()]
        if len(selected_ids_list) != after - before:
            raise ValueError("Teacher top-k span length differs from the selected natural span")
        for row in selected_ids_list:
            if len(row) != self.topk or len(set(row)) != self.topk:
                raise ValueError("Teacher top-k span rows must contain unique 30-token rankings")
        validate_topk_mass(torch.tensor(selected_lps_list, dtype=torch.float32))
        receipt = {
            "kind": "teacher_trajectory_interval_topk",
            "model_role": "frozen_teacher",
            "model": self.teacher_model_path,
            "request_id": request_id,
            "request_hash": _request_hash(
                {"prompt_ids": sequence_ids, "sampling_params": params, "image_count": 2}
            ),
            "prompt_token_count": len(prompt_ids),
            "prompt_ids_sha256": _ids_hash(prompt_ids),
            "sequence_token_count": len(sequence_ids),
            "sequence_ids_sha256": _ids_hash(sequence_ids),
            "causal_prefix_token_count": len(causal_prefix),
            "causal_prefix_ids_sha256": _ids_hash(causal_prefix),
            "before_offset": before,
            "after_offset": after,
            "span_length": after - before,
            "sampling_params": _jsonable(params),
            "image_count": 2,
            "image_roles": ["original", "gt_overlay"],
            # The packed request includes the original trajectory through the
            # target interval. Causal attention still prevents any prediction
            # row from reading a later trajectory token.
            "causal_future_tokens_visible": False,
            "selected_topk_ids": selected_ids_list,
            "selected_topk_logprobs": selected_lps_list,
            "raw_prompt_rows_sha256": _request_hash({"ids": topk_ids, "logprobs": topk_lps}),
            "server_extra_fields": _backend_metadata(extra, excluded={"prompt_ids", "prompt_logprobs"}),
            "timing": {"seconds": float(time.perf_counter() - request_started)},
            "finite": True,
        }
        return selected_ids_list, selected_lps_list, receipt

    async def _score_boundary(
        self,
        *,
        sample_id: str,
        boundary: Mapping[str, Any],
        prompt_ids: Sequence[int],
        reasoning: Mapping[str, Any],
        targets: Mapping[str, Mapping[str, Any]],
        image: Any,
        mm_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        offset = int(boundary["token_offset"])
        fixed_prompt = (
            [int(token_id) for token_id in prompt_ids]
            + [int(token_id) for token_id in reasoning["reasoning_ids"][:offset]]
            + [int(token_id) for token_id in reasoning["close_ids"]]
            + [int(token_id) for token_id in reasoning["bbox_prefix_ids"]]
        )
        gt_score, gt_receipt = await self._score_student_target(
            sample_id=sample_id,
            boundary=boundary,
            condition="ground_truth",
            score_prompt_ids=fixed_prompt,
            target=targets["ground_truth"],
            image=image,
            mm_kwargs=mm_kwargs,
        )
        wrong_score, wrong_receipt = await self._score_student_target(
            sample_id=sample_id,
            boundary=boundary,
            condition="own_final_wrong_bbox",
            score_prompt_ids=fixed_prompt,
            target=targets["own_final_wrong_bbox"],
            image=image,
            mm_kwargs=mm_kwargs,
        )
        advantage = float(gt_score - wrong_score)
        if not math.isfinite(advantage):
            raise ValueError("nonfinite GT-minus-own-final Student advantage")
        return {
            "sample_id": sample_id,
            "checkpoint_index": int(boundary["checkpoint_index"]),
            "checkpoint_kind": str(boundary["checkpoint_kind"]),
            "before_offset": offset,
            "after_offset": offset,
            "token_offset": offset,
            "reasoning_progress": float(boundary["reasoning_progress"]),
            "correct_logprob": float(gt_score),
            "own_final_wrong_logprob": float(wrong_score),
            "correct_advantage": advantage,
            "gt_score_receipt": gt_receipt,
            "own_final_score_receipt": wrong_receipt,
            "score_prompt_ids_sha256": _ids_hash(fixed_prompt),
            "score_prompt_token_count": len(fixed_prompt),
        }

    @staticmethod
    def _payload_ids(payload: Mapping[str, Any], key: str) -> list[int]:
        values = payload.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"payload.{key} must be a token-id sequence")
        ids: list[int] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError(f"payload.{key} contains a boolean token id")
            try:
                ids.append(int(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"payload.{key} contains a non-integer token id") from error
        return ids

    async def _annotate_rosd(
        self,
        *,
        payload: Mapping[str, Any],
        sample_id: str,
        expression: str,
        prompt_mode: str,
        student_prompt: Mapping[str, Any],
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        ground_truth_bbox: Sequence[Any],
        target_category_record: Mapping[str, Any] | None,
        image: Any,
        overlay: Any,
        mm_kwargs: Mapping[str, Any],
        started: float,
    ) -> dict[str, Any]:
        del prompt_ids
        response_ids = [int(token_id) for token_id in response_ids]
        if not response_ids:
            raise ValueError("ROSD requires a nonempty recorded student response")
        sample_success = bool(payload.get("sample_success", False))
        current_solution = decode_ids(self.tokenizer, response_ids)
        reference = payload.get("reference_full_response")
        if sample_success:
            correct_solution = current_solution
        elif isinstance(reference, str) and reference.strip():
            correct_solution = reference.strip()
        else:
            raise ValueError("ROSD wrong trajectory requires a same-group successful response")

        try:
            error_quote, explanation, reflection_receipt = await self._generate_rosd_reflection(
                sample_id=sample_id,
                expression=expression,
                current_solution=current_solution,
                correct_solution=correct_solution,
                sample_success=sample_success,
                image=image,
                overlay=overlay,
                mm_kwargs=mm_kwargs,
            )
        except ROSDReflectionFormatError as error:
            return {
                "sample_id": sample_id,
                "arm": "ROSD",
                "method_variant": ROSD_METHOD,
                "step": int(payload.get("step", -1)),
                "sample_success": sample_success,
                "selection": None,
                "skip_reason": "rosd_reflection_format_failure",
                "request": {"reflection_attempts": error.attempts},
                "timing": {"total_seconds": float(time.perf_counter() - started)},
                "response_length": len(response_ids),
                "response_token_ids_sha256": _ids_hash(response_ids),
            }
        if sample_success:
            start_offset, quote_found = 0, False
        else:
            start_offset, quote_found = self._locate_rosd_error_quote(
                response_ids, error_quote
            )
        supervision_before, supervision_after = resolve_supervision_offsets(
            "error_suffix",
            response_length=len(response_ids),
            selected_before=start_offset,
            selected_after=min(len(response_ids), start_offset + 1),
        )
        teacher_prompt = self._teacher_prompt(
            expression=expression,
            ground_truth_bbox=ground_truth_bbox,
            image=image,
            overlay=overlay,
            arm="ROSD",
            reference_reasoning=None,
            target_category=None
            if target_category_record is None
            else target_category_record["target_category"],
            target_category_record=target_category_record,
            rosd_correct_solution=correct_solution,
            rosd_reflection=explanation,
        )
        teacher_ids, teacher_lps, teacher_receipt = await self._teacher_topk(
            sample_id=sample_id,
            boundary={},
            teacher_prompt=teacher_prompt,
            trajectory_ids=response_ids,
            before=supervision_before,
            after=supervision_after,
            image=image,
            overlay=overlay,
            mm_kwargs=mm_kwargs,
        )
        selection = {
            "before_offset": start_offset,
            "after_offset": min(len(response_ids), start_offset + 1),
            "supervision_scope": "error_suffix",
            "supervision_before_offset": supervision_before,
            "supervision_after_offset": supervision_after,
            "supervision_token_count": supervision_after - supervision_before,
            "error_quote": error_quote,
            "error_quote_found": quote_found,
            "fallback_to_full_response": bool(not sample_success and not quote_found),
            "rule": "ROSD exact error_quote start through recorded response end; full-response fallback if quote is absent",
        }
        return {
            "sample_id": sample_id,
            "arm": "ROSD",
            "method_variant": ROSD_METHOD,
            "step": int(payload.get("step", -1)),
            "sample_success": sample_success,
            "model": {
                "student": self.student_model_path,
                "teacher": self.teacher_model_path,
                "student_role": "current_policy",
                "teacher_role": "frozen_teacher_and_reflector",
                "tokenizer_alignment": self.tokenizer_alignment,
            },
            "request": {
                "student_prompt": student_prompt,
                "reflection": reflection_receipt,
                "teacher_prompt": teacher_prompt,
                "teacher": teacher_receipt,
            },
            "timing": {
                "reflection_seconds": float(reflection_receipt["timing"]["seconds"]),
                "teacher_seconds": float(teacher_receipt["timing"]["seconds"]),
                "total_seconds": float(time.perf_counter() - started),
            },
            "boundary_scores": [],
            "score_receipts": [],
            "teacher_topk_ids": teacher_ids,
            "teacher_topk_logprobs": teacher_lps,
            "selection": selection,
            "raw_score": None,
            "rawscore": None,
            "response_length": len(response_ids),
            "response_token_ids_sha256": _ids_hash(response_ids),
            "supervision_scope": "error_suffix",
            "token_contract": {
                "student_prompt_ids_exact_payload": True,
                "teacher_prompt_has_original_and_gt_overlay": True,
                "teacher_prompt_mode": prompt_mode,
                "teacher_prompt_contains_numeric_gt_bbox": prompt_mode
                == LEGACY_TEACHER_PROMPT_MODE,
                "teacher_prediction_is_causal": True,
                "student_prompt_contains_reflection": False,
                "student_prompt_contains_successful_response": False,
            },
        }

    async def annotate_error_span(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Score every natural boundary and return one local Teacher span.

        The method intentionally lets any processor, context, or backend error
        propagate.  The manager records successful annotations atomically and
        must see a failed request instead of silently turning it into an
        unlabelled trajectory.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("error-span annotation payload must be a mapping")
        started = time.perf_counter()
        sample_id = str(payload.get("sample_id"))
        expression = payload.get("expression")
        if not isinstance(expression, str) or not expression:
            raise ValueError("payload.expression must be a nonempty string")
        arm = str(payload.get("arm", "A"))
        if arm.upper() not in {"A", "B", "SINGLE", "ROSD"}:
            # The manager may carry the native loop arm name while the
            # scientific payload uses A/B.  Treat standard_opd as A only.
            if arm != "standard_opd":
                raise ValueError(f"error-span arm must be A, B, single, or ROSD, got {arm!r}")
            arm = "A"
        prompt_mode = getattr(self, "teacher_prompt_mode", None)
        if prompt_mode is None:
            prompt_mode = _teacher_prompt_mode(getattr(self, "error_span_config", {}))
        else:
            prompt_mode = _teacher_prompt_mode({"teacher_prompt_mode": prompt_mode})
        prompt_ids = self._payload_ids(payload, "prompt_ids")
        response_ids = self._payload_ids(payload, "response_ids")
        if not prompt_ids:
            raise ValueError("payload.prompt_ids must be nonempty")
        gt_bbox = payload.get("ground_truth_bbox")
        wrong_bbox = payload.get("original_final_bbox")
        if not isinstance(gt_bbox, Sequence) or isinstance(gt_bbox, (str, bytes)) or len(gt_bbox) != 4:
            raise ValueError("payload.ground_truth_bbox must contain four coordinates")
        method_variant = getattr(self, "method_variant", LOCAL_SPAN_METHOD)
        if method_variant != ROSD_METHOD and (
            not isinstance(wrong_bbox, Sequence)
            or isinstance(wrong_bbox, (str, bytes))
            or len(wrong_bbox) != 4
        ):
            raise ValueError("payload.original_final_bbox must contain four coordinates")
        reference = None if method_variant == ROSD_METHOD else self._validate_reference(payload, arm)
        target_category_record = None
        if prompt_mode == OVERLAY_VERIFIED_TARGET_CATEGORY_MODE:
            # Resolve source provenance before any selectable-boundary shortcut
            # so absent or inconsistent metadata can never silently become a
            # Standard-OPD/no-target result.
            target_category_record = self._verified_target_category(payload)
        image_processor = getattr(self.processor, "image_processor", None)
        image_patch_size = getattr(image_processor, "patch_size", 14) or 14
        image = _load_rgb_image(payload.get("image_path"), image_patch_size=image_patch_size)
        if image is None:
            raise ValueError("payload.image_path is required")
        mm_kwargs = self._mm_kwargs()
        student_prompt = self._student_prompt(expression, image, prompt_ids)
        if method_variant == ROSD_METHOD:
            from verl.experimental.routed_grounding.teacher import gt_overlay

            overlay = gt_overlay(image, gt_bbox)
            return await self._annotate_rosd(
                payload=payload,
                sample_id=sample_id,
                expression=expression,
                prompt_mode=prompt_mode,
                student_prompt=student_prompt,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                ground_truth_bbox=gt_bbox,
                target_category_record=target_category_record,
                image=image,
                overlay=overlay,
                mm_kwargs=mm_kwargs,
                started=started,
            )
        reasoning = self._split_response(response_ids)
        reasoning_ids = reasoning["reasoning_ids"]
        reasoning_length = int(reasoning["reasoning_length"])
        if reasoning_length <= 0:
            raise ValueError("Student reasoning is empty; no strict natural span can be selected")
        checkpoints = extract_reasoning_checkpoints(
            self.tokenizer,
            reasoning_ids,
            sample_id=sample_id,
            final_bbox=wrong_bbox,
            bbox_iou=None,
            bbox_hit_gt=False,
        )
        strict = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint["checkpoint_kind"] == "natural_boundary"
            and 0 < int(checkpoint["token_offset"]) < reasoning_length
        ]
        base_receipt: dict[str, Any] = {
            "sample_id": sample_id,
            "arm": arm,
            "step": int(payload.get("step", -1)),
            "model": {
                "student": self.student_model_path,
                "teacher": self.teacher_model_path,
                "student_role": "current_policy",
                "teacher_role": "frozen_teacher",
                "tokenizer_alignment": self.tokenizer_alignment,
            },
            "request": {"student_prompt": student_prompt},
            "timing": {},
            "boundary_scores": [],
            "score_receipts": [],
            "teacher_topk_ids": [],
            "teacher_topk_logprobs": [],
            "selection": None,
            "raw_score": None,
            "rawscore": None,
            "reasoning_length": reasoning_length,
            "response_length": len(response_ids),
            "reasoning_token_ids_sha256": _ids_hash(reasoning_ids),
            "token_contract": {
                "student_prompt_ids_exact_payload": True,
                "reasoning_response_slice_exact": True,
                "selected_span_is_strict_intermediate": True,
                "teacher_prompt_has_original_and_gt_overlay": True,
                "teacher_prompt_mode": prompt_mode,
                "teacher_prompt_contains_numeric_gt_bbox": prompt_mode == LEGACY_TEACHER_PROMPT_MODE,
                "teacher_prediction_is_causal": True,
            },
        }
        if not strict:
            base_receipt["skip_reason"] = "no_selectable_strict_intermediate_natural_boundary"
            base_receipt["skipreason"] = base_receipt["skip_reason"]
            base_receipt["timing"] = {"total_seconds": float(time.perf_counter() - started)}
            return base_receipt

        import copy

        target_specs = {
            "ground_truth": canonical_target(self.tokenizer, gt_bbox),
            "own_final_wrong_bbox": canonical_target(self.tokenizer, wrong_bbox),
        }
        if target_specs["ground_truth"]["digit_token_ids"] != target_specs["own_final_wrong_bbox"]["digit_token_ids"]:
            raise ValueError("GT and own-final target digit token maps differ")
        overlay_started = time.perf_counter()
        from verl.experimental.routed_grounding.teacher import gt_overlay

        overlay = gt_overlay(image, gt_bbox)
        boundaries = sorted(checkpoints, key=lambda item: int(item["token_offset"]))
        score_started = time.perf_counter()
        scores: list[dict[str, Any]] = []
        for start_index in range(0, len(boundaries), self.scoring_concurrency):
            chunk = boundaries[start_index : start_index + self.scoring_concurrency]
            # Chunking bounds both coroutine count and in-flight requests.  A
            # boundary's GT and wrong-tail calls remain paired and sequential.
            chunk_scores = await asyncio.gather(
                *(
                    self._score_boundary(
                        sample_id=sample_id,
                        boundary=boundary,
                        prompt_ids=prompt_ids,
                        reasoning=reasoning,
                        targets=target_specs,
                        image=image,
                        mm_kwargs=mm_kwargs,
                    )
                    for boundary in chunk
                )
            )
            scores.extend(chunk_scores)
        base_receipt["boundary_scores"] = scores
        base_receipt["score_receipts"] = [
            {
                "request_id": receipt["request_id"],
                "condition": receipt["condition"],
                "sequence_ids_sha256": receipt["sequence_ids_sha256"],
                "target_ids_sha256": receipt["target_ids_sha256"],
                "target_logprobs": receipt["target_logprobs"],
                "mean_logprob": receipt["mean_logprob"],
            }
            for score in scores
            for receipt in (score["gt_score_receipt"], score["own_final_score_receipt"])
        ]
        selected = select_maximum_decline(scores)
        if selected is None:
            raise ValueError("natural boundary scores produced no selectable strict intermediate span")
        before = int(selected["before_offset"])
        after = int(selected["after_offset"])
        if not (0 <= before < after < reasoning_length):
            raise ValueError("selector produced a span outside strict intermediate reasoning")
        before_record = next(score for score in scores if int(score["after_offset"]) == before)
        after_record = next(score for score in scores if int(score["after_offset"]) == after)
        selected = dict(selected)
        selected.update(
            {
                "before_checkpoint_index": int(before_record["checkpoint_index"]),
                "after_checkpoint_index": int(after_record["checkpoint_index"]),
                "span_ids_sha256": _ids_hash(reasoning_ids[before:after]),
                "span_text": decode_ids(self.tokenizer, reasoning_ids[before:after]),
            }
        )
        supervision_before, supervision_after = resolve_supervision_offsets(
            getattr(self, "supervision_scope", SELECTED_SPAN_SCOPE),
            response_length=len(response_ids),
            selected_before=before,
            selected_after=after,
        )
        selected.update(
            {
                "supervision_scope": getattr(
                    self, "supervision_scope", SELECTED_SPAN_SCOPE
                ),
                "supervision_before_offset": supervision_before,
                "supervision_after_offset": supervision_after,
                "supervision_token_count": supervision_after - supervision_before,
            }
        )
        teacher_prompt = self._teacher_prompt(
            expression=expression,
            ground_truth_bbox=gt_bbox,
            image=image,
            overlay=overlay,
            arm=arm,
            reference_reasoning=reference,
            target_category=None if target_category_record is None else target_category_record["target_category"],
            target_category_record=target_category_record,
        )
        teacher_ids, teacher_lps, teacher_receipt = await self._teacher_topk(
            sample_id=sample_id,
            boundary=after_record,
            teacher_prompt=teacher_prompt,
            trajectory_ids=response_ids,
            before=supervision_before,
            after=supervision_after,
            image=image,
            overlay=overlay,
            mm_kwargs=mm_kwargs,
        )
        base_receipt["selection"] = selected
        base_receipt["raw_score"] = float(selected["score"])
        base_receipt["rawscore"] = float(selected["score"])
        base_receipt["teacher_topk_ids"] = teacher_ids
        base_receipt["teacher_topk_logprobs"] = teacher_lps
        base_receipt["supervision_scope"] = selected["supervision_scope"]
        base_receipt["request"]["teacher_prompt"] = teacher_prompt
        base_receipt["request"]["teacher"] = teacher_receipt
        base_receipt["timing"] = {
            "overlay_seconds": float(time.perf_counter() - overlay_started),
            "score_seconds": float(time.perf_counter() - score_started),
            "teacher_seconds": float(teacher_receipt["timing"]["seconds"]),
            "total_seconds": float(time.perf_counter() - started),
        }
        return copy.deepcopy(base_receipt)


__all__ = [
    "ContextOverflowError",
    "ROSDReflectionFormatError",
    "ErrorSpanAgentLoop",
    "ErrorSpanWorker",
    "LEGACY_TEACHER_PROMPT_MODE",
    "OVERLAY_VERIFIED_TARGET_CATEGORY_MODE",
    "load_finecops_target_category",
]
