"""Frozen five-image comparison of saved and instance-description bridges.

This runner is deliberately read-only.  It loads the completed pilot records,
prefills each image once, replays the unchanged reasoning IDs to ``c0``,
``c1`` and ``cT``, and samples the same four bbox seeds under six arms:
saved-pilot entity versus the new ``instance_v2`` bridge, crossed with
``late_entity``, ``early_step0`` and ``early_span1``.  A failed new bridge
keeps all four draws for each affected arm and assigns them IoU zero.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Select the physical device before importing torch/model modules.  Importing
# this module for its CPU helpers never loads a model or initializes CUDA.
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
    DEFAULT_SAMPLE_IDS,
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
from timeline_self_distillation.entity_bridge import ENTITY_REGEX  # noqa: E402
from timeline_self_distillation.run_teacher_pilot import (  # noqa: E402
    BOX_OPEN,
    BOX_REGEX,
    MODEL,
    QUERY,
    SEED,
    constrained_generate,
    fork,
    generate_entity_bridge,
)
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou  # noqa: E402

CONDITIONS = ("late_entity", "early_step0", "early_span1")
OLD_ARMS = tuple(f"old__{condition}" for condition in CONDITIONS)
NEW_ARMS = tuple(f"new__{condition}" for condition in CONDITIONS)
ARM_NAMES = OLD_ARMS + NEW_ARMS
DRAWS_PER_SAMPLE = 4
TOTAL_EXPECTED_DRAWS = len(DEFAULT_SAMPLE_IDS) * DRAWS_PER_SAMPLE * len(ARM_NAMES)
_ROW_ID_RE = re.compile(r"^row-(?P<index>[0-9]+)$")
_DIRECT_DEPENDENCIES = (
    "timeline_self_distillation/run_entity_bridge_comparison.py",
    "timeline_self_distillation/run_teacher_pilot.py",
    "timeline_self_distillation/entity_bridge.py",
    "live_kv_probe_prototype/run_hf_fork.py",
    "reasoning_checkpoints/extractor.py",
    "reasoning_checkpoints/run_pilot.py",
    "verl/experimental/routed_grounding/router.py",
)

__all__ = [
    "ARM_NAMES",
    "CONDITIONS",
    "DRAWS_PER_SAMPLE",
    "NEW_ARMS",
    "OLD_ARMS",
    "TOTAL_EXPECTED_DRAWS",
    "entity_arm_inputs",
    "micro_eval_seed",
    "pilot_draw_seed",
    "pilot_draw_seeds",
    "summarize_records",
]


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not math.isfinite(converted):
        raise FloatingPointError(f"explicit nonfinite value: {value!r}")
    return converted


def _sample_index(sample_id: str) -> int:
    match = _ROW_ID_RE.fullmatch(str(sample_id))
    if match is None:
        raise ValueError(f"sample_id must use fixed row-N form, got {sample_id!r}")
    return int(match.group("index"))


def micro_eval_seed(base_seed: int, source_index: int, draw: int) -> int:
    """Return the old micro-run seed, used only as a negative-test contrast."""

    return int(base_seed) + 9_000_000 + int(source_index) * 1000 + int(draw)


def pilot_draw_seeds(record: Mapping[str, Any]) -> list[int]:
    """Extract the four seeds saved by the original pilot, without derivation."""

    conditions = record.get("conditions")
    if not isinstance(conditions, Mapping) or "early_span1" not in conditions:
        raise ValueError(f"{record.get('sample_id')}: pilot conditions/early_span1 are missing")
    canonical = conditions["early_span1"]
    if not isinstance(canonical, Mapping) or not isinstance(canonical.get("draws"), list):
        raise ValueError(f"{record.get('sample_id')}: early_span1 draws are missing")
    draws = list(canonical["draws"])
    if len(draws) != DRAWS_PER_SAMPLE:
        raise ValueError(f"{record.get('sample_id')}: expected {DRAWS_PER_SAMPLE} saved pilot draws, got {len(draws)}")
    if all(isinstance(draw, Mapping) and "draw" in draw for draw in draws):
        draws.sort(key=lambda draw: int(draw["draw"]))
        observed_draws = [int(draw["draw"]) for draw in draws]
        if observed_draws != list(range(DRAWS_PER_SAMPLE)):
            raise ValueError(f"{record.get('sample_id')}: saved draw indices are not 0..3")
    seeds = [int(draw["seed"]) for draw in draws]
    if len(set(seeds)) != DRAWS_PER_SAMPLE:
        raise ValueError(f"{record.get('sample_id')}: saved pilot seeds are not distinct")
    # All old pilot conditions are expected to use exactly these same seeds.
    for condition, value in conditions.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("draws"), list):
            continue
        condition_draws = list(value["draws"])
        if len(condition_draws) != DRAWS_PER_SAMPLE:
            raise ValueError(f"{record.get('sample_id')}: condition {condition} does not have four draws")
        if all(isinstance(draw, Mapping) and "draw" in draw for draw in condition_draws):
            condition_draws.sort(key=lambda draw: int(draw["draw"]))
        condition_seeds = [int(draw["seed"]) for draw in condition_draws]
        if condition_seeds != seeds:
            raise ValueError(f"{record.get('sample_id')}: condition {condition} seed drifted from pilot")
    return seeds


def pilot_draw_seed(record: Mapping[str, Any], draw: int) -> int:
    if not 0 <= int(draw) < DRAWS_PER_SAMPLE:
        raise IndexError(f"draw must be in [0, {DRAWS_PER_SAMPLE}), got {draw}")
    return pilot_draw_seeds(record)[int(draw)]


def _bridge_warning(audit: Mapping[str, Any]) -> bool:
    return bool(audit.get("echoes_request", False)) or audit.get("semantic_status") == "request_echo"


def entity_arm_inputs(record: Mapping[str, Any], bridge: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build old/new arm metadata without replacing saved text or fallback."""

    old_entity = str(record.get("entity", ""))
    old_failed = not bool(old_entity)
    audit = bridge.get("entity_audit")
    audit = dict(audit) if isinstance(audit, Mapping) else {"status": "missing_audit"}
    raw = bridge.get("entity_result")
    new_usable = bool(audit.get("usable_for_probe"))
    # The bridge contract decides usability.  In particular, an unresolved or
    # malformed result becomes empty; no original entity or GT fallback exists.
    new_entity = str(bridge.get("entity", "")) if new_usable else ""
    new_failed = not new_usable
    result: dict[str, dict[str, Any]] = {}
    for arm in OLD_ARMS:
        result[arm] = {
            "entity": old_entity,
            "entity_source": "pilot_saved_entity",
            "entity_failed": old_failed,
            "entity_result": record.get("entity_result"),
            "entity_audit": {"status": "saved_pilot_entity"},
            "request_echo_warning": False,
        }
    for arm in NEW_ARMS:
        result[arm] = {
            "entity": new_entity,
            "entity_source": "instance_v2_bridge",
            "entity_failed": new_failed,
            "entity_result": raw,
            "entity_audit": audit,
            "request_echo_warning": _bridge_warning(audit),
        }
    return result


def _draw_summary(draws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ious = [_finite_float(draw.get("iou"), 0.0) for draw in draws]
    return {
        "draw_count": len(draws),
        "mean_iou": _mean(ious),
        "acc_05": _mean([float(bool(draw.get("hit_05", False))) for draw in draws]),
        "parse_valid": sum(bool(draw.get("parse_valid", False)) for draw in draws),
        "numeric_valid": sum(bool(draw.get("numeric_valid", False)) for draw in draws),
        "entity_failed_draws": sum(bool(draw.get("entity_failed", False)) for draw in draws),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate every draw and every sample; never select a best draw."""

    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        draws = [draw for record in records for draw in record.get("conditions", {}).get(arm, {}).get("draws", [])]
        arms[arm] = _draw_summary(draws)
        arms[arm]["sample_count"] = sum(arm in record.get("conditions", {}) for record in records)
    per_sample: list[dict[str, Any]] = []
    for record in records:
        conditions = record.get("conditions", {})
        sample = {"sample_id": str(record.get("sample_id")), "arms": {}}
        for arm in ARM_NAMES:
            entry = conditions.get(arm, {})
            sample["arms"][arm] = {
                **_draw_summary(entry.get("draws", [])),
                "entity": entry.get("entity", ""),
                "entity_failed": bool(entry.get("entity_failed", False)),
            }
        for condition in CONDITIONS:
            old = sample["arms"][f"old__{condition}"]["mean_iou"]
            new = sample["arms"][f"new__{condition}"]["mean_iou"]
            sample["arms"][f"new__{condition}"]["delta_iou_vs_old"] = new - old
        per_sample.append(sample)
    old_parity = {}
    for arm in OLD_ARMS:
        parity_entries = [
            record.get("old_parity", {}).get(arm, {}) for record in records if arm in record.get("old_parity", {})
        ]
        old_parity[arm] = {
            "sample_count": len(parity_entries),
            "draw_count": sum(int(entry.get("draw_count", 0)) for entry in parity_entries),
            "ids_exact_count": sum(int(entry.get("ids_exact_count", 0)) for entry in parity_entries),
            "bbox_exact_count": sum(int(entry.get("bbox_exact_count", 0)) for entry in parity_entries),
            "iou_exact_count": sum(int(entry.get("iou_exact_count", 0)) for entry in parity_entries),
            "all_ids_exact": bool(parity_entries)
            and all(entry.get("all_ids_exact", False) for entry in parity_entries),
            "all_bbox_exact": bool(parity_entries)
            and all(entry.get("all_bbox_exact", False) for entry in parity_entries),
            "all_iou_exact": bool(parity_entries)
            and all(entry.get("all_iou_exact", False) for entry in parity_entries),
        }
    return {
        "sample_count": len(records),
        "draw_count": sum(value["draw_count"] for value in arms.values()),
        "arms": arms,
        "per_sample": per_sample,
        "old_parity": old_parity,
    }


def _tensor_digest(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().cpu()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def _cache_tensors(cache: Any) -> list[tuple[str, torch.Tensor]]:
    result: list[tuple[str, torch.Tensor]] = []
    for layer_index, layer in enumerate(cache.layers):
        for name in sorted(vars(layer)):
            value = getattr(layer, name)
            if isinstance(value, torch.Tensor):
                result.append((f"layers[{layer_index}].{name}", value))
    return result


def _snapshot_cache(cache: Any) -> list[dict[str, Any]]:
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


def _compare_cache(cache: Any, expected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actual = _snapshot_cache(cache)
    version_same = len(actual) == len(expected) and all(
        int(a["version"]) == int(e["version"]) for a, e in zip(actual, expected, strict=True)
    )
    content_same = len(actual) == len(expected) and all(
        a["sha256"] == e["sha256"] for a, e in zip(actual, expected, strict=True)
    )
    return {
        "tensor_count": len(actual),
        "version_unchanged": bool(version_same),
        "content_unchanged": bool(content_same),
        "unchanged": bool(actual == [dict(item) for item in expected]),
    }


def _restore_rope(model: Any, rope_delta: torch.Tensor) -> None:
    model.model.rope_deltas = rope_delta.clone()


def _build_states(model: Any, processor: Any, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prefill each fixed image once and retain C0/C1/CT branches."""

    states: list[dict[str, Any]] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for row in records:
            sample_id = str(row["sample_id"])
            reasoning_ids = [int(value) for value in row["reasoning_ids"]]
            if decode_ids(processor.tokenizer, reasoning_ids) != str(row["reasoning_text"]):
                raise RuntimeError(f"{sample_id}: reasoning IDs/text changed from pilot")
            _, processed = _render_and_process(
                processor,
                str(row["expression"]),
                _load_rgb_image(str(row["image_path"])),
            )
            prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
            if "prompt_token_count" in row and len(prompt_ids) != int(row["prompt_token_count"]):
                raise RuntimeError(f"{sample_id}: prompt token count drifted")
            if "prompt_input_ids" in row and prompt_ids != [int(value) for value in row["prompt_input_ids"]]:
                raise RuntimeError(f"{sample_id}: prompt IDs drifted")
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in processed.items()
            }
            print(f"PREFILL {sample_id} one_image_prefill", flush=True)
            output = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            c0 = output.past_key_values
            rope_delta = model.model.rope_deltas.detach().clone()
            c0_snapshot = _snapshot_cache(c0)
            first_span_offset = int(row["first_span_offset"])
            if not 0 <= first_span_offset <= len(reasoning_ids):
                raise ValueError(f"{sample_id}: invalid first_span_offset {first_span_offset}")
            _restore_rope(model, rope_delta)
            c1, _ = _advance(model, fork(c0), reasoning_ids[:first_span_offset])
            _restore_rope(model, rope_delta)
            cT, _ = _advance(model, fork(c0), reasoning_ids)
            states.append(
                {
                    "row": dict(row),
                    "c0": c0,
                    "c1": c1,
                    "cT": cT,
                    "c0_snapshot": c0_snapshot,
                    "rope_delta": rope_delta,
                }
            )
    return states


def _score_bbox(raw: Mapping[str, Any], ground_truth: Sequence[float]) -> dict[str, Any]:
    text = raw.get("text", "")
    parsed = parse_response("<think>probe" + BOX_OPEN + str(text))
    bbox = None if parsed.bbox is None else [float(value) for value in parsed.bbox]
    numeric_valid = bool(parsed.parse_valid and bbox is not None and all(math.isfinite(value) for value in bbox))
    if numeric_valid:
        iou = float(xyxy_iou(bbox, ground_truth))
        if not math.isfinite(iou):
            raise FloatingPointError("bbox IoU returned a nonfinite value")
    else:
        iou = 0.0
    result = dict(raw)
    result.update(
        bbox=bbox if numeric_valid else None,
        parse_valid=bool(parsed.parse_valid),
        parse_error=parsed.error,
        numeric_valid=numeric_valid,
        iou=float(iou),
        hit_05=bool(iou >= 0.5),
    )
    return result


def _failed_draw(seed: int, reason: str, entity_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "token_ids": [],
        "text": "",
        "logprobs": [],
        "completed": False,
        "seed": int(seed),
        "bbox": None,
        "parse_valid": False,
        "parse_error": reason,
        "numeric_valid": False,
        "iou": 0.0,
        "hit_05": False,
        "entity_failed": True,
        "failure_reason": reason,
        "entity_audit": dict(entity_audit),
    }


def _old_parity(
    record: Mapping[str, Any],
    arm_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pilot_conditions = record.get("conditions", {})
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        arm = f"old__{condition}"
        expected_entry = pilot_conditions.get(condition, {}) if isinstance(pilot_conditions, Mapping) else {}
        expected_draws = expected_entry.get("draws", []) if isinstance(expected_entry, Mapping) else []
        actual_draws = arm_results.get(arm, {}).get("draws", [])
        entries: list[dict[str, Any]] = []
        for draw, (expected, actual) in enumerate(zip(expected_draws, actual_draws, strict=False)):
            expected_ids = [int(value) for value in expected.get("token_ids", expected.get("ids", []))]
            actual_ids = [int(value) for value in actual.get("token_ids", [])]
            expected_bbox = expected.get("bbox")
            actual_bbox = actual.get("bbox")
            bbox_match = expected_bbox == actual_bbox
            expected_iou = _finite_float(expected.get("iou"), 0.0)
            actual_iou = _finite_float(actual.get("iou"), 0.0)
            entries.append(
                {
                    "draw": draw,
                    "seed": int(actual.get("seed", expected.get("seed", 0))),
                    "ids_match": actual_ids == expected_ids,
                    "expected_token_ids": expected_ids,
                    "actual_token_ids": actual_ids,
                    "bbox_match": bbox_match,
                    "expected_bbox": expected_bbox,
                    "actual_bbox": actual_bbox,
                    "iou_match": actual_iou == expected_iou,
                    "iou_close_1e-6": math.isclose(actual_iou, expected_iou, rel_tol=0.0, abs_tol=1e-6),
                    "expected_iou": expected_iou,
                    "actual_iou": actual_iou,
                    "iou_abs_diff": abs(actual_iou - expected_iou),
                }
            )
        output[arm] = {
            "draw_count": len(entries),
            "ids_exact_count": sum(bool(entry["ids_match"]) for entry in entries),
            "bbox_exact_count": sum(bool(entry["bbox_match"]) for entry in entries),
            "iou_exact_count": sum(bool(entry["iou_match"]) for entry in entries),
            "all_ids_exact": len(entries) == DRAWS_PER_SAMPLE and all(entry["ids_match"] for entry in entries),
            "all_bbox_exact": len(entries) == DRAWS_PER_SAMPLE and all(entry["bbox_match"] for entry in entries),
            "all_iou_exact": len(entries) == DRAWS_PER_SAMPLE and all(entry["iou_match"] for entry in entries),
            "entries": entries,
        }
    return output


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "xgrammar"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def run(args: argparse.Namespace, *, bridge_generator=None, bridge_version="instance_v2") -> dict[str, Any]:
    if bridge_generator is None:
        bridge_generator = generate_entity_bridge
    if str(args.device).lower() == "cpu":
        raise ValueError("real bridge comparison requires CUDA; CPU scope is helper tests only")
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(f"set PYTHONHASHSEED={SEED} to preserve the pilot protocol")
    records = _read_jsonl(args.pilot_records)
    expected_ids = list(DEFAULT_SAMPLE_IDS)
    if [str(row.get("sample_id")) for row in records] != expected_ids:
        raise ValueError(f"pilot records must contain fixed five IDs in order {expected_ids}")
    for row in records:
        required = {
            "image_path",
            "expression",
            "entity",
            "reasoning_ids",
            "reasoning_text",
            "first_span_offset",
            "ground_truth_bbox",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"{row.get('sample_id')}: missing pilot fields {missing}")
        ground_truth_bbox = row["ground_truth_bbox"]
        if (
            not isinstance(ground_truth_bbox, Sequence)
            or isinstance(ground_truth_bbox, str)
            or len(ground_truth_bbox) != 4
        ):
            raise ValueError(f"{row.get('sample_id')}: ground_truth_bbox must contain four coordinates")
        if any(not math.isfinite(float(value)) for value in ground_truth_bbox):
            raise ValueError(f"{row.get('sample_id')}: ground_truth_bbox must be finite")
        _sample_index(str(row["sample_id"]))
        pilot_draw_seeds(row)
    args.output_dir.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    command = [sys.executable, *sys.argv]
    dependency_receipts = {
        str(ROOT / relative): _receipt(ROOT / relative)
        for relative in _DIRECT_DEPENDENCIES
        if (ROOT / relative).is_file()
    }
    if bridge_version == "extractive_v3":
        for relative in (
            "timeline_self_distillation/extractive_entity_bridge.py",
            "timeline_self_distillation/run_entity_bridge_extractive.py",
        ):
            dependency_receipts[str(ROOT / relative)] = _receipt(ROOT / relative)
    input_receipt = {"pilot_records": _receipt(args.pilot_records)}
    protocol = {
        "kind": "five_image_saved_vs_instance_entity_bridge_comparison",
        "sample_ids": expected_ids,
        "sample_count": 5,
        "arms": list(ARM_NAMES),
        "conditions": list(CONDITIONS),
        "bridge_versions": {"old": "pilot_saved_entity", "new": bridge_version},
        "model": str(MODEL),
        "seed": int(SEED),
        "bbox_sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "grammar": BOX_REGEX,
            "draws_per_sample": DRAWS_PER_SAMPLE,
            "seed_source": "actual seeds saved in pilot records conditions.early_span1.draws",
            "micro_eval_seeds_forbidden": True,
        },
        "primary": "per-arm mean IoU across all five images and four draws each; invalid=0; no selection",
        "secondary": "per-arm Acc@0.5 across all five images and four draws each; invalid=0",
        "entity_sampling": {
            "grammar": ENTITY_REGEX
            if bridge_version == "instance_v2"
            else "source-candidate ID plus </evidence_id>; exact regex recorded per image",
            "max_tokens": 96,
            "new_bridge_once_from": "cT",
            "all_new_arms_share_entity": True,
            "old_bridge_regeneration": False,
            "failure": "usable_for_probe=false => entity empty; all four draws per new arm remain with IoU=0",
            "request_echo": "warning only; usable echo remains evaluated",
        },
        "cache_protocol": (
            "one image prefill per sample; _advance per reasoning token; retain c0,c1,cT; restore rope_delta"
        ),
        "query_cache": "suffix prefilled once per arm; four fresh forks reuse identical first-coordinate logits",
        "reasoning_protocol": "reasoning_ids, reasoning_text, first_span_offset, and saved entity copied unchanged",
        "entity_input_contract": (
            "new bridge receives cT, expression, and fixed seed; no ground_truth_bbox or GT coordinates"
        ),
        "bbox_input_contract": "QUERY(entity, expression)+BOX_OPEN for every arm; GT used only for post-hoc IoU",
        "postprocess": "parse bbox without coordinate reordering; invalid/nonnumeric result IoU=0",
        "training": {"optimizer": False, "adapter": False, "parameter_update": False, "backbone_frozen": True},
        "scope": "five fixed seen images; bridge/mechanism comparison, not a generalization result",
    }
    _write_json(args.output_dir / "protocol.json", protocol)
    _write_json(
        args.output_dir / "manifest.json",
        {
            **protocol,
            "command": command,
            "git": _git_receipt(),
            "versions": _package_versions(),
            "direct_dependency_hashes": dependency_receipts,
            "input_hashes": input_receipt,
        },
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(MODEL)
    tokenizer = processor.tokenizer
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen_versions_before = {
        name: int(parameter._version) for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    vocab_size = int(model.config.text_config.vocab_size)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size))
    bbox_grammar = compiler.compile_regex(BOX_REGEX)
    entity_grammar = compiler.compile_regex(ENTITY_REGEX)
    states_list = _build_states(model, processor, records)
    states = {str(state["row"]["sample_id"]): state for state in states_list}

    output_records: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row["sample_id"])
        state = states[sample_id]
        seeds = pilot_draw_seeds(row)
        _restore_rope(model, state["rope_delta"])
        bridge = bridge_generator(
            model,
            tokenizer,
            entity_grammar,
            state["cT"],
            str(row["expression"]),
            int(SEED),
            version=bridge_version,
        )
        bridge_audit = bridge.get("entity_audit", {})
        bridge_audit = bridge_audit if isinstance(bridge_audit, Mapping) else {}
        new_entity = str(bridge.get("entity", "")) if bridge_audit.get("usable_for_probe") else ""
        if _bridge_warning(bridge_audit):
            print(f"ENTITY_WARNING {sample_id} request_echo=True; evaluating anyway", flush=True)
        print(
            f"ENTITY {sample_id} status={bridge_audit.get('semantic_status', 'unknown')} "
            f"usable={bool(bridge_audit.get('usable_for_probe'))} entity={new_entity!r}",
            flush=True,
        )
        arm_inputs = entity_arm_inputs(row, bridge)
        arm_results: dict[str, dict[str, Any]] = {}
        for arm in ARM_NAMES:
            bridge_kind, condition = arm.split("__", maxsplit=1)
            metadata = arm_inputs[arm]
            cache_key = {"late_entity": "cT", "early_step0": "c0", "early_span1": "c1"}[condition]
            cache = state[cache_key]
            query_suffix = QUERY.format(entity=metadata["entity"], question=str(row["expression"])) + BOX_OPEN
            prepared_cache, prepared_logits = None, None
            if not metadata["entity_failed"]:
                _restore_rope(model, state["rope_delta"])
                prepared_cache, prepared_logits = _advance(
                    model, fork(cache), tokenizer.encode(query_suffix, add_special_tokens=False)
                )
            draws: list[dict[str, Any]] = []
            for draw, seed in enumerate(seeds):
                if metadata["entity_failed"]:
                    scored = _failed_draw(int(seed), "entity_bridge_failed", metadata["entity_audit"])
                else:
                    _restore_rope(model, state["rope_delta"])
                    raw = constrained_generate(
                        model,
                        tokenizer,
                        bbox_grammar,
                        fork(prepared_cache),
                        query_suffix,
                        int(seed),
                        prefilled_logits=prepared_logits,
                    )
                    scored = _score_bbox(raw, row.get("ground_truth_bbox", []))
                    scored["entity_failed"] = False
                scored.update(
                    {
                        "draw": int(draw),
                        "seed": int(seed),
                        "arm": arm,
                        "bridge": bridge_kind,
                        "condition": condition,
                        "entity": metadata["entity"],
                    }
                )
                draws.append(scored)
            arm_results[arm] = {
                **metadata,
                "condition": condition,
                "bridge": bridge_kind,
                "query_suffix": query_suffix,
                "draws": draws,
                **_draw_summary(draws),
            }
            print(
                f"ARM {sample_id} {arm} draws={len(draws)} mean_iou={arm_results[arm]['mean_iou']:.6f} "
                f"failed={arm_results[arm]['entity_failed']}",
                flush=True,
            )

        cache_audit = _compare_cache(state["c0"], state["c0_snapshot"])
        output_record = dict(row)
        output_record.update(
            {
                "pilot_conditions": row.get("conditions"),
                "old_entity": str(row["entity"]),
                "new_entity": new_entity,
                "new_entity_result": bridge.get("entity_result"),
                "new_entity_audit": bridge.get("entity_audit"),
                "draw_seeds": seeds,
                "conditions": arm_results,
                "old_parity": _old_parity(row, arm_results),
                "master_c0_cache": cache_audit,
            }
        )
        output_records.append(output_record)
        _append_jsonl(
            args.output_dir / "entities.jsonl",
            {
                "sample_id": sample_id,
                "old_entity": str(row["entity"]),
                "old_entity_result": row.get("entity_result"),
                "new_entity": new_entity,
                "new_entity_result": bridge.get("entity_result"),
                "new_entity_audit": bridge.get("entity_audit"),
                "draw_seeds": seeds,
            },
        )
        _append_jsonl(args.output_dir / "records.jsonl", output_record)

    frozen_versions_after = {
        name: int(parameter._version) for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    frozen_unchanged = frozen_versions_before == frozen_versions_after
    cache_checks = {record["sample_id"]: record["master_c0_cache"] for record in output_records}
    cache_unchanged = all(bool(value["unchanged"]) for value in cache_checks.values())
    summary = summarize_records(output_records)
    summary.update(
        {
            "protocol": protocol,
            "total_expected_draws": TOTAL_EXPECTED_DRAWS,
            "total_draws": summary["draw_count"],
            "entity_failure_samples": sum(
                bool(record["new_entity_audit"] and not record["new_entity_audit"].get("usable_for_probe", False))
                for record in output_records
            ),
            "frozen_parameter_versions": {
                "count": len(frozen_versions_before),
                "unchanged": frozen_unchanged,
                "changed_names": sorted(
                    name
                    for name in set(frozen_versions_before) | set(frozen_versions_after)
                    if frozen_versions_before.get(name) != frozen_versions_after.get(name)
                ),
            },
            "master_c0_cache": {
                "per_sample": cache_checks,
                "all_unchanged": cache_unchanged,
            },
            "training_performed": False,
            "seconds": time.perf_counter() - started,
            "limitation": (
                "Five fixed seen images and four saved pilot seeds; exploratory bridge audit, not generalization."
            ),
        }
    )
    _write_json(args.output_dir / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3")
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
