"""Read-only five-image decode comparison for the existing terminal adapters.

The runner replays the saved full reasoning and saved entity from the pilot,
then samples only the legal ``BOX_REGEX`` continuation.  It does not call an
entity bridge, construct an optimizer, or update a parameter.  The three
conditions are base (adapter disabled), the reverse checkpoint, and the
forward checkpoint.  Every condition gets one greedy decode and sixteen
fixed-seed temperature-one draws per image; the random draws are the primary
metric and the greedy rows are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set the physical device before importing torch/model code.  Importing this
# module for CPU tests remains side-effect free apart from this environment
# selection; model loading is performed only by ``run``.
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
from reasoning_checkpoints.run_pilot import _processor  # noqa: E402
from timeline_self_distillation.run_checkpoint_diagnostic import load_adapter_checkpoint  # noqa: E402
from timeline_self_distillation.run_opd_micro import build_states  # noqa: E402
from timeline_self_distillation.run_teacher_pilot import (  # noqa: E402
    BOX_OPEN,
    BOX_REGEX,
    MODEL,
    SEED,
    fork,
)
from timeline_self_distillation.terminal_adapter import (  # noqa: E402
    install_terminal_query_lora,
    terminal_parameter_whitelist,
)
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou  # noqa: E402

MODELS = ("base", "reverse", "forward")
MODES = ("greedy", "random")
RANDOM_DRAWS = 16
GREEDY_DRAWS = 1
EXPECTED_SAMPLES = ("row-0", "row-1", "row-3", "row-4", "row-5")
EXPECTED_RECORDS = len(MODELS) * len(EXPECTED_SAMPLES) * (RANDOM_DRAWS + GREEDY_DRAWS)
CHECKPOINT_NAME = "final_adapter.pt"

_DEPENDENCY_FILES = (
    "timeline_self_distillation/run_decode_diagnostic.py",
    "timeline_self_distillation/run_opd_micro.py",
    "timeline_self_distillation/run_teacher_pilot.py",
    "timeline_self_distillation/terminal_adapter.py",
    "timeline_self_distillation/run_checkpoint_diagnostic.py",
    "timeline_self_distillation/decode_diagnostic_metrics.py",
    "live_kv_probe_prototype/run_hf_fork.py",
    "reasoning_checkpoints/extractor.py",
    "reasoning_checkpoints/run_pilot.py",
    "verl/experimental/routed_grounding/router.py",
)


def micro_seed(base_seed: int, sample_order: int, draw: int) -> int:
    """Return the frozen micro-evaluation seed for a fixed record order."""

    if int(sample_order) < 0 or int(draw) < 0:
        raise ValueError("sample_order and draw must be non-negative")
    return int(base_seed) + 9_000_000 + int(sample_order) * 1000 + int(draw)


def _finite(value: Any, *, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(converted):
        raise FloatingPointError(f"{name} is nonfinite: {value!r}")
    return converted


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_pilot_records(path: Path) -> list[dict[str, Any]]:
    """Load exactly the original five pilot records without regeneration."""

    rows = _read_jsonl(path)
    observed = [str(row.get("sample_id")) for row in rows]
    if observed != list(EXPECTED_SAMPLES):
        raise ValueError(f"pilot records must contain {EXPECTED_SAMPLES} in order, got {observed}")
    required = {
        "image_path",
        "expression",
        "entity",
        "reasoning_ids",
        "reasoning_text",
        "first_span_offset",
        "ground_truth_bbox",
    }
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"{row.get('sample_id')}: missing required fields {missing}")
        bbox = row["ground_truth_bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"{row['sample_id']}: ground_truth_bbox must contain four values")
        for value in bbox:
            _finite(value, name=f"{row['sample_id']} ground_truth_bbox")
        if not isinstance(row["reasoning_ids"], list) or not row["reasoning_ids"]:
            raise ValueError(f"{row['sample_id']}: reasoning_ids must be a non-empty list")
    return rows


def read_old_eval_step(path: Path, step: int, sample_ids: Sequence[str]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Index four old micro-eval rows per sample by ``(sample, seed, step)``.

    The old JSONL has no ``draw`` field.  Sorting by or pairing through a
    synthetic row number would hide seed mismatches, so this helper validates
    the four expected seeds and returns an explicit composite-key index.
    """

    wanted = tuple(str(value) for value in sample_ids)
    rows = [row for row in _read_jsonl(path) if int(row.get("step", -1)) == int(step)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in wanted:
            raise ValueError(f"{path}: unexpected sample_id {sample_id!r}")
        ids = row.get("ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path}: {sample_id} step={step} has no actual bbox IDs")
        if "seed" not in row:
            raise ValueError(f"{path}: {sample_id} step={step} row lacks seed")
        grouped[sample_id].append(row)
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for sample_order, sample_id in enumerate(wanted):
        entries = grouped.get(sample_id, [])
        if len(entries) != 4:
            raise ValueError(f"{path}: expected four rows for {sample_id} step={step}, got {len(entries)}")
        for draw in range(4):
            expected_seed = micro_seed(SEED, sample_order, draw)
            matching = [row for row in entries if int(row["seed"]) == expected_seed]
            if len(matching) != 1:
                raise ValueError(
                    f"{path}: expected exactly one {sample_id}/seed={expected_seed}/step={step}, "
                    f"found {len(matching)}"
                )
            indexed[(sample_id, expected_seed, int(step))] = matching[0]
    if len(indexed) != len(wanted) * 4:
        raise ValueError(f"{path}: expected 20 indexed rows for step={step}, got {len(indexed)}")
    return indexed


def score_bbox(prediction: Mapping[str, Any], ground_truth: Sequence[float]) -> dict[str, Any]:
    """Score one decoded row; parser-invalid boxes (including ``bbox=None``) get zero.

    A finite four-coordinate box accepted by ``parse_response`` is scored by
    ``xyxy_iou``; malformed parser output is recorded as invalid instead of
    being sent to that scorer.  Unexpected scoring errors and nonfinite values
    are raised rather than being converted into a favorable or silent result.
    """

    parse_valid = bool(prediction.get("parse_valid"))
    raw_bbox = prediction.get("bbox")
    if not parse_valid or raw_bbox is None:
        return {
            "parse_valid": parse_valid,
            "numeric_valid": False,
            "bbox": None,
            "iou": 0.0,
            "acc_05": False,
        }
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return {
            "parse_valid": parse_valid,
            "numeric_valid": False,
            "bbox": None,
            "iou": 0.0,
            "acc_05": False,
        }
    bbox = [_finite(value, name="decoded bbox coordinate") for value in raw_bbox]
    try:
        value = float(xyxy_iou(bbox, ground_truth))
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(f"bbox scoring failed for a malformed numeric box: {bbox!r}") from error
    if not math.isfinite(value):
        raise FloatingPointError(f"decoded bbox IoU is nonfinite: {value!r}")
    return {
        "parse_valid": parse_valid,
        "numeric_valid": True,
        "bbox": bbox,
        "iou": value,
        "acc_05": bool(value >= 0.5),
    }


def _metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ious = [_finite(row.get("iou", 0.0), name="record iou") for row in rows]
    invalid = [not bool(row.get("numeric_valid", row.get("parse_valid", False))) for row in rows]
    hits = [bool(row.get("acc_05", iou >= 0.5)) for row, iou in zip(rows, ious, strict=True)]
    return {
        "sample_count": len({str(row.get("sample_id")) for row in rows}),
        "draw_count": len(rows),
        "mean_iou": float(mean(ious)) if ious else 0.0,
        "acc_05": float(mean(float(hit) for hit in hits)) if hits else 0.0,
        "invalid_count": sum(invalid),
        "invalid_ratio": float(mean(float(value) for value in invalid)) if invalid else 0.0,
        "per_sample": {},
    }


def summarize_outputs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize all 255 outputs, with random draws as the primary metric."""

    # Validate numeric fields before coverage checks so a malformed partial
    # output cannot hide an explicit NaN behind a count error.
    for record in records:
        _finite(record.get("iou", 0.0), name="record iou")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} output records, got {len(records)}")
    sample_ids = list(EXPECTED_SAMPLES)
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_key: set[tuple[str, str, str, int]] = set()
    for record in records:
        model = str(record.get("model"))
        sample_id = str(record.get("sample_id"))
        mode = str(record.get("mode"))
        draw = int(record.get("draw", -1))
        if model not in MODELS or sample_id not in sample_ids or mode not in MODES:
            raise ValueError(f"unexpected output key: {model}/{sample_id}/{mode}/{draw}")
        key = (model, sample_id, mode, draw)
        if key in by_key:
            raise ValueError(f"duplicate output key {key}")
        by_key.add(key)
        by_model[model].append(record)
    expected_keys = {
        (model, sample_id, mode, draw)
        for model in MODELS
        for sample_id in sample_ids
        for mode, count in (("greedy", GREEDY_DRAWS), ("random", RANDOM_DRAWS))
        for draw in range(count)
    }
    if by_key != expected_keys:
        raise ValueError("output records do not cover exactly all model/sample/mode/draw keys")

    result: dict[str, Any] = {
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "total_count": len(records),
        "random_draws_per_model": len(sample_ids) * RANDOM_DRAWS,
        "greedy_draws_per_model": len(sample_ids) * GREEDY_DRAWS,
        "primary_metric": "random_draws_mean_iou; invalid=0; no draw selection",
        "models": {},
    }
    for model in MODELS:
        model_rows = by_model[model]
        random_rows = [row for row in model_rows if row["mode"] == "random"]
        greedy_rows = [row for row in model_rows if row["mode"] == "greedy"]
        random_metrics = _metric_rows(random_rows)
        greedy_metrics = _metric_rows(greedy_rows)
        all_metrics = _metric_rows(model_rows)
        for subset_name, subset_metrics, subset_rows in (
            ("random", random_metrics, random_rows),
            ("greedy", greedy_metrics, greedy_rows),
        ):
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in subset_rows:
                grouped[str(row["sample_id"])].append(row)
            subset_metrics["per_sample"] = {
                sample_id: _metric_rows(grouped[sample_id]) for sample_id in sample_ids
            }
            subset_metrics["per_sample"] = {
                sample_id: {
                    key: value for key, value in metrics.items() if key != "per_sample"
                }
                for sample_id, metrics in subset_metrics["per_sample"].items()
            }
        result["models"][model] = {
            "random_count": len(random_rows),
            "greedy_count": len(greedy_rows),
            "random": random_metrics,
            "greedy": greedy_metrics,
            "all": {key: value for key, value in all_metrics.items() if key != "per_sample"},
            "primary_metric": "random_draws_mean_iou",
            "greedy_is_diagnostic_only": True,
        }
    return result


def _pair_key(record: Mapping[str, Any]) -> tuple[str, int, str, int, int]:
    return (
        str(record["sample_id"]),
        int(record["sample_order"]),
        str(record["mode"]),
        int(record["draw"]),
        int(record["seed"]),
    )


def build_pairs(
    base_records: Sequence[Mapping[str, Any]],
    checkpoint_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    metrics: Any = None,
) -> list[dict[str, Any]]:
    """Pair base/checkpoint decodes and attach the two direct metric dicts."""

    if metrics is None:
        from timeline_self_distillation import decode_diagnostic_metrics as metrics  # noqa: PLC0415

    base_by_key = {_pair_key(record): record for record in base_records}
    if len(base_by_key) != len(base_records):
        raise ValueError("duplicate base decode key")
    checkpoint_by_key = {_pair_key(record): record for record in checkpoint_records}
    if len(checkpoint_by_key) != len(checkpoint_records):
        raise ValueError("duplicate checkpoint decode key")
    pairs: list[dict[str, Any]] = []
    for key in sorted(checkpoint_by_key, key=lambda value: (value[1], value[0], value[2], value[3])):
        checkpoint = checkpoint_by_key[key]
        base = base_by_key.get(key)
        if base is None:
            same_prefix = [candidate for candidate in base_by_key if candidate[:4] == key[:4]]
            if same_prefix:
                raise ValueError(f"paired records disagree on seed for prefix {key[:4]}")
            raise ValueError(f"checkpoint decode has no matching base key {key}")
        if int(base["seed"]) != int(checkpoint["seed"]):
            raise ValueError(f"paired records disagree on seed for {key}")
        base_text = str(base.get("text", ""))
        checkpoint_text = str(checkpoint.get("text", ""))
        first_numeric = metrics.first_numeric_difference(base_text, checkpoint_text)
        first_token = metrics.first_token_divergence(base["ids"], checkpoint["ids"], tokenizer)
        pairs.append(
            {
                "sample_id": key[0],
                "sample_order": key[1],
                "mode": key[2],
                "draw": key[3],
                "seed": key[4],
                "checkpoint_model": str(checkpoint["model"]),
                "base_model": str(base["model"]),
                "base_ids": [int(value) for value in base["ids"]],
                "checkpoint_ids": [int(value) for value in checkpoint["ids"]],
                "base_text": base_text,
                "checkpoint_text": checkpoint_text,
                "base_iou": _finite(base.get("iou", 0.0), name="base pair iou"),
                "checkpoint_iou": _finite(checkpoint.get("iou", 0.0), name="checkpoint pair iou"),
                "first_numeric_difference": first_numeric,
                "first_token_divergence": first_token,
            }
        )
    if set(base_by_key) != set(checkpoint_by_key):
        missing = sorted(set(base_by_key) - set(checkpoint_by_key))
        extra = sorted(set(checkpoint_by_key) - set(base_by_key))
        raise ValueError(f"base/checkpoint decode key mismatch; missing={missing}, extra={extra}")
    return pairs


def _sample_bbox(
    model: Any,
    tokenizer: Any,
    grammar: Any,
    state: Mapping[str, Any],
    seed: int,
    *,
    greedy: bool,
) -> dict[str, Any]:
    """Minimal copy of the frozen micro sampler with an optional argmax path."""

    cache = fork(state["student"])
    model.model.rope_deltas = state["rope_deltas"].clone()
    current_input = int(state["last_opening_id"])
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    vocab_size = int(model.config.text_config.vocab_size)
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    token_ids: list[int] = []
    supports: list[list[int]] = []
    with torch.no_grad():
        for _ in range(48):
            cache, logits = _advance(model, cache, [current_input])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite unmasked student logits")
            scores = logits.float().clone()
            xgr.reset_token_bitmask(bitmask)
            if matcher.fill_next_token_bitmask(bitmask):
                xgr.apply_token_bitmask_inplace(scores, bitmask.to(scores.device), vocab_size=vocab_size)
            support = torch.isfinite(scores[0]).nonzero(as_tuple=False).flatten()
            if not support.numel():
                raise RuntimeError("grammar produced empty support")
            supports.append([int(value) for value in support.tolist()])
            if greedy:
                token = int(scores.argmax(-1).item())
            else:
                # Deliberately preserve run_opd_micro.sample_bbox's exact
                # temperature-one torch.softmax + multinomial path.
                token = int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator).item())
            if not matcher.accept_token(token):
                raise RuntimeError("grammar rejected sampled token")
            token_ids.append(token)
            if matcher.is_completed():
                break
            current_input = token
    text = decode_ids(tokenizer, token_ids)
    parsed = parse_response("<think>probe" + BOX_OPEN + text)
    parsed_bbox = None if parsed.bbox is None else list(parsed.bbox)
    parse_valid = bool(parsed.parse_valid)
    return {
        "ids": token_ids,
        "token_ids": token_ids,
        "text": text,
        "raw_text": text,
        "pieces": [decode_ids(tokenizer, [token]) for token in token_ids],
        "token_pieces": [decode_ids(tokenizer, [token]) for token in token_ids],
        "supports": supports,
        "bbox": parsed_bbox,
        "parse_valid": parse_valid,
        "parse_error": parsed.error,
        "completed": bool(matcher.is_completed()),
        "seed": int(seed),
    }


def _score_sample(
    model: Any,
    tokenizer: Any,
    grammar: Any,
    state: Mapping[str, Any],
    seed: int,
    row: Mapping[str, Any],
    *,
    greedy: bool,
    model_name: str,
    mode: str,
    draw: int,
) -> dict[str, Any]:
    prediction = _sample_bbox(model, tokenizer, grammar, state, seed, greedy=greedy)
    scored = score_bbox(prediction, row["ground_truth_bbox"])
    result = {key: value for key, value in prediction.items() if key != "supports"}
    result.update(scored)
    result.update(
        {
            "sample_id": str(row["sample_id"]),
            "sample_order": int(row["sample_order"]),
            "model": model_name,
            "mode": mode,
            "draw": int(draw),
            "seed": int(seed),
        }
    )
    return result


def _bbox_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(float(a) == float(b) for a, b in zip(left, right, strict=True))


def parity_for_model(
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    step: int,
    label: str,
) -> dict[str, Any]:
    """Compare random draws 0--3 to old JSONL by sample, seed and step."""

    entries: list[dict[str, Any]] = []
    selected = [row for row in records if row["mode"] == "random" and int(row["draw"]) < 4]
    for actual in selected:
        key = (str(actual["sample_id"]), int(actual["seed"]), int(step))
        old = expected.get(key)
        if old is None:
            raise ValueError(f"missing old parity row {key}")
        expected_iou = _finite(old.get("iou", 0.0), name="old eval iou")
        actual_iou = _finite(actual.get("iou", 0.0), name="new eval iou")
        expected_ids = [int(value) for value in old["ids"]]
        actual_ids = [int(value) for value in actual["ids"]]
        expected_bbox = old.get("bbox")
        actual_bbox = actual.get("bbox")
        entries.append(
            {
                "sample_id": str(actual["sample_id"]),
                "seed": int(actual["seed"]),
                "step": int(step),
                "label": label,
                "ids_match": actual_ids == expected_ids,
                "bbox_match": _bbox_equal(actual_bbox, expected_bbox),
                "iou_equal": actual_iou == expected_iou,
                "iou_match": math.isclose(actual_iou, expected_iou, rel_tol=0.0, abs_tol=1e-6),
                "expected_ids": expected_ids,
                "actual_ids": actual_ids,
                "expected_bbox": expected_bbox,
                "actual_bbox": actual_bbox,
                "expected_iou": expected_iou,
                "actual_iou": actual_iou,
            }
        )
    return {
        "label": label,
        "count": len(entries),
        "ids_exact_count": sum(bool(entry["ids_match"]) for entry in entries),
        "bbox_exact_count": sum(bool(entry["bbox_match"]) for entry in entries),
        "iou_equal_count": sum(bool(entry["iou_equal"]) for entry in entries),
        "iou_match_count": sum(bool(entry["iou_match"]) for entry in entries),
        "all_ids_match": all(bool(entry["ids_match"]) for entry in entries),
        "all_bbox_match": all(bool(entry["bbox_match"]) for entry in entries),
        "all_iou_equal": all(bool(entry["iou_equal"]) for entry in entries),
        "entries": entries,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "xgrammar"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _frozen_versions(model: Any) -> dict[str, int]:
    return {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _load_checkpoint(model: Any, adapter: Any, checkpoint: Path) -> dict[str, Any]:
    copied = load_adapter_checkpoint(model, adapter, checkpoint)
    return {
        "path": str(checkpoint),
        "sha256": _sha256_file(checkpoint),
        "keys": sorted(copied),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CUDA-only decode audit.  CPU is intentionally test scope only."""

    if str(args.device).lower() == "cpu":
        raise ValueError("real decode diagnostic requires CUDA; CPU scope is unit tests only")
    pilot_path = Path(args.pilot_records)
    reverse_dir = Path(args.reverse_dir)
    forward_dir = Path(args.forward_dir)
    output_dir = Path(args.output_dir)
    records = read_pilot_records(pilot_path)
    reverse_checkpoint = reverse_dir / CHECKPOINT_NAME
    forward_checkpoint = forward_dir / CHECKPOINT_NAME
    reverse_eval = reverse_dir / "eval.jsonl"
    forward_eval = forward_dir / "eval.jsonl"
    required_paths = (reverse_checkpoint, forward_checkpoint, reverse_eval, forward_eval)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    reverse_step0 = read_old_eval_step(reverse_eval, 0, EXPECTED_SAMPLES)
    reverse_step20 = read_old_eval_step(reverse_eval, 20, EXPECTED_SAMPLES)
    forward_step20 = read_old_eval_step(forward_eval, 20, EXPECTED_SAMPLES)
    output_dir.mkdir(parents=False, exist_ok=False)

    command = [sys.executable, *sys.argv]
    source_receipts = {
        relative: _file_receipt(ROOT / relative)
        for relative in _DEPENDENCY_FILES
        if (ROOT / relative).is_file()
    }
    input_receipts = {"pilot_records": _file_receipt(pilot_path)}
    input_receipts.update(
        {
            "reverse_eval": _file_receipt(reverse_eval),
            "forward_eval": _file_receipt(forward_eval),
            "reverse_checkpoint": _file_receipt(reverse_checkpoint),
            "forward_checkpoint": _file_receipt(forward_checkpoint),
        }
    )
    protocol = {
        "kind": "read_only_checkpoint_decode_diagnostic",
        "model": str(MODEL),
        "sample_ids": list(EXPECTED_SAMPLES),
        "sample_order": "pilot record order; row-3 is order 2",
        "seed_formula": "SEED + 9000000 + sample_order*1000 + draw",
        "seed": int(SEED),
        "models": list(MODELS),
        "modes": {"greedy": 1, "random": RANDOM_DRAWS},
        "total_outputs": EXPECTED_RECORDS,
        "bbox_sampling": {"temperature": 1.0, "top_p": 1.0, "grammar": BOX_REGEX, "max_tokens": 48},
        "state": {
            "builder": "run_opd_micro.build_states",
            "teacher": "late_entity",
            "prefill_per_image": 1,
            "saved_reasoning_and_entity": True,
            "entity_bridge_called": False,
        },
        "primary": "per-model random 80 draws meanIoU; invalid=0; no draw selection",
        "secondary": "per-model random Acc@.5 and invalid ratio",
        "greedy": "five rows per model, diagnostic only",
        "parity": (
            "random draws 0..3 indexed by (sample_id, seed, step): base/old reverse step0, "
            "reverse/step20, forward/step20"
        ),
        "scoring": "invalid or nonnumeric box gets IoU=0; no NaN is swallowed",
        "training": {"optimizer": False, "updates": False, "backbone_frozen": True},
        "ground_truth": "only scoring target; never passed to generation or an update",
        "scope": "five fixed seen pilot images; not a transfer/generalization claim",
    }
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(
        output_dir / "manifest.json",
        {
            "command": command,
            "git": _git_receipt(),
            "versions": _package_versions(),
            "sources": source_receipts,
            "inputs_and_checkpoints": input_receipts,
            "entry": str(Path(__file__).resolve()),
        },
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(MODEL)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    tokenizer = processor.tokenizer
    adapter = install_terminal_query_lora(model, rank=8)
    whitelist = terminal_parameter_whitelist(model, adapter)
    if set(whitelist) != {"q_lora_down.weight", "q_lora_up.weight"}:
        raise RuntimeError("terminal adapter whitelist drifted")
    adapter.enabled = False
    frozen_before = _frozen_versions(model)
    for row in records:
        if decode_ids(tokenizer, row["reasoning_ids"]) != str(row["reasoning_text"]):
            raise RuntimeError(f"{row['sample_id']}: reasoning IDs/text changed from pilot")
        row["sample_order"] = EXPECTED_SAMPLES.index(str(row["sample_id"]))

    # This is the sole state builder and therefore the sole image prefill per
    # row.  It uses the original reasoning IDs and saved entity verbatim.
    states_list = build_states(model, processor, records, "late_entity")
    if [str(state["row"]["sample_id"]) for state in states_list] != list(EXPECTED_SAMPLES):
        raise RuntimeError("state builder returned samples in an unexpected order")
    states = {str(state["row"]["sample_id"]): state for state in states_list}
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.text_config.vocab_size)
    ).compile_regex(BOX_REGEX)

    all_outputs: list[dict[str, Any]] = []
    outputs_by_model: dict[str, list[dict[str, Any]]] = {}
    checkpoint_receipts: dict[str, dict[str, Any] | None] = {"base": None, "reverse": None, "forward": None}
    for model_name, checkpoint in (
        ("base", None),
        ("reverse", reverse_checkpoint),
        ("forward", forward_checkpoint),
    ):
        if checkpoint is None:
            adapter.enabled = False
        else:
            checkpoint_receipts[model_name] = _load_checkpoint(model, adapter, checkpoint)
            adapter.enabled = True
        model_outputs: list[dict[str, Any]] = []
        for sample_order, row in enumerate(records):
            sample_id = str(row["sample_id"])
            state = states[sample_id]
            for mode, count, greedy in (("greedy", GREEDY_DRAWS, True), ("random", RANDOM_DRAWS, False)):
                for draw in range(count):
                    seed = micro_seed(SEED, sample_order, draw)
                    result = _score_sample(
                        model,
                        tokenizer,
                        grammar,
                        state,
                        seed,
                        row,
                        greedy=greedy,
                        model_name=model_name,
                        mode=mode,
                        draw=draw,
                    )
                    model_outputs.append(result)
                    all_outputs.append(result)
                    _append_jsonl(output_dir / "records.jsonl", result)
                print(f"DECODE model={model_name} sample={sample_id} mode={mode} count={count}", flush=True)
        outputs_by_model[model_name] = model_outputs

    if len(all_outputs) != EXPECTED_RECORDS:
        raise RuntimeError(f"generated {len(all_outputs)} records, expected {EXPECTED_RECORDS}")
    base_outputs = outputs_by_model["base"]
    pair_records: list[dict[str, Any]] = []
    for checkpoint_name in ("reverse", "forward"):
        pair_records.extend(build_pairs(base_outputs, outputs_by_model[checkpoint_name], tokenizer))
    for pair in pair_records:
        _append_jsonl(output_dir / "pairs.jsonl", pair)

    parity = {
        "base_vs_old_reverse_step0": parity_for_model(
            outputs_by_model["base"], reverse_step0, step=0, label="base_vs_reverse_step0"
        ),
        "reverse_vs_old_reverse_step20": parity_for_model(
            outputs_by_model["reverse"], reverse_step20, step=20, label="reverse_checkpoint_vs_reverse_step20"
        ),
        "forward_vs_old_forward_step20": parity_for_model(
            outputs_by_model["forward"], forward_step20, step=20, label="forward_checkpoint_vs_forward_step20"
        ),
    }
    parity["total_entries"] = sum(int(value["count"]) for value in parity.values())
    parity["expected_total_entries"] = 60
    parity["all_ids_match"] = all(bool(value["all_ids_match"]) for value in parity.values() if isinstance(value, dict))
    parity["all_bbox_match"] = all(
        bool(value["all_bbox_match"]) for value in parity.values() if isinstance(value, dict)
    )
    parity["all_iou_equal"] = all(
        bool(value["all_iou_equal"]) for value in parity.values() if isinstance(value, dict)
    )
    _write_json(output_dir / "parity.json", parity)

    frozen_after = _frozen_versions(model)
    summary = summarize_outputs(all_outputs)
    summary.update(
        {
            "protocol": protocol,
            "pair_count": len(pair_records),
            "pair_models": {"base_vs_reverse": 85, "base_vs_forward": 85},
            "checkpoint_receipts": checkpoint_receipts,
            "frozen_backbone_parameter_versions": {
                "count": len(frozen_before),
                "unchanged": frozen_before == frozen_after,
                "changed_names": sorted(
                    name
                    for name in set(frozen_before) | set(frozen_after)
                    if frozen_before.get(name) != frozen_after.get(name)
                ),
            },
            "optimizer_constructed": False,
            "training_performed": False,
            "parity_summary": {
                "total_entries": parity["total_entries"],
                "expected_total_entries": parity["expected_total_entries"],
                "all_ids_match": parity["all_ids_match"],
                "all_bbox_match": parity["all_bbox_match"],
            },
        }
    )
    _write_json(output_dir / "summary.json", summary)
    adapter.enabled = False
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
    run(parse_args())
