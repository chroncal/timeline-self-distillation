"""Second-round five-image teacher-refinement effect runner.

This runner consumes only the completed first-round ``records.jsonl``.  It
keeps each saved reasoning trajectory untouched in the output record, then
compares three fixed branches before sampling the same four bbox draws:

``pre_coordinate``
    Stop at the last natural boundary before the first detected coordinate
    value (or use the full original reasoning when no coordinate is detected).
``scrub_all``
    Replay the complete reasoning from the image-prefill cache after replacing
    coordinate-value digit tokens with a one-token ``?`` placeholder.
``sham_all``
    Replay the complete reasoning from the same image-prefill cache after
    replacing the same number of nearest nonnumeric tokens.

The model is loaded once and each image is prefetched once.  This is an
exploratory five-seen-image diagnostic, not a training run or a generalization
claim.
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
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bind the physical GPU before importing torch/model code.  The actual model
# path is reached only by ``run``; importing this module for CPU tests does not
# initialize CUDA or load a checkpoint.
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
from reasoning_checkpoints.extractor import decode_ids, natural_boundary_offsets  # noqa: E402
from reasoning_checkpoints.run_pilot import _load_rgb_image, _processor, _render_and_process  # noqa: E402
from timeline_self_distillation.run_teacher_pilot import (  # noqa: E402
    BOX_OPEN,
    BOX_REGEX,
    DEFAULT_SAMPLE_IDS,
    MODEL,
    QUERY,
    SEED,
    constrained_generate,
    fork,
)
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou  # noqa: E402

CONDITIONS = ("pre_coordinate", "scrub_all", "sham_all")
_NUMBER = r"\d+(?:\.\d+)?"
_TUPLE_VALUE_RE = re.compile(
    rf"[\[(]\s*(?P<v0>{_NUMBER})\s*,\s*(?P<v1>{_NUMBER})\s*,\s*"
    rf"(?P<v2>{_NUMBER})\s*,\s*(?P<v3>{_NUMBER})\s*[\])]"
)
_NAMED_VALUE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<name>[xy](?:1|2)?)\s*(?::|=)\s*~?\s*(?P<value>{_NUMBER})",
    re.IGNORECASE,
)
_ROW_ID_RE = re.compile(r"^row-(?P<index>[0-9]+)$")


def _coordinate_value_spans(text: str) -> list[dict[str, Any]]:
    """Return tuple/named-coordinate matches with value-only char spans."""

    matches: list[dict[str, Any]] = []
    for match in _TUPLE_VALUE_RE.finditer(text):
        for value_name in ("v0", "v1", "v2", "v3"):
            start, end = match.span(value_name)
            matches.append(
                {
                    "kind": "four_number_tuple",
                    "name": None,
                    "match_char_span": [int(match.start()), int(match.end())],
                    "value_char_span": [int(start), int(end)],
                    "value_text": match.group(value_name),
                }
            )
    for match in _NAMED_VALUE_RE.finditer(text):
        start, end = match.span("value")
        matches.append(
            {
                "kind": "named_coordinate_assignment",
                "name": match.group("name").lower(),
                "match_char_span": [int(match.start()), int(match.end())],
                "value_char_span": [int(start), int(end)],
                "value_text": match.group("value"),
            }
        )
    # Regexes are independently useful, so deduplicate only exact value spans
    # while retaining deterministic source order.
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for match in sorted(matches, key=lambda item: (item["value_char_span"], item["kind"])):
        unique.setdefault(tuple(match["value_char_span"]), match)
    return list(unique.values())


def _value_token_positions(
    tokenizer: Any,
    token_ids: Sequence[int],
    value_spans: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Map value-only character spans to tokens that consist solely of digits."""

    ids = [int(value) for value in token_ids]
    prefixes = [decode_ids(tokenizer, ids[:offset]) for offset in range(len(ids) + 1)]
    positions: list[int] = []
    for index, token_id in enumerate(ids):
        piece = decode_ids(tokenizer, [token_id])
        # Replacing a token containing punctuation/variable text would mask
        # more than the requested value digit.  Coordinate values in the
        # frozen tokenizer are digit-token aligned; fail closed otherwise.
        if not piece or not piece.isdigit():
            continue
        start, end = len(prefixes[index]), len(prefixes[index + 1])
        if end <= start:
            continue
        if any(
            start >= int(span["value_char_span"][0]) and end <= int(span["value_char_span"][1])
            for span in value_spans
        ):
            positions.append(index)
    return sorted(set(positions))


def pre_coordinate_anchor(tokenizer: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    """Compute the deterministic pre-coordinate anchor metadata.

    ``pre_coordinate_offset`` is ``None`` when no coordinate value is
    detected; callers then use the late/full-reasoning cache.  Otherwise it is
    the final natural boundary strictly before the first value token, or zero
    when no such boundary exists.
    """

    ids = [int(value) for value in token_ids]
    text = decode_ids(tokenizer, ids)
    detections = _coordinate_value_spans(text)
    positions = _value_token_positions(tokenizer, ids, detections)
    boundaries = sorted({int(offset) for offset in natural_boundary_offsets(tokenizer, ids)})
    if not detections:
        return {
            "coordinate_detections": [],
            "coordinate_value_char_spans": [],
            "coordinate_token_positions": [],
            "natural_boundary_offsets": boundaries,
            "first_coordinate_token_position": None,
            "pre_coordinate_offset": None,
            "anchor_kind": "late_no_coordinate",
        }
    if not positions:
        raise RuntimeError("coordinate values were detected but no value-aligned digit tokens were found")
    first = positions[0]
    prior = [offset for offset in boundaries if 0 < offset < first]
    anchor = max(prior, default=0)
    return {
        "coordinate_detections": detections,
        "coordinate_value_char_spans": [item["value_char_span"] for item in detections],
        "coordinate_token_positions": positions,
        "natural_boundary_offsets": boundaries,
        "first_coordinate_token_position": first,
        "pre_coordinate_offset": anchor,
        "anchor_kind": "last_prior_natural_boundary" if prior else "step0_no_prior_natural_boundary",
    }


def _placeholder_id(tokenizer: Any, expected: int | None = None) -> int:
    replacement = tokenizer.encode("?", add_special_tokens=False)
    if len(replacement) != 1:
        raise RuntimeError("scrub placeholder must be exactly one token")
    value = int(replacement[0])
    if expected is not None and value != int(expected):
        raise RuntimeError(f"placeholder token drifted from pilot ({value} != {int(expected)})")
    return value


def scrub_and_sham(tokenizer: Any, token_ids: Sequence[int]) -> tuple[list[int], list[int], dict[str, Any]]:
    """Create value-digit scrub and matched-count nearest-nonnumeric controls."""

    ids = [int(value) for value in token_ids]
    info = pre_coordinate_anchor(tokenizer, ids)
    positions = list(info["coordinate_token_positions"])
    placeholder = _placeholder_id(tokenizer)
    scrubbed = list(ids)
    for index in positions:
        scrubbed[index] = placeholder

    # A sham token cannot itself contain a digit.  Whitespace-only/zero-width
    # pieces are excluded; all remaining nonnumeric pieces are eligible and
    # are ranked globally by distance to the nearest coordinate value token.
    coordinate_set = set(positions)
    candidates: list[int] = []
    for index, token_id in enumerate(ids):
        if index in coordinate_set:
            continue
        piece = decode_ids(tokenizer, [token_id])
        if piece.strip() and not any(character.isdigit() for character in piece):
            candidates.append(index)
    candidates.sort(key=lambda index: (min((abs(index - position) for position in positions), default=0), index))
    sham_positions = sorted(candidates[: len(positions)])
    if len(sham_positions) != len(positions):
        raise RuntimeError(
            f"matched-count sham needs {len(positions)} nonnumeric tokens, found {len(sham_positions)}"
        )
    sham = list(ids)
    for index in sham_positions:
        sham[index] = placeholder

    info.update(
        {
            "placeholder_token_id": placeholder,
            "sham_token_positions": sham_positions,
            "masked_token_count": len(positions),
            "scrubbed_token_count": sum(old != new for old, new in zip(ids, scrubbed, strict=True)),
            "sham_token_count": sum(old != new for old, new in zip(ids, sham, strict=True)),
        }
    )
    return scrubbed, sham, info


def source_index_from_sample_id(sample_id: str) -> int:
    """Recover the fixed source index from pilot IDs such as ``row-3``."""

    match = _ROW_ID_RE.fullmatch(str(sample_id))
    if match is None:
        raise ValueError(f"sample_id must have fixed row-N form, got {sample_id!r}")
    return int(match.group("index"))


def read_pilot_records(path: Path) -> list[dict[str, Any]]:
    """Read exactly the fixed five successful pilot records, with no fallback source read."""

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = list(DEFAULT_SAMPLE_IDS)
    observed = [str(record.get("sample_id")) for record in records]
    if observed != expected:
        raise RuntimeError(f"pilot records must contain fixed five IDs in order {expected}, got {observed}")
    required = {"image_path", "expression", "entity", "reasoning_ids", "reasoning_text", "ground_truth_bbox"}
    for record in records:
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"pilot record {record.get('sample_id')} missing fields: {missing}")
        source_index_from_sample_id(str(record["sample_id"]))
    return records


def _snapshot_rope_deltas(model: Any) -> torch.Tensor | None:
    value = getattr(getattr(model, "model", None), "rope_deltas", None)
    return value.detach().clone() if isinstance(value, torch.Tensor) else None


def _restore_rope_deltas(model: Any, value: torch.Tensor | None) -> None:
    if value is not None and hasattr(getattr(model, "model", None), "rope_deltas"):
        model.model.rope_deltas = value.clone()


def _replay_ids(
    model: Any,
    base_cache: Any,
    token_ids: Sequence[int],
    *,
    rope_deltas: torch.Tensor | None,
) -> Any:
    """Replay one token at a time from a C0 branch (required for Qwen GDN state)."""

    cache = fork(base_cache)
    _restore_rope_deltas(model, rope_deltas)
    for token_id in token_ids:
        cache, _ = _advance(model, cache, [int(token_id)])
    return cache


def build_condition_caches(
    model: Any,
    c0: Any,
    reasoning_ids: Sequence[int],
    scrubbed_ids: Sequence[int],
    sham_ids: Sequence[int],
    pre_coordinate_offset: int | None,
    *,
    rope_deltas: torch.Tensor | None,
) -> dict[str, Any]:
    """Build all three condition caches while preserving C0 and CT branches."""

    original_cache = fork(c0)
    _restore_rope_deltas(model, rope_deltas)
    if pre_coordinate_offset == 0:
        pre_cache = fork(c0)
    else:
        pre_cache = None
    for offset, token_id in enumerate(reasoning_ids, start=1):
        original_cache, _ = _advance(model, original_cache, [int(token_id)])
        if pre_coordinate_offset == offset:
            pre_cache = fork(original_cache)
    if pre_coordinate_offset is None:
        # No detected coordinates means "pre_coordinate" is deliberately the
        # late/full-reasoning branch rather than a shortened trajectory.
        pre_cache = fork(original_cache)
    elif pre_cache is None:
        raise RuntimeError(f"pre-coordinate offset {pre_coordinate_offset} was not reached during replay")

    # Both controls start from the untouched image-prefill C0.  In particular,
    # never delete coordinate tokens or mutate CT to construct a branch.
    scrub_cache = _replay_ids(model, c0, scrubbed_ids, rope_deltas=rope_deltas)
    sham_cache = _replay_ids(model, c0, sham_ids, rope_deltas=rope_deltas)
    return {"pre_coordinate": pre_cache, "scrub_all": scrub_cache, "sham_all": sham_cache}


def _query_suffix(record: Mapping[str, Any], tokenizer: Any) -> str:
    suffix = QUERY.format(entity=str(record["entity"]), question=str(record["expression"])) + BOX_OPEN
    ids = tokenizer.encode(suffix, add_special_tokens=False)
    if decode_ids(tokenizer, ids) != suffix:
        raise RuntimeError(f"{record['sample_id']}: teacher query suffix does not round-trip exactly")
    return suffix


def score_bbox_draw(result: Mapping[str, Any], ground_truth_bbox: Sequence[float]) -> dict[str, Any]:
    """Parse and score one constrained draw; every invalid/nonnumeric result gets IoU=0."""

    parsed = parse_response("<think>probe" + BOX_OPEN + str(result.get("text", "")))
    bbox = None if parsed.bbox is None else [float(value) for value in parsed.bbox]
    numeric_valid = bool(parsed.parse_valid and bbox is not None and all(math.isfinite(value) for value in bbox))
    if numeric_valid:
        try:
            iou = float(xyxy_iou(bbox, ground_truth_bbox))
        except (TypeError, ValueError, RuntimeError):
            iou = 0.0
        if not math.isfinite(iou):
            iou = 0.0
    else:
        iou = 0.0
    return {
        **dict(result),
        "bbox": bbox if numeric_valid else None,
        "parse_valid": bool(parsed.parse_valid),
        "parse_error": parsed.error,
        "numeric_valid": numeric_valid,
        "iou": iou,
        "hit_05": bool(iou >= 0.5),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "xgrammar"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _seed_for_draw(base_seed: int, source_index: int, draw: int) -> int:
    return int(base_seed) + 1_000_000 + int(source_index) * 1000 + int(draw)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed second-round comparison and write protocol/records/summary."""

    if args.draws <= 0:
        raise ValueError("--draws must be positive")
    records = read_pilot_records(args.pilot_records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, *sys.argv]
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    git = _git_receipt()
    common_metadata = {
        "code_sha256": code_sha256,
        "git": git,
        "command": command,
        "pilot_records": str(args.pilot_records),
        "sample_ids": [str(record["sample_id"]) for record in records],
        "conditions": list(CONDITIONS),
        "draws": int(args.draws),
        "seed": int(args.seed),
        "draw_seed_formula": "seed + 1000000 + source_index*1000 + draw",
        "query_suffix": "QUERY.format(entity, question) + BOX_OPEN; identical for all conditions",
        "model": str(args.model),
        "device": str(args.device),
    }
    _write_json(
        args.output_dir / "protocol.json",
        {
            **common_metadata,
            "kind": "exploratory_five_sample_teacher_refinement",
            "hypothesis": (
                "Coordinate-aware pre-anchor or full-reasoning scrub changes final bbox draws "
                "relative to matched nonnumeric sham."
            ),
            "coordinate_detection": {
                "tuple": _TUPLE_VALUE_RE.pattern,
                "named": _NAMED_VALUE_RE.pattern,
                "value_only": True,
                "named_names": ["x", "y", "x1", "x2", "y1", "y2"],
            },
            "anchor": (
                "last natural_boundary_offsets strictly before first coordinate VALUE token; "
                "otherwise Step0; no coordinate means late/CT"
            ),
            "cache_protocol": (
                "one image prefill per sample; one-token replay; pre cache from original replay; "
                "scrub/sham replay from untouched C0; never delete from CT"
            ),
            "trajectory": "original reasoning IDs/text/logprobs and entity are copied unchanged from pilot records",
            "invalid_box_policy": "parse/numeric invalid outputs receive IoU=0 and Acc05=false",
            "ground_truth_policy": "ground_truth_bbox is used only by score_bbox_draw, never in model inputs",
            "scope": "five fixed seen images, exploratory effect diagnostic, not a test-set/generalization result",
        },
    )
    _write_json(args.output_dir / "manifest.json", {**common_metadata, "versions": _package_versions()})

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(args.model)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.model), dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    tokenizer = processor.tokenizer
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.text_config.vocab_size)
    ).compile_regex(BOX_REGEX)

    output_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for record in records:
            sample_id = str(record["sample_id"])
            source_index = source_index_from_sample_id(sample_id)
            reasoning_ids = [int(value) for value in record["reasoning_ids"]]
            if decode_ids(tokenizer, reasoning_ids) != str(record["reasoning_text"]):
                raise RuntimeError(f"{sample_id}: pilot reasoning text/IDs are not an exact pair")
            scrubbed_ids, sham_ids, mask_info = scrub_and_sham(tokenizer, reasoning_ids)
            image = _load_rgb_image(str(record["image_path"]))
            rendered, processed = _render_and_process(processor, str(record["expression"]), image)
            prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
            if "prompt_token_count" in record and len(prompt_ids) != int(record["prompt_token_count"]):
                raise RuntimeError(f"{sample_id}: prompt token count drifted from pilot")
            if "prompt_input_ids" in record and prompt_ids != [int(value) for value in record["prompt_input_ids"]]:
                raise RuntimeError(f"{sample_id}: prompt IDs drifted from pilot")
            if "rendered_prompt" in record and rendered != record["rendered_prompt"]:
                raise RuntimeError(f"{sample_id}: rendered prompt drifted from pilot")

            device_inputs = {
                key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                for key, value in processed.items()
            }
            print(f"PREFILL {sample_id} source_index={source_index}", flush=True)
            prefill = model(**device_inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            c0 = prefill.past_key_values
            rope_deltas = _snapshot_rope_deltas(model)
            if int(c0.get_seq_length()) != len(prompt_ids):
                raise RuntimeError(f"{sample_id}: image-prefill cache length mismatch")
            caches = build_condition_caches(
                model,
                c0,
                reasoning_ids,
                scrubbed_ids,
                sham_ids,
                mask_info["pre_coordinate_offset"],
                rope_deltas=rope_deltas,
            )
            query_suffix = _query_suffix(record, tokenizer)
            condition_results: dict[str, Any] = {}
            for condition in CONDITIONS:
                draws: list[dict[str, Any]] = []
                for draw in range(int(args.draws)):
                    seed = _seed_for_draw(args.seed, source_index, draw)
                    _restore_rope_deltas(model, rope_deltas)
                    sampled = constrained_generate(
                        model,
                        tokenizer,
                        grammar,
                        fork(caches[condition]),
                        query_suffix,
                        seed,
                        limit=48,
                    )
                    scored = score_bbox_draw(sampled, record["ground_truth_bbox"])
                    scored["draw"] = draw
                    scored["seed"] = seed
                    scored["condition"] = condition
                    draws.append(scored)
                    print(
                        f"BOX {sample_id} {condition} draw={draw} valid={scored['parse_valid']} "
                        f"iou={scored['iou']:.4f}",
                        flush=True,
                    )
                condition_results[condition] = {
                    "query_suffix": query_suffix,
                    "reasoning_prefix_token_count": (
                        len(reasoning_ids)
                        if condition != "pre_coordinate" or mask_info["pre_coordinate_offset"] is None
                        else int(mask_info["pre_coordinate_offset"])
                    ),
                    "draws": draws,
                    "mean_iou": mean(float(draw["iou"]) for draw in draws),
                    "acc_05": mean(bool(draw["hit_05"]) for draw in draws),
                }

            # Copy the pilot row so all original reasoning/evidence fields
            # remain byte-for-byte represented; only add refinement metadata
            # and replace the condition view with this round's conditions.
            output_record = dict(record)
            output_record["pilot_conditions"] = record.get("conditions")
            output_record["source_index"] = source_index
            output_record["original_reasoning_ids"] = list(reasoning_ids)
            output_record["original_reasoning_text"] = str(record["reasoning_text"])
            output_record["refinement_mask"] = mask_info
            output_record["scrubbed_reasoning_ids"] = scrubbed_ids
            output_record["sham_reasoning_ids"] = sham_ids
            output_record["conditions"] = condition_results
            output_records.append(output_record)
            _append_jsonl(args.output_dir / "records.jsonl", output_record)
            _append_jsonl(
                args.output_dir / "progress.jsonl",
                {"sample_id": sample_id, "source_index": source_index, "conditions": condition_results},
            )
            del c0, caches

    summary_conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        entries = [record["conditions"][condition] for record in output_records]
        draws = [draw for entry in entries for draw in entry["draws"]]
        summary_conditions[condition] = {
            "mean_iou": mean(float(entry["mean_iou"]) for entry in entries),
            "acc_05": mean(float(entry["acc_05"]) for entry in entries),
            "parse_valid": sum(bool(draw["parse_valid"]) for draw in draws),
            "numeric_valid": sum(bool(draw["numeric_valid"]) for draw in draws),
            "draw_count": len(draws),
        }
    summary = {
        **common_metadata,
        "sample_count": len(output_records),
        "seconds": time.perf_counter() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "conditions": summary_conditions,
        "per_sample": [
            {
                "sample_id": record["sample_id"],
                "source_index": record["source_index"],
                "coordinate_token_count": record["refinement_mask"]["masked_token_count"],
                "pre_coordinate_offset": record["refinement_mask"]["pre_coordinate_offset"],
                **{
                    condition: {
                        "mean_iou": record["conditions"][condition]["mean_iou"],
                        "acc_05": record["conditions"][condition]["acc_05"],
                    }
                    for condition in CONDITIONS
                },
            }
            for record in output_records
        ],
        "limitation": (
            "Five fixed seen images; exploratory teacher-refinement effect, "
            "not generalization or training gain."
        ),
    }
    _write_json(args.output_dir / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="3")
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", type=Path, default=MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
