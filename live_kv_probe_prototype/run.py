#!/usr/bin/env python3
"""PROTOTYPE: probe live reasoning prefixes through vLLM prefix caching.

This answers one narrow question: can independent bbox requests reuse a live
main request's raw-token prefix without changing the main sampled trajectory?
It does not claim to clone a live request's KV state.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

# Bind the requested physical GPU before importing torch/vLLM/transformers.
if "--device" in sys.argv:
    _device_index = sys.argv.index("--device") + 1
    if _device_index >= len(sys.argv):
        raise ValueError("--device requires a value")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_device_index]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
if "--full-determinism" in sys.argv:
    # These must be present before torch/vLLM are imported. PYTHONHASHSEED is
    # additionally validated later because it only takes effect at interpreter
    # startup and therefore must be exported by the launcher.
    os.environ["VERL_FULL_DETERMINISM"] = "1"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["NCCL_ALGO"] = "Ring"

from reasoning_checkpoints.extractor import (  # noqa: E402
    decode_ids,
    natural_boundary_offsets,
    split_reasoning_close,
)
from reasoning_checkpoints.run_pilot import (  # noqa: E402
    BBOX_PREFIX_TEXT,
    BBOX_TAIL_MAX_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    RESPONSE_LENGTH,
    _load_rgb_image,
    _processor,
    _protocol_ids,
    _render_and_process,
)
from scripts.routed_grounding.run_diagnostics import (  # noqa: E402
    BBOX_TAIL_REGEX,
    DEFAULT_SEED,
    DEFAULT_STUDENT_MODEL,
)
from verl.experimental.routed_grounding.router import parse_response  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/research_experiments/reasoning_checkpoints/"
    "pilot_seed260600564_n50_boundary_v3/trajectories.jsonl"
)
DEFAULT_SAMPLE_IDS = ("row-0", "row-1", "row-3", "row-4", "row-5")
CONDITIONS = ("baseline", "forced_close", "natural_instruction")
NATURAL_INSTRUCTION = "\nNow stop and give your current best bounding box estimate. "
PROBE_SEED_OFFSETS = {"forced_close": 1_000_000, "natural_instruction": 2_000_000}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_receipt() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def _ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in token_ids], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_source_rows(path: Path, sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    wanted = set(sample_ids)
    found: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            if sample_id in wanted and row.get("status") == "ok":
                found[sample_id] = row
    missing = [sample_id for sample_id in sample_ids if sample_id not in found]
    if missing:
        raise ValueError(f"successful source rows not found: {missing}")
    return [found[sample_id] for sample_id in sample_ids]


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if int(left_id) != int(right_id):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare_logprobs(
    left: Sequence[float | None], right: Sequence[float | None]
) -> dict[str, Any]:
    same_length = len(left) == len(right)
    none_pattern_equal = same_length and all(
        (left_value is None) == (right_value is None)
        for left_value, right_value in zip(left, right, strict=True)
    )
    exact = same_length and all(
        left_value == right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    differences = [
        abs(float(left_value) - float(right_value))
        for left_value, right_value in zip(left, right, strict=False)
        if left_value is not None and right_value is not None
    ]
    return {
        "chosen_logprobs_equal": exact,
        "chosen_logprob_lengths_equal": same_length,
        "chosen_logprob_none_pattern_equal": none_pattern_equal,
        "chosen_logprob_max_abs_diff": None if not differences else max(differences),
    }


def _chosen_logprobs(completion: Any) -> list[float | None]:
    rows = getattr(completion, "logprobs", None)
    token_ids = [int(value) for value in completion.token_ids]
    if not rows:
        return []
    values: list[float | None] = []
    for token_id, row in zip(token_ids, rows, strict=False):
        entry = row.get(token_id) if row else None
        values.append(None if entry is None else float(entry.logprob))
    return values


def _stable_causal_boundaries(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    """Return exact raw-ID boundaries stable with one-token lookahead.

    Newline boundaries are causal immediately. Punctuation boundaries wait
    until at least one later token is visible, allowing the existing detector
    to reject decimals, ellipses, abbreviations, and unclosed structures.
    """

    ids = [int(value) for value in token_ids]
    offsets = natural_boundary_offsets(tokenizer, ids)
    stable: list[int] = []
    for offset in offsets:
        prefix = decode_ids(tokenizer, ids[:offset])
        if offset < len(ids) or prefix.endswith(("\n", "\r")):
            stable.append(int(offset))
    return stable


def _probe_suffix_ids(tokenizer: Any, condition: str) -> list[int]:
    if condition == "forced_close":
        text = "</think>" + BBOX_PREFIX_TEXT
    elif condition == "natural_instruction":
        text = NATURAL_INSTRUCTION + "</think>" + BBOX_PREFIX_TEXT
    else:
        raise ValueError(f"condition has no probe suffix: {condition}")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if decode_ids(tokenizer, ids) != text:
        raise ValueError(f"probe suffix does not round-trip exactly for {condition}")
    return [int(value) for value in ids]


def _main_params(seed: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        seed=int(seed),
        max_tokens=RESPONSE_LENGTH,
        stop=["</think>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )


def _probe_params(seed: int) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    return SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        seed=int(seed),
        max_tokens=BBOX_TAIL_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(regex=BBOX_TAIL_REGEX),
    )


def _make_request(prompt_ids: Sequence[int], image: Any) -> dict[str, Any]:
    return {
        "prompt_token_ids": [int(value) for value in prompt_ids],
        "multi_modal_data": {"image": image},
        "mm_processor_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    }


async def _run_probe(
    *,
    engine: Any,
    tokenizer: Any,
    image: Any,
    base_prompt_ids: Sequence[int],
    reasoning_ids: Sequence[int],
    token_offset: int,
    checkpoint_index: int,
    condition: str,
    sample_id: str,
    source_index: int,
    main_finished_event: asyncio.Event,
    admission_event: asyncio.Event,
    base_seed: int,
    cache_accounting_unit: int,
) -> dict[str, Any]:
    prefix_ids = [int(value) for value in reasoning_ids[:token_offset]]
    suffix_ids = _probe_suffix_ids(tokenizer, condition)
    prompt_ids = [int(value) for value in base_prompt_ids] + prefix_ids + suffix_ids
    shared_prefix_count = len(base_prompt_ids) + len(prefix_ids)
    seed = (
        int(base_seed)
        + PROBE_SEED_OFFSETS[condition]
        + int(source_index) * 10_000
        + int(checkpoint_index)
    )
    request_id = f"probe-{condition}-{sample_id}-{checkpoint_index}-{seed}"
    start = time.perf_counter()
    final_output = None
    queue = None
    engine_admitted = False
    main_stream_unfinished_at_admission = False
    try:
        # Use add_request directly so the main coroutine can wait for an actual
        # EngineCore acknowledgement rather than merely scheduling a task.
        queue = await engine.add_request(
            request_id=request_id,
            prompt=_make_request(prompt_ids, image),
            params=_probe_params(seed),
        )
        engine_admitted = True
        main_stream_unfinished_at_admission = not main_finished_event.is_set()
        admission_event.set()

        from vllm.outputs import STREAM_FINISHED

        finished = False
        while not finished:
            output = queue.get_nowait() or await queue.get()
            finished = bool(output.finished)
            if output is not STREAM_FINISHED:
                final_output = output
    except (asyncio.CancelledError, Exception):
        admission_event.set()
        if queue is not None:
            await engine.abort(queue.request_id, internal=True)
        raise
    finally:
        # Prevent a failed admission from deadlocking the main stream. The task
        # exception is still propagated by asyncio.gather below.
        admission_event.set()
    elapsed = time.perf_counter() - start
    if final_output is None or not final_output.outputs:
        raise RuntimeError(f"{request_id}: vLLM returned no completion")
    completion = final_output.outputs[0]
    tail_ids = [int(value) for value in completion.token_ids]
    response_ids = prefix_ids + suffix_ids + tail_ids
    response_text = "<think>" + decode_ids(
        tokenizer, _protocol_ids(tokenizer, response_ids)
    )
    parsed = parse_response(response_text)
    bbox = None if parsed.bbox is None else [float(value) for value in parsed.bbox]
    cached = getattr(final_output, "num_cached_tokens", None)
    cached = None if cached is None else int(cached)
    base_prompt_aligned_ceiling = (
        math.ceil(len(base_prompt_ids) / cache_accounting_unit) * cache_accounting_unit
    )
    cache_alignment_overrun = (
        None if cached is None else max(0, cached - len(prompt_ids))
    )
    record = {
        "schema_version": 2,
        "condition": condition,
        "sample_id": sample_id,
        "source_index": int(source_index),
        "checkpoint_index": int(checkpoint_index),
        "token_offset": int(token_offset),
        "reasoning_prefix_token_count": len(prefix_ids),
        "reasoning_prefix_ids_sha256": _ids_sha256(prefix_ids),
        "incremental_boundary_text": decode_ids(tokenizer, prefix_ids[-32:]),
        "suffix_text": decode_ids(tokenizer, suffix_ids),
        "suffix_token_ids": suffix_ids,
        "prompt_token_count": len(prompt_ids),
        "base_prompt_token_count": len(base_prompt_ids),
        "shared_prefix_token_count": shared_prefix_count,
        "cache_accounting_unit": int(cache_accounting_unit),
        "base_prompt_aligned_ceiling": base_prompt_aligned_ceiling,
        "num_cached_tokens": cached,
        "cache_hit_present": cached is not None and cached > 0,
        "cache_hit_beyond_base_prompt_aligned_ceiling": (
            cached is not None and cached > base_prompt_aligned_ceiling
        ),
        "cache_alignment_overrun_tokens": cache_alignment_overrun,
        "cache_accounting_caveat": (
            "Qwen3.5 hybrid GDN/Mamba cache reports aligned cache-state units; "
            "num_cached_tokens is not interpreted as exact logical tokens or FLOPs saved."
        ),
        "seed": int(seed),
        "engine_admitted": engine_admitted,
        "main_stream_unfinished_at_admission": main_stream_unfinished_at_admission,
        "tail_token_ids": tail_ids,
        "tail_text": decode_ids(tokenizer, tail_ids),
        "parse_valid": bool(parsed.parse_valid),
        "parse_error": parsed.error,
        "bbox_xyxy": bbox,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "latency_seconds": elapsed,
    }
    print(
        f"  probe cp={checkpoint_index:03d} off={token_offset:04d} "
        f"cached_raw={cached} shared={shared_prefix_count} "
        f"live_admit={main_stream_unfinished_at_admission} "
        f"parse={record['parse_valid']} "
        f"seconds={elapsed:.3f}",
        flush=True,
    )
    return record


async def _run_sample_condition(
    *,
    engine: Any,
    processor: Any,
    source_row: Mapping[str, Any],
    condition: str,
    main_seed: int,
    cache_accounting_unit: int,
) -> dict[str, Any]:
    sample_id = str(source_row["sample_id"])
    source_index = int(source_row["source_index"])
    image = _load_rgb_image(str(source_row["image_path"]))
    rendered, processed = _render_and_process(
        processor, str(source_row["expression"]), image
    )
    prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
    expected_prompt_ids = [int(value) for value in source_row["prompt_input_ids"]]
    if prompt_ids != expected_prompt_ids or rendered != source_row["rendered_prompt"]:
        raise RuntimeError(f"{sample_id}: prompt/image preprocessing drifted from saved pilot")

    request_id = f"main-{condition}-{sample_id}"
    main_start = time.perf_counter()
    final_output = None
    scheduled_offsets: set[int] = set()
    probe_tasks: list[asyncio.Task[dict[str, Any]]] = []
    latest_ids: list[int] = []
    main_finished_event = asyncio.Event()
    print(f"[{condition}] {sample_id}: main request started", flush=True)
    try:
        async for output in engine.generate(
            _make_request(prompt_ids, image), _main_params(main_seed), request_id=request_id
        ):
            final_output = output
            if not output.outputs:
                continue
            latest_ids = [int(value) for value in output.outputs[0].token_ids]
            if condition == "baseline" or "</think>" in decode_ids(processor.tokenizer, latest_ids):
                continue
            for offset in _stable_causal_boundaries(processor.tokenizer, latest_ids):
                if offset <= 0 or offset in scheduled_offsets:
                    continue
                scheduled_offsets.add(offset)
                checkpoint_index = len(scheduled_offsets) - 1
                admission_event = asyncio.Event()
                probe_tasks.append(
                    asyncio.create_task(
                        _run_probe(
                            engine=engine,
                            tokenizer=processor.tokenizer,
                            image=image,
                            base_prompt_ids=prompt_ids,
                            reasoning_ids=list(latest_ids),
                            token_offset=offset,
                            checkpoint_index=checkpoint_index,
                            condition=condition,
                            sample_id=sample_id,
                            source_index=source_index,
                            main_finished_event=main_finished_event,
                            admission_event=admission_event,
                            base_seed=main_seed,
                            cache_accounting_unit=cache_accounting_unit,
                        )
                    )
                )
                # Admission, not probe completion, is the synchronization point.
                await admission_event.wait()
    finally:
        main_finished_event.set()

    main_elapsed = time.perf_counter() - main_start
    if final_output is None or not final_output.outputs:
        raise RuntimeError(f"{sample_id}/{condition}: main request returned no completion")
    completion = final_output.outputs[0]
    generation_ids = [int(value) for value in completion.token_ids]
    reasoning_ids, close_ids = split_reasoning_close(processor.tokenizer, generation_ids)

    # Resolve any last punctuation boundary plus the mandatory reasoning-end
    # checkpoint after the main request naturally closes.
    if condition != "baseline":
        final_offsets = sorted(set(natural_boundary_offsets(processor.tokenizer, reasoning_ids)) | {len(reasoning_ids)})
        for offset in final_offsets:
            if offset <= 0 or offset in scheduled_offsets:
                continue
            scheduled_offsets.add(offset)
            checkpoint_index = len(scheduled_offsets) - 1
            admission_event = asyncio.Event()
            probe_tasks.append(
                asyncio.create_task(
                    _run_probe(
                        engine=engine,
                        tokenizer=processor.tokenizer,
                        image=image,
                        base_prompt_ids=prompt_ids,
                        reasoning_ids=reasoning_ids,
                        token_offset=offset,
                        checkpoint_index=checkpoint_index,
                        condition=condition,
                        sample_id=sample_id,
                        source_index=source_index,
                        main_finished_event=main_finished_event,
                        admission_event=admission_event,
                        base_seed=main_seed,
                        cache_accounting_unit=cache_accounting_unit,
                    )
                )
            )
    probes = await asyncio.gather(*probe_tasks) if probe_tasks else []
    probes.sort(key=lambda row: int(row["token_offset"]))
    print(
        f"[{condition}] {sample_id}: main_tokens={len(generation_ids)} "
        f"probes={len(probes)} seconds={main_elapsed:.3f}",
        flush=True,
    )
    return {
        "schema_version": 2,
        "condition": condition,
        "sample_id": sample_id,
        "source_index": source_index,
        "image_path": str(source_row["image_path"]),
        "expression": str(source_row["expression"]),
        "main_seed": int(main_seed),
        "prompt_token_count": len(prompt_ids),
        "prompt_ids_sha256": _ids_sha256(prompt_ids),
        "main_generation_token_ids": generation_ids,
        "main_generation_ids_sha256": _ids_sha256(generation_ids),
        "main_reasoning_token_ids": reasoning_ids,
        "main_close_token_ids": close_ids,
        "main_reasoning_text": decode_ids(processor.tokenizer, reasoning_ids),
        "main_chosen_token_logprobs": _chosen_logprobs(completion),
        "main_finish_reason": completion.finish_reason,
        "main_stop_reason": completion.stop_reason,
        "main_num_cached_tokens": getattr(final_output, "num_cached_tokens", None),
        "main_latency_seconds": main_elapsed,
        "probe_count": len(probes),
        "probes": probes,
    }


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baselines = {
        str(row["sample_id"]): row for row in records if row["condition"] == "baseline"
    }
    parity: list[dict[str, Any]] = []
    for row in records:
        if row["condition"] == "baseline":
            continue
        baseline = baselines[str(row["sample_id"])]
        mismatch = _first_mismatch(
            baseline["main_generation_token_ids"], row["main_generation_token_ids"]
        )
        logprob_comparison = _compare_logprobs(
            baseline["main_chosen_token_logprobs"],
            row["main_chosen_token_logprobs"],
        )
        finish_reason_equal = (
            baseline["main_finish_reason"] == row["main_finish_reason"]
        )
        stop_reason_equal = baseline["main_stop_reason"] == row["main_stop_reason"]
        parity.append(
            {
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "token_ids_equal": mismatch is None,
                "first_mismatch_index": mismatch,
                "baseline_token_count": len(baseline["main_generation_token_ids"]),
                "condition_token_count": len(row["main_generation_token_ids"]),
                **logprob_comparison,
                "finish_reason_equal": finish_reason_equal,
                "stop_reason_equal": stop_reason_equal,
                "trajectory_exact": (
                    mismatch is None
                    and bool(logprob_comparison["chosen_logprobs_equal"])
                    and finish_reason_equal
                    and stop_reason_equal
                ),
            }
        )

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in records if row["condition"] == condition]
        probes = [probe for row in condition_rows for probe in row["probes"]]
        live_probes = [
            probe
            for probe in probes
            if bool(probe["main_stream_unfinished_at_admission"])
        ]
        raw_cached_values = [
            int(probe["num_cached_tokens"])
            for probe in probes
            if probe["num_cached_tokens"] is not None
        ]
        by_condition[condition] = {
            "sample_count": len(condition_rows),
            "main_latency_seconds_total": sum(float(row["main_latency_seconds"]) for row in condition_rows),
            "probe_count": len(probes),
            "probe_parse_valid_count": sum(bool(probe["parse_valid"]) for probe in probes),
            "probe_parse_valid_rate": None if not probes else sum(bool(probe["parse_valid"]) for probe in probes) / len(probes),
            "probe_latency_seconds_total": sum(float(probe["latency_seconds"]) for probe in probes),
            "probe_latency_seconds_mean": None if not probes else mean(float(probe["latency_seconds"]) for probe in probes),
            "engine_admitted_count": sum(bool(probe["engine_admitted"]) for probe in probes),
            "engine_admitted_while_main_stream_unfinished_count": len(live_probes),
            "engine_admitted_after_main_stream_finished_count": len(probes) - len(live_probes),
            "raw_num_cached_tokens_min": None if not raw_cached_values else min(raw_cached_values),
            "raw_num_cached_tokens_max": None if not raw_cached_values else max(raw_cached_values),
            "cache_hit_count": sum(bool(probe["cache_hit_present"]) for probe in probes),
            "live_cache_hit_count": sum(bool(probe["cache_hit_present"]) for probe in live_probes),
            "live_cache_hit_beyond_base_prompt_aligned_ceiling_count": sum(
                bool(probe["cache_hit_beyond_base_prompt_aligned_ceiling"])
                for probe in live_probes
            ),
            "aligned_cache_count_exceeded_logical_prompt_count": sum(
                int(probe["cache_alignment_overrun_tokens"] or 0) > 0
                for probe in probes
            ),
            "exact_token_or_flop_savings_reported": False,
        }
    probed_conditions = ["forced_close", "natural_instruction"]
    api_gate = all(
        by_condition[condition]["probe_count"] > 0
        and by_condition[condition]["engine_admitted_count"]
        == by_condition[condition]["probe_count"]
        for condition in probed_conditions
    )
    live_admission_gate = all(
        by_condition[condition]["engine_admitted_while_main_stream_unfinished_count"] > 0
        for condition in probed_conditions
    )
    incremental_cache_gate = all(
        by_condition[condition]["live_cache_hit_beyond_base_prompt_aligned_ceiling_count"] > 0
        for condition in probed_conditions
    )
    parity_gate = bool(parity) and all(item["trajectory_exact"] for item in parity)
    validity_gates = {
        "probe_api_completed": api_gate,
        "probe_admitted_before_main_stream_finished": live_admission_gate,
        "live_cache_hit_beyond_base_prompt_aligned_ceiling_observed": incremental_cache_gate,
        "main_tokens_logprobs_and_stop_exactly_match_baseline": parity_gate,
    }
    return {
        "schema_version": 2,
        "experiment_valid": all(validity_gates.values()),
        "validity_gates": validity_gates,
        "main_trajectory_parity": parity,
        "all_probed_main_token_ids_equal_baseline": bool(parity) and all(item["token_ids_equal"] for item in parity),
        "all_probed_main_trajectories_exact": parity_gate,
        "conditions": by_condition,
        "interpretation": (
            "Same-engine request admission plus prefix-cache reuse, not a live KV-cache clone. "
            "Qwen3.5 hybrid GDN/Mamba cache counters are aligned state units, so no exact logical "
            "token/FLOP saving is inferred. Wall times are descriptive and condition-order confounded."
        ),
    }


def _protocol(args: argparse.Namespace, sample_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "frozen_before_gpu_run",
        "hypothesis": (
            "Same-engine asynchronous bbox probes can reuse live reasoning prefix-cache blocks "
            "without changing the main sampled token trajectory."
        ),
        "control": "baseline main reasoning with no probes",
        "conditions": {
            "forced_close": "raw reasoning prefix + </think><answer>{\\\"bbox\\\":[ + constrained tail",
            "natural_instruction": (
                "raw reasoning prefix + natural-language current-best-box instruction + "
                "</think><answer>{\\\"bbox\\\":[ + constrained tail"
            ),
        },
        "only_key_variable_per_comparison": "presence and suffix type of independent probe requests",
        "primary_validity_metric": (
            "exact main token IDs, chosen-token logprobs, finish reason, and stop reason versus baseline"
        ),
        "primary_efficiency_metric": (
            "conservative evidence of live admission and a cache hit beyond the aligned base-prompt ceiling"
        ),
        "secondary_metrics": ["probe bbox parse rate", "probe latency", "main latency"],
        "result_updates": {
            "parity_and_cache_hit": "supports adopting same-engine prefix-cache probes",
            "main_divergence": "probe scheduling contaminates the sampled rollout under this stack",
            "low_cache_hit": "the proposed optimization is ineffective in this stack",
            "suffix_specific_parse_failure": "that probe wording/protocol is not operationally usable",
        },
        "sample_ids": list(sample_ids),
        "sample_count": len(sample_ids),
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "main_seed": int(args.seed),
        "probe_draws_per_boundary": 1,
        "response_length": RESPONSE_LENGTH,
        "bbox_tail_max_tokens": BBOX_TAIL_MAX_TOKENS,
        "cache_accounting_unit": int(args.cache_accounting_unit),
        "full_determinism": bool(args.full_determinism),
        "engine": "vLLM AsyncLLM same-engine prefix caching; no live request fork API",
    }


async def _run(args: argparse.Namespace) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    sample_ids = list(DEFAULT_SAMPLE_IDS[: 1 if args.mode == "smoke" else args.limit])
    if args.full_determinism and os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError(
            "--full-determinism requires PYTHONHASHSEED to equal --seed before "
            f"interpreter startup (expected {args.seed!s})"
        )
    rows = _read_source_rows(args.source.resolve(), sample_ids)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "protocol.json", _protocol(args, sample_ids))
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 2,
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "vllm", "transformers")
            },
            "git": _git_receipt(),
            "sample_ids": sample_ids,
            "model": str(args.model.resolve()),
            "source": str(args.source.resolve()),
            "determinism_env": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONHASHSEED",
                    "VERL_FULL_DETERMINISM",
                    "VLLM_BATCH_INVARIANT",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "FLASH_ATTENTION_DETERMINISTIC",
                    "NCCL_DETERMINISTIC",
                    "NCCL_ALGO",
                )
            },
        },
    )

    processor = _processor(args.model.resolve())
    engine_args = AsyncEngineArgs(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=True,
        max_model_len=18433,
        max_num_batched_tokens=24576,
        max_num_seqs=32,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        seed=int(args.seed),
        disable_log_stats=False,
        stream_interval=1,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    records: list[dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    try:
        for condition in CONDITIONS:
            if records:
                reset = await engine.reset_prefix_cache()
                print(f"prefix cache reset before {condition}: {reset}", flush=True)
                if not reset:
                    raise RuntimeError("vLLM refused to reset prefix cache between conditions")
                await engine.reset_mm_cache()
                print(f"multimodal cache reset before {condition}", flush=True)
            for row in rows:
                record = await _run_sample_condition(
                    engine=engine,
                    processor=processor,
                    source_row=row,
                    condition=condition,
                    main_seed=int(args.seed),
                    cache_accounting_unit=int(args.cache_accounting_unit),
                )
                records.append(record)
                _append_jsonl(records_path, record)
    finally:
        engine.shutdown()
    _write_json(output_dir / "summary.json", _summarize(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "3"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--cache-accounting-unit",
        type=int,
        default=544,
        help="Observed Qwen3.5 hybrid-cache accounting alignment; used only for a conservative gate.",
    )
    parser.add_argument(
        "--full-determinism",
        action="store_true",
        help="Enable vLLM batch invariance and deterministic CUDA environment flags.",
    )
    args = parser.parse_args()
    if args.mode == "formal" and args.limit != 5:
        parser.error("the frozen formal protocol requires --limit 5")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be in [0.1, 0.95]")
    if args.cache_accounting_unit <= 0:
        parser.error("--cache-accounting-unit must be positive")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
