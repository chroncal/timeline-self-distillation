#!/usr/bin/env python3
"""PROTOTYPE: true copy-on-write KV/state forks for live bbox probes.

Unlike the vLLM prefix-cache approximation in ``run.py``, this runner owns the
Transformers decode loop.  At each causal reasoning boundary it forks the
current DynamicCache, generates a short bbox continuation on the fork, drops
the fork, and resumes the untouched main cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

# Bind the physical GPU and deterministic CUDA settings before importing torch.
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
STRICT_BBOX_TAIL_REGEX = BBOX_TAIL_REGEX.replace(
    r"\]\}\s*</answer>", r"\]\}</answer>"
)


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


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _fork_cache_cow(cache: Any) -> tuple[Any, dict[str, int]]:
    """Fork DynamicCache without copying immutable attention-prefix tensors.

    Dynamic attention layers replace ``keys``/``values`` with ``torch.cat`` on
    update, so their existing prefix tensors can be shared safely.  Qwen3.5's
    GDN layers mutate convolutional and recurrent states in place, so those
    tensors must be cloned for the branch.
    """

    forked = copy.copy(cache)
    forked.layers = []
    shared_attention_bytes = 0
    cloned_linear_state_bytes = 0
    for layer in cache.layers:
        branch_layer = copy.copy(layer)
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(branch_layer, name, value)
                shared_attention_bytes += _tensor_bytes(value)
        for name in ("conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                cloned = value.clone()
                setattr(branch_layer, name, cloned)
                cloned_linear_state_bytes += _tensor_bytes(cloned)
        forked.layers.append(branch_layer)
    return forked, {
        "shared_attention_prefix_bytes": shared_attention_bytes,
        "cloned_linear_state_bytes": cloned_linear_state_bytes,
    }


def _sample_top_p(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> tuple[int, float]:
    scores = logits.float() / float(temperature)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    remove = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scores, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
    probs = torch.softmax(filtered, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    token_id = int(token.item())
    logprob = float(torch.log_softmax(filtered, dim=-1)[0, token_id].item())
    return token_id, logprob


def _advance(model: Any, cache: Any, token_ids: Sequence[int]) -> tuple[Any, torch.Tensor]:
    # Transformers 5.5.3 Qwen3.5 GDN consumes the previous recurrent state
    # only for seq_len == 1. A multi-token cached suffix instead starts its
    # chunk recurrence from None, silently discarding the prefix GDN state.
    # Advance every suffix token through the same recurrent decode path.
    if not token_ids:
        raise ValueError("cached continuation requires at least one token")
    for token_id in token_ids:
        inputs = torch.tensor([[int(token_id)]], dtype=torch.long, device="cuda")
        outputs = model(
            input_ids=inputs,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = outputs.past_key_values
    return cache, outputs.logits[:, -1, :]


def _run_probe(
    *,
    model: Any,
    tokenizer: Any,
    compiled_bbox_grammar: Any,
    branch_cache: Any,
    fork_stats: Mapping[str, int],
    base_prompt_token_count: int,
    reasoning_ids: Sequence[int],
    token_offset: int,
    checkpoint_index: int,
    condition: str,
    sample_id: str,
    source_index: int,
    base_seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    prefix_ids = [int(value) for value in reasoning_ids[:token_offset]]
    suffix_ids = _probe_suffix_ids(tokenizer, condition)
    expected_cache_length = base_prompt_token_count + len(prefix_ids)
    if int(branch_cache.get_seq_length()) != expected_cache_length:
        raise RuntimeError(
            f"{sample_id}/{condition}/cp{checkpoint_index}: cache length "
            f"{branch_cache.get_seq_length()} != expected {expected_cache_length}"
        )

    seed = (
        int(base_seed)
        + PROBE_SEED_OFFSETS[condition]
        + int(source_index) * 10_000
        + int(checkpoint_index)
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    matcher = xgr.GrammarMatcher(compiled_bbox_grammar, terminate_without_stop_token=True)
    bitmask_cpu = xgr.allocate_token_bitmask(1, int(model.config.text_config.vocab_size))

    branch_cache, logits = _advance(model, branch_cache, suffix_ids)
    tail_ids: list[int] = []
    tail_logprobs: list[float] = []
    for _ in range(BBOX_TAIL_MAX_TOKENS):
        xgr.reset_token_bitmask(bitmask_cpu)
        if matcher.fill_next_token_bitmask(bitmask_cpu):
            bitmask_gpu = bitmask_cpu.to(device=logits.device, non_blocking=True)
            xgr.apply_token_bitmask_inplace(
                logits,
                bitmask_gpu,
                vocab_size=int(model.config.text_config.vocab_size),
            )
        token_id, logprob = _sample_top_p(logits, generator=generator)
        if not matcher.accept_token(token_id):
            raise RuntimeError(f"xgrammar rejected its own allowed token {token_id}")
        tail_ids.append(token_id)
        tail_logprobs.append(logprob)
        if matcher.is_completed():
            break
        branch_cache, logits = _advance(model, branch_cache, [token_id])

    prefix_and_suffix = prefix_ids + suffix_ids + tail_ids
    response_text = "<think>" + decode_ids(
        tokenizer, _protocol_ids(tokenizer, prefix_and_suffix)
    )
    parsed = parse_response(response_text)
    bbox = None if parsed.bbox is None else [float(value) for value in parsed.bbox]
    elapsed = time.perf_counter() - start
    record = {
        "schema_version": 1,
        "condition": condition,
        "sample_id": sample_id,
        "source_index": int(source_index),
        "checkpoint_index": int(checkpoint_index),
        "token_offset": int(token_offset),
        "reasoning_prefix_token_count": len(prefix_ids),
        "reasoning_prefix_ids_sha256": _ids_sha256(prefix_ids),
        "suffix_text": decode_ids(tokenizer, suffix_ids),
        "suffix_token_ids": suffix_ids,
        "tail_token_ids": tail_ids,
        "tail_chosen_logprobs": tail_logprobs,
        "tail_text": decode_ids(tokenizer, tail_ids),
        "parse_valid": bool(parsed.parse_valid),
        "parse_error": parsed.error,
        "bbox_xyxy": bbox,
        "grammar_completed": bool(matcher.is_completed()),
        "seed": int(seed),
        "main_prefix_tokens_reused_without_forward": expected_cache_length,
        "branch_forward_input_tokens": len(suffix_ids) + max(0, len(tail_ids) - 1),
        "fork_shared_attention_prefix_bytes": int(fork_stats["shared_attention_prefix_bytes"]),
        "fork_cloned_linear_state_bytes": int(fork_stats["cloned_linear_state_bytes"]),
        "latency_seconds": elapsed,
    }
    print(
        f"  {condition} cp={checkpoint_index:03d} off={token_offset:04d} "
        f"tail={len(tail_ids):02d} parse={record['parse_valid']} seconds={elapsed:.3f}",
        flush=True,
    )
    return record


def _run_sample_condition(
    *,
    model: Any,
    processor: Any,
    compiled_bbox_grammar: Any,
    source_row: Mapping[str, Any],
    condition: str,
    main_seed: int,
) -> dict[str, Any]:
    sample_id = str(source_row["sample_id"])
    source_index = int(source_row["source_index"])
    image = _load_rgb_image(str(source_row["image_path"]))
    rendered, processed = _render_and_process(
        processor, str(source_row["expression"]), image
    )
    prompt_ids = [int(value) for value in processed["input_ids"][0].tolist()]
    if prompt_ids != [int(value) for value in source_row["prompt_input_ids"]]:
        raise RuntimeError(f"{sample_id}: prompt preprocessing drifted from saved pilot")
    if rendered != source_row["rendered_prompt"]:
        raise RuntimeError(f"{sample_id}: rendered prompt drifted from saved pilot")

    device_inputs = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
    }
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(main_seed))
    close_ids = processor.tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise RuntimeError(f"expected </think> to be one token, got {close_ids}")
    close_id = int(close_ids[0])
    eos_id = int(processor.tokenizer.eos_token_id)

    print(f"[{condition}] {sample_id}: prompt prefill", flush=True)
    started = time.perf_counter()
    outputs = model(
        **device_inputs,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    main_cache = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    if int(main_cache.get_seq_length()) != len(prompt_ids):
        raise RuntimeError("prompt cache length mismatch")

    generation_ids: list[int] = []
    main_logprobs: list[float] = []
    scheduled_offsets: set[int] = set()
    pending: dict[int, tuple[list[int], Any, dict[str, int]]] = {}
    probes: list[dict[str, Any]] = []
    finish_reason = "length"
    checkpoint_index = 0

    for _ in range(RESPONSE_LENGTH):
        token_id, logprob = _sample_top_p(logits, generator=generator)
        generation_ids.append(token_id)
        main_logprobs.append(logprob)

        if token_id in (close_id, eos_id):
            finish_reason = "stop" if token_id == close_id else "eos"
            reasoning_ids = generation_ids[:-1]
            if condition != "baseline" and reasoning_ids and len(reasoning_ids) not in scheduled_offsets:
                branch_cache, fork_stats = _fork_cache_cow(main_cache)
                probes.append(
                    _run_probe(
                        model=model,
                        tokenizer=processor.tokenizer,
                        compiled_bbox_grammar=compiled_bbox_grammar,
                        branch_cache=branch_cache,
                        fork_stats=fork_stats,
                        base_prompt_token_count=len(prompt_ids),
                        reasoning_ids=reasoning_ids,
                        token_offset=len(reasoning_ids),
                        checkpoint_index=checkpoint_index,
                        condition=condition,
                        sample_id=sample_id,
                        source_index=source_index,
                        base_seed=main_seed,
                    )
                )
                scheduled_offsets.add(len(reasoning_ids))
            break

        main_cache, logits = _advance(model, main_cache, [token_id])
        if condition == "baseline":
            continue

        raw_offsets = natural_boundary_offsets(processor.tokenizer, generation_ids)
        stable_offsets = {
            int(offset)
            for offset in raw_offsets
            if offset < len(generation_ids)
            or decode_ids(processor.tokenizer, generation_ids[:offset]).endswith(("\n", "\r"))
        }
        for offset in sorted(stable_offsets):
            if offset <= 0 or offset in scheduled_offsets:
                continue
            if offset == len(generation_ids):
                branch_cache, fork_stats = _fork_cache_cow(main_cache)
                prefix_snapshot = list(generation_ids)
            elif offset in pending:
                prefix_snapshot, branch_cache, fork_stats = pending.pop(offset)
            else:
                raise RuntimeError(f"{sample_id}: missing exact cache snapshot for offset {offset}")
            probes.append(
                _run_probe(
                    model=model,
                    tokenizer=processor.tokenizer,
                    compiled_bbox_grammar=compiled_bbox_grammar,
                    branch_cache=branch_cache,
                    fork_stats=fork_stats,
                    base_prompt_token_count=len(prompt_ids),
                    reasoning_ids=prefix_snapshot,
                    token_offset=offset,
                    checkpoint_index=checkpoint_index,
                    condition=condition,
                    sample_id=sample_id,
                    source_index=source_index,
                    base_seed=main_seed,
                )
            )
            scheduled_offsets.add(offset)
            checkpoint_index += 1

        for offset in list(pending):
            if offset < len(generation_ids) and offset not in stable_offsets:
                del pending[offset]
        current_offset = len(generation_ids)
        if (
            current_offset in raw_offsets
            and current_offset not in stable_offsets
            and current_offset not in scheduled_offsets
            and current_offset not in pending
        ):
            branch_cache, fork_stats = _fork_cache_cow(main_cache)
            pending[current_offset] = (list(generation_ids), branch_cache, fork_stats)

    elapsed = time.perf_counter() - started
    reasoning_ids, reasoning_close_ids = split_reasoning_close(
        processor.tokenizer, generation_ids
    )
    probes.sort(key=lambda row: int(row["token_offset"]))
    print(
        f"[{condition}] {sample_id}: main_tokens={len(generation_ids)} "
        f"probes={len(probes)} seconds={elapsed:.3f}",
        flush=True,
    )
    return {
        "schema_version": 1,
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
        "main_close_token_ids": reasoning_close_ids,
        "main_chosen_token_logprobs": main_logprobs,
        "main_finish_reason": finish_reason,
        "main_latency_seconds": elapsed,
        "probe_count": len(probes),
        "probes": probes,
    }


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if int(left_id) != int(right_id):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


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
        logprobs_equal = baseline["main_chosen_token_logprobs"] == row["main_chosen_token_logprobs"]
        finish_equal = baseline["main_finish_reason"] == row["main_finish_reason"]
        parity.append(
            {
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "token_ids_equal": mismatch is None,
                "first_mismatch_index": mismatch,
                "chosen_logprobs_equal": logprobs_equal,
                "finish_reason_equal": finish_equal,
                "trajectory_exact": mismatch is None and logprobs_equal and finish_equal,
            }
        )

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        probes = [probe for row in rows for probe in row["probes"]]
        conditions[condition] = {
            "sample_count": len(rows),
            "main_latency_seconds_total": sum(float(row["main_latency_seconds"]) for row in rows),
            "probe_count": len(probes),
            "probe_parse_valid_count": sum(bool(probe["parse_valid"]) for probe in probes),
            "probe_parse_valid_rate": None if not probes else mean(bool(probe["parse_valid"]) for probe in probes),
            "grammar_completed_count": sum(bool(probe["grammar_completed"]) for probe in probes),
            "probe_latency_seconds_total": sum(float(probe["latency_seconds"]) for probe in probes),
            "probe_latency_seconds_mean": None if not probes else mean(float(probe["latency_seconds"]) for probe in probes),
            "prefix_tokens_reused_without_forward_total": sum(
                int(probe["main_prefix_tokens_reused_without_forward"]) for probe in probes
            ),
            "branch_forward_input_tokens_total": sum(
                int(probe["branch_forward_input_tokens"]) for probe in probes
            ),
            "fork_shared_attention_prefix_bytes_mean": None if not probes else mean(
                int(probe["fork_shared_attention_prefix_bytes"]) for probe in probes
            ),
            "fork_cloned_linear_state_bytes_mean": None if not probes else mean(
                int(probe["fork_cloned_linear_state_bytes"]) for probe in probes
            ),
        }
    parity_gate = bool(parity) and all(item["trajectory_exact"] for item in parity)
    probe_gate = all(
        conditions[condition]["probe_count"] > 0
        and conditions[condition]["grammar_completed_count"] == conditions[condition]["probe_count"]
        for condition in ("forced_close", "natural_instruction")
    )
    return {
        "schema_version": 1,
        "experiment_valid": parity_gate and probe_gate,
        "validity_gates": {
            "main_tokens_logprobs_and_finish_exactly_match_baseline": parity_gate,
            "both_probe_variants_completed_constrained_bbox_grammar": probe_gate,
        },
        "main_trajectory_parity": parity,
        "conditions": conditions,
        "interpretation": (
            "Transformers-owned decode loop with copy-on-write DynamicCache forks. Existing full-attention "
            "prefix tensors are shared; mutable Qwen3.5 GDN recurrent/conv states are cloned. Probe forward "
            "passes consume only suffix/tail tokens, never the main prefix."
        ),
    }


def _protocol(args: argparse.Namespace, sample_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_before_gpu_run",
        "hypothesis": (
            "Copy-on-write cache branches at causal reasoning boundaries can emit bbox probes "
            "without changing the main sampled trajectory."
        ),
        "control": "same Transformers decode loop and seed with no cache forks",
        "conditions": {
            "forced_close": "fork + </think><answer>{\"bbox\":[ + constrained tail",
            "natural_instruction": (
                "fork + natural current-best-box instruction + "
                "</think><answer>{\"bbox\":[ + constrained tail"
            ),
        },
        "primary_validity_metric": (
            "exact main token IDs, chosen-token logprobs, and finish reason versus baseline"
        ),
        "primary_efficiency_metric": (
            "prefix tokens reused with zero branch forward plus shared/cloned cache bytes"
        ),
        "sample_ids": list(sample_ids),
        "sample_count": len(sample_ids),
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "main_seed": int(args.seed),
        "probe_draws_per_boundary": 1,
        "cache_fork": (
            "share immutable DynamicLayer K/V prefix tensors; clone mutable GDN conv/recurrent states"
        ),
    }


def run(args: argparse.Namespace) -> None:
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError(
            f"export PYTHONHASHSEED={args.seed} before launching this deterministic comparison"
        )
    sample_ids = list(DEFAULT_SAMPLE_IDS[: 1 if args.mode == "smoke" else args.limit])
    rows = _read_source_rows(args.source.resolve(), sample_ids)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "protocol.json", _protocol(args, sample_ids))
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "xgrammar")
            },
            "git": _git_receipt(),
            "sample_ids": sample_ids,
            "model": str(args.model.resolve()),
            "source": str(args.source.resolve()),
        },
    )

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(args.model.resolve())
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model.resolve()),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        processor.tokenizer,
        vocab_size=int(model.config.text_config.vocab_size),
    )
    compiled_bbox_grammar = xgr.GrammarCompiler(tokenizer_info).compile_regex(
        STRICT_BBOX_TAIL_REGEX
    )

    records: list[dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    with torch.inference_mode():
        for condition in CONDITIONS:
            for row in rows:
                record = _run_sample_condition(
                    model=model,
                    processor=processor,
                    compiled_bbox_grammar=compiled_bbox_grammar,
                    source_row=row,
                    condition=condition,
                    main_seed=int(args.seed),
                )
                records.append(record)
                _append_jsonl(records_path, record)
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
    args = parser.parse_args()
    if args.mode == "formal" and args.limit != 5:
        parser.error("the frozen formal protocol requires --limit 5")
    return args


if __name__ == "__main__":
    run(parse_args())
