"""Five-image causal readout diagnostic; no training and no early teacher.

All full-attention layers receive +/- log(4) on original image keys ONLY
during bbox decoding. All conditions use the same explicit-mask/math backend.
This is an inference intervention, not an OPD gain or a mechanism proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

if "--device" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--device") + 1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import xgrammar as xgr
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3_5ForConditionalGeneration

from live_kv_probe_prototype.run_hf_fork import _append_jsonl, _git_receipt, _write_json
from reasoning_checkpoints.run_pilot import _load_rgb_image, _processor, _render_and_process
from timeline_self_distillation.run_opd_micro import build_states, sample_bbox
from timeline_self_distillation.run_teacher_pilot import BOX_REGEX, DEFAULT_SAMPLE_IDS, MODEL, SEED
from timeline_self_distillation.terminal_adapter import install_terminal_query_lora
from verl.experimental.routed_grounding.router import xyxy_iou


def add_visual_bias(mask, hidden, key_length, positions, log_bias):
    if hidden.shape[:2] != (1, 1):
        raise ValueError("intervention restricted to single-token cached bbox decode")
    if not positions or min(positions) < 0 or max(positions) >= key_length:
        raise ValueError("image key positions outside active cache")
    bias = torch.zeros((1, 1, 1, key_length), device=hidden.device, dtype=hidden.dtype)
    bias[..., positions] = log_bias
    if mask is not None:
        if mask.dtype == torch.bool or mask.shape[-2:] != (1, key_length):
            raise ValueError("unexpected original mask")
        bias = bias + mask
    return bias


@contextmanager
def visual_bias(model, positions, log_bias):
    saved = []
    layers = model.model.language_model.layers
    for layer in layers:
        if layer.layer_type != "full_attention":
            continue
        attention = layer.self_attn
        original = attention.forward

        def wrapper(*args, _attention=attention, _original=original, **kwargs):
            if args:
                raise ValueError("expected named decoder attention inputs")
            hidden = kwargs["hidden_states"]
            cache = kwargs["past_key_values"]
            keys = cache.layers[_attention.layer_idx].keys
            length = keys.shape[-2] + hidden.shape[1]
            kwargs["attention_mask"] = add_visual_bias(
                kwargs.get("attention_mask"), hidden, length, positions, log_bias
            )
            with sdpa_kernel(SDPBackend.MATH):
                return _original(**kwargs)

        saved.append((attention, original))
        attention.forward = wrapper
    try:
        yield
    finally:
        for attention, original in saved:
            attention.forward = original


def run(args):
    records = [json.loads(line) for line in args.pilot_records.read_text().splitlines() if line.strip()]
    if tuple(r["sample_id"] for r in records) != tuple(DEFAULT_SAMPLE_IDS):
        raise ValueError("requires original fixed five IDs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    conditions = {"matched_zero": 0.0, "visual_x4": math.log(4), "visual_div4": -math.log(4)}
    _write_json(
        args.output_dir / "protocol.json",
        {
            "hypothesis": "late visual retrieval deficit predicts IoU response to image-key logit bias",
            "primary": "all-case mean IoU; invalid=0; no GT selection",
            "conditions": conditions,
            "intervention": "all six full-attention layers, bbox decode only; no change to r/e/prefix",
            "backend": "explicit 4D additive mask and SDPA MATH in every intervention condition",
            "draw_seed": "SEED + 9000000 + sample_order*1000 + draw; four draws",
            "scope": "inference sensitivity on five seen images; not training or generalization",
            "image_encoding": "one per image in build_states; later processor call only locates token IDs",
        },
    )
    files = [
        Path(__file__),
        Path("timeline_self_distillation/run_opd_micro.py"),
        Path("timeline_self_distillation/terminal_adapter.py"),
        Path("live_kv_probe_prototype/run_hf_fork.py"),
        args.pilot_records,
    ]
    _write_json(
        args.output_dir / "manifest.json",
        {
            "git": _git_receipt(),
            "command": [sys.executable, *sys.argv],
            "model": str(MODEL),
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "seed": SEED,
            "device": args.device,
        },
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(MODEL)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
    )
    adapter = install_terminal_query_lora(model, rank=8)
    adapter.enabled = False
    states = build_states(model, processor, records, "late_entity")
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=model.config.text_config.vocab_size)
    ).compile_regex(BOX_REGEX)
    reference = [json.loads(line) for line in args.reference_eval.read_text().splitlines()]
    reference = {(r["sample_id"], r["seed"]): r for r in reference if r["step"] == 0}
    results = []
    started = time.perf_counter()
    versions = {name: p._version for name, p in model.named_parameters()}
    with torch.no_grad():
        for index, state in enumerate(states):
            row = state["row"]
            _, inputs = _render_and_process(processor, row["expression"], _load_rgb_image(row["image_path"]))
            positions = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten().tolist()
            for condition, bias in conditions.items():
                with visual_bias(model, positions, bias):
                    for draw in range(4):
                        seed = SEED + 9_000_000 + index * 1000 + draw
                        sample = sample_bbox(model, processor.tokenizer, grammar, state, seed)
                        iou = (
                            float(xyxy_iou(sample["bbox"], row["ground_truth_bbox"])) if sample["parse_valid"] else 0.0
                        )
                        record = {k: v for k, v in sample.items() if k != "supports"}
                        record.update(
                            sample_id=row["sample_id"],
                            condition=condition,
                            draw=draw,
                            iou=iou,
                            image_token_count=len(positions),
                            previous_sdpa_ids_equal=sample["ids"] == reference[(row["sample_id"], seed)]["ids"],
                        )
                        results.append(record)
                        _append_jsonl(args.output_dir / "records.jsonl", record)
                print(
                    "[DIAG-VISUAL] "
                    + json.dumps(
                        {
                            "sample": row["sample_id"],
                            "condition": condition,
                            "mean_iou": mean(r["iou"] for r in results[-4:]),
                        }
                    ),
                    flush=True,
                )
    if not all(p._version == versions[name] for name, p in model.named_parameters()):
        raise RuntimeError("model weights changed during read-only diagnostic")
    summary = {
        condition: {
            "mean_iou": mean(r["iou"] for r in results if r["condition"] == condition),
            "acc_05": mean(r["iou"] >= 0.5 for r in results if r["condition"] == condition),
            "valid": sum(r["parse_valid"] for r in results if r["condition"] == condition),
            "count": 20,
            "per_sample": {
                row["sample_id"]: mean(
                    r["iou"] for r in results if r["condition"] == condition and r["sample_id"] == row["sample_id"]
                )
                for row in records
            },
        }
        for condition in conditions
    }
    summary["backend_parity_count"] = sum(
        r["previous_sdpa_ids_equal"] for r in results if r["condition"] == "matched_zero"
    )
    summary["seconds"] = time.perf_counter() - started
    summary["max_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    summary["frozen_weights_unchanged"] = True
    _write_json(args.output_dir / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3")
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--reference-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
