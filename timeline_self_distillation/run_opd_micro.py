"""Bounded five-image bbox-on-policy training diagnostic, not a generalization result.

Replay each saved frozen full reasoning once. Resample bbox tokens on-policy at
every update, score the same prefixes with a frozen teacher, update only the
terminal query LoRA, and evaluate fixed fresh seeds on these same five images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

if "--device" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--device") + 1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import xgrammar as xgr
from transformers import Qwen3_5ForConditionalGeneration

from live_kv_probe_prototype.run_hf_fork import _advance, _append_jsonl, _git_receipt, _write_json
from reasoning_checkpoints.extractor import decode_ids
from reasoning_checkpoints.run_pilot import _load_rgb_image, _processor, _render_and_process
from timeline_self_distillation.run_teacher_pilot import BOX_OPEN, BOX_REGEX, MODEL, QUERY, SEED, fork
from timeline_self_distillation.terminal_adapter import install_terminal_query_lora, terminal_parameter_whitelist
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou


def build_states(model, processor, records, teacher):
    states = []
    model.eval()
    with torch.no_grad():
        for row in records:
            print(f"CACHE {row['sample_id']} replay full frozen reasoning", flush=True)
            _, inputs = _render_and_process(processor, row["expression"], _load_rgb_image(row["image_path"]))
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            out = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            c0 = out.past_key_values
            delta = model.model.rope_deltas.detach().clone()
            cT, _ = _advance(model, fork(c0), row["reasoning_ids"])
            if teacher == "early_step0":
                cteacher = c0
            elif teacher == "early_span1":
                cteacher, _ = _advance(model, fork(c0), row["reasoning_ids"][: row["first_span_offset"]])
            elif teacher == "late_scrub":
                cteacher, _ = _advance(model, fork(c0), row["scrubbed_reasoning_ids"])
            elif teacher == "late_entity":
                cteacher = cT
            else:
                raise ValueError(teacher)
            suffix = QUERY.format(entity=row["entity"], question=row["expression"]) + BOX_OPEN
            suffix_ids = processor.tokenizer.encode(suffix, add_special_tokens=False)
            cstudent, _ = _advance(model, fork(cT), suffix_ids[:-1])
            cteacher, _ = _advance(model, fork(cteacher), suffix_ids[:-1])
            states.append(
                {
                    "row": row,
                    "student": cstudent,
                    "teacher": cteacher,
                    "last_opening_id": suffix_ids[-1],
                    "rope_deltas": delta,
                }
            )
    return states


def sample_bbox(model, tokenizer, grammar, state, seed):
    cache = fork(state["student"])
    model.model.rope_deltas = state["rope_deltas"]
    current_input = state["last_opening_id"]
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    mask = xgr.allocate_token_bitmask(1, model.config.text_config.vocab_size)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids, supports = [], []
    with torch.no_grad():
        for _ in range(48):
            cache, logits = _advance(model, cache, [current_input])
            scores = logits.float().clone()
            if not torch.isfinite(scores).all():
                raise RuntimeError("nonfinite student logits")
            xgr.reset_token_bitmask(mask)
            if matcher.fill_next_token_bitmask(mask):
                xgr.apply_token_bitmask_inplace(
                    scores, mask.to(scores.device), vocab_size=model.config.text_config.vocab_size
                )
            support = torch.isfinite(scores[0]).nonzero().flatten()
            if not support.numel():
                raise RuntimeError("empty grammar support")
            token = int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator).item())
            if not matcher.accept_token(token):
                raise RuntimeError("grammar rejected sample")
            ids.append(token)
            supports.append(support)
            if matcher.is_completed():
                break
            current_input = token
    parsed = parse_response("<think>probe" + BOX_OPEN + decode_ids(tokenizer, ids))
    bbox = None if parsed.bbox is None else list(parsed.bbox)
    return {
        "ids": ids,
        "supports": supports,
        "bbox": bbox,
        "parse_valid": bool(parsed.parse_valid),
        "completed": bool(matcher.is_completed()),
        "seed": seed,
    }


def score_teacher(model, adapter, state, sample):
    adapter.enabled = False
    cache = fork(state["teacher"])
    model.model.rope_deltas = state["rope_deltas"]
    current_input, logqs = state["last_opening_id"], []
    with torch.no_grad():
        for token, support in zip(sample["ids"], sample["supports"], strict=True):
            cache, logits = _advance(model, cache, [current_input])
            logqs.append(torch.log_softmax(logits[0, support].float(), -1).detach())
            current_input = token
    return logqs


def update(model, tokenizer, adapter, parameters, optimizer, grammar, state, seed, divergence):
    adapter.enabled = True
    sample = sample_bbox(model, tokenizer, grammar, state, seed)
    numeric = [decode_ids(tokenizer, [token]).isdigit() for token in sample["ids"]]
    if not any(numeric):
        raise RuntimeError("no coordinate tokens; not widening loss mask")
    logqs = score_teacher(model, adapter, state, sample)
    adapter.enabled = True
    cache = fork(state["student"])
    model.model.rope_deltas = state["rope_deltas"]
    current_input = state["last_opening_id"]
    optimizer.zero_grad(set_to_none=True)
    value = 0.0
    for token, support, is_numeric, logq in zip(sample["ids"], sample["supports"], numeric, logqs, strict=True):
        with torch.set_grad_enabled(is_numeric):
            cache, logits = _advance(model, cache, [current_input])
            if is_numeric:
                logp = torch.log_softmax(logits[0, support].float(), -1)
                if divergence == "reverse":
                    loss = (logp.exp() * (logp - logq)).sum() / sum(numeric)
                else:
                    loss = (logq.exp() * (logq - logp)).sum() / sum(numeric)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite coordinate KL")
                loss.backward()
                value += float(loss.detach())
        current_input = token
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("gradient leaked into frozen parameters")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)):
        raise RuntimeError("nonfinite adapter gradient")
    optimizer.step()
    return {
        "loss": value,
        "grad_norm": grad_norm,
        "sampled_bbox": sample["bbox"],
        "numeric_token_count": sum(numeric),
        "seed": seed,
    }


def evaluate(model, tokenizer, adapter, grammar, states, output, step, draws):
    adapter.enabled = True
    results = []
    for index, state in enumerate(states):
        row = state["row"]
        for draw in range(draws):
            # Held-out draws, not held-out images: never call this generalization.
            seed = SEED + 9_000_000 + index * 1000 + draw
            pred = sample_bbox(model, tokenizer, grammar, state, seed)
            iou = float(xyxy_iou(pred["bbox"], row["ground_truth_bbox"])) if pred["parse_valid"] else 0.0
            result = {k: v for k, v in pred.items() if k != "supports"}
            result.update(sample_id=row["sample_id"], step=step, iou=iou, hit_05=iou >= 0.5)
            results.append(result)
            _append_jsonl(output / "eval.jsonl", result)
    summary = {
        "step": step,
        "mean_iou": mean(r["iou"] for r in results),
        "acc_05": mean(r["hit_05"] for r in results),
        "valid": sum(r["parse_valid"] for r in results),
        "n": len(results),
    }
    _append_jsonl(output / "eval_summary.jsonl", summary)
    print("EVAL " + json.dumps(summary), flush=True)
    return summary


def run(args):
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(f"set PYTHONHASHSEED={SEED}")
    records = [json.loads(line) for line in args.pilot_records.read_text().splitlines() if line.strip()]
    if len(records) != 5:
        raise RuntimeError("this diagnostic requires exactly five fixed samples")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        args.output_dir / "protocol.json",
        {
            "kind": "five_seen_image_bbox_on_policy_micro_training",
            "teacher": args.teacher,
            "steps": args.steps,
            "lr": args.lr,
            "rank": 8,
            "loss": f"numeric-only {args.divergence} KL; task=0; no GT in updates",
            "invalid_box_policy": "grammar numeric rows distilled even if geometry invalid; eval IoU=0",
            "main_policy": "frozen complete r/e, replayed once; bbox freshly sampled each update",
            "eval": "same five seen images, four fresh fixed draw seeds; steps 0,10,20; not generalization",
            "eval_draws": args.eval_draws,
            "source": str(args.pilot_records),
            "seed": SEED,
            "sample_ids": [r["sample_id"] for r in records],
            "model": str(MODEL),
        },
    )
    _write_json(
        args.output_dir / "manifest.json",
        {
            "git": _git_receipt(),
            "command": [sys.executable, *sys.argv],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    parameters = list(terminal_parameter_whitelist(model, adapter).values())
    adapter.enabled = False
    states = build_states(model, processor, records, args.teacher)
    vocab = model.config.text_config.vocab_size
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=vocab)
    ).compile_regex(BOX_REGEX)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    initial_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
    initial = evaluate(model, processor.tokenizer, adapter, grammar, states, args.output_dir, 0, args.eval_draws)
    start = time.perf_counter()
    final = initial
    for step in range(1, args.steps + 1):
        state = states[(step - 1) % len(states)]
        metrics = update(
            model,
            processor.tokenizer,
            adapter,
            parameters,
            optimizer,
            grammar,
            state,
            SEED + 20_000_000 + step,
            args.divergence,
        )
        metrics.update(step=step, sample_id=state["row"]["sample_id"])
        _append_jsonl(args.output_dir / "train.jsonl", metrics)
        print("TRAIN " + json.dumps(metrics), flush=True)
        if step % 10 == 0 or step == args.steps:
            final = evaluate(
                model, processor.tokenizer, adapter, grammar, states, args.output_dir, step, args.eval_draws
            )
    unchanged = all(p._version == initial_versions[name] for name, p in model.named_parameters() if not p.requires_grad)
    if not unchanged:
        raise RuntimeError("frozen parameter version changed")
    torch.save(
        {k: p.detach().cpu() for k, p in terminal_parameter_whitelist(model, adapter).items()},
        args.output_dir / "final_adapter.pt",
    )
    _write_json(
        args.output_dir / "summary.json",
        {
            "initial": initial,
            "final": final,
            "frozen_parameter_versions_unchanged": unchanged,
            "seconds": time.perf_counter() - start,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "limitation": "Five seen images; exploratory optimization/transfer diagnostic, not test-set gain.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-records", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--teacher", choices=("early_step0", "early_span1", "late_scrub", "late_entity"), required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--divergence", choices=("reverse", "forward"), default="reverse")
    p.add_argument("--eval-draws", type=int, default=4)
    p.add_argument("--device", default="3")
    run(p.parse_args())
