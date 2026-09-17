"""Five-image exploratory comparison of entity-conditioned timeline teachers.

Run as a module; preserves full natural reasoning, uses sequential HF cache
continuations, and never supplies ground truth to a model branch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
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

from live_kv_probe_prototype.run_hf_fork import (
    DEFAULT_SAMPLE_IDS,
    DEFAULT_SOURCE,
    _advance,
    _append_jsonl,
    _fork_cache_cow,
    _git_receipt,
    _read_source_rows,
    _sample_top_p,
    _write_json,
)
from reasoning_checkpoints.extractor import decode_ids, natural_boundary_offsets
from reasoning_checkpoints.run_pilot import RESPONSE_LENGTH, _load_rgb_image, _processor, _render_and_process
from timeline_self_distillation.entity_bridge import (
    BRIDGE_VERSIONS,
    ENTITY_REGEX,
    INSTANCE_ENTITY_OPEN,
    bridge_prompt,
    inspect_bridge_result,
)
from verl.experimental.routed_grounding.router import parse_response, xyxy_iou

MODEL = Path("/mnt/sda/sujingyang/models/Qwen3.5-0.8B")
SEED = 260600564
BOX_OPEN = '</think><answer>{"bbox":['
NUMBER = r"(?:1000|0|[1-9][0-9]{0,2})"
BOX_REGEX = rf"{NUMBER},{NUMBER},{NUMBER},{NUMBER}\]\}}</answer>"
ENTITY_OPEN = INSTANCE_ENTITY_OPEN
QUERY = (
    "\nThe final target is: {entity}. Locate exactly this target described by the original request: "
    "{question}. Return only its bounding box in the original image, coordinates 0 to 1000.\n"
)
CONDITIONS = ("late_direct", "late_entity", "early_step0", "early_span1", "late_scrub", "late_sham")
COORD_TUPLE = re.compile(r"[\[(]\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){3}\s*[\])]")


def fork(cache):
    return _fork_cache_cow(cache)[0]


def scrub_and_sham(tokenizer, ids):
    """Replace only digit tokens inside four-number tuples; match token count."""
    decoded = [decode_ids(tokenizer, ids[:i]) for i in range(len(ids) + 1)]
    spans = [m.span() for m in COORD_TUPLE.finditer(decoded[-1])]
    positions = []
    for i, token in enumerate(ids):
        piece = decode_ids(tokenizer, [token])
        if piece.isdigit() and any(len(decoded[i]) >= a and len(decoded[i + 1]) <= b for a, b in spans):
            positions.append(i)
    replacement = tokenizer.encode("?", add_special_tokens=False)
    if len(replacement) != 1:
        raise RuntimeError("scrub placeholder must be exactly one token")
    scrubbed = list(ids)
    for i in positions:
        scrubbed[i] = replacement[0]
    candidates = [i for i, tok in enumerate(ids) if decode_ids(tokenizer, [tok]).strip().isalpha()]
    # Match the number of changed tokens, prefer nearby nonnumeric positions.
    candidates.sort(key=lambda i: (min((abs(i - p) for p in positions), default=0), i))
    sham_positions = sorted(candidates[: len(positions)])
    if len(sham_positions) != len(positions):
        raise RuntimeError("not enough non-coordinate tokens for the declared sham control")
    sham = list(ids)
    for i in sham_positions:
        sham[i] = replacement[0]
    return (
        scrubbed,
        sham,
        {
            "coordinate_token_positions": positions,
            "sham_token_positions": sham_positions,
            "tuple_char_spans": spans,
            "placeholder_token_id": replacement[0],
        },
    )


def constrained_generate(
    model, tokenizer, compiled, cache, suffix, seed, *, greedy=False, limit=48, prefilled_logits=None
):
    """Full support, temperature=1 for bbox; greedy for the fixed entity bridge."""
    start = time.perf_counter()
    if prefilled_logits is None:
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        cache, logits = _advance(model, cache, suffix_ids)
    else:
        # The entire suffix was already consumed on the caller's cache fork.
        # Retain suffix in the receipt; do not feed its tokens twice.
        logits = prefilled_logits
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, model.config.text_config.vocab_size)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    ids, logps = [], []
    for _ in range(limit):
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite unmasked model logits")
        scores = logits.float().clone()
        xgr.reset_token_bitmask(bitmask)
        if matcher.fill_next_token_bitmask(bitmask):
            xgr.apply_token_bitmask_inplace(
                scores, bitmask.to(scores.device), vocab_size=model.config.text_config.vocab_size
            )
        logp = torch.log_softmax(scores, dim=-1)
        if not torch.isfinite(logp).any():
            raise RuntimeError("grammar produced empty support")
        token = (
            int(scores.argmax(-1).item())
            if greedy
            else int(torch.multinomial(logp.exp(), 1, generator=generator).item())
        )
        if not matcher.accept_token(token):
            raise RuntimeError("grammar rejected sampled token")
        ids.append(token)
        logps.append(float(logp[0, token].item()))
        if matcher.is_completed():
            break
        cache, logits = _advance(model, cache, [token])
    return {
        "token_ids": ids,
        "text": decode_ids(tokenizer, ids),
        "logprobs": logps,
        "completed": bool(matcher.is_completed()),
        "seed": int(seed),
        "suffix": suffix,
        "latency_seconds": time.perf_counter() - start,
    }


def generate_entity_bridge(model, tokenizer, entity_grammar, cache, expression, seed, *, version="instance_v2"):
    """Read the completed reasoning on a disposable fork, then audit the text.

    The expression is used only to report request echoes, never as a fallback.
    This audit deliberately does not certify visual or semantic correctness.
    """
    raw = constrained_generate(
        model, tokenizer, entity_grammar, fork(cache), bridge_prompt(version), seed, greedy=True, limit=96
    )
    audit = inspect_bridge_result(raw, expression, version)
    return {"entity": audit["entity"], "entity_result": raw, "entity_audit": audit}


def rollout_main(model, processor, row, seed):
    image = _load_rgb_image(row["image_path"])
    rendered, processed = _render_and_process(processor, row["expression"], image)
    prompt_ids = processed["input_ids"][0].tolist()
    if prompt_ids != row["prompt_input_ids"] or rendered != row["rendered_prompt"]:
        raise RuntimeError("prompt/preprocessing differs from fixed source")
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in processed.items()}
    out = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
    main, logits = out.past_key_values, out.logits[:, -1, :]
    early = fork(main)
    first, first_offset, pending = None, None, {}
    ids, logps = [], []
    close_id = processor.tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_id) != 1:
        raise RuntimeError("unexpected think-close tokenization")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    finish = "length"
    for _ in range(RESPONSE_LENGTH):
        token, lp = _sample_top_p(logits, generator=generator)
        if token in (close_id[0], processor.tokenizer.eos_token_id):
            finish = "stop" if token == close_id[0] else "eos"
            break  # cache excludes closing token, consistently for all arms
        ids.append(token)
        logps.append(lp)
        main, logits = _advance(model, main, [token])
        if first is None:
            offsets = natural_boundary_offsets(processor.tokenizer, ids)
            stable = [
                o for o in offsets if o < len(ids) or decode_ids(processor.tokenizer, ids[:o]).endswith(("\n", "\r"))
            ]
            if stable:
                first_offset = min(stable)
                first = fork(main) if first_offset == len(ids) else pending[first_offset]
                pending.clear()
            else:
                pending = {o: c for o, c in pending.items() if o in offsets}
                if len(ids) in offsets:
                    pending[len(ids)] = fork(main)
    if finish != "stop":
        raise RuntimeError(f"{row['sample_id']}: incomplete main reasoning ({finish}); no shortened substitution")
    if first is None:
        first, first_offset = fork(main), len(ids)
    return (
        early,
        first,
        main,
        {
            "reasoning_ids": ids,
            "reasoning_text": decode_ids(processor.tokenizer, ids),
            "reasoning_logprobs": logps,
            "reasoning_tokens": len(ids),
            "first_span_offset": first_offset,
            "finish_reason": finish,
            "prompt_token_count": len(prompt_ids),
        },
    )


def run(args):
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError(f"set PYTHONHASHSEED={args.seed}")
    rows = _read_source_rows(args.source, DEFAULT_SAMPLE_IDS)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "kind": "exploratory_five_sample_effect_pilot",
        "sample_ids": list(DEFAULT_SAMPLE_IDS),
        "question": "Does same-final-entity early grounding outperform matched late grounding?",
        "pivot": "Also evaluate coordinate-scrub and matched-count nonnumeric-sham teachers.",
        "conditions": CONDITIONS,
        "draws": args.draws,
        "seed": args.seed,
        "model": str(args.model),
        "source": str(args.source),
        "main_sampling": {"temperature": 0.8, "top_p": 0.95, "cap": RESPONSE_LENGTH},
        "bbox_sampling": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 48},
        "bbox_regex": BOX_REGEX,
        "entity_regex": ENTITY_REGEX,
        "primary": "mean IoU across all draws, averaged per image; invalid=0",
        "secondary": "Acc@0.5; no best-of-draw selection",
        "query_template": QUERY,
        "entity_open": bridge_prompt(args.entity_bridge),
        "entity_bridge_version": args.entity_bridge,
        "entity_audit": "format/echo/unresolved recorded; non-echo is not certified instance binding",
        "limitations": "Five prior execution-successful cases, not selected for localization; exploratory only.",
    }
    _write_json(args.output_dir / "protocol.json", protocol)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "command": [sys.executable, *sys.argv],
            "git": _git_receipt(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "xgrammar")},
            "device": os.environ["CUDA_VISIBLE_DEVICES"],
        },
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.text_config.vocab_size)
    )
    bbox_grammar, entity_grammar = compiler.compile_regex(BOX_REGEX), compiler.compile_regex(ENTITY_REGEX)
    records = []
    start = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            print(f"MAIN {row['sample_id']} starting full rollout", flush=True)
            c0, c1, cT, main = rollout_main(model, processor, row, args.seed)
            bridge = generate_entity_bridge(
                model, tokenizer, entity_grammar, cT, row["expression"], args.seed, version=args.entity_bridge
            )
            entity, entity_result, entity_audit = bridge["entity"], bridge["entity_result"], bridge["entity_audit"]
            _append_jsonl(args.output_dir / "entity_audits.jsonl", {"sample_id": row["sample_id"], **bridge})
            if not entity_audit["usable_for_probe"]:
                _write_json(args.output_dir / f"entity_failure_{row['sample_id']}.json", bridge)
                raise RuntimeError("entity bridge unusable; no hidden substitution by GT/original request")
            print(
                f"ENTITY {row['sample_id']} len={main['reasoning_tokens']} span={main['first_span_offset']} "
                f"status={entity_audit['semantic_status']} {entity!r}",
                flush=True,
            )
            scrubbed, sham, scrub_info = scrub_and_sham(tokenizer, main["reasoning_ids"])
            sample = {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "expression": row["expression"],
                "entity": entity,
                "entity_result": entity_result,
                "entity_audit": entity_audit,
                **main,
                "scrub": scrub_info,
                "scrubbed_reasoning_ids": scrubbed,
                "sham_reasoning_ids": sham,
                "ground_truth_bbox": row["ground_truth_bbox"],
                "conditions": {},
            }
            query_suffix = QUERY.format(entity=entity, question=row["expression"]) + BOX_OPEN
            # Same complete main r/e and common RNG seeds; each condition gets a fresh fork.
            for condition in CONDITIONS:
                replay_seconds = 0.0
                if condition in ("late_scrub", "late_sham"):
                    replay_start = time.perf_counter()
                    replacement_ids = scrubbed if condition == "late_scrub" else sham
                    cache = fork(c0)
                    if replacement_ids:
                        cache, _ = _advance(model, cache, replacement_ids)
                    replay_seconds = time.perf_counter() - replay_start
                else:
                    cache = {"late_direct": cT, "late_entity": cT, "early_step0": c0, "early_span1": c1}[condition]
                suffix = BOX_OPEN if condition == "late_direct" else query_suffix
                results = []
                for draw in range(args.draws):
                    seed = args.seed + 1_000_000 + int(row["source_index"]) * 1000 + draw
                    result = constrained_generate(model, tokenizer, bbox_grammar, fork(cache), suffix, seed)
                    parsed = parse_response("<think>probe" + BOX_OPEN + result["text"])
                    box = None if parsed.bbox is None else list(parsed.bbox)
                    iou = float(xyxy_iou(box, row["ground_truth_bbox"])) if parsed.parse_valid else 0.0
                    result.update(
                        bbox=box,
                        parse_valid=bool(parsed.parse_valid),
                        parse_error=parsed.error,
                        iou=iou,
                        hit_05=iou >= 0.5,
                        draw=draw,
                    )
                    results.append(result)
                    print(
                        f"BOX {row['sample_id']} {condition} draw={draw} valid={parsed.parse_valid} "
                        f"iou={iou:.4f} bbox={box}",
                        flush=True,
                    )
                sample["conditions"][condition] = {
                    "draws": results,
                    "mean_iou": mean(r["iou"] for r in results),
                    "acc_05": mean(r["hit_05"] for r in results),
                    "replay_seconds": replay_seconds,
                }
                _append_jsonl(
                    args.output_dir / "progress.jsonl",
                    {"sample_id": row["sample_id"], "condition": condition, **sample["conditions"][condition]},
                )
                if condition in ("late_scrub", "late_sham"):
                    del cache
            records.append(sample)
            _append_jsonl(args.output_dir / "records.jsonl", sample)
            del c0, c1, cT
    summary = {
        "sample_count": len(records),
        "entity_bridge": {
            "version": args.entity_bridge,
            "request_echo_count": sum(r["entity_audit"]["echoes_request"] for r in records),
            "instance_binding_verified": False,
            "note": "Non-echo text still requires semantic review; syntax is not instance identification.",
        },
        "draws_per_condition_per_sample": args.draws,
        "seconds": time.perf_counter() - start,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "conditions": {},
        "per_sample": [],
    }
    for condition in CONDITIONS:
        entries = [r["conditions"][condition] for r in records]
        summary["conditions"][condition] = {
            "mean_iou": mean(r["mean_iou"] for r in entries),
            "acc_05": mean(r["acc_05"] for r in entries),
            "parse_valid": sum(d["parse_valid"] for r in entries for d in r["draws"]),
            "draw_count": sum(len(r["draws"]) for r in entries),
        }
    for r in records:
        summary["per_sample"].append(
            {
                "sample_id": r["sample_id"],
                "entity": r["entity"],
                "entity_audit": r["entity_audit"],
                "reasoning_tokens": r["reasoning_tokens"],
                **{c: r["conditions"][c]["mean_iou"] for c in CONDITIONS},
            }
        )
    _write_json(args.output_dir / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="3")
    parser.add_argument("--entity-bridge", choices=BRIDGE_VERSIONS, default="instance_v2")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
