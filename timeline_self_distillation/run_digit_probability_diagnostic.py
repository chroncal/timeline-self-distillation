"""Read-only attribution of saved bbox failures to teacher versus adapter changes.

Score every distinct prefix of all 255 saved decodes under the original teacher,
base and both final checkpoints. Reconstruct all decodes with the original RNG.
Then replace exactly one first-divergence distribution, leaving the rest of the
checkpoint policy unchanged. These oracle-selected interventions are diagnostics,
not a deployable repair or a new method evaluation. No optimizer or update exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if "--device" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--device") + 1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import xgrammar as xgr
from transformers import Qwen3_5ForConditionalGeneration

from live_kv_probe_prototype.run_hf_fork import _advance, _append_jsonl, _git_receipt, _write_json
from reasoning_checkpoints.extractor import decode_ids
from reasoning_checkpoints.run_pilot import _processor
from timeline_self_distillation.run_checkpoint_diagnostic import build_grammar_supports, load_adapter_checkpoint
from timeline_self_distillation.run_decode_diagnostic import read_pilot_records, score_bbox, summarize_outputs
from timeline_self_distillation.run_opd_micro import build_states
from timeline_self_distillation.run_teacher_pilot import BOX_OPEN, BOX_REGEX, MODEL, SEED, fork
from timeline_self_distillation.terminal_adapter import install_terminal_query_lora, terminal_parameter_whitelist
from verl.experimental.routed_grounding.router import parse_response

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("base", "teacher", "reverse", "forward")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution_summary(logits, support):
    values = torch.tensor(logits, dtype=torch.float32)
    logp = torch.log_softmax(values, -1)
    p = logp.exp()
    return {"logits": values.tolist(), "log_probs": logp.tolist(), "probs": p.tolist(),
            "entropy": float(-(p * logp).sum()), "top_id": support[int(values.argmax())]}


def gradient_direction(logp_values, logq_values, reference_index, alternative_index, divergence):
    """Unit logit-space GD direction; not an Adam/parameter-space attribution."""
    logp = torch.tensor(logp_values, dtype=torch.float64)
    logq = torch.tensor(logq_values, dtype=torch.float64)
    p, q = logp.exp(), logq.exp()
    if divergence == "forward":
        gradient = p - q
    elif divergence == "reverse":
        kl = (p * (logp - logq)).sum()
        gradient = p * (logp - logq - kl)
    else:
        raise ValueError(divergence)
    velocity = -p * (gradient - (p * gradient).sum())
    return {"alternative_probability_velocity": float(velocity[alternative_index]),
            "alternative_vs_reference_log_odds_velocity": float(gradient[reference_index] - gradient[alternative_index])}


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def pair_attribution(pair, row):
    position = first_difference(pair["base_ids"], pair["checkpoint_ids"])
    if position is None:
        raise ValueError("identical pair has no probability attribution")
    a, b = pair["base_ids"][position], pair["checkpoint_ids"][position]
    ia, ib = row["support_ids"].index(a), row["support_ids"].index(b)
    distributions = row["distributions"]
    measures = {}
    for name, dist in distributions.items():
        pa, pb = dist["probs"][ia], dist["probs"][ib]
        measures[name] = {"reference_probability": pa, "alternative_probability": pb,
                          "alternative_vs_reference_log_odds": dist["log_probs"][ib] - dist["log_probs"][ia],
                          "reference_rank": 1 + sum(x > pa for x in dist["probs"]),
                          "alternative_rank": 1 + sum(x > pb for x in dist["probs"]),
                          "entropy": dist["entropy"], "top_id": dist["top_id"]}
    for name in ("teacher", "reverse", "forward"):
        measures[name]["alternative_probability_delta_from_base"] = measures[name]["alternative_probability"] - measures["base"]["alternative_probability"]
        measures[name]["log_odds_delta_from_base"] = measures[name]["alternative_vs_reference_log_odds"] - measures["base"]["alternative_vs_reference_log_odds"]
    directions = {}
    for policy in ("base", pair["checkpoint_model"]):
        directions[policy] = gradient_direction(distributions[policy]["log_probs"], distributions["teacher"]["log_probs"], ia, ib, pair["checkpoint_model"])
    return {**pair, "prefix_ids": pair["base_ids"][:position], "prefix_text": row["prefix_text"],
            "position_one_based": position + 1, "reference_token_id": a, "alternative_token_id": b,
            "distributions": measures, "local_logit_gradient_directions": directions,
            "iou_delta": pair["checkpoint_iou"] - pair["base_iou"],
            "warning": "Alternative names a sampled branch, not a globally wrong token; attribution is same-prefix and endpoint-local."}


def full_scores(distribution, support, vocab):
    scores = torch.full((1, vocab), -torch.inf, dtype=torch.float32, device="cuda")
    scores[0, torch.tensor(support, device="cuda")] = torch.tensor(distribution["logits"], device="cuda")
    return scores


def replay_saved(record, rows, vocab):
    generator = torch.Generator(device="cuda").manual_seed(record["seed"])
    for index, expected in enumerate(record["ids"]):
        row = rows[(record["sample_id"], tuple(record["ids"][:index]))]
        scores = full_scores(row["distributions"][record["model"]], row["support_ids"], vocab)
        actual = int(scores.argmax(-1)) if record["mode"] == "greedy" else int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator))
        if actual != expected:
            raise RuntimeError(f"Historical RNG parity failed: {record['model']}/{record['sample_id']}/{record['mode']}/{record['draw']}/{index}: {actual}!={expected}")


def intervene(model, tokenizer, grammar, state, pair, donor_row, donor):
    position = first_difference(pair["base_ids"], pair["checkpoint_ids"])
    prefix = pair["checkpoint_ids"][:position]
    cache = fork(state["student"])
    model.model.rope_deltas = state["rope_deltas"].clone()
    current = state["last_opening_id"]
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    vocab = model.config.text_config.vocab_size
    bitmask = xgr.allocate_token_bitmask(1, vocab)
    generator = torch.Generator(device="cuda").manual_seed(pair["seed"])
    ids = []
    with torch.no_grad():
        for index in range(48):
            cache, logits = _advance(model, cache, [current])
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite intervention logits")
            scores = logits.float().clone()
            xgr.reset_token_bitmask(bitmask)
            if matcher.fill_next_token_bitmask(bitmask):
                xgr.apply_token_bitmask_inplace(scores, bitmask.to(scores.device), vocab_size=vocab)
            if index == position:
                if ids != prefix:
                    raise RuntimeError("Intervention did not reach the saved common prefix")
                support = torch.isfinite(scores[0]).nonzero().flatten().tolist()
                if support != donor_row["support_ids"]:
                    raise RuntimeError("Intervention grammar support mismatch")
                scores = full_scores(donor_row["distributions"][donor], support, vocab)
            token = int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator))
            if not matcher.accept_token(token):
                raise RuntimeError("Intervention grammar rejected token")
            ids.append(token)
            if index < position and token != prefix[index]:
                raise RuntimeError("Pre-intervention RNG parity failed")
            if index == position and donor == "base" and token != pair["base_ids"][index]:
                raise RuntimeError("Base distribution failed to restore the base first token")
            if matcher.is_completed():
                break
            current = token
    text = decode_ids(tokenizer, ids)
    parsed = parse_response("<think>probe" + BOX_OPEN + text)
    result = {"ids": ids, "text": text, "bbox": None if parsed.bbox is None else list(parsed.bbox),
              "parse_valid": bool(parsed.parse_valid), "parse_error": parsed.error, "completed": matcher.is_completed()}
    result.update(score_bbox(result, state["row"]["ground_truth_bbox"]))
    result.update({k: pair[k] for k in ("sample_id", "checkpoint_model", "draw", "seed")})
    result.update(donor=donor, position_one_based=position + 1, base_iou=pair["base_iou"],
                  original_checkpoint_iou=pair["checkpoint_iou"], base_text=pair["base_text"],
                  original_checkpoint_text=pair["checkpoint_text"],
                  restored_base_bbox_text=text.split("]}")[0] == pair["base_text"].split("]}")[0])
    return result


def run(args):
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(f"Set PYTHONHASHSEED={SEED}")
    start = time.perf_counter()
    output = args.output_dir
    output.mkdir(parents=False, exist_ok=False)
    old_manifest = json.loads((args.decode_dir / "manifest.json").read_text())
    for package, expected in old_manifest["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Historical package version changed: {package}")
    receipts = {**old_manifest["sources"], **old_manifest["inputs_and_checkpoints"]}
    for receipt in receipts.values():
        if digest(receipt["path"]) != receipt["sha256"]:
            raise RuntimeError(f"Historical input/source changed: {receipt['path']}")
    for path in (Path(__file__), args.decode_dir / "records.jsonl", args.decode_dir / "pairs.jsonl"):
        receipts[str(path)] = {"path": str(path), "sha256": digest(path)}
    records = read_jsonl(args.decode_dir / "records.jsonl")
    pairs = read_jsonl(args.decode_dir / "pairs.jsonl")
    original_summary = summarize_outputs(records)
    pilot = read_pilot_records(args.pilot_records)
    expected_inputs = old_manifest["inputs_and_checkpoints"]
    for name, path in (("pilot_records", args.pilot_records),
                       ("reverse_checkpoint", args.reverse_dir / "final_adapter.pt"),
                       ("forward_checkpoint", args.forward_dir / "final_adapter.pt")):
        if digest(path) != expected_inputs[name]["sha256"]:
            raise RuntimeError(f"Requested input differs from historical input: {name}")
    for record in pilot:
        path = Path(record["image_path"])
        receipts[str(path)] = {"path": str(path), "sha256": digest(path)}
    changed = [p for p in pairs if p["mode"] == "random" and p["first_numeric_difference"]["different"]]
    if len(pairs) != 170 or len(changed) != 76:
        raise RuntimeError("Historical pair coverage drift")
    for pair in changed:
        if pair["first_token_divergence"]["kind"] != "digit":
            raise RuntimeError("Protocol expects the observed first numeric change to be a token digit change")
    protocol = {"kind": "read_only_digit_probability_attribution", "teacher": "original early_span1, old saved entity, adapter OFF",
                "policies": POLICIES, "scope": "all 255 saved decodes and all 160 random checkpoint/base pairs, five seen images",
                "scoring": "every unique prefix, all grammar-legal candidates including numeric termination, original temperature 1",
                "primary_attribution": "p_base, q_teacher, p_checkpoint at identical first-divergence prefix; absolute probabilities and relative log odds",
                "interventions": "all 76 numeric-changed pairs: replace only first differing token distribution with base or teacher; rest remains checkpoint; identical seed",
                "intervention_warning": "oracle-selected diagnostic, not deployable repair; no filtering by improvement/degradation",
                "unchanged_pairs": "retain original outputs for all-model 80-draw summaries",
                "gradient": "analytic unit logit-space KL descent, not actual Adam/parameter-space trajectory",
                "training": False, "backbone_updates": False, "ground_truth": "evaluation only; never used for generation or teacher",
                "model": str(MODEL), "seed": SEED, "invalid_iou": 0,
                "hypotheses": ["teacher raises harmful alternatives", "adapter overshoots or moves against teacher", "off-prefix continuation compounds first divergence"],
                "historical_limitation": "only final adapters saved; cannot identify a particular optimizer step without new training, which is forbidden"}
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "manifest.json", {"command": [sys.executable, *sys.argv], "git": _git_receipt(), "receipts": receipts,
                                          "versions": old_manifest["versions"]})
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(MODEL)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    adapter = install_terminal_query_lora(model, rank=8)
    terminal_parameter_whitelist(model, adapter)
    versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    adapter.enabled = False
    states = {s["row"]["sample_id"]: s for s in build_states(model, processor, pilot, "early_span1")}
    tokenizer, vocab = processor.tokenizer, model.config.text_config.vocab_size
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)).compile_regex(BOX_REGEX)
    rows, children = {}, defaultdict(set)
    sequences = sorted({(r["sample_id"], tuple(r["ids"])) for r in records})
    for sample_id, ids in sequences:
        supports = build_grammar_supports(grammar, vocab, ids)
        for index, support in enumerate(supports):
            prefix = ids[:index]
            key = (sample_id, prefix)
            if key in rows:
                if rows[key]["support_ids"] != support:
                    raise RuntimeError("Same prefix has inconsistent grammar support")
            else:
                rows[key] = {"sample_id": sample_id, "prefix_ids": list(prefix), "prefix_text": decode_ids(tokenizer, prefix) if prefix else "",
                             "support_ids": support, "support_pieces": [decode_ids(tokenizer, [t]) for t in support], "distributions": {}}
            if index + 1 < len(ids):
                children[key].add(ids[index])
    print(f"PLAN sequences={len(sequences)} unique_prefixes={len(rows)} changed_pairs={len(changed)}", flush=True)

    def visit(sample_id, prefix, cache, current, policy):
        cache, logits = _advance(model, cache, [current])
        row = rows[(sample_id, prefix)]
        selected = logits[0, row["support_ids"]].float()
        if not torch.isfinite(selected).all():
            raise RuntimeError("Nonfinite probability scores")
        row["distributions"][policy] = distribution_summary(selected.cpu().tolist(), row["support_ids"])
        targets = sorted(children[(sample_id, prefix)])
        for number, token in enumerate(targets):
            child_cache = cache if number == len(targets) - 1 else fork(cache)
            visit(sample_id, prefix + (token,), child_cache, token, policy)

    checkpoints = {"reverse": args.reverse_dir / "final_adapter.pt", "forward": args.forward_dir / "final_adapter.pt"}
    with torch.no_grad():
        for policy in POLICIES:
            if policy in checkpoints:
                load_adapter_checkpoint(model, adapter, checkpoints[policy])
            adapter.enabled = policy in checkpoints
            for sample_id, state in states.items():
                model.model.rope_deltas = state["rope_deltas"].clone()
                cache = state["teacher"] if policy == "teacher" else state["student"]
                visit(sample_id, (), fork(cache), state["last_opening_id"], policy)
                print(f"SCORED {policy} {sample_id}", flush=True)
    for key in sorted(rows):
        _append_jsonl(output / "prefix_distributions.jsonl", rows[key])
    for record in records:
        replay_saved(record, rows, vocab)
    _write_json(output / "parity.json", {"exact_saved_decode_rng_replays": len(records), "expected": 255, "passed": True})
    print("PARITY all 255 saved token sequences reproduced", flush=True)
    for pair in changed:
        position = first_difference(pair["base_ids"], pair["checkpoint_ids"])
        row = rows[(pair["sample_id"], tuple(pair["base_ids"][:position]))]
        _append_jsonl(output / "pair_attribution.jsonl", pair_attribution(pair, row))
    interventions = []
    for policy, checkpoint in checkpoints.items():
        load_adapter_checkpoint(model, adapter, checkpoint)
        adapter.enabled = True
        subset = [p for p in changed if p["checkpoint_model"] == policy]
        for index, pair in enumerate(subset):
            position = first_difference(pair["base_ids"], pair["checkpoint_ids"])
            row = rows[(pair["sample_id"], tuple(pair["base_ids"][:position]))]
            for donor in ("base", "teacher"):
                result = intervene(model, tokenizer, grammar, states[pair["sample_id"]], pair, row, donor)
                _append_jsonl(output / "interventions.jsonl", result)
                interventions.append(result)
            if (index + 1) % 5 == 0 or index + 1 == len(subset):
                print(f"INTERVENTION {policy} {index + 1}/{len(subset)} pairs", flush=True)
    summaries = {}
    for policy in checkpoints:
        original = [r for r in records if r["model"] == policy and r["mode"] == "random"]
        for donor in ("base", "teacher"):
            index = {(r["sample_id"], r["seed"]): r for r in interventions if r["checkpoint_model"] == policy and r["donor"] == donor}
            results = [index.get((r["sample_id"], r["seed"]), r) for r in original]
            summaries[f"{policy}_{donor}_first_digit"] = {"n": len(results), "mean_iou": sum(r["iou"] for r in results) / len(results),
                                                         "invalid": sum(not r["parse_valid"] for r in results),
                                                         "per_sample": {sid: sum(r["iou"] for r in results if r["sample_id"] == sid) / 16 for sid in states}}
    if not all(p._version == versions[n] for n, p in model.named_parameters() if not p.requires_grad):
        raise RuntimeError("Frozen backbone parameter version changed")
    for receipt in receipts.values():
        if digest(receipt["path"]) != receipt["sha256"]:
            raise RuntimeError("Source/input changed during diagnostic")
    summary = {"original": original_summary, "unique_sequences": len(sequences), "unique_prefixes": len(rows),
               "attributed_pairs": len(changed), "intervention_outputs": len(interventions), "interventions": summaries,
               "frozen_backbone_unchanged": True, "input_source_hashes_unchanged": True,
               "seconds": time.perf_counter() - start, "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    _write_json(output / "summary.json", summary)
    print("COMPLETE " + json.dumps({k: v for k, v in summary.items() if k != "original"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3")
    base = Path("outputs/research_experiments/timeline_opd")
    parser.add_argument("--decode-dir", type=Path, default=base / "diagnosis_decode_n5_d16_v1")
    parser.add_argument("--pilot-records", type=Path, default=base / "teacher_pilot_n5_v1/records.jsonl")
    parser.add_argument("--reverse-dir", type=Path, default=base / "opd_span1_reverse_n5_s20_v1")
    parser.add_argument("--forward-dir", type=Path, default=base / "opd_span1_forward_n5_s20_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
