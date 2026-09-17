"""Frozen span-1/2/3 versus full-reasoning bbox probes; no training or new bridge."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

if "--device" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--device") + 1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import xgrammar as xgr  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402

from live_kv_probe_prototype.run_hf_fork import _advance, _append_jsonl, _git_receipt, _write_json  # noqa: E402
from reasoning_checkpoints.extractor import decode_ids, natural_boundary_offsets  # noqa: E402
from reasoning_checkpoints.run_pilot import _load_rgb_image, _processor, _render_and_process  # noqa: E402
from timeline_self_distillation.run_checkpoint_diagnostic import compare_cache_snapshot, snapshot_cache  # noqa: E402
from timeline_self_distillation.run_decode_diagnostic import (  # noqa: E402
    _DEPENDENCY_FILES,
    EXPECTED_SAMPLES,
    _file_receipt,
    _frozen_versions,
    _package_versions,
    _read_jsonl,
    _sample_bbox,
    micro_seed,
    read_pilot_records,
    score_bbox,
)
from timeline_self_distillation.run_teacher_pilot import BOX_OPEN, BOX_REGEX, MODEL, QUERY, SEED, fork  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("span1", "span2", "span3", "late")
DRAWS = 16
TOTAL = len(CONDITIONS) * len(EXPECTED_SAMPLES) * (DRAWS + 1)


def span_anchors(tokenizer, row):
    """Use exactly the existing splitter, not semantic filtering or hand-picked spans."""
    ids = row["reasoning_ids"]
    if decode_ids(tokenizer, ids) != row["reasoning_text"]:
        raise ValueError("saved reasoning text/IDs differ")
    offsets = natural_boundary_offsets(tokenizer, ids)
    if len(offsets) < 3:
        raise ValueError("fewer than three existing natural boundaries; no fallback")
    if offsets[0] != row["first_span_offset"]:
        raise ValueError("span1 differs from saved pilot boundary")
    anchors, previous = {}, 0
    for index, offset in enumerate(offsets[:3], 1):
        prefix = decode_ids(tokenizer, ids[:offset])
        preceding = decode_ids(tokenizer, ids[:previous])
        segment = prefix[len(preceding) :]
        if not prefix.startswith(preceding) or not segment.strip():
            raise ValueError("non-prefix or empty natural span")
        anchors[f"span{index}"] = {"offset": offset, "prefix_text": prefix, "span_text": segment}
        previous = offset
    anchors["late"] = {"offset": len(ids), "prefix_text": row["reasoning_text"], "span_text": None}
    return anchors


def capture_prefixes(model, c0, reasoning_ids, anchors):
    """Replay the unchanged trajectory once, fork at cumulative boundaries."""
    cache, previous, saved = fork(c0), 0, {}
    for condition in CONDITIONS:
        offset = anchors[condition]["offset"]
        if not previous <= offset <= len(reasoning_ids):
            raise ValueError("non-monotonic reasoning boundary")
        if offset > previous:
            cache, _ = _advance(model, cache, reasoning_ids[previous:offset])
        saved[condition] = fork(cache)
        previous = offset
    if previous != len(reasoning_ids):
        raise ValueError("full reasoning was not replayed")
    return saved


def output_key(row):
    return row["sample_id"], row["mode"], row["draw"], row["seed"]


def late_parity(records, reference):
    expected = {output_key(r): r for r in reference if r["model"] == "base"}
    actual_rows = [r for r in records if r["condition"] == "late"]
    actual = {output_key(r): r for r in actual_rows}
    if len(expected) != 85 or len(actual_rows) != 85 or len(actual) != 85 or expected.keys() != actual.keys():
        raise ValueError("late/reference key coverage must match all 85 outputs")
    entries = []
    for key, value in actual.items():
        old = expected[key]
        entries.append(
            {
                "key": list(key),
                "ids_equal": value["ids"] == old["ids"],
                "bbox_equal": value["bbox"] == old["bbox"],
                "iou_equal": value["iou"] == old["iou"],
            }
        )
    return {
        "count": len(entries),
        "all_exact": all(all(e[k] for k in ("ids_equal", "bbox_equal", "iou_equal")) for e in entries),
        "entries": entries,
    }


def summarize(records):
    expected = {
        (c, s, mode, draw, micro_seed(SEED, order, draw))
        for c in CONDITIONS
        for order, s in enumerate(EXPECTED_SAMPLES)
        for mode, n in (("random", DRAWS), ("greedy", 1))
        for draw in range(n)
    }
    actual = {(r["condition"], *output_key(r)) for r in records}
    if len(records) != TOTAL or actual != expected:
        raise ValueError(f"expected exactly {TOTAL} distinct protocol outputs")
    for r in records:
        if not math.isfinite(r["iou"]):
            raise FloatingPointError("nonfinite IoU")
        if not r["parse_valid"] and r["iou"] != 0:
            raise ValueError("invalid bbox must count as zero")

    def metrics(rows):
        return {
            "count": len(rows),
            "mean_iou": mean(r["iou"] for r in rows),
            "iou_sample_sd": stdev(r["iou"] for r in rows) if len(rows) > 1 else None,
            "acc_05": mean(r["iou"] >= 0.5 for r in rows),
            "invalid_count": sum(not r["parse_valid"] for r in rows),
            "invalid_ratio": mean(not r["parse_valid"] for r in rows),
        }

    summary = {"total_outputs": len(records), "primary_metric": "random mean IoU; invalid=0", "conditions": {}}
    for condition in CONDITIONS:
        summary["conditions"][condition] = {}
        for mode in ("random", "greedy"):
            subset = [r for r in records if r["condition"] == condition and r["mode"] == mode]
            summary["conditions"][condition][mode] = {
                **metrics(subset),
                "per_sample": {s: metrics([r for r in subset if r["sample_id"] == s]) for s in EXPECTED_SAMPLES},
            }
    baseline = summary["conditions"]["span1"]["random"]["mean_iou"]
    for condition in CONDITIONS:
        value = summary["conditions"][condition]["random"]["mean_iou"]
        summary["conditions"][condition]["random"].update(
            delta_vs_span1=value - baseline, relative_vs_span1=value / baseline - 1 if baseline else None
        )
    return summary


def run(args):
    started = time.time()
    rows = read_pilot_records(args.pilot_records)
    reference = _read_jsonl(args.reference_records)
    output = args.output_dir
    output.mkdir(parents=False, exist_ok=False)
    dependencies = set(_DEPENDENCY_FILES) | {
        "timeline_self_distillation/run_teacher_span_comparison.py",
        "timeline_self_distillation/entity_bridge.py",
    }
    _write_json(
        output / "manifest.json",
        {
            "command": [sys.executable, *sys.argv],
            "git": _git_receipt(),
            "versions": _package_versions(),
            "sources": {p: _file_receipt(ROOT / p) for p in sorted(dependencies)},
            "inputs": {"pilot": _file_receipt(args.pilot_records), "reference": _file_receipt(args.reference_records)},
            "model_path": str(MODEL),
            "gpu": args.device,
            "environment_spec_hash": "a093f958",
        },
    )
    _write_json(
        output / "protocol.json",
        {
            "conditions": CONDITIONS,
            "sample_ids": EXPECTED_SAMPLES,
            "random_draws": DRAWS,
            "greedy_draws": 1,
            "seed": SEED,
            "seed_formula": "SEED+9000000+sample_order*1000+draw",
            "total_outputs": TOTAL,
            "sampling": {"temperature": 1, "top_p": 1, "regex": BOX_REGEX, "max_tokens": 48},
            "boundary": "first three natural_boundary_offsets; cumulative original reasoning prefixes",
            "entity": "unchanged saved pilot entity; no new bridge",
            "teacher": "frozen same Qwen3.5-0.8B",
            "primary": "random80/condition mean IoU; invalid=0; no selection",
            "greedy": "diagnostic only",
            "ground_truth": "post-generation scoring only",
            "training": False,
            "optimizer": False,
            "image_prefills": 5,
            "late_parity": "85 base rows from diagnosis_decode_n5_d16_v1",
        },
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = _processor(MODEL)
    tokenizer = processor.tokenizer
    anchors = {r["sample_id"]: span_anchors(tokenizer, r) for r in rows}
    for r in rows:
        _append_jsonl(
            output / "anchors.jsonl",
            {"sample_id": r["sample_id"], "entity": r["entity"], "anchors": anchors[r["sample_id"]]},
        )
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    frozen_before = _frozen_versions(model)
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.text_config.vocab_size)
    ).compile_regex(BOX_REGEX)
    records, cache_checks = [], []
    with torch.no_grad():
        for order, row in enumerate(rows):
            sid = row["sample_id"]
            print(f"PREFILL {sid} offsets={[anchors[sid][c]['offset'] for c in CONDITIONS]}", flush=True)
            _, inputs = _render_and_process(processor, row["expression"], _load_rgb_image(row["image_path"]))
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            result = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            c0, rope = result.past_key_values, model.model.rope_deltas.clone()
            c0_snapshot = snapshot_cache(c0)
            prefixes = capture_prefixes(model, c0, row["reasoning_ids"], anchors[sid])
            suffix = QUERY.format(entity=row["entity"], question=row["expression"]) + BOX_OPEN
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            for condition in CONDITIONS:
                model.model.rope_deltas = rope.clone()
                state_cache, _ = _advance(model, fork(prefixes[condition]), suffix_ids[:-1])
                state = {"student": state_cache, "rope_deltas": rope, "last_opening_id": suffix_ids[-1]}
                snapshot = snapshot_cache(state_cache)
                for mode, n in (("greedy", 1), ("random", DRAWS)):
                    for draw in range(n):
                        seed = micro_seed(SEED, order, draw)
                        raw = _sample_bbox(model, tokenizer, grammar, state, seed, greedy=mode == "greedy")
                        record = {k: v for k, v in raw.items() if k != "supports"}
                        record.update(score_bbox(raw, row["ground_truth_bbox"]))
                        record.update(
                            sample_id=sid,
                            sample_order=order,
                            condition=condition,
                            mode=mode,
                            draw=draw,
                            seed=seed,
                            anchor_offset=anchors[sid][condition]["offset"],
                        )
                        records.append(record)
                        _append_jsonl(output / "records.jsonl", record)
                    print(f"DECODE {sid} {condition} {mode} count={n}", flush=True)
                check = {"sample_id": sid, "condition": condition, **compare_cache_snapshot(state_cache, snapshot)}
                cache_checks.append(check)
                _append_jsonl(output / "cache_checks.jsonl", check)
                if not check["unchanged"]:
                    raise RuntimeError("probe modified its source cache")
            check = {"sample_id": sid, "condition": "c0", **compare_cache_snapshot(c0, c0_snapshot)}
            cache_checks.append(check)
            _append_jsonl(output / "cache_checks.jsonl", check)
            if not check["unchanged"]:
                raise RuntimeError("reasoning replay modified original image cache")
    parity = late_parity(records, reference)
    _write_json(output / "parity.json", parity)
    frozen_after = _frozen_versions(model)
    summary = summarize(records)
    summary.update(
        elapsed_seconds=time.time() - started,
        training_performed=False,
        frozen_parameter_count=len(frozen_before),
        frozen_parameters_unchanged=frozen_before == frozen_after,
        cache_checks_unchanged=all(c["unchanged"] for c in cache_checks),
        late_parity_exact=parity["all_exact"],
    )
    _write_json(output / "summary.json", summary)
    if not parity["all_exact"] or frozen_before != frozen_after:
        raise RuntimeError("parity or frozen parameter check failed; retain all outputs for diagnosis")
    print(json.dumps({c: summary["conditions"][c]["random"]["mean_iou"] for c in CONDITIONS}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3")
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
