"""Read-only residual-strength diagnostic of existing adapters, not retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean

if "--device" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--device") + 1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import xgrammar as xgr
from transformers import Qwen3_5ForConditionalGeneration

from live_kv_probe_prototype.run_hf_fork import _append_jsonl, _git_receipt, _write_json
from reasoning_checkpoints.run_pilot import _processor
from timeline_self_distillation.run_opd_micro import build_states, sample_bbox
from timeline_self_distillation.run_teacher_pilot import BOX_REGEX, DEFAULT_SAMPLE_IDS, MODEL, SEED
from timeline_self_distillation.terminal_adapter import install_terminal_query_lora, terminal_parameter_whitelist
from verl.experimental.routed_grounding.router import xyxy_iou


def run(args):
    root = Path("outputs/research_experiments/timeline_opd")
    paths = {name: root / f"opd_span1_{name}_n5_s20_v1" for name in ("reverse", "forward")}
    source = root / "teacher_pilot_n5_v1/records.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert tuple(r["sample_id"] for r in rows) == tuple(DEFAULT_SAMPLE_IDS)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scales = (0.0, 0.25, 0.5, 1.0)
    _write_json(
        args.output_dir / "protocol.json",
        {
            "hypothesis": "if excessive learned residual drives harm, shrinking it may recover IoU",
            "intervention": "loaded final adapter B *= scale, A unchanged; no optimization",
            "scales": scales,
            "primary": "mean IoU on all draws, invalid=0",
            "scope": "five seen images; exploratory amplitude diagnostic, not LR tuning or new training",
            "draw_seed": "SEED + 9000000 + index*1000 + draw; four draws",
        },
    )
    inputs = [
        source,
        Path(__file__),
        Path("timeline_self_distillation/run_opd_micro.py"),
        Path("timeline_self_distillation/terminal_adapter.py"),
        Path("live_kv_probe_prototype/run_hf_fork.py"),
    ]
    inputs += [p / "final_adapter.pt" for p in paths.values()]
    _write_json(
        args.output_dir / "manifest.json",
        {
            "git": _git_receipt(),
            "command": [sys.executable, *sys.argv],
            "model": str(MODEL),
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
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
    parameters = terminal_parameter_whitelist(model, adapter)
    adapter.enabled = False
    states = build_states(model, processor, rows, "late_entity")
    versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    grammar = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=model.config.text_config.vocab_size)
    ).compile_regex(BOX_REGEX)
    summary = {}
    with torch.no_grad():
        for name, path in paths.items():
            weights = torch.load(path / "final_adapter.pt", map_location="cuda", weights_only=True)
            if set(weights) != set(parameters):
                raise RuntimeError("checkpoint keys mismatch")
            reference_rows = [json.loads(line) for line in (path / "eval.jsonl").read_text().splitlines()]
            reference = {(r["step"], r["sample_id"], r["seed"]): r for r in reference_rows}
            for scale in scales:
                for key, parameter in parameters.items():
                    parameter.copy_(weights[key] * (scale if key == "q_lora_up.weight" else 1.0))
                adapter.enabled = True
                results = []
                for index, state in enumerate(states):
                    row = state["row"]
                    for draw in range(4):
                        seed = SEED + 9_000_000 + index * 1000 + draw
                        prediction = sample_bbox(model, processor.tokenizer, grammar, state, seed)
                        record = {k: v for k, v in prediction.items() if k != "supports"}
                        iou = (
                            float(xyxy_iou(prediction["bbox"], row["ground_truth_bbox"]))
                            if prediction["parse_valid"]
                            else 0.0
                        )
                        record.update(adapter=name, scale=scale, sample_id=row["sample_id"], draw=draw, iou=iou)
                        if scale in (0.0, 1.0):
                            expected = reference[(0 if scale == 0 else 20, row["sample_id"], seed)]
                            record["endpoint_ids_equal"] = prediction["ids"] == expected["ids"]
                            if not record["endpoint_ids_equal"] or iou != expected["iou"]:
                                raise RuntimeError("endpoint does not reproduce original run")
                        results.append(record)
                        _append_jsonl(args.output_dir / "records.jsonl", record)
                key = f"{name}_scale_{scale:g}"
                summary[key] = {
                    "mean_iou": mean(r["iou"] for r in results),
                    "acc_05": mean(r["iou"] >= 0.5 for r in results),
                    "valid": sum(r["parse_valid"] for r in results),
                    "count": len(results),
                    "per_sample": {
                        row["sample_id"]: mean(r["iou"] for r in results if r["sample_id"] == row["sample_id"])
                        for row in rows
                    },
                }
                print("[DIAG-SCALE] " + json.dumps({key: summary[key]}), flush=True)
    if not all(p._version == versions[n] for n, p in model.named_parameters() if not p.requires_grad):
        raise RuntimeError("frozen backbone changed")
    summary["backbone_unchanged"] = True
    _write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
