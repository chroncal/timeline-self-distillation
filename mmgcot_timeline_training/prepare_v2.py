"""Freeze natural MM-GCoT trajectories and the unmodified v3p5 bridge.

This entrypoint never passes labels, reference CoT, or answers to inference.
It deliberately keeps noisy, parse-valid target descriptions unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from mmgcot_diagnostic.protocol import MODEL, file_hash, stable_seed
from mmgcot_timeline_training import bridge_v3


def record_path(root: Path, sample_id: str, trajectory_index: int) -> Path:
    digest = hashlib.sha256(sample_id.encode()).hexdigest()[:20]
    return root / "records" / f"{digest}_t{trajectory_index}.json"


def write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    # Set device visibility before torch/CUDA is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    import torch
    import xgrammar as xgr
    import mmgcot_diagnostic.run as diagnostic_run

    diagnostic_run.torch = torch
    diagnostic_run.xgr = xgr
    rows = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.num_shards == args.shard_index]
    output = args.output_dir.absolute()
    protocol = {
        "schema_version": "mmgcot_formal_v2_frozen_trajectory",
        "selection": str(args.selection.absolute()),
        "selection_sha256": file_hash(args.selection),
        "model": str(Path(args.model).absolute()),
        "bridge_version": bridge_v3.BRIDGE_VERSION,
        "bridge_prompt_sha256": hashlib.sha256(bridge_v3.BRIDGE_PROMPT.encode()).hexdigest(),
        "bridge_regex_sha256": hashlib.sha256(bridge_v3.BRIDGE_REGEX.encode()).hexdigest(),
        "reasoning": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 4096},
        "trajectories_per_image": args.trajectories,
        "model_input_allowlist": ["image_path", "question"],
        "labels_used_by_inference": False,
        "semantic_review_policy": "report_only",
    }
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    try:
        write_once(protocol_path, protocol)
    except FileExistsError:
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("existing frozen trajectory protocol differs")

    torch.manual_seed(20260921)
    torch.cuda.manual_seed_all(20260921)
    engine = diagnostic_run.Engine(args.model)
    bridge_grammar = engine.compiler.compile_regex(bridge_v3.BRIDGE_REGEX)
    start = time.time()
    complete = skipped = 0
    for _, row in selected:
        if file_hash(row["image_path"]) != row["image_sha256"]:
            raise RuntimeError(f"image bytes changed: {row['sample_id']}")
        for ti in range(args.trajectories):
            target = record_path(output, row["sample_id"], ti)
            if target.exists():
                if not args.resume:
                    raise FileExistsError(target)
                saved = json.loads(target.read_text())
                if saved["sample_id"] != row["sample_id"] or saved["trajectory_index"] != ti:
                    raise RuntimeError(f"resume record mismatch: {target}")
                skipped += 1
                continue
            with torch.no_grad():
                c0, logits0, prompt_ids, rendered = engine.prefill({
                    "image_path": row["image_path"], "question": row["question"]})
                seed = stable_seed(row["sample_id"], ti, "reasoning")
                cT, _, reasoning_ids, logprobs, finish = engine.rollout(c0, logits0, seed)
                bridge_input = engine.prepare(cT, bridge_v3.BRIDGE_PROMPT)
                bridge_draw = engine.decode(
                    bridge_input, stable_seed(row["sample_id"], ti, "bridge_v2"),
                    greedy=True, grammar=bridge_grammar, limit=160)
                parsed = bridge_v3.parse_bridge(
                    bridge_draw["text"], completed=bridge_draw["completed"])
            result = {
                "sample_id": row["sample_id"], "image_id": row["image_id"],
                "split": row["split"], "task_type": row["task_type"],
                "image_path": row["image_path"], "image_sha256": row["image_sha256"],
                "question": row["question"], "trajectory_index": ti,
                "prompt_token_ids": prompt_ids, "rendered_prompt": rendered,
                "reasoning_token_ids": reasoning_ids,
                "reasoning_token_ids_sha256": bridge_v3.token_ids_sha256(reasoning_ids),
                "reasoning_logprobs": logprobs, "reasoning_finish": finish,
                "reasoning_seed": seed, "bridge_version": bridge_v3.BRIDGE_VERSION,
                "bridge_generation": bridge_draw, "bridge_seed": stable_seed(row["sample_id"], ti, "bridge_v2"),
                **parsed,
            }
            write_once(target, result)
            complete += 1
            print(f"FROZEN {args.shard_index} {complete}/{len(selected)*args.trajectories} "
                  f"{row['sample_id']} t{ti} {finish} {parsed['bridge_parse_status']}", flush=True)
    engine.unchanged()
    write_once(output / f"shard{args.shard_index}.receipt.json", {
        "selection_sha256": file_hash(args.selection),
        "completed": complete, "skipped_existing": skipped,
        "seconds": time.time()-start, "parameters_unchanged": True,
        "command": [sys.executable, *sys.argv],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.trajectories < 1:
        parser.error("invalid shard or trajectory count")
    return args


if __name__ == "__main__":
    run(parse_args())
