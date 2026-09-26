"""Replay frozen reasoning and run a one-trajectory L0/L/E/R bridge-v2 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from mmgcot_diagnostic.protocol import bbox_suffix, comma_prefix, early_offset, file_hash, stable_seed
from mmgcot_timeline_training import bridge_v2, bridge_v3, bridge_v4, bridge_v5, bridge_v6, bridge_v7, bridge_v8, bridge_v9, bridge_v10
from mmgcot_timeline_training.generate_review_bridges import _record_path


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bridge_contract(version: str):
    return {"v2": bridge_v2, "v3": bridge_v3, "v4": bridge_v4, "v5": bridge_v5,
            "v6": bridge_v6, "v7": bridge_v7, "v8": bridge_v8,
            "v9": bridge_v9, "v10": bridge_v10}[version]


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import xgrammar as xgr
    import mmgcot_diagnostic.run as diagnostic_run

    diagnostic_run.torch = torch
    diagnostic_run.xgr = xgr
    rows = _read(args.manifest)
    contract = _bridge_contract(args.bridge_version)
    bridge_protocol = json.loads((args.bridge_root / "protocol.json").read_text())
    if bridge_protocol.get("schema_version") != contract.BRIDGE_VERSION:
        raise RuntimeError("bridge root protocol differs from --bridge-version")
    selected = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    output = args.output_dir.absolute()
    (output / "records").mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": f"mmgcot_bridge_{args.bridge_version}_geometry_diagnostic_v1",
        "bridge_version": contract.BRIDGE_VERSION,
        "manifest": str(args.manifest.absolute()),
        "manifest_sha256": file_hash(args.manifest),
        "bridge_root": str(args.bridge_root.absolute()),
        "bridge_protocol_sha256": file_hash(args.bridge_root / "protocol.json"),
        "trajectory_index": 0,
        "early_fraction": 0.25,
        "A": {"arms": ["L0", "L", "E", "R"], "greedy": 1, "random": 4},
        "B": {"source": "L/random/draw0", "prefix_coordinates": [1, 2],
              "arms": ["E", "L", "R"], "random": 4},
        "task_answer_used_by_bbox": False,
        "gt_used_only_for_evaluation": True,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("existing geometry protocol differs")
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")

    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    engine = diagnostic_run.Engine(args.model)
    started = time.time()
    completed = skipped = 0
    for row in selected:
        stem = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:20]
        journal = output / "records" / f"{stem}.jsonl"
        if journal.exists():
            if not args.resume:
                raise FileExistsError(journal)
            lines = _read(journal)
            if lines and lines[-1].get("type") == "trajectory_done":
                skipped += 1
                continue
            raise RuntimeError(f"refusing partial journal resume: {journal}")
        bridge_path = _record_path(args.bridge_root, row["sample_id"])
        bridge = json.loads(bridge_path.read_text())
        for key in ("sample_id", "image_id", "question", "image_sha256"):
            if str(bridge[key]) != str(row[key]):
                raise RuntimeError(f"bridge/manifest mismatch on {key}")
        if bridge["reasoning_token_ids_sha256"] != contract.token_ids_sha256(bridge["reasoning_token_ids"]):
            raise RuntimeError("bridge reasoning hash mismatch")
        base = {
            "sample_id": row["sample_id"], "image_id": row["image_id"],
            "trajectory_index": 0, "task_type": row["task_type"],
        }
        c0, _, prompt_ids, _ = engine.prefill(
            {"image_path": row["image_path"], "question": row["question"]}
        )
        if prompt_ids != bridge["prompt_token_ids"]:
            raise RuntimeError("prompt token IDs changed")
        rids = [int(value) for value in bridge["reasoning_token_ids"]]
        late, _ = engine.replay(c0, rids)
        boundaries = diagnostic_run.sentence_offsets(engine.tokenizer, rids)
        k = early_offset(len(rids), boundaries)
        trajectory = {
            **base, "type": "trajectory", "seed": bridge["reasoning_seed"],
            "prompt_token_ids": prompt_ids, "reasoning_ids": rids,
            "reasoning_token_ids_sha256": contract.token_ids_sha256(rids),
            "reasoning_length": len(rids), "finish_reason": bridge["reasoning_finish"],
            "early_offset": k, "entity": bridge["target_entity_reference"],
            "entity_status": bridge["target_entity_reference_status"],
            "task_answer_audit_only": bridge.get("task_answer"),
            "bridge_record_sha256": file_hash(bridge_path),
        }
        _append(journal, trajectory)
        target = bridge["target_entity_reference"]
        if bridge["bridge_parse_status"] != "valid" or not target or target.casefold() == "unresolved":
            _append(journal, {**base, "type": "failure", "reason": "bridge_target_unavailable"})
            _append(journal, {**base, "type": "trajectory_done", "inference_complete": False})
            continue
        early, _ = engine.replay(c0, rids[:k])
        states = {"L0": late, "L": late, "E": early, "R": c0}
        bbox_with_target = (
            contract.bbox_suffix_v4 if args.bridge_version == "v4"
            else contract.bbox_suffix_v5 if args.bridge_version == "v5"
            else contract.bbox_suffix_v6 if args.bridge_version == "v6"
            else contract.bbox_suffix_v7 if args.bridge_version == "v7"
            else contract.bbox_suffix_v8 if args.bridge_version == "v8"
            else contract.bbox_suffix_v9 if args.bridge_version == "v9"
            else contract.bbox_suffix_v10 if args.bridge_version == "v10"
            else contract.bbox_suffix_v2
        )
        prepared = {
            "L0": engine.prepare(late, bbox_suffix(row["question"], None)),
            "L": engine.prepare(late, bbox_with_target(row["question"], target)),
            "E": engine.prepare(early, bbox_with_target(row["question"], target)),
            "R": engine.prepare(c0, bbox_with_target(row["question"], target)),
        }
        late_sequences = []
        for arm in ("L0", "L", "E", "R"):
            for mode, draws in (("greedy", 1), ("random", 4)):
                for draw in range(draws):
                    result = engine.decode(
                        prepared[arm], stable_seed(row["sample_id"], 0, "bridge_v2_A", mode, draw),
                        greedy=mode == "greedy",
                    )
                    record = diagnostic_run.bbox_record(
                        base, result, stage="A", arm=arm, mode=mode, draw=draw,
                        gt=row["ground_truth_bbox"],
                    )
                    _append(journal, record)
                    if arm == "L" and mode == "random":
                        late_sequences.append(record)
        source = next(record for record in late_sequences if record["draw"] == 0)
        for count in (1, 2):
            prefix = comma_prefix(engine.tokenizer, source["token_ids"], count)
            if prefix is None:
                _append(journal, {**base, "type": "prefix_unavailable", "prefix_coordinates": count,
                                  "source_token_ids": source["token_ids"],
                                  "reason": "no_exact_original_token_boundary"})
                continue
            for arm in ("E", "L", "R"):
                for draw in range(4):
                    result = engine.decode(
                        prepared[arm],
                        stable_seed(row["sample_id"], 0, "bridge_v2_B", count, draw),
                        prefix=prefix,
                    )
                    _append(journal, diagnostic_run.bbox_record(
                        base, result, stage="B", arm=arm, mode="random", draw=draw,
                        gt=row["ground_truth_bbox"], prefix_coordinates=count,
                    ))
        engine.unchanged()
        _append(journal, {**base, "type": "trajectory_done", "inference_complete": True,
                          "parity": {"prompt_ids_exact": True, "reasoning_ids_exact": True,
                                     "parameters_unchanged": True}})
        completed += 1
        print(f"GEOMETRY {completed}/{len(selected)} {row['sample_id']}", flush=True)
    receipt = output / f"shard{args.shard_index}.receipt.json"
    receipt.write_text(json.dumps({
        "completed": completed, "skipped": skipped, "seconds": time.time() - started,
        "parameters_unchanged": True, "command": [sys.executable, *sys.argv],
    }, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bridge-version", choices=("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"), default="v2")
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="/mnt/sda/sujingyang/models/Qwen3.5-0.8B")
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")
    run(args)


if __name__ == "__main__":
    main()
