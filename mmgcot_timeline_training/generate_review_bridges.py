"""Generate frozen reasoning/entity bridges for the preselected blind review.

This entrypoint never generates L/E/R boxes and never reads IoU or training
results.  Each shard writes immutable per-sample records; ``--merge`` creates
the reviewer-facing packet after every selected sample has a record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

from mmgcot_diagnostic.protocol import ENTITY_PROMPT, ENTITY_REGEX, MODEL, file_hash, stable_seed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
        handle.write("\n")


def _record_path(output: Path, sample_id: str) -> Path:
    return output / "records" / (hashlib.sha256(sample_id.encode()).hexdigest()[:20] + ".json")


def _protocol(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "mmgcot_timeline_blind_review_bridges_v1",
        "selection": str(args.selection.absolute()),
        "selection_sha256": file_hash(args.selection),
        "model": str(Path(args.model).absolute()),
        "samples": len(rows),
        "reasoning": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 4096},
        "entity_prompt": ENTITY_PROMPT,
        "result_blind": True,
        "forbidden_inputs": ["R/E predictions", "IoU", "teacher probabilities", "training arm results"],
    }


def run_shard(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import xgrammar as xgr
    import mmgcot_diagnostic.run as diagnostic_run

    diagnostic_run.torch = torch
    diagnostic_run.xgr = xgr
    rows = _read_jsonl(args.selection)
    work_rows = _read_jsonl(args.work_selection) if args.work_selection else rows
    frozen_by_id = {row["sample_id"]: row for row in rows}
    for row in work_rows:
        if row.get("sample_id") not in frozen_by_id or row != frozen_by_id[row["sample_id"]]:
            raise RuntimeError("work selection is not an exact subset of the frozen selection")
    output = args.output_dir.absolute()
    (output / "records").mkdir(parents=True, exist_ok=True)
    protocol = _protocol(args, rows)
    protocol_path = output / "protocol.json"
    try:
        _write_exclusive(protocol_path, protocol)
    except FileExistsError:
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("existing blind-review protocol differs")
    selected = [row for index, row in enumerate(work_rows) if index % args.num_shards == args.shard_index]
    receipt_name = args.receipt_label or f"shard{args.shard_index}"
    receipt_path = output / f"{receipt_name}.receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)

    torch.manual_seed(20260921)
    torch.cuda.manual_seed_all(20260921)
    engine = diagnostic_run.Engine(args.model)
    start = time.time()
    completed = 0
    skipped = 0
    for row in selected:
        if file_hash(row["image_path"]) != row["image_sha256"]:
            raise RuntimeError(f"image hash changed: {row['sample_id']}")
        target = _record_path(output, row["sample_id"])
        if target.exists():
            if not args.resume:
                raise FileExistsError(target)
            existing = json.loads(target.read_text())
            if existing.get("sample_id") != row["sample_id"]:
                raise RuntimeError(f"resume record mismatch: {target}")
            skipped += 1
            print(f"SKIP {skipped} {row['sample_id']}", flush=True)
            continue
        inputs = {"image_path": row["image_path"], "question": row["question"]}
        with torch.no_grad():
            c0, logits0, prompt_ids, rendered = engine.prefill(inputs)
            cT, _, reasoning_ids, reasoning_logprobs, finish = engine.rollout(
                c0, logits0, stable_seed(row["sample_id"], 0, "reasoning")
            )
            entity_text = ""
            entity_status = "reasoning_" + finish
            entity_generation = None
            if finish == "stop":
                entity_generation = engine.decode(
                    engine.prepare(cT, ENTITY_PROMPT),
                    stable_seed(row["sample_id"], 0, "entity"),
                    greedy=True,
                    grammar=engine.entity_grammar,
                    limit=96,
                )
                valid = entity_generation["completed"] and re.fullmatch(
                    ENTITY_REGEX, entity_generation["text"]
                ) is not None
                entity_text = entity_generation["text"][:-1].strip() if valid else ""
                entity_status = (
                    "invalid" if not entity_text
                    else "unresolved" if entity_text.casefold() == "unresolved"
                    else "usable"
                )
        record = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "image_id": row["image_id"],
            "task_type": row["task_type"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "question": row["question"],
            "ground_truth_bbox": row["ground_truth_bbox"],
            "prompt_token_ids": prompt_ids,
            "rendered_prompt": rendered,
            "reasoning_token_ids": reasoning_ids,
            "reasoning_logprobs": reasoning_logprobs,
            "reasoning_text": engine.tokenizer.decode(reasoning_ids, skip_special_tokens=False),
            "reasoning_finish": finish,
            "target_description": entity_text,
            "target_description_status": entity_status,
            "entity_generation": entity_generation,
            "reasoning_seed": stable_seed(row["sample_id"], 0, "reasoning"),
            "entity_seed": stable_seed(row["sample_id"], 0, "entity"),
        }
        _write_exclusive(target, record)
        completed += 1
        print(f"BRIDGE {completed}/{len(selected)} {row['sample_id']} {entity_status}", flush=True)
    engine.unchanged()
    _write_exclusive(receipt_path, {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_sample_ids": [row["sample_id"] for row in selected],
        "completed": completed,
        "skipped_existing": skipped,
        "seconds": time.time() - start,
        "parameters_unchanged": True,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "command": [sys.executable, *sys.argv],
        "work_selection": str(args.work_selection.absolute()) if args.work_selection else None,
        "work_selection_sha256": file_hash(args.work_selection) if args.work_selection else None,
    })


def merge(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.selection)
    output = args.output_dir.absolute()
    records = []
    for row in rows:
        path = _record_path(output, row["sample_id"])
        if not path.is_file():
            raise RuntimeError(f"missing bridge record: {row['sample_id']}")
        record = json.loads(path.read_text())
        if record["sample_id"] != row["sample_id"]:
            raise RuntimeError("bridge record/sample mismatch")
        records.append(record)
    packet_path = output / "blind_review_packet.jsonl"
    if packet_path.exists():
        raise FileExistsError(packet_path)
    forbidden = {"iou", "bbox_prediction", "teacher_probability", "training_arm"}
    with packet_path.open("x", encoding="utf-8") as handle:
        for record in records:
            packet = {
                key: record[key] for key in (
                    "sample_id", "split", "image_id", "task_type", "image_path",
                    "image_sha256", "question", "ground_truth_bbox", "target_description",
                    "target_description_status",
                )
            }
            packet.update(
                review_label=None,
                allowed_review_labels=[
                    "same_target", "different_target", "description_not_unique", "cannot_determine"
                ],
            )
            if forbidden & set(packet):
                raise RuntimeError("review packet leaked a forbidden result field")
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    _write_exclusive(output / "merge_manifest.json", {
        "records": len(records),
        "status_counts": dict(sorted(__import__("collections").Counter(
            record["target_description_status"] for record in records
        ).items())),
        "packet": str(packet_path),
        "packet_sha256": file_hash(packet_path),
        "selection_sha256": file_hash(args.selection),
        "result_fields_excluded": sorted(forbidden),
    })
    print(packet_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--work-selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--receipt-label")
    args = parser.parse_args(argv)
    if args.merge:
        return args
    if args.device is None or args.shard_index is None or args.num_shards is None:
        parser.error("generation requires --device, --shard-index, and --num-shards")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    merge(parsed) if parsed.merge else run_shard(parsed)
