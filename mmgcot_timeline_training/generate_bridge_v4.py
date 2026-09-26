"""Generate the frozen three-stage MM-GCoT target bridge v4."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from mmgcot_diagnostic.protocol import MODEL, file_hash, stable_seed
from mmgcot_timeline_training import bridge_v4
from mmgcot_timeline_training.generate_review_bridges import _record_path


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
        handle.write("\n")


def _generation_error(generation: dict[str, Any], parse_status: str) -> str:
    if not generation["completed"]:
        return "generation_incomplete"
    if parse_status != "valid":
        return "grammar_mismatch"
    return "none"


def _protocol(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": bridge_v4.BRIDGE_VERSION,
        "selection": str(args.selection.absolute()),
        "selection_sha256": file_hash(args.selection),
        "model": str(Path(args.model).absolute()),
        "samples": len(rows),
        "reasoning_source": "frozen_replay",
        "frozen_reasoning_root": str(args.frozen_reasoning_root.absolute()),
        "stages": {
            "frame": {"context": "c0", "prompt": bridge_v4.FRAME_PROMPT, "regex": bridge_v4.FRAME_REGEX},
            "candidate": {"context": "cT", "prompt_template": "candidate_prompt", "regex": bridge_v4.ENTITY_REGEX},
            "verify": {"context": "independent_c0_fork", "prompt_template": "verify_prompt", "regex": bridge_v4.VERIFY_REGEX},
        },
        "decode": {"greedy": True, "max_tokens_per_stage": 96},
        "model_input_allowlist": ["image_path", "question", "frozen reasoning token IDs", "prior stage text"],
        "gt_used_for_inference": False,
        "task_answer_generated_or_used": False,
        "student_choice_repair_allowed": False,
        "sealed_independent_confirmation_accessed": False,
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
    rows = _read(args.selection)
    work_rows = _read(args.work_selection) if args.work_selection else rows
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
            raise RuntimeError("existing bridge-v4 protocol differs")

    selected = [row for index, row in enumerate(work_rows) if index % args.num_shards == args.shard_index]
    receipt_label = args.receipt_label or f"shard{args.shard_index}"
    receipt = output / f"{receipt_label}.receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)

    torch.manual_seed(20260923)
    torch.cuda.manual_seed_all(20260923)
    engine = diagnostic_run.Engine(args.model)
    frame_grammar = engine.compiler.compile_regex(bridge_v4.FRAME_REGEX)
    entity_grammar = engine.compiler.compile_regex(bridge_v4.ENTITY_REGEX)
    verify_grammar = engine.compiler.compile_regex(bridge_v4.VERIFY_REGEX)
    started = time.time()
    completed = skipped = 0
    for row in selected:
        if file_hash(row["image_path"]) != row["image_sha256"]:
            raise RuntimeError(f"image hash changed: {row['sample_id']}")
        target_path = _record_path(output, row["sample_id"])
        if target_path.exists():
            if not args.resume:
                raise FileExistsError(target_path)
            skipped += 1
            continue

        source_path = _record_path(args.frozen_reasoning_root, row["sample_id"])
        if not source_path.is_file():
            raise FileNotFoundError(f"missing frozen reasoning source: {source_path}")
        source = json.loads(source_path.read_text())
        for key in ("sample_id", "image_id", "question", "image_sha256"):
            if str(source[key]) != str(row[key]):
                raise RuntimeError(f"frozen source differs on {key}: {row['sample_id']}")

        c0, _, prompt_ids, rendered = engine.prefill(
            {"image_path": row["image_path"], "question": row["question"]}
        )
        if prompt_ids != source["prompt_token_ids"]:
            raise RuntimeError(f"prompt token IDs changed: {row['sample_id']}")
        reasoning_ids = [int(value) for value in source["reasoning_token_ids"]]
        cT, _ = engine.replay(c0, reasoning_ids)

        frame_generation = engine.decode(
            engine.prepare(c0, bridge_v4.FRAME_PROMPT),
            stable_seed(row["sample_id"], 0, "bridge_v4_frame"),
            greedy=True, grammar=frame_grammar, limit=96,
        )
        frame = bridge_v4.parse_frame(
            frame_generation["text"], completed=frame_generation["completed"]
        )
        candidate_generation = engine.decode(
            engine.prepare(cT, bridge_v4.candidate_prompt(
                frame.get("target_source", "unresolved") or "unresolved",
                frame.get("query_entity", "NONE") or "NONE",
            )),
            stable_seed(row["sample_id"], 0, "bridge_v4_candidate"),
            greedy=True, grammar=entity_grammar, limit=96,
        )
        candidate = bridge_v4.parse_candidate(
            candidate_generation["text"], completed=candidate_generation["completed"]
        )
        verify_generation = engine.decode(
            engine.prepare(c0, bridge_v4.verify_prompt(
                frame.get("target_source", "unresolved") or "unresolved",
                frame.get("query_entity", "NONE") or "NONE",
                candidate.get("candidate_entity", "UNRESOLVED") or "UNRESOLVED",
            )),
            stable_seed(row["sample_id"], 0, "bridge_v4_verify"),
            greedy=True, grammar=verify_grammar, limit=96,
        )
        verified = bridge_v4.parse_verified(
            verify_generation["text"], completed=verify_generation["completed"]
        )
        parsed = bridge_v4.combine(frame, candidate, verified)
        stage_errors = {
            "frame": _generation_error(frame_generation, frame["parse_status"]),
            "candidate": _generation_error(candidate_generation, candidate["parse_status"]),
            "verify": _generation_error(verify_generation, verified["parse_status"]),
        }
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
            "reasoning_token_ids_sha256": bridge_v4.token_ids_sha256(reasoning_ids),
            "reasoning_logprobs": source.get("reasoning_logprobs"),
            "reasoning_text": engine.tokenizer.decode(reasoning_ids, skip_special_tokens=False),
            "reasoning_finish": source["reasoning_finish"],
            "reasoning_origin": "frozen_replay",
            "reasoning_seed": source.get("reasoning_seed"),
            "frozen_source_record": str(source_path.absolute()),
            "frozen_source_record_sha256": file_hash(source_path),
            "bridge_version": bridge_v4.BRIDGE_VERSION,
            "frame_generation": frame_generation,
            "candidate_generation": candidate_generation,
            "verify_generation": verify_generation,
            "stage_serialization_errors": stage_errors,
            "extractor_serialization_error": (
                "none" if all(value == "none" for value in stage_errors.values()) else "stage_failure"
            ),
            **parsed,
        }
        _write_exclusive(target_path, record)
        completed += 1
        print(
            f"BRIDGE_V4 {completed}/{len(selected)} {row['sample_id']} "
            f"{parsed['bridge_parse_status']} {parsed['target_entity_reference']}",
            flush=True,
        )

    engine.unchanged()
    _write_exclusive(receipt, {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "completed": completed,
        "skipped_existing": skipped,
        "seconds": time.time() - started,
        "parameters_unchanged": True,
        "command": [sys.executable, *sys.argv],
        "work_selection": str(args.work_selection.absolute()) if args.work_selection else None,
        "work_selection_sha256": file_hash(args.work_selection) if args.work_selection else None,
    })


def merge(args: argparse.Namespace) -> None:
    rows = _read(args.selection)
    output = args.output_dir.absolute()
    protocol_path = output / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != bridge_v4.BRIDGE_VERSION:
        raise RuntimeError("output protocol is not bridge v4")
    if protocol.get("selection_sha256") != file_hash(args.selection):
        raise RuntimeError("output protocol selection differs from merge selection")
    packet_path = output / "semantic_review_packet.jsonl"
    if packet_path.exists():
        raise FileExistsError(packet_path)

    records = []
    for row in rows:
        record_path = _record_path(output, row["sample_id"])
        if not record_path.is_file():
            raise RuntimeError(f"missing bridge record: {row['sample_id']}")
        record = json.loads(record_path.read_text())
        if record.get("bridge_version") != bridge_v4.BRIDGE_VERSION:
            raise RuntimeError(f"record bridge version differs: {row['sample_id']}")
        records.append(record)
    with packet_path.open("x", encoding="utf-8") as handle:
        for record in records:
            packet = {key: record[key] for key in (
                "sample_id", "split", "image_id", "task_type", "image_path",
                "image_sha256", "question", "ground_truth_bbox", "reasoning_text",
                "reasoning_finish", "reasoning_token_ids_sha256", "target_source",
                "query_entity", "candidate_entity", "target_entity_reference",
                "target_entity_reference_status", "frame_parse_status",
                "candidate_parse_status", "verify_parse_status", "bridge_parse_status",
                "extractor_serialization_error",
            )}
            packet.update(
                student_target_selection_error=None,
                extractor_content_error=None,
                target_reference_review_label=None,
                student_target_selection_error_labels=list(bridge_v4.STUDENT_SELECTION_LABELS),
                extractor_content_error_labels=list(bridge_v4.EXTRACTOR_CONTENT_ERROR_LABELS),
                target_reference_review_labels=list(bridge_v4.TARGET_REFERENCE_REVIEW_LABELS),
            )
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    _write_exclusive(output / "merge_manifest.json", {
        "records": len(records),
        "parse_status_counts": dict(sorted(Counter(record["bridge_parse_status"] for record in records).items())),
        "packet": str(packet_path),
        "packet_sha256": file_hash(packet_path),
        "selection_sha256": file_hash(args.selection),
        "bridge_version": bridge_v4.BRIDGE_VERSION,
        "gt_used_for_inference": False,
        "task_answer_generated_or_used": False,
        "sealed_independent_confirmation_accessed": False,
    })
    print(packet_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--frozen-reasoning-root", type=Path, required=True)
    parser.add_argument("--work-selection", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--receipt-label")
    args = parser.parse_args(argv)
    if not args.merge and (args.device is None or args.shard_index is None or args.num_shards is None):
        parser.error("generation requires device, shard-index, and num-shards")
    if not args.merge and not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    merge(parsed) if parsed.merge else run_shard(parsed)
