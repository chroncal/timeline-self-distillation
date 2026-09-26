"""Generate target bridge v2 from frozen or newly sampled natural reasoning.

GT boxes are copied only to the reviewer packet after inference.  They are not
passed to model input, prompt construction, replay, or bridge decoding.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

from mmgcot_diagnostic.protocol import MODEL, file_hash, stable_seed
from mmgcot_timeline_training import bridge_v2, bridge_v3, bridge_v5, bridge_v6, bridge_v7, bridge_v8, bridge_v9, bridge_v10
from mmgcot_timeline_training.generate_review_bridges import _record_path


def _bridge_contract(version: str):
    return {"v2": bridge_v2, "v3": bridge_v3, "v5": bridge_v5, "v6": bridge_v6,
            "v7": bridge_v7, "v8": bridge_v8, "v9": bridge_v9,
            "v10": bridge_v10}[version]


def _bridge_seed(sample_id: str) -> int:
    """Keep the bridge draw identical across prompt-only versions."""
    return stable_seed(sample_id, 0, "bridge_v2")


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_exclusive(path: Path, value: Any) -> None:
    # Multiple GPU shards can create the same protocol concurrently.  Write a
    # complete temporary file, then link it exclusively so readers never see
    # an empty or partially serialized protocol.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_record(root: Path | None, sample_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if root is None:
        return None, None
    path = _record_path(root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen reasoning source: {path}")
    return json.loads(path.read_text()), file_hash(path)


def _protocol(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    contract = _bridge_contract(args.bridge_version)
    return {
        "schema_version": contract.BRIDGE_VERSION,
        "selection": str(args.selection.absolute()),
        "selection_sha256": file_hash(args.selection),
        "model": str(Path(args.model).absolute()),
        "samples": len(rows),
        "reasoning_source": "frozen_v1_replay" if args.frozen_reasoning_root else "new_natural_rollout",
        "frozen_reasoning_root": (
            str(args.frozen_reasoning_root.absolute()) if args.frozen_reasoning_root else None
        ),
        "new_reasoning": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 4096},
        "bridge_prompt": contract.BRIDGE_PROMPT,
        "bridge_regex": contract.BRIDGE_REGEX,
        "model_input_allowlist": ["image_path", "question", "frozen reasoning token IDs"],
        "gt_used_for_inference": False,
        "student_choice_repair_allowed": False,
        "bridge_seed_tag": "bridge_v2",
        "sealed_independent_confirmation_accessed": False,
        "candidate_roots": ([str(path.absolute()) for path in args.candidate_root]
                            if args.bridge_version == "v8" else []),
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
    contract = _bridge_contract(args.bridge_version)
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
            raise RuntimeError("existing bridge-v2 protocol differs")
    selected = [row for index, row in enumerate(work_rows) if index % args.num_shards == args.shard_index]
    receipt_label = args.receipt_label or f"shard{args.shard_index}"
    receipt = output / f"{receipt_label}.receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)

    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    engine = diagnostic_run.Engine(args.model)
    bridge_grammar = engine.compiler.compile_regex(contract.BRIDGE_REGEX)
    started = time.time()
    completed = skipped = 0
    for row in selected:
        if file_hash(row["image_path"]) != row["image_sha256"]:
            raise RuntimeError(f"image hash changed: {row['sample_id']}")
        target = _record_path(output, row["sample_id"])
        if target.exists():
            if not args.resume:
                raise FileExistsError(target)
            skipped += 1
            continue
        c0, logits0, prompt_ids, rendered = engine.prefill(
            {"image_path": row["image_path"], "question": row["question"]}
        )
        source, source_hash = _source_record(args.frozen_reasoning_root, row["sample_id"])
        if source is not None:
            for key in ("sample_id", "image_id", "question", "image_sha256"):
                if str(source[key]) != str(row[key]):
                    raise RuntimeError(f"frozen source differs on {key}: {row['sample_id']}")
            if prompt_ids != source["prompt_token_ids"]:
                raise RuntimeError(f"prompt token IDs changed: {row['sample_id']}")
            reasoning_ids = [int(value) for value in source["reasoning_token_ids"]]
            reasoning_logprobs = source["reasoning_logprobs"]
            reasoning_finish = source["reasoning_finish"]
            cT, _ = engine.replay(c0, reasoning_ids)
            reasoning_origin = "frozen_v1_replay"
            reasoning_seed = source["reasoning_seed"]
        else:
            cT, _, reasoning_ids, reasoning_logprobs, reasoning_finish = engine.rollout(
                c0, logits0, stable_seed(row["sample_id"], 0, "reasoning")
            )
            reasoning_origin = "new_natural_rollout"
            reasoning_seed = stable_seed(row["sample_id"], 0, "reasoning")
        candidate = None
        if args.bridge_version == "v8":
            matches = [(path, _record_path(path, row["sample_id"])) for path in args.candidate_root]
            matches = [(path, item) for path, item in matches if item.is_file()]
            if len(matches) != 1:
                raise RuntimeError(f"expected exactly one v3p5 candidate: {row['sample_id']}")
            candidate_path = matches[0][1]
            candidate = json.loads(candidate_path.read_text())
            for key in ("sample_id", "image_id", "question", "image_sha256"):
                if str(candidate[key]) != str(row[key]):
                    raise RuntimeError(f"candidate differs on {key}: {row['sample_id']}")
            if candidate["reasoning_token_ids_sha256"] != contract.token_ids_sha256(reasoning_ids):
                raise RuntimeError(f"candidate reasoning differs: {row['sample_id']}")
            if candidate["bridge_parse_status"] != "valid":
                raise RuntimeError(f"invalid v3p5 candidate: {row['sample_id']}")
            prompt = contract.bridge_prompt(row["question"],
                                            candidate["target_entity_reference"],
                                            candidate["task_answer"])
        else:
            prompt = contract.BRIDGE_PROMPT
        audit = None
        if args.bridge_version == "v9":
            prompt = contract.analysis_prompt(row["question"])
            audit_state = engine.prepare(cT, prompt)
            audit_cache, _, audit_ids, audit_logprobs, audit_finish = engine.rollout(
                audit_state[0], audit_state[1],
                stable_seed(row["sample_id"], 0, "bridge_v9_audit"), limit=192,
            )
            audit = {
                "token_ids": audit_ids,
                "text": engine.tokenizer.decode(audit_ids, skip_special_tokens=False),
                "logprobs": audit_logprobs,
                "finish": audit_finish,
                "max_tokens": 192,
            }
            final_state = engine.prepare(audit_cache, contract.FINAL_SUFFIX)
        else:
            final_state = engine.prepare(cT, prompt)
        generation = engine.decode(
            final_state,
            _bridge_seed(row["sample_id"]),
            greedy=True,
            grammar=bridge_grammar,
            limit=160,
        )
        parsed = contract.parse_bridge(generation["text"], completed=generation["completed"])
        if not generation["completed"]:
            serialization_error = "generation_incomplete"
        elif parsed["bridge_parse_status"] != "valid":
            serialization_error = "grammar_mismatch"
        else:
            serialization_error = "none"
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
            "reasoning_token_ids_sha256": contract.token_ids_sha256(reasoning_ids),
            "reasoning_logprobs": reasoning_logprobs,
            "reasoning_text": engine.tokenizer.decode(reasoning_ids, skip_special_tokens=False),
            "reasoning_finish": reasoning_finish,
            "reasoning_origin": reasoning_origin,
            "reasoning_seed": reasoning_seed,
            "frozen_source_record_sha256": source_hash,
            "bridge_version": contract.BRIDGE_VERSION,
            "bridge_generation": generation,
            "bridge_seed": _bridge_seed(row["sample_id"]),
            "bridge_prompt_rendered": prompt if args.bridge_version in ("v8", "v9") else None,
            "candidate_record_sha256": file_hash(candidate_path) if candidate is not None else None,
            "candidate_target_entity": candidate["target_entity_reference"] if candidate is not None else None,
            "candidate_task_answer": candidate["task_answer"] if candidate is not None else None,
            "audit_generation": audit,
            "extractor_serialization_error": serialization_error,
            **parsed,
        }
        _write_exclusive(target, record)
        completed += 1
        print(f"BRIDGE_{args.bridge_version.upper()} {completed}/{len(selected)} {row['sample_id']} "
              f"{parsed['bridge_parse_status']}", flush=True)
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
    contract = _bridge_contract(args.bridge_version)
    rows = _read(args.selection)
    output = args.output_dir.absolute()
    protocol_path = output / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != contract.BRIDGE_VERSION:
        raise RuntimeError("output protocol bridge version differs from --bridge-version")
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
        record_version = record.get("bridge_version")
        if record_version is not None and record_version != contract.BRIDGE_VERSION:
            raise RuntimeError(f"record bridge version differs: {row['sample_id']}")
        if record_version is None and args.bridge_version != "v2":
            raise RuntimeError(f"v3 record lacks bridge version: {row['sample_id']}")
        records.append(record)
    with packet_path.open("x", encoding="utf-8") as handle:
        for record in records:
            packet = {key: record[key] for key in (
                "sample_id", "split", "image_id", "task_type", "image_path",
                "image_sha256", "question", "ground_truth_bbox", "reasoning_text",
                "reasoning_finish", "reasoning_token_ids_sha256", "target_entity_reference",
                "target_entity_reference_status", "bridge_parse_status",
                "extractor_serialization_error",
            )}
            if args.bridge_version not in ("v5", "v7", "v8", "v9"):
                packet["task_answer"] = record["task_answer"]
                packet["task_answer_status"] = record["task_answer_status"]
            packet.update(
                student_target_selection_error=None,
                extractor_content_error=None,
                target_reference_review_label=None,
                student_target_selection_error_labels=list(contract.STUDENT_SELECTION_LABELS),
                extractor_content_error_labels=list(contract.EXTRACTOR_CONTENT_ERROR_LABELS),
                target_reference_review_labels=list(contract.TARGET_REFERENCE_REVIEW_LABELS),
            )
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "records": len(records),
        "parse_status_counts": dict(sorted(Counter(
            record["bridge_parse_status"] for record in records
        ).items())),
        "reasoning_origin_counts": dict(sorted(Counter(
            record["reasoning_origin"] for record in records
        ).items())),
        "packet": str(packet_path),
        "packet_sha256": file_hash(packet_path),
        "selection_sha256": file_hash(args.selection),
        "bridge_version": contract.BRIDGE_VERSION,
        "gt_used_for_inference": False,
        "sealed_independent_confirmation_accessed": False,
    }
    _write_exclusive(output / "merge_manifest.json", manifest)
    print(packet_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bridge-version", choices=("v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10"), default="v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--frozen-reasoning-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--work-selection", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--receipt-label")
    args = parser.parse_args(argv)
    if not args.merge and args.bridge_version in ("v3", "v5", "v6", "v7", "v8", "v9", "v10") and args.frozen_reasoning_root is None:
        parser.error("prompt-only generation requires --frozen-reasoning-root")
    if not args.merge and args.bridge_version == "v8" and not args.candidate_root:
        parser.error("v8 requires --candidate-root")
    if not args.merge and (args.device is None or args.shard_index is None or args.num_shards is None):
        parser.error("generation requires device, shard-index, and num-shards")
    if not args.merge and not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    merge(parsed) if parsed.merge else run_shard(parsed)
