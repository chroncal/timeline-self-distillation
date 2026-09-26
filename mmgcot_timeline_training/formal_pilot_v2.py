"""Fixed, small v3p5 L/R/E free-box and shared-student-prefix readout."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from mmgcot_diagnostic.protocol import MODEL, comma_prefix, file_hash, iou, stable_seed
from mmgcot_timeline_training.prepare_v2 import record_path


def main(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    import torch
    import xgrammar as xgr
    import mmgcot_timeline_training.formal_train_v2 as formal

    formal.torch = torch
    formal.xgr = xgr
    rows = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
    protocol = json.loads((args.prepared / "protocol.json").read_text())
    if protocol["selection_sha256"] != file_hash(args.selection):
        raise RuntimeError("pilot selection/trajectory protocol mismatch")
    engine = formal.FormalEngine(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for row in rows[:args.images]:
            path = record_path(args.prepared, row["sample_id"], 0)
            frozen = json.loads(path.read_text())
            state = engine.build_state(frozen)
            first_student = None
            for branch in ("L", "R", "E"):
                for draw in range(4):
                    seed = stable_seed(row["sample_id"], 0, draw, "formal_v2_pilot_free")
                    result = engine.sample_bbox(state, branch=branch, seed=seed)
                    if branch == "L" and draw == 0:
                        first_student = result
                    record = {"sample_id": row["sample_id"], "image_id": row["image_id"],
                              "trajectory_index": 0, "arm": branch, "mode": "free", "draw": draw,
                              "seed": seed, "iou": iou(result["bbox"], row["ground_truth_bbox"], result["valid"]),
                              "valid": result["valid"], "completed": result["completed"],
                              "bbox": result["bbox"], "response_token_ids": result["token_ids"],
                              "has_target": state["has_target"], "early_offset": state["early_offset"]}
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            assert first_student is not None
            for coordinate_count in (1, 2):
                prefix = comma_prefix(engine.tokenizer, first_student["token_ids"], coordinate_count)
                if prefix is None:
                    output.write(json.dumps({"sample_id": row["sample_id"],
                        "mode": "continuation", "prefix_coordinates": coordinate_count,
                        "status": "unscorable_prefix", "student_token_ids": first_student["token_ids"]}) + "\n")
                    continue
                for branch in ("L", "R", "E"):
                    for draw in range(4):
                        seed = stable_seed(row["sample_id"], coordinate_count, draw,
                                           "formal_v2_pilot_continuation")
                        result = engine.sample_bbox(state, branch=branch, seed=seed,
                                                    forced_prefix_ids=prefix)
                        output.write(json.dumps({"sample_id": row["sample_id"],
                            "image_id": row["image_id"], "trajectory_index": 0,
                            "arm": branch, "mode": "continuation", "draw": draw,
                            "prefix_coordinates": coordinate_count, "raw_prefix_ids": prefix,
                            "response_token_ids": result["token_ids"], "seed": seed,
                            "bbox": result["bbox"], "valid": result["valid"],
                            "completed": result["completed"],
                            "iou": iou(result["bbox"], row["ground_truth_bbox"], result["valid"]),
                            "status": "ok"}, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(f"PILOT {row['sample_id']}", flush=True)
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    unscorable = 0
    for record in records:
        if record.get("status") == "unscorable_prefix":
            unscorable += 1
            continue
        condition = ("free" if record["mode"] == "free"
                     else f"prefix_{record['prefix_coordinates']}")
        scores[condition][record["arm"]][record["sample_id"]].append(record["iou"])
    summary: dict[str, object] = {"images": min(args.images, len(rows)),
                                  "unscorable_prefixes": unscorable,
                                  "selection_sha256": file_hash(args.selection),
                                  "conditions": {}}
    for condition, arms in scores.items():
        means = {arm: {sample: sum(values)/len(values) for sample, values in samples.items()}
                 for arm, samples in arms.items()}
        common = set.intersection(*(set(values) for values in means.values()))
        condition_report: dict[str, object] = {
            "arm_miou": {arm: sum(values.values())/len(values) for arm, values in means.items()},
            "common_images": len(common),
        }
        if common:
            condition_report["r_minus_l"] = sum(means["R"][x]-means["L"][x] for x in common)/len(common)
            condition_report["e_minus_l"] = sum(means["E"][x]-means["L"][x] for x in common)/len(common)
            condition_report["e_minus_r"] = sum(means["E"][x]-means["R"][x] for x in common)/len(common)
        summary["conditions"][condition] = condition_report
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2,
        ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    if args.images < 1:
        parser.error("--images must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
