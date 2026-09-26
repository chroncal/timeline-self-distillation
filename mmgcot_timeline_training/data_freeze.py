"""Freeze train/dev, independent-confirmation, and blind-review manifests.

The selector is deliberately result-blind.  It reads source annotations,
pre-existing eligibility decisions, image files, and historical launch
manifests; it never reads model predictions, IoU, or training metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mmgcot_diagnostic.data import _default_image_dirs, _raw_candidates, discover_local_images
from mmgcot_diagnostic.protocol import file_hash

SEED = 20260921
TASKS = ("attribute", "object")
TRAIN_PER_TASK = 155
DEV_PER_TASK = 30
CONFIRM_PER_TASK = 24
REVIEW_QUOTAS = {("train", "attribute"): 20, ("train", "object"): 20,
                 ("dev", "attribute"): 5, ("dev", "object"): 5}

DEFAULT_RAW_ROOT = Path("/mnt/sda/sujingyang/research/datasets/mmgcot_20260920")
DEFAULT_FROZEN_ROOT = Path("/mnt/sda/sujingyang/research/datasets/mmgcot_20260920_frozen_final_v1")
DEFAULT_TIMELINE_OUTPUTS = Path(
    "/mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline"
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _stable_key(*parts: object, seed: int = SEED) -> str:
    blob = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_json_bytes(dict(row)))


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))


def _validate_gt(box: Sequence[Any]) -> list[float]:
    if len(box) != 4:
        raise ValueError("ground_truth_bbox must have four values")
    result = [float(value) for value in box]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"non-finite or out-of-range ground truth: {result}")
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"non-positive ground-truth extent: {result}")
    return result


def _public_row(candidate: Mapping[str, Any], *, split: str, image_path: Path) -> dict[str, Any]:
    image_path = image_path.absolute()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    task = str(candidate["task_type"])
    source_id = str(candidate["source_id"])
    occurrence = int(candidate["source_occurrence"])
    return {
        "sample_id": f"{split}:{task}:{source_id}:{occurrence}",
        "split": split,
        "image_id": str(candidate["image_id"]),
        "task_type": task,
        "image_path": str(image_path),
        "image_sha256": file_hash(image_path),
        "question": str(candidate["question"]),
        "ground_truth_bbox": _validate_gt(candidate["ground_truth_bbox"]),
        "source_id": source_id,
        "source_file": str(candidate["source_file"]),
        "source_row_index": int(candidate["source_row_index"]),
        "source_occurrence": occurrence,
    }


def _ranked(rows: Iterable[Mapping[str, Any]], namespace: str) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: _stable_key(
            namespace, row.get("image_id"), row.get("source_id"), row.get("source_row_index")
        ),
    )


def assign_balanced_images(
    candidates_by_task: Mapping[str, Sequence[Mapping[str, Any]]], *, per_task: int
) -> dict[str, list[dict[str, Any]]]:
    """Choose exact balanced task assignments without reusing an image."""
    best: dict[str, dict[str, dict[str, Any]]] = {task: {} for task in TASKS}
    for task in TASKS:
        for row in _ranked(candidates_by_task.get(task, ()), f"row:{task}"):
            best[task].setdefault(str(row["image_id"]), row)
    attr_ids, object_ids = set(best["attribute"]), set(best["object"])
    only_attr = sorted(attr_ids - object_ids, key=lambda x: _stable_key("only:attribute", x))
    only_object = sorted(object_ids - attr_ids, key=lambda x: _stable_key("only:object", x))
    both = sorted(attr_ids & object_ids, key=lambda x: _stable_key("both", x))

    selected_ids = {
        "attribute": only_attr[:per_task],
        "object": only_object[:per_task],
    }
    need_attr = per_task - len(selected_ids["attribute"])
    need_object = per_task - len(selected_ids["object"])
    if need_attr < 0 or need_object < 0 or need_attr + need_object > len(both):
        raise RuntimeError(
            f"cannot form disjoint balanced assignments: only_attr={len(only_attr)}, "
            f"only_object={len(only_object)}, both={len(both)}, target={per_task}"
        )
    selected_ids["attribute"].extend(both[:need_attr])
    selected_ids["object"].extend(both[need_attr : need_attr + need_object])
    result = {
        task: [best[task][image_id] for image_id in selected_ids[task]] for task in TASKS
    }
    flat_ids = [str(row["image_id"]) for rows in result.values() for row in rows]
    if len(flat_ids) != 2 * per_task or len(flat_ids) != len(set(flat_ids)):
        raise RuntimeError("balanced assignment reused an image or missed a quota")
    return result


def split_train_dev(assignments: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    dev: list[dict] = []
    for task in TASKS:
        ordered = _ranked(assignments[task], f"split:{task}")
        if len(ordered) != TRAIN_PER_TASK + DEV_PER_TASK:
            raise RuntimeError(f"unexpected {task} assignment count: {len(ordered)}")
        dev.extend(ordered[:DEV_PER_TASK])
        train.extend(ordered[DEV_PER_TASK:])
    return _ranked(train, "train:output"), _ranked(dev, "dev:output")


def choose_review_rows(train: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]]) -> list[dict]:
    selected: list[dict] = []
    for split, rows in (("train", train), ("dev", dev)):
        for task in TASKS:
            eligible = [row for row in rows if row["task_type"] == task]
            chosen = _ranked(eligible, f"blind-review:{split}:{task}")[: REVIEW_QUOTAS[(split, task)]]
            if len(chosen) != REVIEW_QUOTAS[(split, task)]:
                raise RuntimeError(f"blind-review quota unavailable for {split}/{task}")
            for row in chosen:
                selected.append({
                    "sample_id": row["sample_id"],
                    "split": split,
                    "image_id": row["image_id"],
                    "task_type": task,
                    "image_path": row["image_path"],
                    "image_sha256": row["image_sha256"],
                    "question": row["question"],
                    "ground_truth_bbox": row["ground_truth_bbox"],
                    "target_description": None,
                    "review_label": None,
                    "allowed_review_labels": [
                        "same_target", "different_target", "description_not_unique", "cannot_determine"
                    ],
                })
    return _ranked(selected, "blind-review:output")


def historical_image_ids(timeline_outputs: Path, frozen_root: Path) -> tuple[set[str], list[dict[str, str]]]:
    manifests = {
        frozen_root / "pilot_frozen_final.jsonl",
        frozen_root / "formal_frozen_final.jsonl",
    }
    for launch_path in sorted(timeline_outputs.glob("*/launch.json")):
        launch = json.loads(launch_path.read_text())
        manifest = Path(str(launch.get("manifest", "")))
        if manifest.is_file():
            manifests.add(manifest)
    ids: set[str] = set()
    receipts: list[dict[str, str]] = []
    for manifest in sorted(manifests):
        rows = _read_jsonl(manifest)
        ids.update(str(row["image_id"]) for row in rows)
        receipts.append({"path": str(manifest.absolute()), "sha256": file_hash(manifest)})
    return ids, receipts


def _load_confirmation_candidates(frozen_root: Path) -> list[dict[str, Any]]:
    mapping_path = frozen_root / "eligibility_review_mapping.jsonl"
    mapping = [
        row for row in _read_jsonl(mapping_path)
        if row.get("source") == "reserve_formal"
        and row.get("decision") == "quota_not_selected"
        and row.get("status") == "eligible"
    ]
    manifests = {Path(str(row["manifest"])) for row in mapping}
    source_rows: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for row in _read_jsonl(manifest):
            source_rows[str(row["sample_id"])] = row
    result = []
    for receipt in mapping:
        row = dict(source_rows[str(receipt["sample_id"])])
        if str(row["image_id"]) != str(receipt["image_id"]):
            raise RuntimeError("confirmation review mapping disagrees with source manifest")
        row["eligibility_case_id"] = receipt["case_id"]
        row["eligibility_status"] = receipt["status"]
        row["eligibility_decision"] = receipt["decision"]
        result.append(row)
    return result


def choose_confirmation(frozen_root: Path, history_ids: set[str]) -> list[dict[str, Any]]:
    candidates = _load_confirmation_candidates(frozen_root)
    by_task = defaultdict(list)
    for row in candidates:
        if str(row["image_id"]) in history_ids:
            raise RuntimeError("a quota_not_selected confirmation row appeared in diagnostic history")
        by_task[str(row["task_type"])].append(row)
    selected: list[dict[str, Any]] = []
    for task in TASKS:
        ranked = _ranked(by_task[task], f"confirmation:{task}")
        if len(ranked) < CONFIRM_PER_TASK:
            raise RuntimeError(f"confirmation quota unavailable for {task}: {len(ranked)}")
        selected.extend(ranked[:CONFIRM_PER_TASK])
    result = []
    for row in _ranked(selected, "confirmation:output"):
        path = Path(str(row["image_path"]))
        if not path.is_file() or file_hash(path) != row["image_sha256"]:
            raise RuntimeError(f"confirmation image missing or hash changed: {path}")
        result.append({
            "sample_id": row["sample_id"],
            "split": "independent_confirmation",
            "image_id": str(row["image_id"]),
            "task_type": str(row["task_type"]),
            "image_path": str(path.absolute()),
            "image_sha256": row["image_sha256"],
            "question": row["question"],
            "ground_truth_bbox": _validate_gt(row["ground_truth_bbox"]),
            "source_id": str(row["source_id"]),
            "source_file": row["source_file"],
            "source_row_index": int(row["source_row_index"]),
            "source_occurrence": int(row["source_occurrence"]),
            "eligibility_case_id": row["eligibility_case_id"],
            "eligibility_status": row["eligibility_status"],
            "eligibility_decision": row["eligibility_decision"],
        })
    ids = [row["image_id"] for row in result]
    if len(ids) != 48 or len(ids) != len(set(ids)):
        raise RuntimeError("independent confirmation must contain 48 unique images")
    return result


def freeze(args: argparse.Namespace) -> Path:
    output = args.output_dir.absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    history_ids, history_receipts = historical_image_ids(args.timeline_outputs, args.frozen_root)
    confirmation = choose_confirmation(args.frozen_root, history_ids)
    confirmation_ids = {row["image_id"] for row in confirmation}

    candidates, _, raw_stats = _raw_candidates(args.raw_root)
    local_index = discover_local_images(_default_image_dirs([]))
    eligible: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for task in TASKS:
        for row in candidates["pilot"][task]:
            image_id = str(row["image_id"])
            if image_id in history_ids or image_id in confirmation_ids or image_id not in local_index:
                continue
            eligible[task].append(row)
    assigned = assign_balanced_images(eligible, per_task=TRAIN_PER_TASK + DEV_PER_TASK)
    train_candidates, dev_candidates = split_train_dev(assigned)
    train = [_public_row(row, split="train", image_path=local_index[str(row["image_id"])]) for row in train_candidates]
    dev = [_public_row(row, split="dev", image_path=local_index[str(row["image_id"])]) for row in dev_candidates]
    review = choose_review_rows(train, dev)

    cohorts = {"train": train, "dev": dev, "independent_confirmation": confirmation}
    cohort_ids = {name: {row["image_id"] for row in rows} for name, rows in cohorts.items()}
    for left, right in (("train", "dev"), ("train", "independent_confirmation"), ("dev", "independent_confirmation")):
        if cohort_ids[left] & cohort_ids[right]:
            raise RuntimeError(f"image leakage between {left} and {right}")
    if set.union(*cohort_ids.values()) & history_ids:
        raise RuntimeError("new cohort overlaps diagnostic history")

    paths = {
        "train": output / "train_frozen.jsonl",
        "dev": output / "dev_frozen.jsonl",
        "independent_confirmation": output / "independent_confirmation_frozen.jsonl",
        "blind_review_selection": output / "blind_review_selection.jsonl",
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl_exclusive(paths["train"], train)
    _write_jsonl_exclusive(paths["dev"], dev)
    _write_jsonl_exclusive(paths["independent_confirmation"], confirmation)
    _write_jsonl_exclusive(paths["blind_review_selection"], review)
    artifact_hashes = {name: file_hash(path) for name, path in paths.items()}
    manifest = {
        "schema_version": "mmgcot_timeline_training_data_v1",
        "seed": SEED,
        "selection_is_result_blind": True,
        "historical_diagnostic_image_count": len(history_ids),
        "historical_manifests": history_receipts,
        "source_hashes": {
            "raw_train": file_hash(args.raw_root / "raw/Train/train_dataset.json"),
            "raw_test_attribute": file_hash(args.raw_root / "raw/Test/CoP_dataset_attributes_test.json"),
            "raw_test_object": file_hash(args.raw_root / "raw/Test/CoP_dataset_things_test.json"),
            "eligibility_review_mapping": file_hash(args.frozen_root / "eligibility_review_mapping.jsonl"),
        },
        "counts": {
            name: {"rows": len(rows), "images": len(cohort_ids.get(name, {r['image_id'] for r in rows})),
                   "tasks": dict(Counter(row["task_type"] for row in rows))}
            for name, rows in cohorts.items()
        } | {"blind_review_selection": {"rows": len(review), "tasks": dict(Counter(r["task_type"] for r in review)),
                                         "splits": dict(Counter(r["split"] for r in review))}},
        "raw_candidate_stats": raw_stats,
        "artifacts": {name: {"path": str(path), "sha256": artifact_hashes[name]} for name, path in paths.items()},
        "confirmation_policy": "reserve_formal eligible quota_not_selected; stable 24 per task; never diagnosed",
        "blind_review_policy": "train 20/task + dev 5/task; selected before target descriptions or model results",
    }
    _write_json_exclusive(output / "manifest.json", manifest)
    _write_json_exclusive(output / "hashes.json", {
        **artifact_hashes,
        "manifest.json": file_hash(output / "manifest.json"),
    })
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--timeline-outputs", type=Path, default=DEFAULT_TIMELINE_OUTPUTS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(freeze(parse_args()))
