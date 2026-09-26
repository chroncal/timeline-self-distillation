"""CPU-only aggregation for the preregistered MM-GCoT diagnostic.

The runner writes one immutable JSONL journal per trajectory.  This module
reads only those journals, validates the metric-bearing fields, and computes
image-equal summaries.  It deliberately does not read the potentially large
``distributions/*.jsonl.gz`` files.

The public entry points are :func:`analyze`, :func:`write_outputs`, and
:func:`main`.  The command line interface is also usable with a run directory;
when ``--records`` points at a run directory, only its ``records`` child is
scanned.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import glob
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2_026_09_20
IOU_THRESHOLD = 0.5
EXPECTED_TRAJECTORIES = 3

A_ARMS = ("L0", "L", "E", "R")
B_ARMS = ("E", "L", "R")
A_CONTRASTS = (("E", "L"), ("E", "R"), ("R", "L"), ("L", "L0"))
B_CONTRASTS = (("E", "L"), ("E", "R"), ("R", "L"))
ALLOWED_RECORD_TYPES = {
    "trajectory",
    "trajectory_done",
    "bbox",
    "prefix_unavailable",
    "failure",
}
ALLOWED_FINISH_REASONS = {"stop", "length", "eos"}
ALLOWED_ENTITY_STATUSES = {"usable", "unresolved", "invalid", "not_generated"}
ALLOWED_REVIEW_STATUSES = {
    "correct_unique",
    "wrong_target",
    "ambiguous",
    "unresolved",
    "uncertain",
}

TrajectoryKey = tuple[str, str, int]
PrefixKey = tuple[str, str, int, int]
BboxKey = tuple[str, str, int, str, str, str, int, int | None]
Slot = tuple[str, str, str, int, int | None]


class ValidationError(ValueError):
    """Raised when an input violates the frozen analysis interface."""


@dataclass
class RecordIndex:
    """Validated record rows indexed by their protocol keys."""

    trajectories: dict[TrajectoryKey, dict[str, Any]] = field(default_factory=dict)
    trajectory_done: dict[TrajectoryKey, dict[str, Any]] = field(default_factory=dict)
    bbox: dict[BboxKey, dict[str, Any]] = field(default_factory=dict)
    failure_outputs: dict[BboxKey, dict[str, Any]] = field(default_factory=dict)
    prefix_unavailable: dict[PrefixKey, dict[str, Any]] = field(default_factory=dict)
    trajectory_failures: list[dict[str, Any]] = field(default_factory=list)
    trajectory_failure_keys: set[TrajectoryKey] = field(default_factory=set)
    files: list[str] = field(default_factory=list)
    truncated_tail_files: list[dict[str, Any]] = field(default_factory=list)
    rows_read: int = 0


def _context(path: str | Path | None, line: int | None = None) -> str:
    if path is None:
        return ""
    return f" in {path}" + (f":{line}" if line is not None else "")


def _error(message: str, *, path: str | Path | None = None, line: int | None = None) -> None:
    raise ValidationError(message + _context(path, line))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any, name: str, *, path: str | Path | None = None, line: int | None = None) -> float:
    if not _is_number(value):
        _error(f"{name} must be a finite number, got {value!r}", path=path, line=line)
    result = float(value)
    if not math.isfinite(result):
        _error(f"{name} is non-finite", path=path, line=line)
    return result


def _integer(value: Any, name: str, *, path: str | Path | None = None, line: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _error(f"{name} must be an integer, got {value!r}", path=path, line=line)
    return int(value)


def _boolean(value: Any, name: str, *, path: str | Path | None = None, line: int | None = None) -> bool:
    if not isinstance(value, bool):
        _error(f"{name} must be boolean, got {value!r}", path=path, line=line)
    return value


def _id(value: Any, name: str, *, path: str | Path | None = None, line: int | None = None) -> str:
    if isinstance(value, bool) or value is None:
        _error(f"{name} must be a non-empty identifier", path=path, line=line)
    result = str(value)
    if not result:
        _error(f"{name} must be a non-empty identifier", path=path, line=line)
    return result


def _strict_json_loads(text: str, *, path: str | Path | None = None, line: int | None = None) -> Any:
    def reject_constant(value: str) -> Any:
        _error(f"JSON constant {value!r} is not allowed", path=path, line=line)

    try:
        return json.loads(text, parse_constant=reject_constant)
    except ValidationError:
        raise
    except json.JSONDecodeError as exc:
        _error(f"invalid JSON: {exc.msg}", path=path, line=line)
    raise AssertionError("unreachable")


def _read_json_lines(path: Path) -> tuple[list[tuple[dict[str, Any], int]], int | None]:
    if path.suffix.lower() == ".json":
        try:
            value = _strict_json_loads(path.read_text(encoding="utf-8"), path=path)
        except UnicodeDecodeError as exc:
            _error(f"not UTF-8 JSON: {exc}", path=path)
        if isinstance(value, dict):
            if "records" in value:
                value = value["records"]
            else:
                value = [value]
        if not isinstance(value, list):
            _error("JSON trajectory file must contain an object or a records list", path=path)
        rows: list[tuple[dict[str, Any], int]] = []
        for number, row in enumerate(value, start=1):
            if not isinstance(row, dict):
                _error("record must be a JSON object", path=path, line=number)
            rows.append((row, number))
        return rows, None

    rows = []
    truncated_tail_line: int | None = None
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        _error(f"cannot open records file: {exc}", path=path)
    with stream:
        raw_lines = list(stream)
    for number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            row = _strict_json_loads(raw, path=path, line=number)
        except ValidationError:
            # A journal may be observed while the runner is writing its
            # final JSON object.  A non-newline-terminated malformed final
            # line is excluded and surfaced as collection partial.  Any
            # malformed internal line, or a malformed line terminated by
            # newline, remains a hard validation error.
            is_final_nonempty = not any(item.strip() for item in raw_lines[number:])
            if is_final_nonempty and not raw.endswith(("\n", "\r")):
                truncated_tail_line = number
                break
            raise
        if not isinstance(row, dict):
            _error("record must be a JSON object", path=path, line=number)
        rows.append((row, number))
    return rows, truncated_tail_line


def _record_filename(path: Path) -> bool:
    if path.name.startswith(".") or path.suffix.lower() not in {".jsonl", ".json"}:
        return False
    # A run directory may contain protocol/receipt JSON files.  The actual
    # runner journals are the files below records/, so this filter keeps those
    # auxiliary files out even when a broad directory is supplied.
    return True


def expand_record_paths(records: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve files, globs, or run directories without scanning distributions."""

    if isinstance(records, (str, Path)):
        requested: list[str | Path] = [records]
    else:
        requested = list(records)
    found: dict[str, Path] = {}

    def add(path: Path) -> None:
        if path.is_file() and _record_filename(path):
            found[str(path.resolve())] = path.resolve()

    for item in requested:
        raw = str(item)
        if glob.has_magic(raw):
            for matched in sorted(glob.glob(raw, recursive=True)):
                add(Path(matched))
            continue
        path = Path(item)
        if path.is_file():
            add(path)
            continue
        if not path.exists():
            _error("records path does not exist", path=path)
        if not path.is_dir():
            _error("records path is neither a file nor a directory", path=path)

        if path.name == "records":
            roots = [path]
        elif (path / "records").is_dir():
            roots = [path / "records"]
        else:
            roots = [path]
        for root in roots:
            for candidate in sorted(root.rglob("*")):
                if candidate.is_file() and "distributions" not in candidate.parts:
                    add(candidate)

    result = sorted(found.values(), key=lambda value: str(value))
    if not result:
        _error("no trajectory JSON/JSONL files found under --records")
    return result


def _validate_gt(value: Any, *, path: str | Path | None = None, line: int | None = None) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        _error("ground_truth_bbox must be normalized xyxy with four values", path=path, line=line)
    gt = [_finite(item, "ground_truth_bbox coordinate", path=path, line=line) for item in value]
    if any(item < 0.0 or item > 1.0 for item in gt):
        _error("ground_truth_bbox must lie in [0, 1]; dataset GT conversion is out of scope", path=path, line=line)
    if gt[0] > gt[2] or gt[1] > gt[3]:
        _error("ground_truth_bbox must be xyxy with non-decreasing edges", path=path, line=line)
    return gt


def _cohort(row: Mapping[str, Any]) -> str:
    for key in ("cohort", "phase", "split"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return "unlabeled"


def _validated_manifest_rows(rows: Iterable[Mapping[str, Any]], *, source: str = "manifest") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, original in enumerate(rows, start=1):
        if not isinstance(original, Mapping):
            _error("manifest row must be an object", path=source, line=number)
        row = dict(original)
        sample_id = _id(row.get("sample_id"), "sample_id", path=source, line=number)
        image_id = _id(row.get("image_id"), "image_id", path=source, line=number)
        if sample_id in seen:
            _error(f"duplicate manifest sample_id {sample_id!r}", path=source, line=number)
        seen.add(sample_id)
        task_type = row.get("task_type")
        question = row.get("question")
        if not isinstance(task_type, str) or not task_type:
            _error("task_type must be a non-empty string", path=source, line=number)
        if not isinstance(question, str):
            _error("question must be a string", path=source, line=number)
        result.append(
            {
                **row,
                "sample_id": sample_id,
                "image_id": image_id,
                "task_type": task_type,
                "question": question,
                "ground_truth_bbox": _validate_gt(row.get("ground_truth_bbox"), path=source, line=number),
                "cohort": _cohort(row),
            }
        )
    if not result:
        _error("manifest is empty", path=source)
    return result


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the required sample JSONL manifest.

    The GT is retained as normalized xyxy.  Only the metric's internal unit
    comparison scales that normalized box to the runner's 0--1000 coordinate
    system; dataset-specific GT conversion is intentionally absent.
    """

    source = str(path)
    rows: list[dict[str, Any]] = []
    file_path = Path(path)
    try:
        stream = file_path.open("r", encoding="utf-8")
    except OSError as exc:
        _error(f"cannot open manifest: {exc}", path=path)
    with stream:
        for number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            value = _strict_json_loads(raw, path=path, line=number)
            if not isinstance(value, dict):
                _error("manifest row must be a JSON object", path=path, line=number)
            rows.append(value)
    return _validated_manifest_rows(rows, source=source)


def _common_key(row: Mapping[str, Any], *, path: str | Path | None, line: int | None) -> TrajectoryKey:
    sample_id = _id(row.get("sample_id"), "sample_id", path=path, line=line)
    image_id = _id(row.get("image_id"), "image_id", path=path, line=line)
    trajectory_index = _integer(row.get("trajectory_index"), "trajectory_index", path=path, line=line)
    if trajectory_index not in range(EXPECTED_TRAJECTORIES):
        _error("trajectory_index must be 0, 1, or 2", path=path, line=line)
    return sample_id, image_id, trajectory_index


def _check_common_against_manifest(
    row: Mapping[str, Any],
    manifest_by_sample: Mapping[str, Mapping[str, Any]],
    *,
    path: str | Path | None,
    line: int | None,
) -> TrajectoryKey:
    key = _common_key(row, path=path, line=line)
    sample_id, image_id, _ = key
    if sample_id not in manifest_by_sample:
        _error(f"record references unknown sample_id {sample_id!r}", path=path, line=line)
    expected_image = str(manifest_by_sample[sample_id]["image_id"])
    if image_id != expected_image:
        _error(
            f"record image_id {image_id!r} disagrees with manifest image_id {expected_image!r}",
            path=path,
            line=line,
        )
    row_task = row.get("task_type")
    if row_task is not None and row_task != manifest_by_sample[sample_id]["task_type"]:
        _error("record task_type disagrees with manifest", path=path, line=line)
    return key


def _validate_bbox_coordinates(
    value: Any, *, path: str | Path | None, line: int | None
) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        _error("bbox must be null or 0--1000 xyxy with four values", path=path, line=line)
    bbox = [_finite(item, "bbox coordinate", path=path, line=line) for item in value]
    # Do not clip, reorder, or otherwise repair invalid raw boxes.  The runner
    # records such boxes with valid=false and iou=0; the analysis must retain
    # them for coverage while excluding them from geometric success.
    return bbox


def _raw_bbox_is_geometrically_valid(bbox: Sequence[float] | None) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    return (
        all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1000.0 for value in bbox)
        and float(bbox[0]) < float(bbox[2])
        and float(bbox[1]) < float(bbox[3])
    )


def bbox_iou(bbox: Sequence[float] | None, gt_normalized: Sequence[float]) -> float:
    """Compute IoU between a 0--1000 prediction and normalized GT xyxy."""

    gt = [_finite(value, "ground-truth coordinate") for value in gt_normalized]
    if len(gt) != 4 or any(value < 0 or value > 1 for value in gt):
        raise ValidationError("gt_normalized must contain four finite [0, 1] coordinates")
    if bbox is None:
        return 0.0
    pred = [_finite(value, "bbox coordinate") for value in bbox]
    if len(pred) != 4 or any(value < 0 or value > 1000 for value in pred):
        raise ValidationError("bbox must contain four finite [0, 1000] coordinates")
    gt1000 = [value * 1000.0 for value in gt]
    inter_width = max(0.0, min(pred[2], gt1000[2]) - max(pred[0], gt1000[0]))
    inter_height = max(0.0, min(pred[3], gt1000[3]) - max(pred[1], gt1000[1]))
    intersection = inter_width * inter_height
    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    gt_area = max(0.0, gt1000[2] - gt1000[0]) * max(0.0, gt1000[3] - gt1000[1])
    union = pred_area + gt_area - intersection
    result = intersection / union if union > 0.0 else 0.0
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValidationError("recomputed IoU is non-finite or outside [0, 1]")
    return result


def _slot_from_row(
    row: Mapping[str, Any], *, path: str | Path | None, line: int | None
) -> Slot:
    stage = row.get("stage")
    arm = row.get("arm")
    mode = row.get("mode")
    if stage not in {"A", "B"}:
        _error("bbox/failure stage must be 'A' or 'B'", path=path, line=line)
    if not isinstance(arm, str) or not isinstance(mode, str):
        _error("bbox/failure arm and mode are required strings", path=path, line=line)
    draw = _integer(row.get("draw"), "draw", path=path, line=line)
    prefix_value = row.get("prefix_coordinates")
    prefix: int | None
    if prefix_value is None:
        prefix = None
    else:
        prefix = _integer(prefix_value, "prefix_coordinates", path=path, line=line)

    if stage == "A":
        if arm not in A_ARMS or mode not in {"random", "greedy"}:
            _error("unsupported A arm/mode", path=path, line=line)
        if mode == "random" and draw not in range(4):
            _error("A random draw must be 0..3", path=path, line=line)
        if mode == "greedy" and draw != 0:
            _error("A greedy draw must be 0", path=path, line=line)
        if prefix is not None:
            _error("A records cannot have prefix_coordinates", path=path, line=line)
    else:
        if arm not in B_ARMS or mode != "random" or draw not in range(4):
            _error("B records must be E/L/R random draw 0..3", path=path, line=line)
        if prefix not in {1, 2}:
            _error("B records require prefix_coordinates 1 or 2", path=path, line=line)
    return stage, arm, mode, draw, prefix


def _bbox_key(common: TrajectoryKey, slot: Slot) -> BboxKey:
    return (*common, *slot)


def _validate_trajectory_row(
    row: Mapping[str, Any], *, path: str | Path | None, line: int | None
) -> dict[str, Any]:
    result = dict(row)
    finish = result.get("finish_reason")
    if finish is not None and finish not in ALLOWED_FINISH_REASONS:
        _error("invalid finish_reason", path=path, line=line)
    status = result.get("entity_status")
    if status is not None and status not in ALLOWED_ENTITY_STATUSES:
        _error("invalid entity_status", path=path, line=line)
    if "reasoning_length" in result and result["reasoning_length"] is not None:
        length = _integer(result["reasoning_length"], "reasoning_length", path=path, line=line)
        if length < 0:
            _error("reasoning_length must be non-negative", path=path, line=line)
        result["reasoning_length"] = length
    if "entity" in result and result["entity"] is not None and not isinstance(result["entity"], str):
        _error("entity must be a string or null", path=path, line=line)
    return result


def _validate_bbox_row(
    row: Mapping[str, Any],
    common: TrajectoryKey,
    gt: Sequence[float],
    *,
    path: str | Path | None,
    line: int | None,
) -> tuple[BboxKey, dict[str, Any]]:
    result = dict(row)
    slot = _slot_from_row(row, path=path, line=line)
    result["stage"], result["arm"], result["mode"], result["draw"], result["prefix_coordinates"] = slot
    valid = _boolean(row.get("valid"), "valid", path=path, line=line)
    bbox = _validate_bbox_coordinates(row.get("bbox"), path=path, line=line)
    result["bbox"] = bbox
    completed = row.get("completed")
    if completed is not None:
        _boolean(completed, "completed", path=path, line=line)
    expected_valid = _raw_bbox_is_geometrically_valid(bbox) and completed is not False
    if valid != expected_valid:
        _error(
            f"valid flag {valid} disagrees with raw bbox/completed validity {expected_valid}",
            path=path,
            line=line,
        )
    result["valid"] = valid
    recorded_iou = _finite(row.get("iou"), "iou", path=path, line=line)
    if recorded_iou < 0.0 or recorded_iou > 1.0:
        _error("iou must lie in [0, 1]", path=path, line=line)
    if not valid and recorded_iou != 0.0:
        _error("valid=false requires iou=0", path=path, line=line)
    if valid and bbox is None:
        _error("valid=true requires a bbox", path=path, line=line)
    recomputed = bbox_iou(bbox, gt) if valid else 0.0
    if valid and not math.isclose(recorded_iou, recomputed, rel_tol=1e-7, abs_tol=1e-7):
        _error(
            f"recorded iou {recorded_iou} disagrees with independently recomputed {recomputed}",
            path=path,
            line=line,
        )
    result["iou"] = recorded_iou
    if "seed" in result and result["seed"] is not None:
        _integer(result["seed"], "seed", path=path, line=line)
    if "text" in result and result["text"] is not None and not isinstance(result["text"], str):
        _error("text must be a string or null", path=path, line=line)
    if "token_ids" in result and result["token_ids"] is not None:
        if not isinstance(result["token_ids"], list):
            _error("token_ids must be a list or null", path=path, line=line)
        for token in result["token_ids"]:
            _integer(token, "token_ids entry", path=path, line=line)
    return _bbox_key(common, slot), result


def _validate_file_order(rows: Sequence[tuple[dict[str, Any], int]], path: Path) -> None:
    types = [row.get("type") for row, _ in rows]
    if any(kind not in ALLOWED_RECORD_TYPES for kind in types):
        unknown = sorted({str(kind) for kind in types if kind not in ALLOWED_RECORD_TYPES})
        _error(f"unknown record type(s): {unknown}", path=path)
    # The runner's per-trajectory journal is explicitly headed and terminated.
    # A combined records*.jsonl is also accepted when it contains multiple
    # journals; in that case ordering is checked by the individual keys below.
    if types.count("trajectory") == 1:
        if types[0] != "trajectory":
            _error("trajectory journal must begin with type=trajectory", path=path)
        if "trajectory_done" in types and types[-1] != "trajectory_done":
            _error("trajectory journal must end with type=trajectory_done", path=path)


def ingest_records(
    manifest: Sequence[Mapping[str, Any]], record_paths: str | Path | Sequence[str | Path]
) -> RecordIndex:
    """Read, validate, and index record journals.

    Duplicate keys, non-finite numbers, invalid raw boxes, and inconsistent IoU
    values fail closed.  Missing expected cells are represented later as zero;
    they are never fabricated as present records here.
    """

    manifest_rows = _validated_manifest_rows(manifest, source="manifest")
    manifest_by_sample = {row["sample_id"]: row for row in manifest_rows}
    paths = expand_record_paths(record_paths)
    index = RecordIndex(files=[str(path) for path in paths])
    seen_failure_keys: set[tuple[Any, ...]] = set()

    for path in paths:
        file_rows, truncated_tail_line = _read_json_lines(path)
        if truncated_tail_line is not None:
            index.truncated_tail_files.append(
                {"path": str(path), "excluded_tail_line": truncated_tail_line}
            )
        _validate_file_order(file_rows, path)
        for row, line in file_rows:
            index.rows_read += 1
            kind = row.get("type")
            if kind not in ALLOWED_RECORD_TYPES:
                _error(f"unknown record type {kind!r}", path=path, line=line)
            common = _check_common_against_manifest(
                row, manifest_by_sample, path=path, line=line
            )
            sample_id, _, _ = common
            gt = manifest_by_sample[sample_id]["ground_truth_bbox"]
            if kind == "trajectory":
                if common in index.trajectories:
                    _error(f"duplicate trajectory key {common!r}", path=path, line=line)
                index.trajectories[common] = _validate_trajectory_row(row, path=path, line=line)
            elif kind == "trajectory_done":
                if common in index.trajectory_done:
                    _error(f"duplicate trajectory_done key {common!r}", path=path, line=line)
                if "inference_complete" in row:
                    _boolean(row["inference_complete"], "inference_complete", path=path, line=line)
                index.trajectory_done[common] = dict(row)
            elif kind == "bbox":
                key, validated = _validate_bbox_row(
                    row, common, gt, path=path, line=line
                )
                if key in index.bbox or key in index.failure_outputs:
                    _error(f"duplicate bbox output key {key!r}", path=path, line=line)
                index.bbox[key] = validated
            elif kind == "prefix_unavailable":
                prefix = _integer(
                    row.get("prefix_coordinates"),
                    "prefix_coordinates",
                    path=path,
                    line=line,
                )
                if prefix not in {1, 2}:
                    _error("prefix_unavailable requires prefix_coordinates 1 or 2", path=path, line=line)
                pkey = (*common, prefix)
                if pkey in index.prefix_unavailable:
                    _error(f"duplicate prefix_unavailable key {pkey!r}", path=path, line=line)
                index.prefix_unavailable[pkey] = dict(row)
            else:  # failure
                has_target = any(name in row for name in ("stage", "arm", "mode", "draw"))
                reason = row.get("reason", "unspecified")
                if not isinstance(reason, str):
                    _error("failure reason must be a string", path=path, line=line)
                if has_target:
                    slot = _slot_from_row(row, path=path, line=line)
                    key = _bbox_key(common, slot)
                    if key in index.bbox or key in index.failure_outputs:
                        _error(f"duplicate bbox output key {key!r}", path=path, line=line)
                    index.failure_outputs[key] = dict(row)
                    seen_failure_keys.add((key, reason))
                else:
                    failure_key = (*common, reason)
                    if failure_key in seen_failure_keys:
                        _error(f"duplicate failure key {failure_key!r}", path=path, line=line)
                    seen_failure_keys.add(failure_key)
                    index.trajectory_failures.append(dict(row))
                    index.trajectory_failure_keys.add(common)

    for pkey in index.prefix_unavailable:
        common = pkey[:3]
        prefix = pkey[3]
        conflicting = [
            key
            for key in [*index.bbox, *index.failure_outputs]
            if key[:3] == common and key[0 + 3] == "B" and key[-1] == prefix
        ]
        if conflicting:
            _error(f"prefix_unavailable conflicts with B outputs for {pkey!r}")
    return index


def load_records(record_paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load raw validated-JSON objects without scanning distribution files."""

    rows: list[dict[str, Any]] = []
    for path in expand_record_paths(record_paths):
        file_rows, _ = _read_json_lines(path)
        _validate_file_order(file_rows, path)
        rows.extend(row for row, _ in file_rows)
    return rows


def _panel_slots(name: str) -> tuple[str, str, tuple[str, ...], tuple[int, ...], tuple[int | None, ...]]:
    if name == "A_random":
        return "A", "random", A_ARMS, (0, 1, 2, 3), (None,)
    if name == "A_greedy":
        return "A", "greedy", A_ARMS, (0,), (None,)
    if name == "B_prefix1":
        return "B", "random", B_ARMS, (0, 1, 2, 3), (1,)
    if name == "B_prefix2":
        return "B", "random", B_ARMS, (0, 1, 2, 3), (2,)
    raise KeyError(name)


def _slots_for_panel(name: str) -> list[Slot]:
    stage, mode, arms, draws, prefixes = _panel_slots(name)
    return [
        (stage, arm, mode, draw, prefix)
        for arm in arms
        for draw in draws
        for prefix in prefixes
    ]


def _cell(
    index: RecordIndex, trajectory: TrajectoryKey, slot: Slot
) -> tuple[float, str]:
    key = _bbox_key(trajectory, slot)
    row = index.bbox.get(key)
    if row is not None:
        return (float(row["iou"]) if row["valid"] else 0.0), (
            "valid" if row["valid"] else "invalid"
        )
    if key in index.failure_outputs:
        return 0.0, "failed"
    if slot[0] == "B" and (trajectory[0], trajectory[1], trajectory[2], int(slot[4])) in index.prefix_unavailable:
        return 0.0, "prefix_unavailable"
    return 0.0, "missing"


def _trajectory_panel(
    index: RecordIndex, trajectory: TrajectoryKey, panel: str
) -> dict[str, Any]:
    stage, mode, arms, draws, prefixes = _panel_slots(panel)
    values: dict[str, list[float]] = {arm: [] for arm in arms}
    statuses: list[str] = []
    for arm in arms:
        for draw in draws:
            prefix = prefixes[0]
            value, status = _cell(index, trajectory, (stage, arm, mode, draw, prefix))
            values[arm].append(value)
            statuses.append(status)
    arm_means = {
        arm: float(statistics.fmean(values[arm])) if values[arm] else 0.0 for arm in arms
    }
    present = sum(status != "missing" and status != "prefix_unavailable" for status in statuses)
    complete = all(status in {"valid", "invalid", "failed"} for status in statuses)
    source_present = True
    source_usable = True
    if stage == "B":
        source_key = _bbox_key(trajectory, ("A", "L", "random", 0, None))
        source_row = index.bbox.get(source_key)
        source_present = source_row is not None or source_key in index.failure_outputs
        # B intentionally probes erroneous student states too.  Geometric
        # validity of the completed L box is therefore not an eligibility
        # condition; only an actual saved token sequence and exact comma
        # boundary are required.
        source_usable = source_row is not None and isinstance(source_row.get("token_ids"), list)
        prefix_key = (*trajectory, int(prefixes[0]))
        if prefix_key in index.prefix_unavailable:
            complete = False
        complete = complete and source_present and source_usable
    return {
        "arm_means": arm_means,
        "statuses": Counter(statuses),
        "present_cells": present,
        "expected_cells": len(statuses),
        "complete": complete,
        "source_present": source_present,
        "source_usable": source_usable,
    }


def _trajectory_keys(manifest: Sequence[Mapping[str, Any]]) -> list[TrajectoryKey]:
    return [
        (str(row["sample_id"]), str(row["image_id"]), index)
        for row in manifest
        for index in range(EXPECTED_TRAJECTORIES)
    ]


def _panel_coverage(
    index: RecordIndex,
    manifest: Sequence[Mapping[str, Any]],
    panel: str,
) -> dict[str, Any]:
    slots = _slots_for_panel(panel)
    counts = Counter()
    by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    sample_cohorts = {str(row["sample_id"]): str(row["cohort"]) for row in manifest}
    complete = 0
    for trajectory in _trajectory_keys(manifest):
        sample_id = trajectory[0]
        panel_row = _trajectory_panel(index, trajectory, panel)
        if panel_row["complete"]:
            complete += 1
        for slot in slots:
            _, status = _cell(index, trajectory, slot)
            counts[status] += 1
            by_cohort[sample_cohorts[sample_id]][status] += 1
    expected = len(_trajectory_keys(manifest)) * len(slots)
    return {
        "expected_cells": expected,
        "present_cells": counts["valid"] + counts["invalid"] + counts["failed"],
        "valid_bbox_cells": counts["valid"],
        "invalid_bbox_cells": counts["invalid"],
        "failed_output_cells": counts["failed"],
        "missing_cells": counts["missing"] + counts["prefix_unavailable"],
        "prefix_unavailable_cells": counts["prefix_unavailable"],
        "complete_trajectories": complete,
        "expected_trajectories": len(_trajectory_keys(manifest)),
        "by_cohort": {cohort: dict(values) for cohort, values in sorted(by_cohort.items())},
    }


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_bootstrap(
    differences: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return an image-paired percentile bootstrap for one difference vector."""

    values = [float(value) for value in differences]
    if any(not math.isfinite(value) for value in values):
        raise ValidationError("bootstrap differences contain a non-finite value")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not values:
        return {
            "n_images": 0,
            "mean": None,
            "ci_95": [None, None],
            "replicates": replicates,
            "seed": seed,
        }
    rng = random.Random(seed)
    n = len(values)
    boot_means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    ]
    return {
        "n_images": n,
        "mean": float(statistics.fmean(values)),
        "ci_95": [_percentile(boot_means, 0.025), _percentile(boot_means, 0.975)],
        "replicates": replicates,
        "seed": seed,
    }


def _bootstrap_vectors(
    vectors: Mapping[str, Sequence[float]], *, replicates: int, seed: int
) -> dict[str, dict[str, Any]]:
    if not vectors:
        return {}
    n = len(next(iter(vectors.values())))
    if any(len(values) != n for values in vectors.values()):
        raise ValueError("bootstrap vectors must have the same image denominator")
    if n == 0:
        return {
            name: paired_bootstrap([], replicates=replicates, seed=seed)
            for name in vectors
        }
    rng = random.Random(seed)
    bootstrap_values = {name: [] for name in vectors}
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        for name, values in vectors.items():
            bootstrap_values[name].append(sum(values[index] for index in indices) / n)
    result: dict[str, dict[str, Any]] = {}
    for name, values in vectors.items():
        raw = [float(value) for value in values]
        result[name] = {
            "n_images": n,
            "mean": float(statistics.fmean(raw)),
            "ci_95": [
                _percentile(bootstrap_values[name], 0.025),
                _percentile(bootstrap_values[name], 0.975),
            ],
            "replicates": replicates,
            "seed": seed,
        }
    return result


def _transition(left: float, right: float) -> str:
    left_correct = left > IOU_THRESHOLD
    right_correct = right > IOU_THRESHOLD
    if not right_correct and left_correct:
        return "repair"
    if right_correct and not left_correct:
        return "damage"
    if left_correct and right_correct:
        return "correct_to_correct"
    return "incorrect_to_incorrect"


def _population_summary(
    image_values: Mapping[str, Mapping[str, float]],
    *,
    arms: Sequence[str],
    contrasts: Sequence[tuple[str, str]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    image_ids = sorted(image_values)
    arm_means = {
        arm: _mean([float(image_values[image_id][arm]) for image_id in image_ids])
        for arm in arms
    }
    arm_accuracy = {
        arm: (
            None
            if not image_ids
            else sum(image_values[image_id][arm] > IOU_THRESHOLD for image_id in image_ids)
            / len(image_ids)
        )
        for arm in arms
    }
    difference_vectors: dict[str, list[float]] = {}
    raw_pairs: dict[str, list[tuple[float, float]]] = {}
    for left, right in contrasts:
        name = f"{left}-{right}"
        pairs = [
            (float(image_values[image_id][left]), float(image_values[image_id][right]))
            for image_id in image_ids
        ]
        raw_pairs[name] = pairs
        difference_vectors[name] = [left_value - right_value for left_value, right_value in pairs]
    boot = _bootstrap_vectors(
        difference_vectors, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    contrast_rows: dict[str, Any] = {}
    for name, pairs in raw_pairs.items():
        wins = ties = losses = 0
        transitions = Counter()
        for left_value, right_value in pairs:
            difference = left_value - right_value
            if difference > 1e-12:
                wins += 1
            elif difference < -1e-12:
                losses += 1
            else:
                ties += 1
            transitions[_transition(left_value, right_value)] += 1
        contrast_rows[name] = {
            **boot[name],
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_tie_loss_unit": "image",
            "repair_damage_threshold": f"strict IoU > {IOU_THRESHOLD}",
            "threshold_transitions": {
                label: transitions[label]
                for label in (
                    "repair",
                    "damage",
                    "correct_to_correct",
                    "incorrect_to_incorrect",
                )
            },
        }
    return {
        "n_images": len(image_ids),
        "image_ids": image_ids,
        "image_values": {
            image_id: {arm: float(values[arm]) for arm in arms}
            for image_id, values in sorted(image_values.items())
        },
        "arm_mean_iou": arm_means,
        "arm_accuracy_strict_iou_gt_0_5": arm_accuracy,
        "contrasts": contrast_rows,
    }


def _panel_summary(
    index: RecordIndex,
    manifest: Sequence[Mapping[str, Any]],
    panel: str,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    eligible_trajectories: set[TrajectoryKey] | None = None,
) -> dict[str, Any]:
    _, _, arms, _, _ = _panel_slots(panel)
    contrasts = A_CONTRASTS if panel.startswith("A_") else B_CONTRASTS
    trajectories = _trajectory_keys(manifest)
    if eligible_trajectories is not None:
        trajectories = [key for key in trajectories if key in eligible_trajectories]
    trajectory_rows = {
        key: _trajectory_panel(index, key, panel) for key in trajectories
    }
    samples_by_image: dict[str, list[TrajectoryKey]] = defaultdict(list)
    for key in trajectories:
        samples_by_image[key[1]].append(key)
    all_image_values: dict[str, dict[str, float]] = {}
    comparable_trajectories: dict[str, list[TrajectoryKey]] = defaultdict(list)
    for key in trajectories:
        if trajectory_rows[key]["complete"]:
            comparable_trajectories[key[1]].append(key)
    for image_id, image_trajectories in samples_by_image.items():
        all_image_values[image_id] = {
            arm: float(
                statistics.fmean(
                    [trajectory_rows[key]["arm_means"][arm] for key in image_trajectories]
                )
            )
            for arm in arms
        }
    comparable_image_values: dict[str, dict[str, float]] = {}
    for image_id, image_trajectories in comparable_trajectories.items():
        comparable_image_values[image_id] = {
            arm: float(
                statistics.fmean(
                    [trajectory_rows[key]["arm_means"][arm] for key in image_trajectories]
                )
            )
            for arm in arms
        }
    return {
        "definition": {
            "stage": _panel_slots(panel)[0],
            "mode": _panel_slots(panel)[1],
            "arms": list(arms),
            "draws_per_arm": len(_panel_slots(panel)[3]),
            "prefix_coordinates": _panel_slots(panel)[4][0],
            "aggregation": "mean draws -> trajectory -> image",
            "missing_and_failed_value_in_all_cohort": 0,
        },
        "coverage": _panel_coverage(index, manifest, panel),
        "all_cohort": _population_summary(
            all_image_values,
            arms=arms,
            contrasts=contrasts,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "comparable": _population_summary(
            comparable_image_values,
            arms=arms,
            contrasts=contrasts,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        ),
    }


def load_reviews(
    path: str | Path | None,
    manifest: Sequence[Mapping[str, Any]],
    index: RecordIndex,
) -> dict[str, Any]:
    """Read optional consensus-qualified entity reviews without gating metrics."""

    if path is None:
        return {
            "available": False,
            "path": None,
            "rows": 0,
            "covered_trajectories": 0,
            "coverage_fraction": 0.0,
            "status_counts": {},
            "consensus_qualified_only": None,
            "mechanism_claim_allowed": False,
            "correct_unique_trajectory_keys": [],
        }
    file_path = Path(path)
    try:
        stream = file_path.open("r", encoding="utf-8")
    except OSError as exc:
        _error(f"cannot open reviews: {exc}", path=path)
    manifest_by_sample = {str(row["sample_id"]): row for row in manifest}
    seen: set[TrajectoryKey] = set()
    statuses: Counter[str] = Counter()
    correct_unique: list[TrajectoryKey] = []
    with stream:
        for number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            row = _strict_json_loads(raw, path=path, line=number)
            if not isinstance(row, dict):
                _error("review row must be an object", path=path, line=number)
            sample_id = _id(row.get("sample_id"), "review sample_id", path=path, line=number)
            trajectory_index = _integer(
                row.get("trajectory_index"), "review trajectory_index", path=path, line=number
            )
            if trajectory_index not in range(EXPECTED_TRAJECTORIES):
                _error("review trajectory_index must be 0, 1, or 2", path=path, line=number)
            if sample_id not in manifest_by_sample:
                _error("review references unknown sample_id", path=path, line=number)
            image_id = row.get("image_id")
            if image_id is not None and str(image_id) != str(manifest_by_sample[sample_id]["image_id"]):
                _error("review image_id disagrees with manifest", path=path, line=number)
            status = row.get("status")
            if status not in ALLOWED_REVIEW_STATUSES:
                _error("invalid consensus review status", path=path, line=number)
            for flag in ("consensus_qualified", "consensus"):
                if flag in row and row[flag] is not True:
                    _error(f"review {flag}=false is not a consensus-qualified row", path=path, line=number)
            key = (sample_id, str(manifest_by_sample[sample_id]["image_id"]), trajectory_index)
            if key in seen:
                _error(f"duplicate review key {key!r}", path=path, line=number)
            seen.add(key)
            statuses[status] += 1
            if status == "correct_unique":
                correct_unique.append(key)
    expected = len(manifest) * EXPECTED_TRAJECTORIES
    return {
        "available": True,
        "path": str(file_path.resolve()),
        "rows": len(seen),
        "covered_trajectories": len(seen),
        "coverage_fraction": (len(seen) / expected if expected else 0.0),
        "status_counts": dict(sorted(statuses.items())),
        "consensus_qualified_only": True,
        "mechanism_claim_allowed": False,
        "used_as_selection_gate": False,
        "correct_unique_trajectory_keys": [list(key) for key in sorted(correct_unique)],
    }


def _length_bin(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length < 128:
        return "0-127"
    if length < 512:
        return "128-511"
    if length < 1024:
        return "512-1023"
    return "1024+"


def _stratification(manifest: Sequence[Mapping[str, Any]], index: RecordIndex) -> dict[str, Any]:
    by_group: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_by_sample = {str(row["sample_id"]): row for row in manifest}
    expected_keys = _trajectory_keys(manifest)
    for key in expected_keys:
        sample_id, image_id, trajectory_index = key
        row = index.trajectories.get(key)
        task_type = str(manifest_by_sample[sample_id]["task_type"])
        length = row.get("reasoning_length") if row else None
        if length is not None:
            length = int(length)
        group_key = (task_type, _length_bin(length))
        group = by_group.setdefault(
            group_key,
            {
                "task_type": task_type,
                "natural_length_bin": group_key[1],
                "expected_trajectories": 0,
                "trajectory_records": 0,
                "images": set(),
                "reasoning_lengths": [],
                "finish_reason_counts": Counter(),
                "entity_status_counts": Counter(),
            },
        )
        group["expected_trajectories"] += 1
        group["images"].add(image_id)
        if row is not None:
            group["trajectory_records"] += 1
            if length is not None:
                group["reasoning_lengths"].append(length)
            if row.get("finish_reason") is not None:
                group["finish_reason_counts"][row["finish_reason"]] += 1
            if row.get("entity_status") is not None:
                group["entity_status_counts"][row["entity_status"]] += 1
    output = {}
    for (task_type, length_bin), group in sorted(by_group.items()):
        key = f"{task_type}::{length_bin}"
        output[key] = {
            "task_type": task_type,
            "natural_length_bin": length_bin,
            "expected_trajectories": group["expected_trajectories"],
            "trajectory_records": group["trajectory_records"],
            "images": len(group["images"]),
            "mean_reasoning_length": _mean(group["reasoning_lengths"]),
            "finish_reason_counts": dict(sorted(group["finish_reason_counts"].items())),
            "entity_status_counts": dict(sorted(group["entity_status_counts"].items())),
            "selection_gate": None,
        }
    return {
        "length_bins": ["0-127", "128-511", "512-1023", "1024+", "unknown"],
        "groups": output,
        "selection_gate": "none; strata are descriptive only",
    }


def _trajectory_coverage(
    manifest: Sequence[Mapping[str, Any]], index: RecordIndex
) -> dict[str, Any]:
    expected = _trajectory_keys(manifest)
    present = [key for key in expected if key in index.trajectories]
    done = [key for key in expected if key in index.trajectory_done]
    complete_done = [
        key
        for key in done
        if index.trajectory_done[key].get("inference_complete") is True
    ]
    finish = Counter(
        index.trajectories[key].get("finish_reason")
        for key in present
        if index.trajectories[key].get("finish_reason") is not None
    )
    entity = Counter(
        index.trajectories[key].get("entity_status")
        for key in present
        if index.trajectories[key].get("entity_status") is not None
    )
    return {
        "expected_trajectories": len(expected),
        "trajectory_records": len(present),
        "missing_trajectory_records": len(expected) - len(present),
        "trajectory_done_records": len(done),
        "missing_trajectory_done_records": len(expected) - len(done),
        "inference_complete_true": len(complete_done),
        "trajectory_failure_records": len(index.trajectory_failures),
        "finish_reason_counts": dict(sorted(finish.items())),
        "entity_status_counts": dict(sorted(entity.items())),
    }


def _cohort_coverage(manifest: Sequence[Mapping[str, Any]], index: RecordIndex) -> dict[str, Any]:
    by_cohort: dict[str, dict[str, Any]] = {}
    for cohort in sorted({str(row["cohort"]) for row in manifest}):
        rows = [row for row in manifest if str(row["cohort"]) == cohort]
        samples = {str(row["sample_id"]) for row in rows}
        images = {str(row["image_id"]) for row in rows}
        expected = [key for key in _trajectory_keys(rows)]
        by_cohort[cohort] = {
            "samples": len(samples),
            "images": len(images),
            "expected_trajectories": len(expected),
            "trajectory_records": sum(key in index.trajectories for key in expected),
            "trajectory_done_records": sum(key in index.trajectory_done for key in expected),
            "protocol_expected_images": (
                30 if cohort == "pilot" else 200 if cohort == "formal" else None
            ),
        }
    return by_cohort


def _load_manifest_input(manifest: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        return load_manifest(manifest)
    return _validated_manifest_rows(manifest, source="manifest")


def analyze(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    records: str | Path | Sequence[str | Path],
    reviews: str | Path | None = None,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate records and return the complete JSON-serializable summary."""

    manifest_rows = _load_manifest_input(manifest)
    index = ingest_records(manifest_rows, records)
    review_summary = load_reviews(reviews, manifest_rows, index)
    unique_images = sorted({str(row["image_id"]) for row in manifest_rows})
    panels = {
        "A": {
            "random": _panel_summary(
                index,
                manifest_rows,
                "A_random",
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            ),
            "greedy": _panel_summary(
                index,
                manifest_rows,
                "A_greedy",
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            ),
        },
        "B": {
            "prefix1": _panel_summary(
                index,
                manifest_rows,
                "B_prefix1",
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            ),
            "prefix2": _panel_summary(
                index,
                manifest_rows,
                "B_prefix2",
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            ),
        },
    }
    panel_coverage = {
        f"{stage}_{name}": panels[stage][name]["coverage"]
        for stage in panels
        for name in panels[stage]
    }
    trajectory_coverage = _trajectory_coverage(manifest_rows, index)
    collection_partial = (
        trajectory_coverage["missing_trajectory_records"] > 0
        or trajectory_coverage["missing_trajectory_done_records"] > 0
        or bool(index.truncated_tail_files)
    )
    workflow_has_missing_branches = any(
        values["missing_cells"] > 0 for values in panel_coverage.values()
    )
    correct_unique_keys = {
        (str(key[0]), str(key[1]), int(key[2]))
        for key in review_summary.get("correct_unique_trajectory_keys", [])
    }
    identity_panels: dict[str, Any] | None = None
    if reviews is not None:
        identity_panels = {
            "definition": (
                "Post-hoc descriptive subset fixed only by blinded entity reviews; "
                "the preregistered all-cohort contrasts remain primary."
            ),
            "n_correct_unique_trajectories": len(correct_unique_keys),
            "A": {
                "random": _panel_summary(
                    index, manifest_rows, "A_random",
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_replicates=bootstrap_replicates,
                    eligible_trajectories=correct_unique_keys,
                ),
                "greedy": _panel_summary(
                    index, manifest_rows, "A_greedy",
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_replicates=bootstrap_replicates,
                    eligible_trajectories=correct_unique_keys,
                ),
            },
            "B": {
                "prefix1": _panel_summary(
                    index, manifest_rows, "B_prefix1",
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_replicates=bootstrap_replicates,
                    eligible_trajectories=correct_unique_keys,
                ),
                "prefix2": _panel_summary(
                    index, manifest_rows, "B_prefix2",
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_replicates=bootstrap_replicates,
                    eligible_trajectories=correct_unique_keys,
                ),
            },
        }
    return {
        "schema_version": 1,
        "protocol": {
            "model": "Qwen3.5-0.8B",
            "frozen_weights": True,
            "pilot_images": 30,
            "formal_images": 200,
            "trajectories_per_sample": EXPECTED_TRAJECTORIES,
            "A": {
                "arms": list(A_ARMS),
                "random_draws": [0, 1, 2, 3],
                "greedy_draws": [0],
                "contrasts": [f"{left}-{right}" for left, right in A_CONTRASTS],
            },
            "B": {
                "source": "A/L/random/draw0 raw token prefix",
                "prefix_coordinates": [1, 2],
                "arms": list(B_ARMS),
                "random_draws": [0, 1, 2, 3],
                "contrasts": [f"{left}-{right}" for left, right in B_CONTRASTS],
            },
            "aggregation": "mean draws -> trajectory -> image",
            "all_cohort_missing_and_failed_value": 0,
            "accuracy": "strict IoU > 0.5",
            "bootstrap": {
                "unit": "image",
                "paired": True,
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "interval": "percentile 95%",
            },
        },
        "inputs": {
            "manifest": str(manifest) if isinstance(manifest, (str, Path)) else "<in-memory>",
            "record_files": index.files,
            "records_scanned": "trajectory JSON/JSONL only; distributions excluded",
            "reviews": str(reviews) if reviews is not None else None,
        },
        "coverage": {
            "manifest_samples": len(manifest_rows),
            "manifest_images": len(unique_images),
            "trajectory": trajectory_coverage,
            "panels": panel_coverage,
            "cohorts": _cohort_coverage(manifest_rows, index),
            "collection_complete": not collection_partial,
            "partial": bool(collection_partial),
            "workflow_has_missing_or_unavailable_branches": workflow_has_missing_branches,
            "truncated_tail_files": index.truncated_tail_files,
        },
        "aggregates": panels,
        "identity_confirmed_subset": identity_panels,
        "reviews": review_summary,
        "stratification": _stratification(manifest_rows, index),
        "limitations": {
            "dataset_gt_conversion": "out of scope; manifest GT is required normalized xyxy",
            "entity_reviews_as_gate": False,
            "mechanism_claim": False,
            "mechanism_claim_reason": (
                "geometry summaries do not establish a mechanism; missing/incomplete blind reviews "
                "must remain explicit"
            ),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(summary: Mapping[str, Any]) -> str:
    """Render the summary as a Chinese audit-oriented Markdown report."""

    coverage = summary["coverage"]
    status = "PARTIAL / 采集未完成" if coverage["partial"] else "COMPLETE / 预定轨迹均已结案"
    reviews = summary["reviews"]
    lines = [
        "# MM-GCoT 独立统计报告",
        "",
        f"统计覆盖状态：**{status}**。本报告只反映输入 records 中已经记录的轨迹和输出。"
        "模型自然推理未结束、目标未解析或精确逗号前缀不存在属于实验观察，"
        "不等同于采集进程中断；它们在全队列结果中按协议保留。",
        "",
        "## 协议与口径",
        "",
        "- 模型：Qwen3.5-0.8B，frozen weights；协议目标为 pilot 30 图像、formal 200 图像，每图像 3 条 trajectory。",
        "- 聚合：先对 random draws 求均值，再对 trajectory 求均值，最后对 image 求均值；all-cohort 面板对缺失和 failed output 填 0。",
        "- Acc：严格使用 IoU > 0.5；greedy 与 random 分开统计。",
        f"- paired bootstrap：以 image 为配对单位，{summary['protocol']['bootstrap']['replicates']} 次，seed={summary['protocol']['bootstrap']['seed']}，percentile 95% CI。",
        "- A 对比：E-L、E-R、R-L、L-L0；B 对每个 prefix_coordinates=1/2 对比 E-L、E-R、R-L。",
        "",
        "## 覆盖与分母",
        "",
        f"- manifest：{coverage['manifest_samples']} samples，{coverage['manifest_images']} images。",
        f"- trajectory：预期 {coverage['trajectory']['expected_trajectories']}，记录 {coverage['trajectory']['trajectory_records']}，缺失 {coverage['trajectory']['missing_trajectory_records']}；trajectory_done 缺失 {coverage['trajectory']['missing_trajectory_done_records']}。",
        f"- trajectory failure 记录：{coverage['trajectory']['trajectory_failure_records']}；缺失记录和失败输出没有从报告中隐去。",
        "",
        "| 面板 | 预期 cells | 已有 cells | valid | invalid | failed | missing/prefix unavailable | 完整 trajectory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, panel in coverage["panels"].items():
        lines.append(
            f"| {name} | {panel['expected_cells']} | {panel['present_cells']} | "
            f"{panel['valid_bbox_cells']} | {panel['invalid_bbox_cells']} | {panel['failed_output_cells']} | "
            f"{panel['missing_cells']} | {panel['complete_trajectories']} |"
        )
    lines.extend(
        [
            "",
            "comparable 分母只保留该面板所有所需 arm/draw 已有记录的 trajectory；B 还要求 source A/L/random/draw0 可用且没有 prefix_unavailable。comparable 先在 image 内平均符合条件的 trajectory，再做 image 等权平均。",
            "",
            "## 几何差异",
            "",
        ]
    )
    for stage, modes in summary["aggregates"].items():
        for mode, panel in modes.items():
            lines.extend([f"### {stage} / {mode}", ""])
            lines.append(
                "| 分母 | 图像数 | 对比 | ΔIoU 均值 | 95% CI | wins/ties/losses | repair | damage |"
            )
            lines.append("|---|---:|---|---:|---|---:|---:|---:|")
            for denominator in ("all_cohort", "comparable"):
                population = panel[denominator]
                for contrast, row in population["contrasts"].items():
                    transition = row["threshold_transitions"]
                    lines.append(
                        f"| {denominator} | {population['n_images']} | {contrast} | "
                        f"{_fmt(row['mean'])} | [{_fmt(row['ci_95'][0])}, {_fmt(row['ci_95'][1])}] | "
                        f"{row['wins']}/{row['ties']}/{row['losses']} | {transition['repair']} | {transition['damage']} |"
                    )
            lines.append("")
            lines.append(
                "arm mean IoU（all_cohort / comparable）："
                + "; ".join(
                    f"{arm}={_fmt(panel['all_cohort']['arm_mean_iou'].get(arm))} / "
                    f"{_fmt(panel['comparable']['arm_mean_iou'].get(arm))}"
                    for arm in panel["definition"]["arms"]
                )
                + "。"
            )
            lines.append("")
    lines.extend(["## task_type 与自然长度分层", ""])
    lines.append("分层仅作描述，没有按 task、长度、实体状态或 IoU 结果设置选择 gate。")
    lines.append("")
    lines.append("| task_type | natural length bin | expected trajectories | trajectory records | images | mean length |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for group in summary["stratification"]["groups"].values():
        lines.append(
            f"| {group['task_type']} | {group['natural_length_bin']} | {group['expected_trajectories']} | "
            f"{group['trajectory_records']} | {group['images']} | {_fmt(group['mean_reasoning_length'], 1)} |"
        )
    lines.extend(["", "## 实体盲审与结论边界", ""])
    if not reviews["available"]:
        lines.append(
            "实体 blind reviews：**缺失**。因此不能声称实体保持、修复/损伤的语义机制；本报告只给出几何 IoU、覆盖、分母和阈值 transition 的描述性统计。"
        )
    else:
        lines.append(
            f"实体 blind reviews：已提供 {reviews['rows']} 条 consensus-qualified 记录，覆盖率 {reviews['coverage_fraction']:.1%}；status 只作描述，不作为统计选择 gate。"
        )
        identity = summary.get("identity_confirmed_subset")
        if identity is not None:
            lines.append(
                f"其中 correct_unique trajectory 为 {identity['n_correct_unique_trajectories']} 条。"
                "下表是预先规定的身份确认子集描述；全队列结果仍为主结果。"
            )
            lines.extend(["", "| 面板 | 图像数 | E-L ΔIoU | 95% CI |", "|---|---:|---:|---|"])
            for label, panel in (
                ("A/random", identity["A"]["random"]),
                ("A/greedy", identity["A"]["greedy"]),
                ("B/prefix1", identity["B"]["prefix1"]),
                ("B/prefix2", identity["B"]["prefix2"]),
            ):
                row = panel["all_cohort"]["contrasts"]["E-L"]
                lines.append(
                    f"| {label} | {panel['all_cohort']['n_images']} | {_fmt(row['mean'])} | "
                    f"[{_fmt(row['ci_95'][0])}, {_fmt(row['ci_95'][1])}] |"
                )
        lines.append(
            "即使存在盲审，几何差异与 review status 也不单独构成机制证据；本模块不输出机制性结论。"
        )
    lines.extend(
        [
            "",
            "## 复核提示",
            "",
            "summary.json 保留 image-level 数值、all_cohort 与 comparable 两套分母、每个面板 coverage 和审查状态。原始 bbox 不做裁剪或修正；manifest 的数据集 GT 转换不属于本模块。",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_rows(summary: Mapping[str, Any]) -> list[tuple[str, str, float | None, float | None, int]]:
    rows: list[tuple[str, str, float | None, float | None, int]] = []
    for stage, modes in summary["aggregates"].items():
        for mode, panel in modes.items():
            for denominator in ("all_cohort", "comparable"):
                population = panel[denominator]
                for contrast, values in population["contrasts"].items():
                    ci = values["ci_95"]
                    label = f"{stage}-{mode} {denominator} {contrast}"
                    rows.append((label, contrast, values["mean"], ci[0] if ci else None, population["n_images"]))
                    # The upper CI is recovered from the next helper below by
                    # replacing the tuple in a local display-only path.
                    rows[-1] = (label, contrast, values["mean"], (ci[0], ci[1]) if ci else None, population["n_images"])  # type: ignore[assignment]
    return rows


def write_plot(summary: Mapping[str, Any], output_dir: str | Path) -> str:
    """Write a forest plot; fall back to SVG when matplotlib is unavailable."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, float, float, float, int]] = []
    for stage, modes in summary["aggregates"].items():
        for mode, panel in modes.items():
            for denominator in ("all_cohort", "comparable"):
                population = panel[denominator]
                for contrast, values in population["contrasts"].items():
                    if values["mean"] is None or values["ci_95"][0] is None:
                        continue
                    records.append(
                        (
                            f"{stage}/{mode} {denominator} {contrast}",
                            float(values["mean"]),
                            float(values["ci_95"][0]),
                            float(values["ci_95"][1]),
                            int(population["n_images"]),
                        )
                    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        height = max(4.0, 0.34 * max(1, len(records)) + 1.5)
        figure, axis = plt.subplots(figsize=(11.5, height), constrained_layout=True)
        if records:
            labels = [f"{label} (n={n})" for label, _, _, _, n in records]
            y = list(range(len(records)))
            means = [mean for _, mean, _, _, _ in records]
            lower = [mean - lo for _, mean, lo, _, _ in records]
            upper = [hi - mean for _, mean, _, hi, _ in records]
            axis.errorbar(means, y, xerr=[lower, upper], fmt="o", color="#1f4e79", ecolor="#1f4e79", capsize=3)
            axis.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
            axis.set_yticks(y)
            axis.set_yticklabels(labels, fontsize=8)
            axis.set_xlabel("paired image-level ΔIoU (left − right), 95% percentile CI")
            axis.set_ylabel("comparison and denominator")
            axis.grid(axis="x", alpha=0.25)
            axis.invert_yaxis()
        else:
            axis.text(0.5, 0.5, "No finite contrast estimates", ha="center", va="center")
            axis.set_axis_off()
        figure.suptitle("MM-GCoT paired image-level differences")
        path = output / "paired-differences.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return str(path)
    except (ImportError, ModuleNotFoundError):
        path = output / "paired-differences.svg"
        width = 1100
        row_height = 26
        height = max(120, 70 + row_height * max(1, len(records)))
        text_rows = []
        for i, (label, mean, low, high, n) in enumerate(records):
            y = 45 + i * row_height
            text_rows.append(
                f'<text x="10" y="{y}" font-size="12">{label} (n={n}) Δ={mean:.4f} [{low:.4f}, {high:.4f}]</text>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="white"/>'
            '<text x="10" y="22" font-size="16">MM-GCoT paired image-level differences</text>'
            + "".join(text_rows)
            + "</svg>"
        )
        path.write_text(svg, encoding="utf-8")
        return str(path)


def write_outputs(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write summary.json, REPORT.md, and the paired-differences plot."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    report_path = output / "REPORT.md"
    summary_value = json.loads(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    plot_path = write_plot(summary_value, output)
    summary_value["outputs"] = {
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "paired_differences_plot": str(plot_path),
    }
    summary_path.write_text(
        json.dumps(summary_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary_value), encoding="utf-8")
    return {key: str(value) for key, value in summary_value["outputs"].items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--records",
        required=True,
        nargs="+",
        help="records directory, run directory, JSONL/JSON file, or recursive glob",
    )
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = analyze(args.manifest, args.records, args.reviews)
    outputs = write_outputs(summary, args.output_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
