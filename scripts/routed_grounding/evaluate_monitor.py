#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only evaluation and monitoring helpers for routed grounding.

The evaluator intentionally has no image, torch, or pandas dependency.  A
manifest supplies the image instances and target annotation, while a
prediction JSONL supplies one response per observed sample.  The router is
used for response parsing and instance matching; target IoU is computed
against the manifest target so a prediction that matches a distractor cannot
accidentally receive the distractor's IoU.

The module can be used as a library or as a small command-line program::

    python scripts/routed_grounding/evaluate_monitor.py \
        --manifest path/to/val.jsonl \
        --predictions path/to/predictions.jsonl \
        --output monitor_results.csv

CSV rows are appended by default.  ``initial_route=all`` denotes the overall
row; other rows are grouped by the optional ``initial_route`` prediction
field.  Missing denominators are written as empty CSV cells and represented
by ``None`` in Python, which keeps missing probes and empty groups distinct
from measured zero rates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import numbers
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_router_path = Path(__file__).resolve().parents[2] / "verl/experimental/routed_grounding/router.py"
if _router_path.is_file():
    # Importing ``verl.experimental`` through the package root also imports
    # optional distributed dependencies such as Ray.  Prefer loading the
    # dependency-free router directly so this evaluator remains genuinely
    # CPU-only even in a fully provisioned training environment.
    _router_spec = importlib.util.spec_from_file_location("_routed_grounding_cpu_router", _router_path)
    if _router_spec is None or _router_spec.loader is None:
        raise ImportError(f"could not load CPU router from {_router_path}")
    _router_module = importlib.util.module_from_spec(_router_spec)
    sys.modules[_router_spec.name] = _router_module
    _router_spec.loader.exec_module(_router_module)
    EXPECTED_PROBE_COUNT = _router_module.EXPECTED_PROBE_COUNT
    Instance = _router_module.Instance
    MatchResult = _router_module.MatchResult
    match_prediction = _router_module.match_prediction
    parse_response = _router_module.parse_response
    route_response = _router_module.route_response
    xywh_to_xyxy = _router_module.xywh_to_xyxy
    xyxy_iou = _router_module.xyxy_iou
else:
    from verl.experimental.routed_grounding.router import (
        EXPECTED_PROBE_COUNT,
        Instance,
        MatchResult,
        match_prediction,
        parse_response,
        route_response,
        xywh_to_xyxy,
        xyxy_iou,
    )


OVERALL_ROUTE = "all"
UNKNOWN_ROUTE = "unknown"
STATUS_NAMES = ("matched", "background", "ambiguous", "invalid")
ROUTE_NAMES = (
    "correct",
    "localization",
    "localization_recoverable",
    "referent_unrecoverable",
    "uncertain",
)

# These are the stable, human-readable columns used by monitor_results.csv.
# The punctuation in the accuracy and IoU names is intentional: it mirrors
# the experiment protocol and keeps the resulting table easy to inspect.
MONITOR_RESULT_COLUMNS = (
    "update",
    "initial_route",
    "sample_count",
    "Acc@0.5",
    "Acc@0.75",
    "Acc@0.9",
    "mean IoU",
    "wrong_instance_rate",
    "target_match_mean_IoU",
    "bbox_valid_rate",
    "probe_0_target_hit_rate",
    "probe_1_target_hit_rate",
    "probe_2_target_hit_rate",
    "probe_3_target_hit_rate",
    "matched_count",
    "background_count",
    "ambiguous_count",
    "invalid_count",
    "wrong_instance_count",
    "target_match_count",
    "matched_wrong_instance_rate",
    "probe_0_calls",
    "probe_1_calls",
    "probe_2_calls",
    "probe_3_calls",
)

# Per-probe status columns make malformed probes auditable instead of
# collapsing invalid, background, and ambiguous probes into the same zero
# target-hit rate.
MONITOR_RESULT_COLUMNS = MONITOR_RESULT_COLUMNS + tuple(
    f"probe_{index}_{status}_count" for index in range(EXPECTED_PROBE_COUNT) for status in STATUS_NAMES
)

# Update logs are deliberately a separate schema from per-sample evaluation
# output.  Nested route maps make it possible to add a route without changing
# this top-level wire format.
UPDATE_LOG_FIELDS = (
    "update",
    "route_counts",
    "route_proportions",
    "probe_calls",
    "teacher_acceptance",
    "localization_loss",
    "referent_loss",
    "standard_loss",
    "invalid_rate",
    "response_length",
    "training_time",
)
UPDATE_LOG_SCHEMA = {field: "required" for field in UPDATE_LOG_FIELDS}


def _json_scalar_key(value: Any) -> str:
    """Return the manifest's stable scalar-id comparison key.

    COCO ids are usually integers, but predictions produced by a JSONL
    pipeline occasionally contain their string representation.  Numeric
    integer spellings therefore compare as the same image id, while bools
    remain distinct from integers.
    """

    if value is None:
        return "null:"
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return f"number:{int(number)}"
        return f"number:{number!r}"
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except (TypeError, ValueError, OverflowError):
            number = math.nan
        if math.isfinite(number) and number.is_integer():
            return f"number:{int(number)}"
    return f"string:{value}"


def _same_id(left: Any, right: Any) -> bool:
    return _json_scalar_key(left) == _json_scalar_key(right)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records and fail with a useful line number."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {source} line {line_number}: {error.msg}") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL record in {source} line {line_number} must be an object")
            records.append(dict(value))
    return records


load_jsonl = read_jsonl


def _as_records(value: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, str | Path):
        return read_jsonl(value)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        records.append(dict(item))
    return records


def _instances_for_record(record: Mapping[str, Any]) -> tuple[Instance | Mapping[str, Any], ...]:
    """Return router-compatible instances, including a target-only fallback.

    The frozen manifest always carries ``instances``.  The fallback is useful
    for small CPU smoke tests and still leaves distractor matching impossible
    rather than inventing distractor annotations.
    """

    values = record.get("instances", ())
    if isinstance(values, Mapping):
        # A mapping is accepted as one annotation when it has a bbox; a map
        # keyed by annotation id is also a convenient hand-written fixture.
        if "bbox" in values or "bbox_xywh" in values:
            values = (values,)
        else:
            values = tuple(
                dict(item, ann_id=item.get("ann_id", key)) for key, item in values.items() if isinstance(item, Mapping)
            )
    elif isinstance(values, str | bytes | bytearray):
        raise ValueError("manifest instances must be a sequence of mappings")
    else:
        try:
            values = tuple(values)
        except TypeError as error:
            raise ValueError("manifest instances must be a sequence of mappings") from error

    if values:
        return tuple(values)

    # Keep a minimal manifest fixture useful without changing the production
    # manifest contract.  The target annotation is sufficient for target
    # accuracy, while match_prediction will still report background when no
    # distractor is present and a malformed target box is never fabricated.
    target_bbox = record.get("target_bbox", record.get("target_bbox_xywh"))
    target_id = record.get("target_ann_id", record.get("target_id"))
    if target_bbox is None or target_id is None:
        return ()
    return (
        {
            "ann_id": target_id,
            "bbox": target_bbox,
            "segmentation": record.get("target_mask"),
            "image_width": record.get("image_width"),
            "image_height": record.get("image_height"),
        },
    )


def _find_target_instance(
    record: Mapping[str, Any], instances: Sequence[Instance | Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | Instance | None, Any]:
    target_id = record.get("target_ann_id", record.get("target_id"))
    if target_id is None:
        return None, None
    for value in instances:
        if isinstance(value, Instance):
            if _same_id(value.ann_id, target_id):
                return value, target_id
            continue
        if isinstance(value, Mapping):
            candidate_id = value.get("ann_id", value.get("id"))
            if _same_id(candidate_id, target_id):
                return value, target_id
    return None, target_id


def _dimension(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _target_xyxy_and_dimensions(
    record: Mapping[str, Any], target: Mapping[str, Any] | Instance | None
) -> tuple[tuple[float, float, float, float] | None, float | None, float | None]:
    width = _dimension(record.get("image_width"))
    height = _dimension(record.get("image_height"))
    bbox: Any = record.get("target_bbox", record.get("target_bbox_xywh"))
    bbox_is_normalized_xyxy = False
    if isinstance(target, Instance):
        bbox = target.bbox_xywh
        width = target.image_width or width
        height = target.image_height or height
    elif isinstance(target, Mapping):
        bbox = target.get("bbox_xywh", target.get("bbox", bbox))
        width = _dimension(target.get("image_width", target.get("width"))) or width
        height = _dimension(target.get("image_height", target.get("height"))) or height
    if bbox is None:
        bbox = record.get("target_bbox_normalized")
        bbox_is_normalized_xyxy = bbox is not None
    if bbox is None:
        return None, width, height
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError):
        return None, width, height
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        return None, width, height
    try:
        if bbox_is_normalized_xyxy:
            target_xyxy = coordinates
        else:
            target_xyxy = xywh_to_xyxy(coordinates)
    except (TypeError, ValueError):
        return None, width, height
    return target_xyxy, width, height


def _normalised_to_source_bbox(
    bbox: Sequence[float], width: float | None, height: float | None
) -> tuple[float, float, float, float]:
    if width is None or height is None:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    )


def _target_iou(
    bbox: Sequence[float],
    record: Mapping[str, Any],
    target: Mapping[str, Any] | Instance | None,
) -> float | None:
    target_xyxy, width, height = _target_xyxy_and_dimensions(record, target)
    if target_xyxy is None:
        return None
    try:
        return xyxy_iou(_normalised_to_source_bbox(bbox, width, height), target_xyxy)
    except (TypeError, ValueError):
        return None


def _response_value(value: Any) -> Any:
    """Unwrap common serialized response receipts before parsing/matching."""

    if isinstance(value, Mapping):
        if "response" in value:
            return _response_value(value["response"])
        if "raw_response" in value:
            return value["raw_response"]
        if "full_response" in value:
            return _response_value(value["full_response"])
    return value


def _match_value(value: Any, instances: Sequence[Instance | Mapping[str, Any]]) -> MatchResult:
    value = _response_value(value)
    if isinstance(value, Mapping) and "bbox" in value and "response" not in value:
        try:
            return match_prediction(value["bbox"], instances)
        except (TypeError, ValueError) as error:
            # match_prediction is fail-closed for malformed boxes, but keep a
            # defensive path for alternate router implementations.
            return _invalid_match(str(error))
    parsed = parse_response(value)
    if not parsed.format_valid or parsed.bbox is None:
        return _invalid_match(parsed.error or "invalid response")
    return match_prediction(parsed.bbox, instances)


def _invalid_match(error: str) -> MatchResult:
    # MatchResult is public and constructing it here avoids importing the
    # router's private helper while retaining the required invalid category.
    return MatchResult(
        status="invalid",
        ann_id=None,
        iou=0.0,
        second_iou=0.0,
        margin=0.0,
        method="invalid",
        center=None,
        candidate_ious=(),
        error=error,
    )


def _probe_values(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Extract exactly the four fixed reasoning probes when present."""

    for field in (
        "probes",
        "probe_responses",
        "reasoning_probes",
        "reasoning_probe_responses",
        "probe_receipts",
    ):
        if field not in record or record[field] is None:
            continue
        value = record[field]
        if isinstance(value, Mapping):
            items = list(value.items())

            def key(item: tuple[Any, Any]) -> tuple[int, str]:
                text = str(item[0])
                digits = "".join(character for character in text if character.isdigit())
                return (int(digits) if digits else 10**9, text)

            items.sort(key=key)
            return tuple(item[1] for item in items[:EXPECTED_PROBE_COUNT])
        if isinstance(value, str | bytes | bytearray):
            return (value,)
        try:
            return tuple(value)[:EXPECTED_PROBE_COUNT]
        except TypeError:
            return (value,)

    receipt = record.get("routed_grounding_receipt")
    if isinstance(receipt, Mapping):
        nested = _probe_values(receipt)
        if nested:
            return nested

    # Support both zero-based and one-based flat field spellings.  A present
    # probe_0 (or reasoning_probe_0) unambiguously selects zero-based order.
    prefixes = ("probe", "reasoning_probe", "probe_reasoning")
    keys = set(record)
    for prefix in prefixes:
        # Only an explicit zero index identifies zero-based spelling.  Merely
        # seeing probe_1 is also compatible with the common one-based form.
        if f"{prefix}_0" in keys or f"{prefix}0" in keys:
            values = []
            for index in range(EXPECTED_PROBE_COUNT):
                name = f"{prefix}_{index}" if f"{prefix}_{index}" in record else f"{prefix}{index}"
                if name in record:
                    values.append(record[name])
            return tuple(values)
        one_based = [f"{prefix}_{index}" for index in range(1, EXPECTED_PROBE_COUNT + 1)]
        compact_one_based = [f"{prefix}{index}" for index in range(1, EXPECTED_PROBE_COUNT + 1)]
        if any(name in keys for name in (*one_based, *compact_one_based)):
            values = []
            for index in range(1, EXPECTED_PROBE_COUNT + 1):
                name = f"{prefix}_{index}" if f"{prefix}_{index}" in record else f"{prefix}{index}"
                if name in record:
                    values.append(record[name])
            return tuple(values)
    return ()


def _empty_status_counts() -> dict[str, int]:
    return {status: 0 for status in STATUS_NAMES}


def _sample_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    if "image_id" not in record:
        raise ValueError("manifest/prediction record is missing image_id")
    if "response" not in record:
        raise ValueError(f"prediction for image_id={record['image_id']!r} is missing response")

    instances = _instances_for_record(record)
    target, target_id = _find_target_instance(record, instances)
    response = record["response"]
    parsed = parse_response(_response_value(response))
    if parsed.format_valid and parsed.bbox is not None:
        original_match = match_prediction(parsed.bbox, instances)
        target_iou = _target_iou(parsed.bbox, record, target)
    else:
        original_match = _invalid_match(parsed.error or "invalid response")
        target_iou = None

    probe_matches = tuple(_match_value(value, instances) for value in _probe_values(record))
    target_match = original_match.status == "matched" and _same_id(original_match.ann_id, target_id)
    wrong_instance = original_match.status == "matched" and not target_match
    return {
        "image_id": record["image_id"],
        "update": record.get("update"),
        "initial_route": record.get("initial_route", UNKNOWN_ROUTE) or UNKNOWN_ROUTE,
        "status": original_match.status,
        "target_iou": target_iou,
        "target_match": target_match,
        "wrong_instance": wrong_instance,
        "parsed_valid": bool(parsed.format_valid and parsed.bbox is not None),
        "probe_matches": probe_matches,
    }


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    values = tuple(float(value) for value in values if math.isfinite(float(value)))
    return sum(values) / len(values) if values else None


def compute_metrics(
    entries: Iterable[Mapping[str, Any]],
    *,
    update: Any = 0,
    initial_route: str = OVERALL_ROUTE,
) -> dict[str, Any]:
    """Compute one overall or ``initial_route``-grouped metrics row.

    Accuracy and ``mean IoU`` use all observed prediction records; invalid
    responses contribute zero IoU. ``target_match_mean_IoU`` is conditional
    on a unique router match to the target.
    """

    records = tuple(entries)
    count = len(records)
    status_counts = _empty_status_counts()
    target_ious = [record["target_iou"] for record in records if record.get("target_iou") is not None]
    target_match_ious = [
        record["target_iou"]
        for record in records
        if record.get("target_match") and record.get("target_iou") is not None
    ]
    wrong_count = 0
    target_match_count = 0
    valid_count = 0
    probe_hit_counts = [0] * EXPECTED_PROBE_COUNT
    probe_call_counts = [0] * EXPECTED_PROBE_COUNT
    probe_status_counts = [_empty_status_counts() for _ in range(EXPECTED_PROBE_COUNT)]

    for record in records:
        status = str(record.get("status", "invalid"))
        if status not in status_counts:
            status = "invalid"
        status_counts[status] += 1
        valid_count += int(bool(record.get("parsed_valid")))
        wrong_count += int(bool(record.get("wrong_instance")))
        target_match_count += int(bool(record.get("target_match")))
        matches = tuple(record.get("probe_matches", ()))
        has_explicit_probe_flags = "probe_target_hits" in record
        for index in range(min(EXPECTED_PROBE_COUNT, len(matches))):
            probe_call_counts[index] += 1
            match = matches[index]
            if isinstance(match, MatchResult):
                probe_status = match.status
            elif isinstance(match, Mapping):
                probe_status = str(match.get("status", "invalid"))
            else:
                probe_status = "invalid"
            if probe_status not in STATUS_NAMES:
                probe_status = "invalid"
            probe_status_counts[index][probe_status] += 1
            if not has_explicit_probe_flags and isinstance(match, MatchResult) and match.status == "matched":
                target_id = record.get("target_ann_id")
                # Evaluation entries created by _sample_entry do not expose
                # target_ann_id; their probe matches already contain the id,
                # so a target hit is set below by _annotate_entries.  This
                # branch remains useful for callers supplying explicit entries.
                if target_id is not None and _same_id(match.ann_id, target_id):
                    probe_hit_counts[index] += 1
            elif isinstance(match, Mapping):
                if match.get("target_hit") is True:
                    probe_hit_counts[index] += 1

    # _sample_entry stores target-hit flags separately to avoid depending on
    # any annotation id serialization in MatchResult.  Count those flags when
    # present; malformed/background/ambiguous probes remain zero hits.
    for record in records:
        flags = tuple(record.get("probe_target_hits", ()))
        for index in range(min(EXPECTED_PROBE_COUNT, len(flags))):
            probe_hit_counts[index] += int(bool(flags[index]))

    metrics: dict[str, Any] = {
        "update": update,
        "initial_route": initial_route,
        "sample_count": count,
        "Acc@0.5": _rate(sum((value is not None and value >= 0.5) for value in target_ious), count),
        "Acc@0.75": _rate(sum((value is not None and value >= 0.75) for value in target_ious), count),
        "Acc@0.9": _rate(sum((value is not None and value >= 0.9) for value in target_ious), count),
        "mean IoU": _rate(sum(target_ious), count),
        # This is measured over all observed predictions.  The companion
        # ``matched_wrong_instance_rate`` below is conditional on matched
        # instances for callers that need that denominator.
        "wrong_instance_rate": _rate(wrong_count, count),
        "matched_wrong_instance_rate": _rate(wrong_count, status_counts["matched"]),
        "target_match_mean_IoU": _mean(target_match_ious),
        "bbox_valid_rate": _rate(valid_count, count),
        "matched_count": status_counts["matched"],
        "background_count": status_counts["background"],
        "ambiguous_count": status_counts["ambiguous"],
        "invalid_count": status_counts["invalid"],
        "wrong_instance_count": wrong_count,
        "target_match_count": target_match_count,
        "probe_calls": sum(probe_call_counts),
        "probe_0_calls": probe_call_counts[0],
        "probe_1_calls": probe_call_counts[1],
        "probe_2_calls": probe_call_counts[2],
        "probe_3_calls": probe_call_counts[3],
        "probe_0_target_hit_rate": _rate(probe_hit_counts[0], probe_call_counts[0]),
        "probe_1_target_hit_rate": _rate(probe_hit_counts[1], probe_call_counts[1]),
        "probe_2_target_hit_rate": _rate(probe_hit_counts[2], probe_call_counts[2]),
        "probe_3_target_hit_rate": _rate(probe_hit_counts[3], probe_call_counts[3]),
        "status_counts": status_counts,
        "probe_status_counts": probe_status_counts,
    }
    for index, counts in enumerate(probe_status_counts):
        for status, value in counts.items():
            metrics[f"probe_{index}_{status}_count"] = value
    # Friendly aliases for library callers; the CSV remains stable and uses
    # the protocol names above.
    metrics.update(
        {
            "acc_at_0_5": metrics["Acc@0.5"],
            "acc_at_0_75": metrics["Acc@0.75"],
            "acc_at_0_9": metrics["Acc@0.9"],
            "mean_iou": metrics["mean IoU"],
            "target_match_mean_iou": metrics["target_match_mean_IoU"],
            "mean_iou_when_target_matched": metrics["target_match_mean_IoU"],
            "matched_target_mean_iou": metrics["target_match_mean_IoU"],
            "bbox_legal_rate": metrics["bbox_valid_rate"],
            "bbox_validity_rate": metrics["bbox_valid_rate"],
            "wrong_match_rate": metrics["wrong_instance_rate"],
            "reasoning_probe_target_hit_rates": tuple(
                metrics[f"probe_{index}_target_hit_rate"] for index in range(EXPECTED_PROBE_COUNT)
            ),
        }
    )
    return metrics


def _annotate_entries(
    manifest_by_image: Mapping[str, Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    default_update: Any = 0,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, prediction in enumerate(prediction_records):
        if not isinstance(prediction, Mapping):
            raise TypeError(f"prediction record {index} must be a mapping")
        if "response" not in prediction:
            raise ValueError(f"prediction record {index} is missing response")
        image_id = prediction.get("image_id")
        key = _json_scalar_key(image_id)
        if key not in manifest_by_image:
            raise ValueError(f"prediction image_id={image_id!r} is absent from val manifest")
        manifest = manifest_by_image[key]
        merged = dict(manifest)
        merged.update(dict(prediction))
        # Ground truth always comes from the fixed validation manifest.  A
        # prediction receipt may carry extra metadata, but must not be able
        # to replace target geometry, dimensions, or distractor instances.
        for truth_field in (
            "target_ann_id",
            "target_id",
            "target_bbox",
            "target_bbox_xywh",
            "target_bbox_normalized",
            "target_mask",
            "instances",
            "image_width",
            "image_height",
        ):
            if truth_field in manifest:
                merged[truth_field] = manifest[truth_field]
        if "update" not in merged or merged["update"] is None:
            merged["update"] = default_update
        entry = _sample_entry(merged)
        target_id = merged.get("target_ann_id", merged.get("target_id"))
        flags = tuple(
            match.status == "matched" and _same_id(match.ann_id, target_id) for match in entry["probe_matches"]
        )
        entry["update"] = merged["update"]
        entry["target_ann_id"] = target_id
        entry["probe_target_hits"] = flags
        entries.append(entry)
    return entries


def evaluate_predictions(
    manifest: str | Path | Iterable[Mapping[str, Any]],
    predictions: str | Path | Iterable[Mapping[str, Any]],
    *,
    default_update: Any = 0,
) -> dict[str, Any]:
    """Evaluate predictions and return rows plus grouped views.

    ``manifest`` and ``predictions`` may be paths or already-loaded records.
    Predictions are grouped by their ``update`` field (or
    ``default_update``), with one overall row and one row per
    ``initial_route`` within each update.
    """

    manifest_records = _as_records(manifest)
    prediction_records = _as_records(predictions)
    manifest_by_image: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(manifest_records):
        if "image_id" not in record:
            raise ValueError(f"manifest record {index} is missing image_id")
        key = _json_scalar_key(record["image_id"])
        if key in manifest_by_image:
            raise ValueError(f"manifest contains duplicate image_id={record['image_id']!r}")
        manifest_by_image[key] = record

    entries = _annotate_entries(manifest_by_image, prediction_records, default_update=default_update)
    by_update: dict[str, list[dict[str, Any]]] = defaultdict(list)
    update_values: dict[str, Any] = {}
    for entry in entries:
        key = _json_scalar_key(entry["update"])
        by_update[key].append(entry)
        update_values.setdefault(key, entry["update"])

    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for update_key, update_entries in by_update.items():
        update_value = update_values[update_key]
        overall = compute_metrics(update_entries, update=update_value, initial_route=OVERALL_ROUTE)
        rows.append(overall)
        grouped[update_key] = {OVERALL_ROUTE: overall}
        route_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in update_entries:
            route_groups[str(entry["initial_route"])].append(entry)
        for route_name in sorted(route_groups):
            row = compute_metrics(route_groups[route_name], update=update_value, initial_route=route_name)
            rows.append(row)
            grouped[update_key][route_name] = row

    # Preserve insertion order for update rows while still making grouping
    # deterministic within each update.  A flat overall view is convenient
    # for the common one-update case and is intentionally empty if no samples
    # were observed.
    overall = rows[0] if rows else None
    return {
        "rows": rows,
        "overall": overall,
        "by_update": grouped,
        "by_initial_route": (
            {route: row for route, row in grouped[next(iter(grouped))].items()} if len(grouped) == 1 else {}
        ),
    }


evaluate_manifest = evaluate_predictions
evaluate = evaluate_predictions


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _rows_from_results(results: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(results, Mapping):
        if "rows" in results:
            rows = results["rows"]
        elif "overall" in results and results["overall"] is not None:
            rows = [results["overall"]]
        else:
            rows = [results]
    else:
        rows = results
    normalised: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"monitor result row {index} must be a mapping")
        normalised.append(dict(row))
    return normalised


def append_monitor_results(
    path: str | Path,
    results: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    append: bool = True,
) -> int:
    """Write monitor rows to CSV, returning the number of appended rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_from_results(results)
    mode = "a" if append else "w"
    existing_fields: list[str] | None = None
    if append and destination.exists() and destination.stat().st_size:
        with destination.open("r", encoding="utf-8", newline="") as handle:
            existing_fields = next(csv.reader(handle), None)
        if existing_fields != list(MONITOR_RESULT_COLUMNS):
            raise ValueError(
                f"existing monitor CSV schema does not match {destination}; use append=False for a new protocol table"
            )

    with destination.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MONITOR_RESULT_COLUMNS), extrasaction="ignore")
        if not existing_fields:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in MONITOR_RESULT_COLUMNS})
    return len(rows)


write_monitor_results = append_monitor_results


def _finite_number(value: Any, *, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"update log field {field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"update log field {field!r} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"update log field {field!r} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"update log field {field!r} must be <= {maximum}")
    return number


def _validate_route_map(value: Any, *, field: str, integral: bool, bounded: bool) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"update log field {field!r} must be an object")
    result: dict[str, float | int] = {}
    for route, raw in value.items():
        if not isinstance(route, str) or not route:
            raise ValueError(f"update log field {field!r} has an invalid route key")
        if integral:
            if isinstance(raw, bool) or not isinstance(raw, numbers.Real) or float(raw) != int(raw):
                raise ValueError(f"update log field {field!r} values must be integers")
            number: float | int = int(raw)
            if number < 0:
                raise ValueError(f"update log field {field!r} values must be non-negative")
        else:
            number = _finite_number(raw, field=f"{field}.{route}", minimum=0.0, maximum=1.0 if bounded else None)
        result[route] = number
    return result


def validate_update_log(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-safe update monitoring record.

    All protocol fields are required.  ``None`` is accepted for losses and
    other unavailable scalar measurements, but malformed values, NaN/Inf,
    and out-of-range rates fail closed.  Unknown fields are retained so
    callers can attach run metadata without weakening required-field checks.
    """

    if not isinstance(record, Mapping):
        raise TypeError("update log record must be a mapping")
    missing = [field for field in UPDATE_LOG_FIELDS if field not in record]
    if missing:
        raise ValueError(f"update log is missing required fields: {', '.join(missing)}")

    validated = dict(record)
    validated["route_counts"] = _validate_route_map(
        record["route_counts"], field="route_counts", integral=True, bounded=False
    )
    validated["route_proportions"] = _validate_route_map(
        record["route_proportions"], field="route_proportions", integral=False, bounded=True
    )
    probe_calls = record["probe_calls"]
    if isinstance(probe_calls, Mapping):
        validated["probe_calls"] = _validate_route_map(probe_calls, field="probe_calls", integral=True, bounded=False)
    else:
        validated["probe_calls"] = int(_finite_number(probe_calls, field="probe_calls", minimum=0.0))

    acceptance = record["teacher_acceptance"]
    if isinstance(acceptance, Mapping):
        accepted = acceptance.get("accepted", acceptance.get("count"))
        total = acceptance.get("total", acceptance.get("calls"))
        if accepted is None or total is None:
            raise ValueError("teacher_acceptance object requires accepted/count and total/calls")
        accepted_number = _finite_number(accepted, field="teacher_acceptance.accepted", minimum=0.0)
        total_number = _finite_number(total, field="teacher_acceptance.total", minimum=0.0)
        if accepted_number > total_number:
            raise ValueError("teacher_acceptance accepted count exceeds total")
        validated["teacher_acceptance"] = dict(acceptance)
    elif acceptance is None:
        validated["teacher_acceptance"] = None
    else:
        validated["teacher_acceptance"] = _finite_number(
            acceptance, field="teacher_acceptance", minimum=0.0, maximum=1.0
        )

    for field in ("localization_loss", "referent_loss", "standard_loss"):
        value = record[field]
        if value is not None:
            validated[field] = _finite_number(value, field=field)
    if record["invalid_rate"] is not None:
        validated["invalid_rate"] = _finite_number(
            record["invalid_rate"], field="invalid_rate", minimum=0.0, maximum=1.0
        )
    for field in ("response_length", "training_time"):
        if record[field] is not None:
            validated[field] = _finite_number(record[field], field=field, minimum=0.0)

    # Reject non-standard Python objects and non-finite nested values before a
    # log line can make the JSONL stream unreadable.
    try:
        json.dumps(validated, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"update log is not JSON serializable: {error}") from error
    return validated


validate_update_log_schema = validate_update_log


def write_update_log(path: str | Path, record: Mapping[str, Any], *, append: bool = True) -> dict[str, Any]:
    """Validate and write one update log JSONL record."""

    validated = validate_update_log(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with destination.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    return validated


append_update_log = write_update_log
write_update_log_jsonl = write_update_log


def write_update_logs(path: str | Path, records: Iterable[Mapping[str, Any]], *, append: bool = True) -> int:
    """Validate and append multiple update records, returning their count."""

    count = 0
    first = True
    for record in records:
        write_update_log(path, record, append=append if first else True)
        first = False
        count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        "--val-manifest",
        dest="manifest",
        type=Path,
        required=True,
        help="fixed validation manifest JSONL",
    )
    parser.add_argument(
        "--predictions",
        "--prediction-jsonl",
        dest="predictions",
        type=Path,
        required=True,
        help="prediction JSONL; each record needs image_id and response",
    )
    parser.add_argument(
        "--output",
        "--output-csv",
        dest="output",
        type=Path,
        default=Path("monitor_results.csv"),
        help="monitor CSV path (default: monitor_results.csv)",
    )
    parser.add_argument(
        "--update",
        default=0,
        help="update value for predictions without an update field (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the monitor CSV instead of appending",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_predictions(args.manifest, args.predictions, default_update=args.update)
    written = append_monitor_results(args.output, report, append=not args.overwrite)
    print(f"Wrote {written} monitor rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXPECTED_PROBE_COUNT",
    "MONITOR_RESULT_COLUMNS",
    "OVERALL_ROUTE",
    "ROUTE_NAMES",
    "STATUS_NAMES",
    "UPDATE_LOG_FIELDS",
    "UPDATE_LOG_SCHEMA",
    "UNKNOWN_ROUTE",
    "append_monitor_results",
    "append_update_log",
    "compute_metrics",
    "evaluate",
    "evaluate_manifest",
    "evaluate_predictions",
    "load_jsonl",
    "read_jsonl",
    "route_response",
    "validate_update_log",
    "validate_update_log_schema",
    "write_monitor_results",
    "write_update_log",
    "write_update_log_jsonl",
    "write_update_logs",
]
