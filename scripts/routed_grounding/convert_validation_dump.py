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

"""Convert legacy RayPPO validation generation dumps for CPU monitoring.

The legacy trainer writes one ``<update>.jsonl`` file containing fields such
as ``input``, ``output``, ``gts``, and ``step``.  Routed Grounding adds two
optional fields to that dump: ``image_id`` (copied from input
``extra_info``) and ``routed_grounding_receipt`` (copied from the agent-loop
output).  The ordinary ``output`` is the post-repair response, so it is not
the response to use when measuring the initial route.  This converter uses
the receipt's ``student_token_response`` and its four probe receipts to
produce the prediction schema consumed by :mod:`evaluate_monitor`.

Typical use::

    python scripts/routed_grounding/convert_validation_dump.py \
        --manifest data/manifests/refcocog_umd_pilot/val.jsonl \
        --validation-dump outputs/.../validation \
        --output monitor_results.csv \
        --overwrite

Use ``--predictions-out`` when the intermediate prediction JSONL is useful
for inspection or for invoking ``evaluate_monitor.py`` separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Keep direct ``python scripts/...`` execution independent of the caller's
# PYTHONPATH.  The monitor evaluator itself uses the same repository layout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.routed_grounding.evaluate_monitor import (  # noqa: E402
    UNKNOWN_ROUTE,
    _instances_for_record,
    _json_scalar_key,
    append_monitor_results,
    evaluate_predictions,
    read_jsonl,
    route_response,
)

_PROBE_FIELDS = (
    "probes",
    "probe_responses",
    "reasoning_probes",
    "reasoning_probe_responses",
)
_RESPONSE_FIELDS = ("full_response", "response", "raw_response", "text", "output")


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Decode a mapping that may have been serialized as a JSON string."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, Mapping):
            return decoded
    return None


def _response_text(value: Any) -> str | None:
    """Extract response text from common receipt wrappers."""

    if isinstance(value, Mapping):
        for field in _RESPONSE_FIELDS:
            if field in value:
                response = _response_text(value[field])
                if response is not None:
                    return response
        return None
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _ordered_values(value: Any) -> list[Any]:
    """Return list-like metadata in numeric probe order when keyed by a map."""

    if isinstance(value, Mapping):
        items = list(value.items())

        def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
            text = str(item[0])
            digits = "".join(character for character in text if character.isdigit())
            return (int(digits) if digits else 10**9, text)

        return [item[1] for item in sorted(items, key=sort_key)]
    if isinstance(value, str | bytes | bytearray):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _receipt(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for field in ("routed_grounding_receipt", "receipt"):
        decoded = _as_mapping(record.get(field))
        if decoded is not None:
            return decoded
    return {}


def _image_id(record: Mapping[str, Any]) -> Any:
    direct = record.get("image_id")
    if direct is not None:
        return direct
    for field in ("extra_info", "metadata"):
        container = _as_mapping(record.get(field))
        if container is not None and container.get("image_id") is not None:
            return container["image_id"]
    raise ValueError("legacy validation record is missing image_id (or extra_info.image_id)")


def _raw_response(record: Mapping[str, Any], receipt: Mapping[str, Any]) -> str:
    """Select the original Student response before the optional repair."""

    candidates: list[Any] = [
        receipt.get("student_protocol_response"),
        receipt.get("student_parse"),
        receipt.get("student_token_response"),
        record.get("raw_response"),
        record.get("response"),
        record.get("output"),
    ]
    for candidate in candidates:
        response = _response_text(candidate)
        if response is not None:
            return response
    raise ValueError("legacy validation record is missing response/output and receipt Student response")


def _probe_responses(record: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str | None]:
    """Extract at most the four fixed probe responses, retaining call slots."""

    for field in _PROBE_FIELDS:
        if field not in record or record[field] is None:
            continue
        values = _ordered_values(record[field])[:4]
        return [_response_text(value) for value in values]

    raw_receipts = receipt.get("probe_receipts")
    if raw_receipts is None:
        return []
    values = _ordered_values(raw_receipts)[:4]
    return [_response_text(value) for value in values]


def _initial_route(
    record: Mapping[str, Any],
    receipt: Mapping[str, Any],
    raw_response: str,
    manifest: Mapping[str, Any] | None,
) -> str:
    direct = record.get("initial_route")
    if direct is None:
        direct = receipt.get("initial_route")
    if direct is not None and str(direct):
        return str(direct)

    # The agent-loop receipt records the route after the probes but before any
    # Teacher repair.  That is the monitor's initial route: it describes the
    # branch whose response is being evaluated, whereas ``output`` may already
    # contain the repaired Teacher suffix.
    routed = receipt.get("route")
    if isinstance(routed, Mapping) and routed.get("route") is not None:
        return str(routed["route"])

    # Older receipts may omit the route field.  Re-run the no-probe router
    # against the immutable manifest as a best-effort fallback.  For a valid
    # non-target response this intentionally yields ``uncertain`` because the
    # frozen router requires all four probes for a final non-direct route.
    if manifest is None:
        return UNKNOWN_ROUTE
    target_id = manifest.get("target_ann_id", manifest.get("target_id"))
    instances = _instances_for_record(manifest)
    try:
        return str(route_response(raw_response, instances, target_id).route)
    except (TypeError, ValueError):
        return UNKNOWN_ROUTE


def convert_validation_record(
    record: Mapping[str, Any],
    *,
    manifest_by_image: Mapping[str, Mapping[str, Any]] | None = None,
    default_update: Any = 0,
) -> dict[str, Any]:
    """Convert one native legacy dump row to evaluator prediction schema."""

    image_id = _image_id(record)
    manifest = None
    if manifest_by_image is not None:
        manifest = manifest_by_image.get(_json_scalar_key(image_id))
        if manifest is None:
            raise ValueError(f"validation dump image_id={image_id!r} is absent from the val manifest")

    receipt = _receipt(record)
    raw_response = _raw_response(record, receipt)
    update = record.get("update", record.get("step", default_update))
    if update is None:
        update = default_update
    probes = _probe_responses(record, receipt)
    return {
        "image_id": image_id,
        "response": raw_response,
        # Keep an explicit alias for downstream audit tools; evaluate_monitor
        # consumes ``response`` while this makes the pre-repair choice visible.
        "raw_response": raw_response,
        "initial_route": _initial_route(record, receipt, raw_response, manifest),
        "probes": probes,
        "update": update,
    }


def _dump_files(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"validation dump path does not exist: {path}")
    files = list(path.glob("*.jsonl"))

    def sort_key(candidate: Path) -> tuple[int, int | str]:
        try:
            return (0, int(candidate.stem))
        except ValueError:
            return (1, candidate.name)

    return sorted(files, key=sort_key)


def read_validation_dumps(source: str | Path) -> list[dict[str, Any]]:
    """Read all legacy validation JSONL files in update order."""

    records: list[dict[str, Any]] = []
    for path in _dump_files(source):
        records.extend(read_jsonl(path))
    return records


def _manifest_index(manifest: str | Path | Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    records = read_jsonl(manifest) if isinstance(manifest, str | Path) else [dict(item) for item in manifest]
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if "image_id" not in record:
            raise ValueError(f"manifest record {index} is missing image_id")
        key = _json_scalar_key(record["image_id"])
        if key in result:
            raise ValueError(f"manifest contains duplicate image_id={record['image_id']!r}")
        result[key] = record
    return result


def convert_validation_dumps(
    validation_dump: str | Path,
    manifest: str | Path | Iterable[Mapping[str, Any]],
    *,
    default_update: Any = 0,
) -> list[dict[str, Any]]:
    """Convert all validation dump rows to evaluator prediction records."""

    manifest_index = _manifest_index(manifest)
    converted = [
        convert_validation_record(
            record,
            manifest_by_image=manifest_index,
            default_update=default_update,
        )
        for record in read_validation_dumps(validation_dump)
    ]
    # Freeze the grouping cohort at the earliest available checkpoint (the
    # configs emit update 0 before training). Later rows retain their current
    # route separately while ``initial_route`` stays fixed for fair deltas.
    baseline: dict[str, tuple[float, str]] = {}
    for record in converted:
        key = _json_scalar_key(record["image_id"])
        try:
            update_order = float(record["update"])
        except (TypeError, ValueError, OverflowError):
            update_order = float("inf")
        if key not in baseline or update_order < baseline[key][0]:
            baseline[key] = (update_order, str(record["initial_route"]))
    for record in converted:
        record["current_route"] = record["initial_route"]
        record["initial_route"] = baseline[_json_scalar_key(record["image_id"])][1]
    return converted


# Friendly aliases for callers that use the singular legacy-dump wording.
convert_legacy_validation_dump = convert_validation_dumps
convert_dump_record = convert_validation_record


def write_predictions(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write converted evaluator prediction JSONL and return row count."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False))
            handle.write("\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dump",
        "--dump",
        "--input",
        dest="validation_dump",
        type=Path,
        required=True,
        help="legacy validation JSONL file or directory of <update>.jsonl files",
    )
    parser.add_argument("--manifest", "--val-manifest", type=Path, required=True, help="validation manifest JSONL")
    parser.add_argument(
        "--output",
        "--output-csv",
        type=Path,
        default=Path("monitor_results.csv"),
        help="monitor CSV path (default: monitor_results.csv)",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="optional path for the converted prediction JSONL",
    )
    parser.add_argument("--default-update", type=int, default=0, help="update for rows without step/update")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the monitor CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    predictions = convert_validation_dumps(
        args.validation_dump,
        args.manifest,
        default_update=args.default_update,
    )
    if args.predictions_out is not None:
        write_predictions(args.predictions_out, predictions)

    report = evaluate_predictions(args.manifest, predictions, default_update=args.default_update)
    written = append_monitor_results(args.output, report, append=not args.overwrite)
    print(f"Wrote {written} monitor rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "convert_dump_record",
    "convert_legacy_validation_dump",
    "convert_validation_dumps",
    "convert_validation_record",
    "main",
    "read_validation_dumps",
    "write_predictions",
]
