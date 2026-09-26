"""Route v6 bridge cases to old-label reuse or a new blind review.

The script is deliberately limited to review routing.  It never evaluates the
image, rewrites a target reference, or infers that two different references
are semantically equivalent.  An old adjudication is reusable only when the
v6 and v3p5 references are valid, non-empty, and equal after the explicitly
documented textual normalization.

The command line accepts two v3p5 packages and one matching old-review set for
each package.  All input validation happens before any output is created.
Outputs are committed with exclusive hard links from fully written temporary
files, so a failed run cannot overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "mmgcot_bridge_review_delta_v1"
PACKAGE_COUNT = 2
METADATA_FIELDS = (
    "image_id",
    "image_sha256",
    "question",
    "reasoning_token_ids_sha256",
    "task_type",
    "split",
)
REVIEW_LABEL_FIELDS = frozenset(
    {
        "student_target_selection_error",
        "extractor_content_error",
        "target_reference_review_label",
        "review_label",
        "label",
    }
)


class ReviewDeltaError(ValueError):
    """Raised when an input cannot be trusted for review routing."""


@dataclass(frozen=True)
class JsonlInput:
    path: Path
    rows: list[dict[str, Any]]
    sha256: str


@dataclass(frozen=True)
class PackageInput(JsonlInput):
    by_sample: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ReviewSet:
    mapping: JsonlInput
    reviewer_a: JsonlInput
    reviewer_b: JsonlInput
    adjudicated: JsonlInput
    mapping_by_sample: dict[str, dict[str, Any]]
    reviewer_a_by_case: dict[str, dict[str, Any]]
    reviewer_b_by_case: dict[str, dict[str, Any]]
    adjudicated_by_case: dict[str, dict[str, Any]]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewDeltaError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ReviewDeltaError(f"non-standard JSON constant: {value}")


def _read_jsonl(path: Path) -> JsonlInput:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewDeltaError(f"{path}: input is not UTF-8") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
            )
        except ReviewDeltaError as exc:
            raise ReviewDeltaError(f"{path}:{line_number}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ReviewDeltaError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ReviewDeltaError(
                f"{path}:{line_number}: each JSONL row must be an object"
            )
        rows.append(value)
    if not rows:
        raise ReviewDeltaError(f"{path}: input contains no JSONL rows")
    return JsonlInput(path=path, rows=rows, sha256=digest)


def _require_string(row: dict[str, Any], field: str, path: Path, row_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ReviewDeltaError(
            f"{path}: row {row_number}: {field} must be a non-empty string"
        )
    return value


def _require_image_id(row: dict[str, Any], path: Path, row_number: int) -> None:
    value = row.get("image_id")
    if value is None or isinstance(value, (bool, list, dict)):
        raise ReviewDeltaError(
            f"{path}: row {row_number}: image_id must be a scalar value"
        )


def _index_by_sample(package: JsonlInput) -> PackageInput:
    by_sample: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(package.rows, start=1):
        sample_id = _require_string(row, "sample_id", package.path, row_number)
        _require_image_id(row, package.path, row_number)
        for field in METADATA_FIELDS:
            if field != "image_id":
                _require_string(row, field, package.path, row_number)
        if not isinstance(row.get("target_entity_reference"), str):
            raise ReviewDeltaError(
                f"{package.path}: row {row_number}: "
                "target_entity_reference must be a string"
            )
        if sample_id in by_sample:
            raise ReviewDeltaError(f"{package.path}: duplicate sample_id: {sample_id}")
        by_sample[sample_id] = row
    return PackageInput(
        path=package.path,
        rows=package.rows,
        sha256=package.sha256,
        by_sample=by_sample,
    )


def _index_by_case(
    source: JsonlInput,
    *,
    role: str,
) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(source.rows, start=1):
        case_id = _require_string(row, "case_id", source.path, row_number)
        if case_id in by_case:
            raise ReviewDeltaError(f"{source.path}: duplicate case_id: {case_id}")
        if role != "mapping" and not REVIEW_LABEL_FIELDS.intersection(row):
            raise ReviewDeltaError(
                f"{source.path}: row {row_number}: {role} row has no review fields"
            )
        by_case[case_id] = row
    return by_case


def _load_package(path: Path) -> PackageInput:
    return _index_by_sample(_read_jsonl(Path(path)))


def _load_review_set(
    mapping_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    adjudicated_path: Path,
    package: PackageInput,
) -> ReviewSet:
    mapping = _read_jsonl(Path(mapping_path))
    reviewer_a = _read_jsonl(Path(reviewer_a_path))
    reviewer_b = _read_jsonl(Path(reviewer_b_path))
    adjudicated = _read_jsonl(Path(adjudicated_path))

    mapping_by_case = _index_by_case(mapping, role="mapping")
    mapping_by_sample: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(mapping.rows, start=1):
        sample_id = _require_string(row, "sample_id", mapping.path, row_number)
        if sample_id in mapping_by_sample:
            raise ReviewDeltaError(
                f"{mapping.path}: duplicate sample_id: {sample_id}"
            )
        mapping_by_sample[sample_id] = row
    expected_samples = set(package.by_sample)
    if set(mapping_by_sample) != expected_samples:
        missing = sorted(expected_samples - set(mapping_by_sample))
        extra = sorted(set(mapping_by_sample) - expected_samples)
        raise ReviewDeltaError(
            f"{mapping.path}: sample coverage mismatch; missing={missing}, extra={extra}"
        )

    reviewer_a_by_case = _index_by_case(reviewer_a, role="reviewer_a")
    reviewer_b_by_case = _index_by_case(reviewer_b, role="reviewer_b")
    adjudicated_by_case = _index_by_case(adjudicated, role="adjudicated")
    expected_cases = set(mapping_by_case)
    for path, role, actual in (
        (reviewer_a.path, "reviewer_a", reviewer_a_by_case),
        (reviewer_b.path, "reviewer_b", reviewer_b_by_case),
        (adjudicated.path, "adjudicated", adjudicated_by_case),
    ):
        if set(actual) != expected_cases:
            missing = sorted(expected_cases - set(actual))
            extra = sorted(set(actual) - expected_cases)
            raise ReviewDeltaError(
                f"{path}: case coverage mismatch for {role}; "
                f"missing={missing}, extra={extra}"
            )

    for sample_id, mapping_row in mapping_by_sample.items():
        case_id = _require_string(mapping_row, "case_id", mapping.path, 0)
        if mapping_by_case[case_id]["sample_id"] != sample_id:
            raise ReviewDeltaError(
                f"{mapping.path}: inconsistent mapping for case_id={case_id}"
            )

    return ReviewSet(
        mapping=mapping,
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        adjudicated=adjudicated,
        mapping_by_sample=mapping_by_sample,
        reviewer_a_by_case=reviewer_a_by_case,
        reviewer_b_by_case=reviewer_b_by_case,
        adjudicated_by_case=adjudicated_by_case,
    )


def normalize_target_entity_reference(reference: str) -> str:
    """Normalize only case, whitespace, and terminal full stops.

    No articles, morphology, punctuation other than a terminal ``.`` or ``。``,
    synonyms, or entity wording are changed.  Whitespace runs are represented
    by one ASCII space so tabs and line breaks cannot create a false change.
    """

    if not isinstance(reference, str):
        raise TypeError("target_entity_reference must be a string")
    normalized = re.sub(r"\s+", " ", reference.strip()).casefold()
    while normalized.endswith((".", "。")):
        normalized = normalized[:-1].rstrip()
    return normalized


# Short alias for callers that use the terminology from the review protocol.
normalize_reference = normalize_target_entity_reference


def _strict_equal(left: Any, right: Any) -> bool:
    """Compare JSON scalar values without Python's bool/int equivalence."""

    return type(left) is type(right) and left == right


def _validate_package_count(
    v3p5_packages: Sequence[Path],
    mappings: Sequence[Path],
    reviewer_a: Sequence[Path],
    reviewer_b: Sequence[Path],
    adjudicated: Sequence[Path],
) -> None:
    lengths = {
        "v3p5_packages": len(v3p5_packages),
        "mappings": len(mappings),
        "reviewer_a": len(reviewer_a),
        "reviewer_b": len(reviewer_b),
        "adjudicated": len(adjudicated),
    }
    if any(length != PACKAGE_COUNT for length in lengths.values()):
        raise ReviewDeltaError(
            "expected exactly two corresponding v3p5/review inputs; "
            + ", ".join(f"{name}={length}" for name, length in lengths.items())
        )


def _input_descriptor(source: JsonlInput) -> dict[str, Any]:
    return {
        "path": str(source.path.absolute()),
        "sha256": source.sha256,
        "rows": len(source.rows),
    }


def _same_metadata(
    v6_row: dict[str, Any],
    v3p5_row: dict[str, Any],
    sample_id: str,
    v6_path: Path,
    v3p5_path: Path,
) -> None:
    for field in METADATA_FIELDS:
        left = v6_row[field]
        right = v3p5_row[field]
        if not _strict_equal(left, right):
            raise ReviewDeltaError(
                f"metadata mismatch for sample_id={sample_id!r}, field={field!r}: "
                f"v6={left!r} from {v6_path}, v3p5={right!r} from {v3p5_path}"
            )


def _route_row(
    v6_row: dict[str, Any],
    v3p5_row: dict[str, Any],
    review_set: ReviewSet,
    *,
    reasons: list[str],
) -> dict[str, Any]:
    sample_id = v6_row["sample_id"]
    mapping_row = review_set.mapping_by_sample[sample_id]
    case_id = mapping_row["case_id"]
    normalized_v6 = normalize_target_entity_reference(
        v6_row["target_entity_reference"]
    )
    normalized_v3p5 = normalize_target_entity_reference(
        v3p5_row["target_entity_reference"]
    )
    unchanged = not reasons
    output = dict(v6_row)
    output.update(
        {
            "review_route": "unchanged" if unchanged else "changed",
            "review_status": (
                "reused_old_adjudicated"
                if unchanged
                else "needs_new_blind_review"
            ),
            "review_reuse_allowed": unchanged,
            "new_blind_review_required": not unchanged,
            "case_id": case_id,
            "case_id_role": "prior_review_case",
            "review_route_reasons": reasons or ["valid_nonempty_reference_equal_after_normalization"],
            "target_entity_reference_normalized": normalized_v6,
            "target_entity_reference_comparison": (
                "equal_after_normalization"
                if normalized_v6 == normalized_v3p5
                else "different_after_normalization"
            ),
        }
    )
    if unchanged:
        output.update(
            {
                "v3p5_target_entity_reference": v3p5_row[
                    "target_entity_reference"
                ],
                "v3p5_target_entity_reference_normalized": normalized_v3p5,
                "reused_review": {
                    "mapping": mapping_row,
                    "reviewer_a": review_set.reviewer_a_by_case[case_id],
                    "reviewer_b": review_set.reviewer_b_by_case[case_id],
                    "adjudicated": review_set.adjudicated_by_case[case_id],
                },
            }
        )
    else:
        # Do not include the old reference or any old review row in the blind
        # review input.  A changed case receives no inferred label here.
        output["new_blind_review_label"] = None
    return output


def _classify(
    v6_package: Path,
    v3p5_packages: Sequence[Path],
    mappings: Sequence[Path],
    reviewer_a: Sequence[Path],
    reviewer_b: Sequence[Path],
    adjudicated: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _validate_package_count(
        v3p5_packages, mappings, reviewer_a, reviewer_b, adjudicated
    )
    v6 = _load_package(Path(v6_package))
    old_packages = [_load_package(Path(path)) for path in v3p5_packages]

    old_by_sample: dict[str, tuple[PackageInput, dict[str, Any], ReviewSet]] = {}
    review_sets: list[ReviewSet] = []
    for package, mapping, reviewer_a_path, reviewer_b_path, adjudicated_path in zip(
        old_packages, mappings, reviewer_a, reviewer_b, adjudicated, strict=True
    ):
        review_set = _load_review_set(
            Path(mapping),
            Path(reviewer_a_path),
            Path(reviewer_b_path),
            Path(adjudicated_path),
            package,
        )
        review_sets.append(review_set)
        for sample_id, row in package.by_sample.items():
            if sample_id in old_by_sample:
                raise ReviewDeltaError(
                    f"sample_id appears in more than one v3p5 package: {sample_id}"
                )
            old_by_sample[sample_id] = (package, row, review_set)

    v6_samples = set(v6.by_sample)
    old_samples = set(old_by_sample)
    if v6_samples != old_samples:
        missing = sorted(old_samples - v6_samples)
        extra = sorted(v6_samples - old_samples)
        raise ReviewDeltaError(
            f"v6/v3p5 sample coverage mismatch; missing_from_v6={missing}, "
            f"extra_in_v6={extra}"
        )

    unchanged_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    text_equal_count = 0
    for sample_id in sorted(v6_samples):
        v6_row = v6.by_sample[sample_id]
        v3p5_package, v3p5_row, review_set = old_by_sample[sample_id]
        _same_metadata(
            v6_row,
            v3p5_row,
            sample_id,
            v6.path,
            v3p5_package.path,
        )
        normalized_v6 = normalize_target_entity_reference(
            v6_row["target_entity_reference"]
        )
        normalized_v3p5 = normalize_target_entity_reference(
            v3p5_row["target_entity_reference"]
        )
        reasons = []
        for version, row in (("v6", v6_row), ("v3p5", v3p5_row)):
            if row.get("bridge_parse_status") != "valid":
                reasons.append(f"{version}_bridge_parse_not_valid")
        if not normalized_v6:
            reasons.append("v6_empty_target")
        if not normalized_v3p5:
            reasons.append("v3p5_empty_target")
        if normalized_v6 != normalized_v3p5:
            reasons.append("target_text_changed")
        else:
            text_equal_count += 1
        routed = _route_row(
            v6_row,
            v3p5_row,
            review_set,
            reasons=reasons,
        )
        if not reasons:
            unchanged_rows.append(routed)
        else:
            changed_rows.append(routed)

    input_hashes: dict[str, str] = {}
    input_hashes[str(v6.path.absolute())] = v6.sha256
    for source in old_packages:
        input_hashes[str(source.path.absolute())] = source.sha256
    for review_set in review_sets:
        for source in (
            review_set.mapping,
            review_set.reviewer_a,
            review_set.reviewer_b,
            review_set.adjudicated,
        ):
            input_hashes[str(source.path.absolute())] = source.sha256

    inputs = {
        "v6_semantic_package": _input_descriptor(v6),
        "v3p5_packages": [_input_descriptor(source) for source in old_packages],
        "old_review_sets": [
            {
                "mapping": _input_descriptor(review_set.mapping),
                "reviewer_a": _input_descriptor(review_set.reviewer_a),
                "reviewer_b": _input_descriptor(review_set.reviewer_b),
                "adjudicated": _input_descriptor(review_set.adjudicated),
            }
            for review_set in review_sets
        ],
    }
    metadata = {
        "sample_count": len(v6_samples),
        "v3p5_package_count": len(old_packages),
        "review_set_count": len(review_sets),
        "metadata_fields_strictly_compared": list(METADATA_FIELDS),
        "reference_equal_after_normalization": text_equal_count,
        "reference_different_after_normalization": len(v6_samples) - text_equal_count,
    }
    return unchanged_rows, changed_rows, {
        "inputs": inputs,
        "input_hashes": input_hashes,
        "metadata": metadata,
    }


def _jsonl_content(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _absolute_destination(path: Path) -> Path:
    return Path(path).resolve(strict=False)


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_destinations_available(paths: Sequence[Path]) -> list[Path]:
    destinations = [_absolute_destination(Path(path)) for path in paths]
    if len(set(destinations)) != len(destinations):
        raise FileExistsError("output destinations must be three distinct paths")
    existing = [str(path) for path in destinations if _destination_exists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite existing output(s): " + ", ".join(existing))
    return destinations


def _write_transaction(files: Sequence[tuple[Path, str]]) -> None:
    destinations = _assert_destinations_available([path for path, _ in files])
    temporary: list[Path] = []
    committed: list[Path] = []
    try:
        for destination, content in zip(destinations, (value for _, value in files), strict=True):
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary_path = Path(temporary_name)
            temporary.append(temporary_path)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

        for temporary_path, destination in zip(temporary, destinations, strict=True):
            # link(2) is atomic and fails if destination appeared after the
            # preflight check; unlike replace(2), it cannot overwrite it.
            os.link(temporary_path, destination)
            committed.append(destination)
            temporary_path.unlink()
    except BaseException:
        for destination in committed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        for temporary_path in temporary:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def prepare_bridge_review_delta(
    v6_package: Path,
    v3p5_packages: Sequence[Path],
    mappings: Sequence[Path],
    reviewer_a: Sequence[Path],
    reviewer_b: Sequence[Path],
    adjudicated: Sequence[Path],
    unchanged_output: Path,
    changed_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    """Validate inputs, route rows, and atomically write the three outputs."""

    destinations = _assert_destinations_available(
        [unchanged_output, changed_output, manifest_output]
    )
    unchanged_rows, changed_rows, audit = _classify(
        Path(v6_package),
        [Path(path) for path in v3p5_packages],
        [Path(path) for path in mappings],
        [Path(path) for path in reviewer_a],
        [Path(path) for path in reviewer_b],
        [Path(path) for path in adjudicated],
    )
    unchanged_content = _jsonl_content(unchanged_rows)
    changed_content = _jsonl_content(changed_rows)
    unchanged_destination, changed_destination, manifest_destination = destinations
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "routing_rule": (
            "reuse the old adjudicated review only when strict metadata matches, "
            "both bridge_parse_status values are valid, and both non-empty "
            "target_entity_reference values are equal after casefold, "
            "whitespace collapse, and terminal period normalization"
        ),
        "normalization": {
            "case": "casefold",
            "whitespace": "strip and collapse runs to one ASCII space",
            "terminal_period": "remove terminal ASCII '.' or fullwidth '。'",
            "semantic_equivalence_inference": False,
        },
        "counts": {
            "total": len(unchanged_rows) + len(changed_rows),
            "unchanged": len(unchanged_rows),
            "changed": len(changed_rows),
            "new_blind_review_required": len(changed_rows),
            "old_adjudicated_reviews_reused": len(unchanged_rows),
        },
        "metadata": audit["metadata"],
        "inputs": audit["inputs"],
        "input_hashes": audit["input_hashes"],
        "outputs": {
            "unchanged": {
                "path": str(unchanged_destination),
                "rows": len(unchanged_rows),
                "sha256": _sha256_text(unchanged_content),
            },
            "changed": {
                "path": str(changed_destination),
                "rows": len(changed_rows),
                "sha256": _sha256_text(changed_content),
            },
            "manifest": {"path": str(manifest_destination)},
        },
        "fail_closed": True,
        "outputs_are_exclusive_and_atomic": True,
    }
    manifest_content = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    _write_transaction(
        (
            (unchanged_destination, unchanged_content),
            (changed_destination, changed_content),
            (manifest_destination, manifest_content),
        )
    )
    return manifest


# A concise name is useful for library callers and tests.
prepare = prepare_bridge_review_delta


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v6-package",
        "--v6",
        dest="v6_package",
        type=Path,
        required=True,
        help="v6 semantic package JSONL",
    )
    parser.add_argument(
        "--v3p5-package",
        "--v3p5",
        dest="v3p5_packages",
        action="append",
        type=Path,
        required=True,
        help="one of the two v3p5 semantic package JSONL files; repeat twice",
    )
    for option, destination, help_text in (
        ("--mapping", "mappings", "old private mapping JSONL; repeat twice"),
        ("--reviewer-a", "reviewer_a", "old reviewer A JSONL; repeat twice"),
        ("--reviewer-b", "reviewer_b", "old reviewer B JSONL; repeat twice"),
        ("--adjudicated", "adjudicated", "old adjudicated JSONL; repeat twice"),
    ):
        parser.add_argument(
            option,
            dest=destination,
            action="append",
            type=Path,
            required=True,
            help=help_text,
        )
    parser.add_argument(
        "--unchanged-output",
        "--unchanged",
        dest="unchanged_output",
        type=Path,
        required=True,
        help="JSONL receiving rows eligible for old-label reuse",
    )
    parser.add_argument(
        "--changed-output",
        "--changed",
        dest="changed_output",
        type=Path,
        required=True,
        help="JSONL receiving rows that require a new blind review",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_output",
        type=Path,
        required=True,
        help="JSON manifest with counts and input hashes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare_bridge_review_delta(
            args.v6_package,
            args.v3p5_packages,
            args.mappings,
            args.reviewer_a,
            args.reviewer_b,
            args.adjudicated,
            args.unchanged_output,
            args.changed_output,
            args.manifest_output,
        )
    except (OSError, ReviewDeltaError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
