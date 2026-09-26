"""Fail-closed MM-GCoT eligibility qualification and reserve sampling.

This module is intentionally separate from :mod:`mmgcot_diagnostic.data`.
It reads an already prepared v2 raw tree, creates an image-disjoint reserve,
and freezes a new final manifest only after independent eligibility reviews
have been reconciled.  It never edits a v2 manifest or JSONL file.

The reserve path is coupled to the private helpers currently implemented by
``mmgcot_diagnostic.data``: ``_raw_candidates``, ``_public_row``,
``_stable_rank``, ``stable_order``, ``materialize_image``,
``_default_image_dirs`` and ``discover_local_images``.  The exact source hash
and helper list are recorded in every provenance/final manifest so that a
future change to ``data.py`` cannot silently change the qualification split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError

try:  # Support both ``python -m mmgcot_diagnostic.qualification`` and imports.
    from . import data as _data
except ImportError:  # pragma: no cover - only useful for direct script loading.
    from mmgcot_diagnostic import data as _data


TASK_TYPES: tuple[str, ...] = tuple(_data.TASK_TYPES)
DEFAULT_SEED = int(_data.SEED)
DEFAULT_RESERVE_PER_TASK: dict[str, int] = {"pilot": 10, "formal": 40}
DEFAULT_FINAL_PER_TASK: dict[str, int] = {"pilot": 15, "formal": 100}

RESERVE_PROVENANCE_NAME = "reserve_provenance.json"
PILOT_FINAL_NAME = "pilot_frozen_final.jsonl"
FORMAL_FINAL_NAME = "formal_frozen_final.jsonl"
REVIEW_MAPPING_NAME = "eligibility_review_mapping.jsonl"
HASHES_NAME = "final_hashes.json"
FINAL_MANIFEST_NAME = "final_manifest.json"

_ALLOWED_REVIEW_STATUSES = frozenset({"eligible", "ambiguous", "invalid"})
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_DATA_PRIVATE_HELPERS = (
    "_raw_candidates",
    "_public_row",
    "_stable_rank",
    "stable_order",
    "materialize_image",
    "_default_image_dirs",
    "discover_local_images",
)


class QualificationError(RuntimeError):
    """A fail-closed input, review, or quota error."""


@dataclass(frozen=True)
class ReserveResult:
    output_dir: Path
    pilot_path: Path
    formal_path: Path
    provenance_path: Path
    counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class FinalizeResult:
    output_dir: Path
    pilot_path: Path
    formal_path: Path
    review_mapping_path: Path
    hashes_path: Path
    manifest_path: Path
    counts: dict[str, Any]


@dataclass(frozen=True)
class EligibilitySource:
    """One manifest and its private eligibility-review artifacts."""

    name: str
    split: str
    role: str
    manifest: Path
    private_key: Path
    review: Path


@dataclass
class _ReviewedCandidate:
    source: EligibilitySource
    row: dict[str, Any]
    case_id: str
    status: str
    review_reason: str
    stable_rank: str
    decision: str | None = None
    fallback_for: str | None = None
    selected_index: int | None = None


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: str | Path) -> str:
    """Hash a file without importing the inference stack."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL while preserving line-oriented validation errors."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QualificationError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise QualificationError(f"{path}:{line_number} must be a JSON object")
                rows.append(dict(value))
    except UnicodeDecodeError as exc:
        raise QualificationError(f"{path} is not UTF-8 JSONL") from exc
    return rows


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a v2 JSONL manifest, with a small JSON-list compatibility path."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot read manifest {path}: {exc}") from exc
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = None
        for key in ("rows", "samples", "records", "manifest"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        if rows is None and "sample_id" in value:
            rows = [value]
        if rows is None:
            raise QualificationError(f"{path} has no JSON manifest row list")
    else:
        raise QualificationError(f"{path} must contain a JSON list or object")
    if not all(isinstance(row, dict) for row in rows):
        raise QualificationError(f"{path} contains a non-object manifest row")
    return [dict(row) for row in rows]


def _nonempty(value: Any, *, label: str) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        raise QualificationError(f"{label} must be a non-empty scalar")
    text = str(value).strip()
    if not text:
        raise QualificationError(f"{label} must be non-empty")
    return text


def _validate_manifest_rows(rows: Sequence[Mapping[str, Any]], *, path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for line_number, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        sample_id = _nonempty(row.get("sample_id"), label=f"{path}:{line_number}.sample_id")
        image_id = _nonempty(row.get("image_id"), label=f"{path}:{line_number}.image_id")
        task_type = _nonempty(row.get("task_type"), label=f"{path}:{line_number}.task_type")
        if task_type not in TASK_TYPES:
            raise QualificationError(f"{path}:{line_number} has unknown task_type {task_type!r}; expected {TASK_TYPES}")
        if sample_id in seen_sample_ids:
            raise QualificationError(f"duplicate sample_id in manifest {path}: {sample_id}")
        seen_sample_ids.add(sample_id)
        row["sample_id"] = sample_id
        row["image_id"] = image_id
        row["task_type"] = task_type
        result.append(row)
    return result


def _manifest_image_ids(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = _validate_manifest_rows(read_manifest(path), path=path)
    return rows, {str(row["image_id"]) for row in rows}


def stable_order_rows(
    rows: Iterable[Mapping[str, Any]], *, split: str, task_type: str, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Expose the exact seeded ordering used by ``data.py`` for one pool."""

    if split not in {"pilot", "formal"}:
        raise ValueError(f"unsupported split: {split}")
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type}")
    return _data.stable_order(rows, seed=seed, namespace=f"{split}:{task_type}")


def stable_rank(row: Mapping[str, Any], *, split: str, seed: int = DEFAULT_SEED) -> str:
    """Return the original data-preparer rank, honoring a persisted rank."""

    persisted = row.get("stable_rank")
    if persisted is not None and str(persisted).strip():
        return str(persisted)
    task_type = str(row.get("task_type", ""))
    if task_type not in TASK_TYPES:
        raise QualificationError(f"cannot rank row with task_type {task_type!r}")
    try:
        return str(_data._stable_rank(row, seed=seed, namespace=f"{split}:{task_type}"))
    except (TypeError, ValueError, KeyError) as exc:
        raise QualificationError(f"cannot compute stable rank for {row.get('sample_id', '<unknown>')}") from exc


def _rank_key(candidate: _ReviewedCandidate) -> tuple[str, str, int, str, str]:
    row = candidate.row
    try:
        source_row_index = int(row.get("source_row_index", -1))
    except (TypeError, ValueError):
        source_row_index = -1
    return (
        candidate.stable_rank,
        str(row.get("source_id", "")),
        source_row_index,
        str(row["sample_id"]),
        candidate.source.name,
    )


def _data_dependency() -> dict[str, Any]:
    module_path = Path(_data.__file__).resolve()
    return {
        "module": str(module_path),
        "sha256": file_sha256(module_path),
        "private_helpers": list(_DATA_PRIVATE_HELPERS),
        "contract": (
            "qualification.py is coupled to the current private data.py candidate schema and stable rank namespace"
        ),
    }


def _assert_new_output_dir(output_dir: Path, names: Sequence[str]) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    existing = [str(output_dir / name) for name in names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError("refusing to overwrite output files: " + ", ".join(existing))


def _assert_outside_root(output_dir: Path, root: Path, *, label: str) -> None:
    output_resolved = output_dir.resolve()
    root_resolved = root.resolve()
    if output_resolved == root_resolved or root_resolved in output_resolved.parents:
        raise QualificationError(f"{label} must be outside protected root {root_resolved}")


def _input_file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _output_file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _raw_paths(root: Path) -> list[Path]:
    missing = [root / "raw" / relative for relative in _data.RAW_FILES if not (root / "raw" / relative).is_file()]
    if missing:
        raise QualificationError("v2 raw tree is incomplete: " + ", ".join(str(path) for path in missing))
    return [root / "raw" / relative for relative in _data.RAW_FILES]


def _resolve_primary_manifest(root: Path, split: str, path: str | Path | None) -> Path:
    if path is not None:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"primary {split} manifest is missing: {resolved}")
        return resolved
    for name in (f"{split}.jsonl", f"{split}_v2.jsonl", f"{split}_primary.jsonl"):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not find primary {split} manifest below {root}")


def _candidate_rank(row: Mapping[str, Any], *, split: str, seed: int) -> str:
    persisted = row.get("stable_rank")
    if persisted is not None and str(persisted).strip():
        return str(persisted)
    task_type = str(row.get("task_type", ""))
    return str(_data._stable_rank(row, seed=seed, namespace=f"{split}:{task_type}"))


def _select_raw_reserve(
    *,
    split: str,
    candidates_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    target_per_task: int,
    seed: int,
    excluded_image_ids: set[str],
    image_dir: Path,
    local_index: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """Select exact per-task reserve quotas from raw candidates.

    The formal pool is called before this function for pilot, so the caller
    can pass its selected image IDs as part of ``excluded_image_ids``.
    Reserve quotas intentionally do not borrow between task types: the reserve
    contract asks for a fixed number of each type.
    """

    selected: list[dict[str, Any]] = []
    selected_image_ids = set(excluded_image_ids)
    counts = {task: 0 for task in TASK_TYPES}
    exclusions: list[dict[str, Any]] = []
    ranked = {
        task: stable_order_rows(candidates_by_task.get(task, ()), split=split, task_type=task, seed=seed)
        for task in TASK_TYPES
    }

    for task in TASK_TYPES:
        for candidate in ranked[task]:
            image_id = _nonempty(candidate.get("image_id"), label="raw candidate image_id")
            if counts[task] >= target_per_task:
                break
            if image_id in selected_image_ids:
                exclusions.append(
                    {
                        "split": split,
                        "task_type": task,
                        "source_id": candidate.get("source_id"),
                        "image_id": image_id,
                        "reason": "duplicate_image_primary_or_formal_reserve",
                    }
                )
                continue
            try:
                image = _data.materialize_image(
                    image_id=image_id,
                    image_ref=str(candidate["image_ref"]),
                    image_dir=image_dir,
                    local_index=local_index,
                )
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError, KeyError) as exc:
                exclusions.append(
                    {
                        "split": split,
                        "task_type": task,
                        "source_id": candidate.get("source_id"),
                        "image_id": image_id,
                        "reason": "image_unreadable_or_download_failed",
                        "detail": str(exc),
                    }
                )
                continue
            row = _data._public_row(candidate, image, split=split)
            row.update(
                {
                    "split": split,
                    "qualification_pool": "reserve",
                    "stable_rank": _candidate_rank(candidate, split=split, seed=seed),
                    "reserve_selection_index": len(selected),
                }
            )
            selected.append(row)
            selected_image_ids.add(image_id)
            counts[task] += 1
    return selected, counts, exclusions


def reserve(
    v2_root: str | Path,
    output_dir: str | Path,
    *,
    primary_pilot: str | Path | None = None,
    primary_formal: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    pilot_per_task: int = DEFAULT_RESERVE_PER_TASK["pilot"],
    formal_per_task: int = DEFAULT_RESERVE_PER_TASK["formal"],
    local_image_dirs: Sequence[str | Path] = (),
) -> ReserveResult:
    """Materialize an image-disjoint reserve from a prepared v2 raw tree."""

    if pilot_per_task < 0 or formal_per_task < 0:
        raise ValueError("reserve quotas must be non-negative")
    root = Path(v2_root).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"v2 data root is missing: {root}")
    _assert_outside_root(output, root, label="reserve output")
    output_names = ("pilot_reserve.jsonl", "formal_reserve.jsonl", RESERVE_PROVENANCE_NAME)
    _assert_new_output_dir(output, output_names)
    raw_paths = _raw_paths(root)
    primary_pilot_path = _resolve_primary_manifest(root, "pilot", primary_pilot)
    primary_formal_path = _resolve_primary_manifest(root, "formal", primary_formal)
    primary_pilot_rows, primary_pilot_ids = _manifest_image_ids(primary_pilot_path)
    primary_formal_rows, primary_formal_ids = _manifest_image_ids(primary_formal_path)
    primary_ids = primary_pilot_ids | primary_formal_ids

    candidates_by_split_task, raw_exclusions, raw_stats = _data._raw_candidates(root)
    image_dirs = _data._default_image_dirs([Path(path) for path in local_image_dirs])
    local_index = _data.discover_local_images(image_dirs)
    image_dir = output / "images"

    formal_rows, formal_counts, formal_exclusions = _select_raw_reserve(
        split="formal",
        candidates_by_task=candidates_by_split_task["formal"],
        target_per_task=formal_per_task,
        seed=seed,
        excluded_image_ids=primary_ids,
        image_dir=image_dir,
        local_index=local_index,
    )
    formal_ids = {str(row["image_id"]) for row in formal_rows}
    if any(formal_counts[task] != formal_per_task for task in TASK_TYPES):
        raise QualificationError(f"formal reserve quota unmet: {formal_counts}; requested {formal_per_task} per task")

    pilot_rows, pilot_counts, pilot_exclusions = _select_raw_reserve(
        split="pilot",
        candidates_by_task=candidates_by_split_task["pilot"],
        target_per_task=pilot_per_task,
        seed=seed,
        excluded_image_ids=primary_ids | formal_ids,
        image_dir=image_dir,
        local_index=local_index,
    )
    if any(pilot_counts[task] != pilot_per_task for task in TASK_TYPES):
        raise QualificationError(f"pilot reserve quota unmet: {pilot_counts}; requested {pilot_per_task} per task")

    pilot_path = output / "pilot_reserve.jsonl"
    formal_path = output / "formal_reserve.jsonl"
    provenance_path = output / RESERVE_PROVENANCE_NAME
    _write_jsonl_exclusive(pilot_path, pilot_rows)
    _write_jsonl_exclusive(formal_path, formal_rows)
    provenance = {
        "schema_version": "mmgcot_qualification_reserve_v1",
        "status": "reserved",
        "generated_at_utc": _now_utc(),
        "v2_root": str(root),
        "protocol": {
            "seed": seed,
            "tasks": list(TASK_TYPES),
            "pilot_reserve_per_task": pilot_per_task,
            "formal_reserve_per_task": formal_per_task,
            "formal_priority": True,
            "stable_order_namespace": "{split}:{task_type}",
            "duplicate_rule": (
                "exclude every primary image_id; pilot additionally excludes every selected formal reserve image_id"
            ),
            "reserve_quota_rule": "exact per-task quotas; no cross-task borrowing",
        },
        "data_dependency": _data_dependency(),
        "inputs": {
            "raw_files": [_input_file_record(path) for path in raw_paths],
            "primary_pilot_manifest": _input_file_record(primary_pilot_path),
            "primary_formal_manifest": _input_file_record(primary_formal_path),
            "primary_pilot_rows": len(primary_pilot_rows),
            "primary_formal_rows": len(primary_formal_rows),
            "primary_image_ids": len(primary_ids),
        },
        "local_image_search_dirs": [str(path.resolve()) for path in image_dirs],
        "raw_candidate_stats": raw_stats,
        "raw_exclusions": raw_exclusions,
        "reserve_exclusions": formal_exclusions + pilot_exclusions,
        "counts": {
            "pilot": pilot_counts,
            "formal": formal_counts,
            "pilot_unique_images": len({str(row["image_id"]) for row in pilot_rows}),
            "formal_unique_images": len({str(row["image_id"]) for row in formal_rows}),
            "primary_formal_overlap": len(primary_formal_ids & primary_pilot_ids),
            "reserve_cross_split_overlap": len(
                {str(row["image_id"]) for row in pilot_rows} & {str(row["image_id"]) for row in formal_rows}
            ),
        },
        "files": {
            "pilot_reserve": _output_file_record(pilot_path),
            "formal_reserve": _output_file_record(formal_path),
        },
    }
    _write_json_exclusive(provenance_path, provenance)
    return ReserveResult(
        output_dir=output,
        pilot_path=pilot_path,
        formal_path=formal_path,
        provenance_path=provenance_path,
        counts={"pilot": pilot_counts, "formal": formal_counts},
    )


def _load_reviewed_source(source: EligibilitySource) -> list[_ReviewedCandidate]:
    rows = _validate_manifest_rows(read_manifest(source.manifest), path=source.manifest)
    rows_by_sample = {str(row["sample_id"]): row for row in rows}

    key_rows = read_jsonl(source.private_key)
    key_by_case: dict[str, dict[str, Any]] = {}
    key_sample_ids: set[str] = set()
    for line_number, key_row in enumerate(key_rows, start=1):
        case_id = _nonempty(key_row.get("case_id"), label=f"{source.private_key}:{line_number}.case_id")
        sample_id = _nonempty(key_row.get("sample_id"), label=f"{source.private_key}:{line_number}.sample_id")
        if case_id in key_by_case:
            raise QualificationError(f"duplicate case_id in private key {source.private_key}: {case_id}")
        if sample_id in key_sample_ids:
            raise QualificationError(f"duplicate sample_id in private key {source.private_key}: {sample_id}")
        key_by_case[case_id] = dict(key_row)
        key_sample_ids.add(sample_id)
        manifest_row = rows_by_sample.get(sample_id)
        if manifest_row is None:
            raise QualificationError(f"private key {source.private_key} references unknown sample_id {sample_id}")
        if "image_id" in key_row and str(key_row["image_id"]) != str(manifest_row["image_id"]):
            raise QualificationError(
                f"private key image mismatch for {case_id}: {key_row['image_id']} != {manifest_row['image_id']}"
            )

    manifest_sample_ids = set(rows_by_sample)
    if key_sample_ids != manifest_sample_ids:
        missing = sorted(manifest_sample_ids - key_sample_ids)
        extra = sorted(key_sample_ids - manifest_sample_ids)
        raise QualificationError(
            f"private key coverage mismatch for {source.name}: missing={missing[:5]} extra={extra[:5]}"
        )

    review_rows = read_jsonl(source.review)
    review_by_case: dict[str, dict[str, Any]] = {}
    for line_number, review_row in enumerate(review_rows, start=1):
        case_id = _nonempty(review_row.get("case_id"), label=f"{source.review}:{line_number}.case_id")
        if case_id in review_by_case:
            raise QualificationError(f"duplicate case_id in review {source.review}: {case_id}")
        status = review_row.get("status")
        if not isinstance(status, str) or status not in _ALLOWED_REVIEW_STATUSES:
            raise QualificationError(
                f"unknown eligibility status {status!r} in {source.review}:{line_number}; "
                f"expected {sorted(_ALLOWED_REVIEW_STATUSES)}"
            )
        # A reviewer supplied sample_id can be useful to a human, but it is
        # never used for mapping.  If present, a contradiction is rejected.
        key_row = key_by_case.get(case_id)
        if key_row is not None and "sample_id" in review_row:
            if str(review_row["sample_id"]) != str(key_row["sample_id"]):
                raise QualificationError(f"review sample_id disagrees with private key for case_id {case_id}")
        review_by_case[case_id] = dict(review_row)

    expected_cases = set(key_by_case)
    actual_cases = set(review_by_case)
    if actual_cases != expected_cases:
        missing = sorted(expected_cases - actual_cases)
        extra = sorted(actual_cases - expected_cases)
        raise QualificationError(f"review coverage mismatch for {source.name}: missing={missing[:5]} extra={extra[:5]}")

    result: list[_ReviewedCandidate] = []
    for key_row in key_rows:
        case_id = str(key_row["case_id"])
        sample_id = str(key_row["sample_id"])
        review_row = review_by_case[case_id]
        manifest_row = rows_by_sample[sample_id]
        result.append(
            _ReviewedCandidate(
                source=source,
                row=dict(manifest_row),
                case_id=case_id,
                status=str(review_row["status"]),
                review_reason=str(review_row.get("reason", "")),
                stable_rank=stable_rank(manifest_row, split=source.split, seed=DEFAULT_SEED),
            )
        )
    return result


def _load_all_reviewed_sources(sources: Sequence[EligibilitySource], *, seed: int) -> list[_ReviewedCandidate]:
    all_candidates: list[_ReviewedCandidate] = []
    sample_to_source: dict[str, str] = {}
    for source in sources:
        candidates = _load_reviewed_source(source)
        for candidate in candidates:
            candidate.stable_rank = stable_rank(candidate.row, split=source.split, seed=seed)
            sample_id = str(candidate.row["sample_id"])
            previous = sample_to_source.get(sample_id)
            if previous is not None:
                raise QualificationError(
                    f"sample_id appears in more than one input manifest: {sample_id} ({previous}, {source.name})"
                )
            sample_to_source[sample_id] = source.name
        all_candidates.extend(candidates)
    return all_candidates


def _select_final_split(
    candidates: Sequence[_ReviewedCandidate],
    *,
    split: str,
    per_task: int,
    seed: int,
    forbidden_image_ids: set[str] | None = None,
) -> tuple[list[_ReviewedCandidate], dict[str, int], list[dict[str, Any]]]:
    """Select task quotas, then use the other task for an allowed fallback."""

    if split not in {"pilot", "formal"}:
        raise ValueError(f"unsupported split: {split}")
    if per_task < 0:
        raise ValueError("per-task quota must be non-negative")
    forbidden = set(forbidden_image_ids or ())
    split_candidates = [candidate for candidate in candidates if candidate.source.split == split]
    for candidate in split_candidates:
        if candidate.status != "eligible":
            candidate.decision = "review_status_not_eligible"
        else:
            candidate.decision = None
    ranked = {
        task: sorted(
            (candidate for candidate in split_candidates if candidate.row["task_type"] == task),
            key=_rank_key,
        )
        for task in TASK_TYPES
    }
    selected: list[_ReviewedCandidate] = []
    selected_sample_ids: set[str] = set()
    selected_image_ids: set[str] = set()
    counts = {task: 0 for task in TASK_TYPES}
    fallback_records: list[dict[str, Any]] = []
    total_target = per_task * len(TASK_TYPES)

    def try_take(candidate: _ReviewedCandidate, *, reason: str, fallback_for: str | None = None) -> bool:
        if candidate.status != "eligible":
            return False
        sample_id = str(candidate.row["sample_id"])
        image_id = str(candidate.row["image_id"])
        if image_id in forbidden:
            candidate.decision = "excluded_formal_image"
            return False
        if sample_id in selected_sample_ids:
            candidate.decision = "duplicate_sample_id_in_selection"
            return False
        if image_id in selected_image_ids:
            candidate.decision = "duplicate_image_in_selection"
            return False
        candidate.decision = reason
        candidate.fallback_for = fallback_for
        candidate.selected_index = len(selected)
        selected.append(candidate)
        selected_sample_ids.add(sample_id)
        selected_image_ids.add(image_id)
        counts[str(candidate.row["task_type"])] += 1
        if fallback_for is not None:
            fallback_records.append(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "selected_task_type": candidate.row["task_type"],
                    "for_short_task_type": fallback_for,
                    "reason": "quota_fallback_other_task_type",
                }
            )
        return True

    for task in TASK_TYPES:
        for candidate in ranked[task]:
            if counts[task] >= per_task:
                break
            try_take(candidate, reason="quota")

    # A task that is short may borrow from the other task.  Each borrower is
    # processed deterministically, and the total remains capped at 2*quota.
    for recipient in TASK_TYPES:
        if counts[recipient] >= per_task or len(selected) >= total_target:
            continue
        donor = next(task for task in TASK_TYPES if task != recipient)
        for candidate in ranked[donor]:
            if len(selected) >= total_target:
                break
            if candidate.selected_index is not None:
                continue
            if try_take(candidate, reason="quota_fallback_other_task_type", fallback_for=recipient):
                continue

    if len(selected) < total_target:
        # This second pass only matters for pathological duplicate layouts in
        # which both task quotas are short but unused eligible rows remain.
        for task in TASK_TYPES:
            if len(selected) >= total_target:
                break
            for candidate in ranked[task]:
                if len(selected) >= total_target:
                    break
                if candidate.selected_index is not None:
                    continue
                try_take(candidate, reason="quota_fallback_available_candidate", fallback_for=task)

    if len(selected) < total_target:
        available = Counter(
            str(candidate.row["task_type"])
            for candidate in split_candidates
            if candidate.status == "eligible" and str(candidate.row["image_id"]) not in forbidden
        )
        raise QualificationError(
            f"{split} final quota unmet: selected={len(selected)}/{total_target}, "
            f"counts={counts}, eligible_available={dict(available)}"
        )
    return selected, counts, fallback_records


def _final_row(candidate: _ReviewedCandidate, *, split: str) -> dict[str, Any]:
    row = dict(candidate.row)
    row.update(
        {
            "split": split,
            "qualification_pool": candidate.source.name,
            "stable_rank": candidate.stable_rank,
            "final_selection_index": candidate.selected_index,
            "final_selection_reason": candidate.decision,
        }
    )
    # case_id deliberately does not enter the frozen data row.  The private
    # audit mapping below is the only artifact that joins it to sample_id.
    return row


def _review_mapping_rows(
    candidates: Sequence[_ReviewedCandidate],
    *,
    formal_image_ids: set[str],
    selected_by_sample: Mapping[str, tuple[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (item.source.name, _rank_key(item))):
        sample_id = str(candidate.row["sample_id"])
        selected = selected_by_sample.get(sample_id)
        decision = candidate.decision
        if (
            selected is None
            and candidate.source.split == "pilot"
            and str(candidate.row["image_id"]) in formal_image_ids
        ):
            decision = "excluded_formal_image"
        if decision is None:
            decision = "quota_not_selected"
        rows.append(
            {
                "source": candidate.source.name,
                "manifest": str(candidate.source.manifest.resolve()),
                "case_id": candidate.case_id,
                "sample_id": sample_id,
                "image_id": str(candidate.row["image_id"]),
                "task_type": str(candidate.row["task_type"]),
                "status": candidate.status,
                "review_reason": candidate.review_reason,
                "selected": selected is not None,
                "final_split": selected[0] if selected is not None else None,
                "final_selection_index": selected[1] if selected is not None else None,
                "decision": decision,
            }
        )
    return rows


def _final_sources(
    *,
    primary_pilot: str | Path,
    primary_formal: str | Path,
    reserve_pilot: str | Path,
    reserve_formal: str | Path,
    primary_pilot_key: str | Path,
    primary_formal_key: str | Path,
    reserve_pilot_key: str | Path,
    reserve_formal_key: str | Path,
    primary_pilot_review: str | Path,
    primary_formal_review: str | Path,
    reserve_pilot_review: str | Path,
    reserve_formal_review: str | Path,
) -> list[EligibilitySource]:
    return [
        EligibilitySource(
            "primary_pilot",
            "pilot",
            "primary",
            Path(primary_pilot).resolve(),
            Path(primary_pilot_key).resolve(),
            Path(primary_pilot_review).resolve(),
        ),
        EligibilitySource(
            "primary_formal",
            "formal",
            "primary",
            Path(primary_formal).resolve(),
            Path(primary_formal_key).resolve(),
            Path(primary_formal_review).resolve(),
        ),
        EligibilitySource(
            "reserve_pilot",
            "pilot",
            "reserve",
            Path(reserve_pilot).resolve(),
            Path(reserve_pilot_key).resolve(),
            Path(reserve_pilot_review).resolve(),
        ),
        EligibilitySource(
            "reserve_formal",
            "formal",
            "reserve",
            Path(reserve_formal).resolve(),
            Path(reserve_formal_key).resolve(),
            Path(reserve_formal_review).resolve(),
        ),
    ]


def finalize(
    *,
    primary_pilot: str | Path,
    primary_formal: str | Path,
    reserve_pilot: str | Path,
    reserve_formal: str | Path,
    primary_pilot_key: str | Path,
    primary_formal_key: str | Path,
    reserve_pilot_key: str | Path,
    reserve_formal_key: str | Path,
    primary_pilot_review: str | Path,
    primary_formal_review: str | Path,
    reserve_pilot_review: str | Path,
    reserve_formal_review: str | Path,
    output_dir: str | Path,
    seed: int = DEFAULT_SEED,
    pilot_per_task: int = DEFAULT_FINAL_PER_TASK["pilot"],
    formal_per_task: int = DEFAULT_FINAL_PER_TASK["formal"],
) -> FinalizeResult:
    """Freeze final manifests after complete private-key review coverage."""

    if pilot_per_task < 0 or formal_per_task < 0:
        raise ValueError("final quotas must be non-negative")
    output = Path(output_dir).resolve()
    sources = _final_sources(
        primary_pilot=primary_pilot,
        primary_formal=primary_formal,
        reserve_pilot=reserve_pilot,
        reserve_formal=reserve_formal,
        primary_pilot_key=primary_pilot_key,
        primary_formal_key=primary_formal_key,
        reserve_pilot_key=reserve_pilot_key,
        reserve_formal_key=reserve_formal_key,
        primary_pilot_review=primary_pilot_review,
        primary_formal_review=primary_formal_review,
        reserve_pilot_review=reserve_pilot_review,
        reserve_formal_review=reserve_formal_review,
    )
    output_names = (
        PILOT_FINAL_NAME,
        FORMAL_FINAL_NAME,
        REVIEW_MAPPING_NAME,
        HASHES_NAME,
        FINAL_MANIFEST_NAME,
    )
    _assert_new_output_dir(output, output_names)

    # Detect a v2 root from any source path that sits below a raw/ directory.
    # This guard prevents accidentally placing final artifacts inside v2.
    protected_roots: set[Path] = set()
    for source in sources:
        for path in (source.manifest, source.private_key, source.review):
            for parent in [path.parent, *path.parents]:
                if parent.name == "raw":
                    protected_roots.add(parent.parent.resolve())
                    break
                if (parent / "raw").is_dir():
                    protected_roots.add(parent.resolve())
                    break
    for root in protected_roots:
        _assert_outside_root(output, root, label="final output")

    input_files: dict[str, Any] = {}
    for source in sources:
        input_files[source.name] = {
            "manifest": _input_file_record(source.manifest),
            "private_key": _input_file_record(source.private_key),
            "review": _input_file_record(source.review),
            "role": source.role,
            "split": source.split,
        }

    candidates = _load_all_reviewed_sources(sources, seed=seed)
    formal_candidates = [candidate for candidate in candidates if candidate.source.split == "formal"]
    pilot_candidates = [candidate for candidate in candidates if candidate.source.split == "pilot"]
    formal_selected, formal_counts, formal_fallbacks = _select_final_split(
        formal_candidates,
        split="formal",
        per_task=formal_per_task,
        seed=seed,
    )
    formal_image_ids = {str(candidate.row["image_id"]) for candidate in formal_candidates}
    pilot_selected, pilot_counts, pilot_fallbacks = _select_final_split(
        pilot_candidates,
        split="pilot",
        per_task=pilot_per_task,
        seed=seed,
        forbidden_image_ids=formal_image_ids,
    )

    formal_output_rows = [_final_row(candidate, split="formal") for candidate in formal_selected]
    pilot_output_rows = [_final_row(candidate, split="pilot") for candidate in pilot_selected]
    selected_by_sample = {
        str(candidate.row["sample_id"]): (split, int(candidate.selected_index))
        for split, selected_rows in (("formal", formal_selected), ("pilot", pilot_selected))
        for candidate in selected_rows
    }
    mapping_rows = _review_mapping_rows(
        candidates,
        formal_image_ids=formal_image_ids,
        selected_by_sample=selected_by_sample,
    )

    pilot_path = output / PILOT_FINAL_NAME
    formal_path = output / FORMAL_FINAL_NAME
    mapping_path = output / REVIEW_MAPPING_NAME
    hashes_path = output / HASHES_NAME
    manifest_path = output / FINAL_MANIFEST_NAME
    _write_jsonl_exclusive(pilot_path, pilot_output_rows)
    _write_jsonl_exclusive(formal_path, formal_output_rows)
    _write_jsonl_exclusive(mapping_path, mapping_rows)

    artifact_records = {
        PILOT_FINAL_NAME: _output_file_record(pilot_path),
        FORMAL_FINAL_NAME: _output_file_record(formal_path),
        REVIEW_MAPPING_NAME: _output_file_record(mapping_path),
    }
    hash_payload = {
        "schema_version": "mmgcot_qualification_hashes_v1",
        "generated_at_utc": _now_utc(),
        "inputs": input_files,
        "artifacts": artifact_records,
        "statement": (
            "Input v2/primary/reserve files are read-only; this hash record covers every "
            "final JSONL and the review mapping."
        ),
    }
    _write_json_exclusive(hashes_path, hash_payload)
    artifact_records[HASHES_NAME] = _output_file_record(hashes_path)

    selected_total = len(pilot_output_rows) + len(formal_output_rows)
    final_manifest = {
        "schema_version": "mmgcot_qualification_final_v1",
        "status": "frozen",
        "generated_at_utc": _now_utc(),
        "protocol": {
            "seed": seed,
            "tasks": list(TASK_TYPES),
            "formal_per_task": formal_per_task,
            "pilot_per_task": pilot_per_task,
            "formal_priority": "select eligible formal rows by original stable rank first",
            "pilot_rule": (
                "select eligible pilot rows by original stable rank after excluding every formal input image_id"
            ),
            "fallback_rule": (
                "when a task is short, eligible rows from the other task may fill the total "
                "quota; each fallback is recorded"
            ),
            "review_rule": (
                "only status=eligible is selectable; key/review coverage, duplicate cases, "
                "and unknown statuses fail closed"
            ),
            "case_id_mapping": "case_id is joined to sample_id only through the supplied private key",
        },
        "data_dependency": _data_dependency(),
        "inputs": input_files,
        "counts": {
            "formal": formal_counts,
            "pilot": pilot_counts,
            "formal_total": len(formal_output_rows),
            "pilot_total": len(pilot_output_rows),
            "selected_total": selected_total,
            "formal_input_unique_images": len(formal_image_ids),
            "formal_selected_unique_images": len({str(row["image_id"]) for row in formal_output_rows}),
            "pilot_selected_unique_images": len({str(row["image_id"]) for row in pilot_output_rows}),
            "pilot_formal_image_overlap": len({str(row["image_id"]) for row in pilot_output_rows} & formal_image_ids),
            "reviewed_rows": len(candidates),
            "eligible_rows": sum(candidate.status == "eligible" for candidate in candidates),
        },
        "fallbacks": {"formal": formal_fallbacks, "pilot": pilot_fallbacks},
        "files": artifact_records,
        "review_mapping": str(mapping_path.resolve()),
        "hashes": str(hashes_path.resolve()),
        "v2_unchanged": True,
    }
    _write_json_exclusive(manifest_path, final_manifest)

    return FinalizeResult(
        output_dir=output,
        pilot_path=pilot_path,
        formal_path=formal_path,
        review_mapping_path=mapping_path,
        hashes_path=hashes_path,
        manifest_path=manifest_path,
        counts={
            "pilot": pilot_counts,
            "formal": formal_counts,
            "formal_fallbacks": formal_fallbacks,
            "pilot_fallbacks": pilot_fallbacks,
        },
    )


def _add_common_quota_args(parser: argparse.ArgumentParser, *, reserve_mode: bool) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    if reserve_mode:
        parser.add_argument("--pilot-per-task", type=int, default=DEFAULT_RESERVE_PER_TASK["pilot"])
        parser.add_argument("--formal-per-task", type=int, default=DEFAULT_RESERVE_PER_TASK["formal"])
    else:
        parser.add_argument("--pilot-per-task", type=int, default=DEFAULT_FINAL_PER_TASK["pilot"])
        parser.add_argument("--formal-per-task", type=int, default=DEFAULT_FINAL_PER_TASK["formal"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reserve_parser = subparsers.add_parser("reserve", help="materialize image-disjoint reserve candidates")
    reserve_parser.add_argument("--v2-root", "--data-root", dest="v2_root", type=Path, required=True)
    reserve_parser.add_argument("--output-dir", type=Path, required=True)
    reserve_parser.add_argument("--primary-pilot", "--primary-pilot-manifest", dest="primary_pilot", type=Path)
    reserve_parser.add_argument("--primary-formal", "--primary-formal-manifest", dest="primary_formal", type=Path)
    reserve_parser.add_argument("--local-image-dir", action="append", type=Path, default=[])
    _add_common_quota_args(reserve_parser, reserve_mode=True)

    finalize_parser = subparsers.add_parser("finalize", help="freeze review-qualified final manifests")
    finalize_parser.add_argument(
        "--primary-pilot", "--primary-pilot-manifest", dest="primary_pilot", type=Path, required=True
    )
    finalize_parser.add_argument(
        "--primary-formal", "--primary-formal-manifest", dest="primary_formal", type=Path, required=True
    )
    finalize_parser.add_argument(
        "--reserve-pilot", "--reserve-pilot-manifest", dest="reserve_pilot", type=Path, required=True
    )
    finalize_parser.add_argument(
        "--reserve-formal", "--reserve-formal-manifest", dest="reserve_formal", type=Path, required=True
    )
    for prefix in ("primary-pilot", "primary-formal", "reserve-pilot", "reserve-formal"):
        normalized = prefix.replace("-", "_")
        finalize_parser.add_argument(
            f"--{prefix}-key", f"--{prefix}-private-key", dest=f"{normalized}_key", type=Path, required=True
        )
        finalize_parser.add_argument(
            f"--{prefix}-review", f"--{prefix}-review-jsonl", dest=f"{normalized}_review", type=Path, required=True
        )
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    _add_common_quota_args(finalize_parser, reserve_mode=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "reserve":
            result = reserve(
                args.v2_root,
                args.output_dir,
                primary_pilot=args.primary_pilot,
                primary_formal=args.primary_formal,
                seed=args.seed,
                pilot_per_task=args.pilot_per_task,
                formal_per_task=args.formal_per_task,
                local_image_dirs=args.local_image_dir,
            )
            payload = {
                "command": "reserve",
                "output_dir": str(result.output_dir),
                "pilot": str(result.pilot_path),
                "formal": str(result.formal_path),
                "provenance": str(result.provenance_path),
                "counts": result.counts,
            }
        else:
            result = finalize(
                primary_pilot=args.primary_pilot,
                primary_formal=args.primary_formal,
                reserve_pilot=args.reserve_pilot,
                reserve_formal=args.reserve_formal,
                primary_pilot_key=args.primary_pilot_key,
                primary_formal_key=args.primary_formal_key,
                reserve_pilot_key=args.reserve_pilot_key,
                reserve_formal_key=args.reserve_formal_key,
                primary_pilot_review=args.primary_pilot_review,
                primary_formal_review=args.primary_formal_review,
                reserve_pilot_review=args.reserve_pilot_review,
                reserve_formal_review=args.reserve_formal_review,
                output_dir=args.output_dir,
                seed=args.seed,
                pilot_per_task=args.pilot_per_task,
                formal_per_task=args.formal_per_task,
            )
            payload = {
                "command": "finalize",
                "output_dir": str(result.output_dir),
                "pilot": str(result.pilot_path),
                "formal": str(result.formal_path),
                "review_mapping": str(result.review_mapping_path),
                "hashes": str(result.hashes_path),
                "manifest": str(result.manifest_path),
                "counts": result.counts,
            }
    except (QualificationError, FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"MM-GCoT qualification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
