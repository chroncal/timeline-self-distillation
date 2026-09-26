"""Standalone aggregation and calibration helpers for formal MM-GCoT v2.

The functions in this module deliberately operate only on records supplied by
the caller.  They do not know about train/dev manifests, split names, or the
independent confirmation cohort.  A formal result file is JSONL with one bbox
frame per row and at least these fields::

    sample_id, image_id, trajectory_index, arm, seed, mode, draw,
    iou, valid, completed

An image score is computed in the declared order: mean over four stochastic
draws for each trajectory, then mean over trajectories for that image.  Missing
queued image frames are zero-filled; an observed frame is also zero whenever
it is invalid or incomplete.  Acc@0.5 uses the same averaging order on each
box's strict ``IoU > 0.5`` indicator, including ``mode="sample"`` draws 0–3.
The public reports use the exact key names
``mIoU`` and ``Acc@0.5`` and also include lower-case aliases where useful for
callers that serialize metric names differently.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "image_id",
        "trajectory_index",
        "arm",
        "seed",
        "mode",
        "draw",
        "iou",
        "valid",
        "completed",
    }
)
DEFAULT_DRAWS_PER_TRAJECTORY = 4
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_TIE_TOLERANCE = 0.002
ACC_THRESHOLD = 0.5

Record = Mapping[str, Any]
ImageKey = tuple[Any, Any]
FrameKey = tuple[Any, Any, Any, Any]
RecordSource = str | Path | Iterable[Record]


def read_bbox_jsonl(source: str | Path) -> list[dict[str, Any]]:
    """Read and minimally validate bbox result records from a JSONL path.

    Blank lines are ignored.  Validation of metric values is deferred until
    aggregation, because an invalid or incomplete frame is allowed to carry a
    null IoU and must contribute zero rather than fail the whole file.
    """

    path = Path(source)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            rows.append(dict(value))
    return rows


# This name is intentionally short for scripts that already call their input
# a result file.  It is an alias rather than a second implementation.
load_bbox_records = read_bbox_jsonl


def _materialize_records(source: RecordSource | Record) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        rows = read_bbox_jsonl(source)
    elif isinstance(source, Mapping):
        rows = [dict(source)]
    else:
        rows = []
        for index, item in enumerate(source):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at iterable item {index}") from exc
            if not isinstance(item, Mapping):
                raise ValueError(f"record at iterable item {index} is not an object")
            rows.append(dict(item))

    for index, row in enumerate(rows):
        missing = REQUIRED_RECORD_FIELDS.difference(row)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"record {index} is missing required fields: {fields}")
        _validate_identity_fields(row, index=index)
    return rows


def _validate_identity_fields(row: Record, *, index: int) -> None:
    for field in ("sample_id", "image_id", "seed", "mode", "arm"):
        if row[field] is None:
            raise ValueError(f"record {index} has null {field}")
        try:
            hash(row[field])
        except TypeError as exc:
            raise ValueError(f"record {index} has unhashable {field}") from exc

    for field in ("trajectory_index", "draw"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"record {index} has invalid nonnegative integer {field}: {value!r}")


def _as_bool(value: Any, *, field: str, index: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"record {index} has non-boolean {field}: {value!r}")


def _frame_iou(row: Record, *, index: int) -> tuple[float, bool]:
    valid = _as_bool(row["valid"], field="valid", index=index)
    completed = _as_bool(row["completed"], field="completed", index=index)
    if not valid or not completed:
        return 0.0, False

    try:
        value = float(row["iou"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {index} has non-numeric IoU for a valid frame") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"record {index} has IoU outside [0, 1]: {value!r}")
    return value, True


def _image_key(row: Record) -> ImageKey:
    return row["sample_id"], row["image_id"]


def _record_frame_key(row: Record) -> FrameKey:
    return (
        row["sample_id"],
        row["image_id"],
        row["trajectory_index"],
        row["draw"],
    )


def _stable_key(value: Any) -> tuple[str, str]:
    """Sort heterogeneous JSON scalar identifiers without comparing types."""

    return type(value).__name__, repr(value)


def _normalise_queued_images(
    queued_images: Iterable[Any] | str | Path | None,
) -> list[ImageKey] | None:
    if queued_images is None:
        return None

    if isinstance(queued_images, (str, Path)):
        queued_items: Iterable[Any] = read_bbox_jsonl(queued_images)
    elif isinstance(queued_images, Mapping):
        # A single manifest row is a useful and unambiguous convenience form.
        if "sample_id" in queued_images or "image_id" in queued_images:
            queued_items = [queued_images]
        else:
            queued_items = queued_images.items()
    else:
        queued_items = queued_images

    result: list[ImageKey] = []
    seen: set[ImageKey] = set()
    for index, item in enumerate(queued_items):
        if isinstance(item, Mapping):
            if "sample_id" not in item or "image_id" not in item:
                raise ValueError(f"queued image {index} lacks sample_id or image_id")
            key = item["sample_id"], item["image_id"]
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            key = item[0], item[1]
        else:
            raise ValueError(
                "queued_images entries must be manifest rows or (sample_id, image_id) pairs"
            )
        try:
            hash(key[0])
            hash(key[1])
        except TypeError as exc:
            raise ValueError(f"queued image {index} has an unhashable identifier") from exc
        if key in seen:
            raise ValueError(f"queued image appears more than once: {key!r}")
        seen.add(key)
        result.append(key)
    return result


def _select_mode(rows: Sequence[dict[str, Any]], mode: Any | None) -> list[dict[str, Any]]:
    if mode is not None:
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            raise ValueError(f"no records found for mode {mode!r}")
        return selected
    modes = {row["mode"] for row in rows}
    if len(modes) > 1:
        names = ", ".join(repr(value) for value in sorted(modes, key=_stable_key))
        raise ValueError(f"multiple modes are present; choose one explicitly: {names}")
    return list(rows)


def _frame_values(
    rows: Sequence[dict[str, Any]],
    *,
    draws_per_trajectory: int,
) -> tuple[
    dict[tuple[Any, ImageKey, int, int], float],
    dict[tuple[Any, ImageKey, int, int], bool],
    dict[Any, set[int]],
]:
    if draws_per_trajectory < 1:
        raise ValueError("draws_per_trajectory must be positive")

    values: dict[tuple[Any, ImageKey, int, int], float] = {}
    observed_valid: dict[tuple[Any, ImageKey, int, int], bool] = {}
    trajectory_ids: dict[Any, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        draw = row["draw"]
        if draw >= draws_per_trajectory:
            raise ValueError(
                f"draw {draw} is outside 0..{draws_per_trajectory - 1} in record {index}"
            )
        key = (row["seed"], _image_key(row), row["trajectory_index"], draw)
        if key in values:
            raise ValueError(f"duplicate bbox frame key: {key!r}")
        value, is_valid = _frame_iou(row, index=index)
        values[key] = value
        observed_valid[key] = is_valid
        trajectory_ids[row["seed"]].add(row["trajectory_index"])
    return values, observed_valid, trajectory_ids


def _metric(value_by_image: Mapping[Any, float]) -> float:
    if not value_by_image:
        raise ValueError("cannot compute an image metric with no queued images")
    return sum(value_by_image.values()) / len(value_by_image)


def _aggregate_internal(
    rows: Sequence[dict[str, Any]],
    queued_images: Iterable[Any] | str | Path | None,
    *,
    mode: Any | None,
    draws_per_trajectory: int,
) -> dict[str, Any]:
    selected = _select_mode(rows, mode)
    queue = _normalise_queued_images(queued_images)
    values, observed_valid, trajectory_ids = _frame_values(
        selected,
        draws_per_trajectory=draws_per_trajectory,
    )

    observed_images = {_image_key(row) for row in selected}
    if queue is None:
        queue = sorted(observed_images, key=_stable_key)
    queue_set = set(queue)
    extra_images = observed_images.difference(queue_set)
    if extra_images:
        raise ValueError(f"records contain images outside queued_images: {sorted(extra_images, key=_stable_key)!r}")
    if not queue:
        raise ValueError("queued_images must contain at least one image")

    # A missing frame or a missing queued image is represented by zero.  This
    # is what keeps every queued image in the denominator while still allowing
    # an incomplete worker record to be reported.
    seed_image_scores: dict[Any, dict[ImageKey, float]] = {}
    seed_image_acc: dict[Any, dict[ImageKey, float]] = {}
    seed_trajectory_scores: dict[Any, dict[ImageKey, dict[int, float]]] = {}
    seed_frame_counts: dict[Any, dict[str, int]] = {}
    for seed in sorted(trajectory_ids, key=_stable_key):
        trajectories = sorted(trajectory_ids[seed])
        image_scores: dict[ImageKey, float] = {}
        image_acc: dict[ImageKey, float] = {}
        image_trajectories: dict[ImageKey, dict[int, float]] = {}
        expected_frame_count = len(queue) * len(trajectories) * draws_per_trajectory
        observed_frame_count = 0
        valid_frame_count = 0
        for image in queue:
            trajectory_scores: dict[int, float] = {}
            trajectory_acc: dict[int, float] = {}
            for trajectory in trajectories:
                draws = [
                    values.get((seed, image, trajectory, draw), 0.0)
                    for draw in range(draws_per_trajectory)
                ]
                trajectory_scores[trajectory] = sum(draws) / draws_per_trajectory
                trajectory_acc[trajectory] = (
                    sum(value > ACC_THRESHOLD for value in draws) / draws_per_trajectory
                )
                for draw in range(draws_per_trajectory):
                    key = (seed, image, trajectory, draw)
                    if key in values:
                        observed_frame_count += 1
                        valid_frame_count += int(observed_valid[key])
            image_score = (
                sum(trajectory_scores.values()) / len(trajectories)
                if trajectories
                else 0.0
            )
            image_scores[image] = image_score
            image_acc[image] = (
                sum(trajectory_acc.values()) / len(trajectories)
                if trajectories
                else 0.0
            )
            image_trajectories[image] = trajectory_scores
        seed_image_scores[seed] = image_scores
        seed_image_acc[seed] = image_acc
        seed_trajectory_scores[seed] = image_trajectories
        seed_frame_counts[seed] = {
            "expected": expected_frame_count,
            "observed": observed_frame_count,
            "valid_completed": valid_frame_count,
        }

    # Keep the internal score maps private to the report builder, but return
    # them here so paired bootstrap can sample whole images without flattening
    # their four draws or their trajectories.
    return {
        "queue": queue,
        "mode": selected[0]["mode"] if selected else mode,
        "seed_image_scores": seed_image_scores,
        "seed_image_acc": seed_image_acc,
        "seed_trajectory_scores": seed_trajectory_scores,
        "seed_frame_counts": seed_frame_counts,
        "draws_per_trajectory": draws_per_trajectory,
    }


def _image_rows(
    queue: Sequence[ImageKey],
    scores: Mapping[ImageKey, float],
    accuracy: Mapping[ImageKey, float],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": image[0],
            "image_id": image[1],
            "mIoU": scores[image],
            "Acc@0.5": accuracy[image],
        }
        for image in queue
    ]


def _public_seed_report(
    internal: Mapping[str, Any],
) -> dict[Any, dict[str, Any]]:
    queue = internal["queue"]
    reports: dict[Any, dict[str, Any]] = {}
    for seed, scores in internal["seed_image_scores"].items():
        image_acc = internal["seed_image_acc"][seed]
        miou, accuracy = _metric(scores), _metric(image_acc)
        frame_counts = internal["seed_frame_counts"][seed]
        reports[seed] = {
            "mIoU": miou,
            "miou": miou,
            "Acc@0.5": accuracy,
            "acc_at_0_5": accuracy,
            "n_images": len(queue),
            "image_count": len(queue),
            "per_image": _image_rows(queue, scores, image_acc),
            "trajectory_scores": [
                {
                    "sample_id": image[0],
                    "image_id": image[1],
                    "trajectories": internal["seed_trajectory_scores"][seed][image],
                }
                for image in queue
            ],
            "frame_counts": frame_counts,
        }
    return reports


def _public_system_report(internal: Mapping[str, Any]) -> dict[str, Any]:
    queue = internal["queue"]
    all_scores = {
        (seed, image): score
        for seed, image_scores in internal["seed_image_scores"].items()
        for image, score in image_scores.items()
    }
    all_acc = {
        (seed, image): score
        for seed, image_scores in internal["seed_image_acc"].items()
        for image, score in image_scores.items()
    }
    if all_scores:
        miou, accuracy = _metric(all_scores), _metric(all_acc)
    else:
        # A queued system with no emitted rows is a complete failure.  It is
        # still a valid report with every queued image contributing zero.
        miou, accuracy = 0.0, 0.0
    per_seed = _public_seed_report(internal)
    return {
        "mIoU": miou,
        "miou": miou,
        "Acc@0.5": accuracy,
        "acc_at_0_5": accuracy,
        "n_images": len(queue),
        "image_count": len(queue),
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "by_seed": per_seed,
        "mode": internal["mode"],
        "draws_per_trajectory": internal["draws_per_trajectory"],
    }


def aggregate_records(
    records: RecordSource | Record,
    queued_images: Iterable[Any] | str | Path | None = None,
    *,
    mode: Any | None = None,
    draws_per_trajectory: int = DEFAULT_DRAWS_PER_TRAJECTORY,
) -> dict[str, Any]:
    """Aggregate one comparable system using the formal v2 image metric.

    ``queued_images`` should contain one ``(sample_id, image_id)`` pair or
    manifest row per queued image.  If omitted, image keys present in the
    records are used; callers must provide the queue to include images with no
    result rows in the denominator.
    """

    rows = _materialize_records(records)
    internal = _aggregate_internal(
        rows,
        queued_images,
        mode=mode,
        draws_per_trajectory=draws_per_trajectory,
    )
    report = _public_system_report(internal)
    report["queued_images"] = [
        {"sample_id": image[0], "image_id": image[1]} for image in internal["queue"]
    ]
    return report


def _system_sources(
    systems: Mapping[Any, RecordSource | Record],
) -> dict[Any, list[dict[str, Any]]]:
    return {name: _materialize_records(source) for name, source in systems.items()}


def assert_comparable_keys(
    systems: Mapping[Any, RecordSource | Record] | RecordSource | Record,
    other: RecordSource | Record | None = None,
    *,
    mode: Any | None = None,
) -> bool:
    """Assert frame-key equality for every comparable seed and mode.

    The compared key is exactly ``(sample_id, image_id, trajectory_index,
    draw)``.  Seed and mode are grouping dimensions, so a missing seed or a
    mode-specific key mismatch cannot be hidden by another group.  Duplicate
    keys are rejected as well.  An :class:`AssertionError` is intentional here:
    a paired comparison with unequal stochastic frames is invalid by contract.
    """

    if other is not None:
        system_mapping: Mapping[Any, RecordSource | Record] = {
            "left": systems,  # type: ignore[dict-item]
            "right": other,
        }
    elif isinstance(systems, Mapping) and not REQUIRED_RECORD_FIELDS.issubset(systems):
        system_mapping = systems
    else:
        raise TypeError("systems must be a name-to-records mapping or two record sources")
    if len(system_mapping) < 2:
        return True
    grouped: dict[Any, dict[tuple[Any, Any], set[FrameKey]]] = {}
    for name, source in system_mapping.items():
        rows = _materialize_records(source)
        if mode is not None:
            rows = [row for row in rows if row["mode"] == mode]
        groups: dict[tuple[Any, Any], set[FrameKey]] = defaultdict(set)
        for row in rows:
            group = row["seed"], row["mode"]
            key = _record_frame_key(row)
            if key in groups[group]:
                raise AssertionError(f"duplicate comparable key in system {name!r}: {group + key!r}")
            groups[group].add(key)
        grouped[name] = groups

    names = list(grouped)
    reference_name = names[0]
    reference = grouped[reference_name]
    for name in names[1:]:
        candidate = grouped[name]
        if set(candidate) != set(reference):
            raise AssertionError(
                f"comparable seed/mode groups differ between {reference_name!r} and {name!r}: "
                f"{set(reference)!r} != {set(candidate)!r}"
            )
        for group in reference:
            if candidate[group] != reference[group]:
                missing = reference[group].difference(candidate[group])
                extra = candidate[group].difference(reference[group])
                raise AssertionError(
                    f"comparable keys differ for group {group!r} between "
                    f"{reference_name!r} and {name!r}; missing={sorted(missing, key=_stable_key)!r}, "
                    f"extra={sorted(extra, key=_stable_key)!r}"
                )
    return True


# Public wording used by a few callers and tests.
assert_same_keys = assert_comparable_keys


def _as_image_score_map(values: Mapping[Any, Any] | Sequence[float]) -> dict[Any, float]:
    if isinstance(values, Mapping):
        result = {key: float(value) for key, value in values.items()}
    else:
        result = {index: float(value) for index, value in enumerate(values)}
    if not result:
        raise ValueError("paired bootstrap requires at least one image")
    for key, value in result.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite image score for {key!r}: {value!r}")
    return result


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_image_bootstrap(
    baseline_scores: Mapping[Any, float] | Sequence[float],
    candidate_scores: Mapping[Any, float] | Sequence[float],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 20260923,
) -> dict[str, Any]:
    """Bootstrap a candidate-minus-baseline image-level delta.

    Resampling happens only at the image level.  Each image score already
    contains all four draws for every trajectory, so no draw or trajectory is
    resampled independently.  The returned ``ci95`` is a percentile 95%
    interval for the mean paired delta.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    baseline = _as_image_score_map(baseline_scores)
    candidate = _as_image_score_map(candidate_scores)
    if set(baseline) != set(candidate):
        missing = set(baseline).difference(candidate)
        extra = set(candidate).difference(baseline)
        raise AssertionError(f"paired image keys differ; missing={missing!r}, extra={extra!r}")

    keys = list(baseline)
    deltas = [candidate[key] - baseline[key] for key in keys]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(replicates):
        total = 0.0
        for _ in keys:
            total += deltas[rng.randrange(len(deltas))]
        bootstrap_means.append(total / len(deltas))
    bootstrap_means.sort()
    lower = _percentile(bootstrap_means, 0.025)
    upper = _percentile(bootstrap_means, 0.975)
    return {
        "delta": point,
        "candidate_minus_baseline": point,
        "baseline_minus_candidate": -point,
        "ci95": [lower, upper],
        "lower": lower,
        "upper": upper,
        "n_images": len(keys),
        "replicates": replicates,
        "seed": seed,
    }


# Concise spelling for callers that use "paired bootstrap" as the metric name.
paired_bootstrap = paired_image_bootstrap


def _default_baseline(names: Sequence[Any]) -> Any:
    preferred = {"bbox_sft", "sft", "bbox-sft", "bboxSFT"}
    for name in names:
        if str(name) in preferred or str(name).lower() in {value.lower() for value in preferred}:
            return name
    return names[0]


def _normalise_comparisons(
    names: Sequence[Any],
    comparisons: Mapping[Any, Sequence[Any]] | Sequence[Sequence[Any]] | None,
) -> list[tuple[Any, Any, Any]]:
    if comparisons is None:
        if len(names) < 2:
            return []
        baseline = _default_baseline(names)
        return [
            (f"{candidate}-{baseline}", candidate, baseline)
            for candidate in names
            if candidate != baseline
        ]
    if isinstance(comparisons, Mapping):
        result = []
        for label, pair in comparisons.items():
            if len(pair) != 2:
                raise ValueError(f"comparison {label!r} must be (candidate, baseline)")
            result.append((label, pair[0], pair[1]))
        return result
    result = []
    for pair in comparisons:
        if len(pair) != 2:
            raise ValueError("comparisons entries must be (candidate, baseline)")
        result.append((f"{pair[0]}-{pair[1]}", pair[0], pair[1]))
    return result


def aggregate_systems(
    systems: Mapping[Any, RecordSource | Record],
    queued_images: Iterable[Any] | str | Path | None = None,
    *,
    mode: Any | None = None,
    draws_per_trajectory: int = DEFAULT_DRAWS_PER_TRAJECTORY,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 20260923,
    comparisons: Mapping[Any, Sequence[Any]] | Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate comparable systems and report per-seed paired deltas."""

    if not systems:
        raise ValueError("at least one system is required")
    sources = _system_sources(systems)
    if len(sources) > 1:
        assert_comparable_keys(sources, mode=mode)

    # Materialize a caller-supplied generator once so every system uses the
    # same queued-image denominator.
    queue = _normalise_queued_images(queued_images) if queued_images is not None else None

    internal = {
        name: _aggregate_internal(
            rows,
            queue,
            mode=mode,
            draws_per_trajectory=draws_per_trajectory,
        )
        for name, rows in sources.items()
    }
    system_reports = {name: _public_system_report(value) for name, value in internal.items()}
    pair_reports: dict[Any, dict[str, Any]] = {}
    for label, candidate_name, baseline_name in _normalise_comparisons(
        list(sources), comparisons
    ):
        if candidate_name not in internal or baseline_name not in internal:
            raise KeyError(f"unknown system in comparison {label!r}")
        candidate = internal[candidate_name]
        baseline = internal[baseline_name]
        candidate_seeds = set(candidate["seed_image_scores"])
        baseline_seeds = set(baseline["seed_image_scores"])
        if candidate_seeds != baseline_seeds:
            raise AssertionError(
                f"paired seed sets differ for {candidate_name!r} and {baseline_name!r}"
            )

        per_seed: dict[Any, dict[str, Any]] = {}
        for seed_value in sorted(candidate_seeds, key=_stable_key):
            candidate_images = candidate["seed_image_scores"][seed_value]
            baseline_images = baseline["seed_image_scores"][seed_value]
            miou_bootstrap = paired_image_bootstrap(
                baseline_images,
                candidate_images,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            candidate_acc = candidate["seed_image_acc"][seed_value]
            baseline_acc = baseline["seed_image_acc"][seed_value]
            acc_bootstrap = paired_image_bootstrap(
                baseline_acc,
                candidate_acc,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 1,
            )
            per_seed[seed_value] = {
                "mIoU": miou_bootstrap,
                "Acc@0.5": acc_bootstrap,
                "delta": miou_bootstrap["delta"],
                "delta_mIoU": miou_bootstrap["delta"],
                "delta_Acc@0.5": acc_bootstrap["delta"],
                "ci95": miou_bootstrap["ci95"],
            }

        # The overall paired bootstrap samples images, while retaining every
        # formal seed inside each sampled image.  Flattening seed-image pairs
        # would give a seed with four draws the same bootstrap status as an
        # additional image, which is not the formal image-level unit.
        image_keys = candidate["queue"]
        all_candidate = {
            image: sum(
                candidate["seed_image_scores"][seed_value][image]
                for seed_value in candidate_seeds
            ) / len(candidate_seeds)
            for image in image_keys
        }
        all_baseline = {
            image: sum(
                baseline["seed_image_scores"][seed_value][image]
                for seed_value in baseline_seeds
            ) / len(baseline_seeds)
            for image in image_keys
        }
        overall_miou = paired_image_bootstrap(
            all_baseline,
            all_candidate,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        overall_candidate_acc = {
            image: sum(
                candidate["seed_image_acc"][seed_value][image]
                for seed_value in candidate_seeds
            ) / len(candidate_seeds)
            for image in image_keys
        }
        overall_baseline_acc = {
            image: sum(
                baseline["seed_image_acc"][seed_value][image]
                for seed_value in baseline_seeds
            ) / len(baseline_seeds)
            for image in image_keys
        }
        overall_acc = paired_image_bootstrap(
            overall_baseline_acc,
            overall_candidate_acc,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        )
        pair_reports[label] = {
            "candidate": candidate_name,
            "baseline": baseline_name,
            "per_seed": per_seed,
            "by_seed": per_seed,
            "mIoU": overall_miou,
            "Acc@0.5": overall_acc,
            "delta": overall_miou["delta"],
            "delta_mIoU": overall_miou["delta"],
            "delta_Acc@0.5": overall_acc["delta"],
            "bootstrap_replicates": bootstrap_replicates,
        }

    queue = next(iter(internal.values()))["queue"]
    return {
        "schema_version": "formal_v2",
        "queued_images": [
            {"sample_id": image[0], "image_id": image[1]} for image in queue
        ],
        "n_images": len(queue),
        "draws_per_trajectory": draws_per_trajectory,
        "bootstrap_replicates": bootstrap_replicates,
        "systems": system_reports,
        "per_system": system_reports,
        "paired": pair_reports,
        "paired_deltas": pair_reports,
    }


def aggregate_formal_v2(
    records: RecordSource | Record | Mapping[Any, RecordSource | Record],
    queued_images: Iterable[Any] | str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Aggregate a combined arm JSONL file or an explicit system mapping.

    A combined source is grouped by its required ``arm`` field.  An explicit
    mapping is useful when two files use different human-readable system
    labels.  All remaining options are those of :func:`aggregate_systems`.
    """

    if isinstance(records, Mapping) and not REQUIRED_RECORD_FIELDS.issubset(records):
        systems = records
    else:
        rows = _materialize_records(records)  # type: ignore[arg-type]
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["arm"]].append(row)
        systems = grouped
    return aggregate_systems(systems, queued_images, **kwargs)


aggregate_evaluation = aggregate_formal_v2


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite: {value!r}")
    return result


def _metric_from_candidate(value: Any, *, label: str) -> float:
    if isinstance(value, Mapping):
        for key in ("mIoU", "miou", "mean_iou", "dev_miou", "score", "value"):
            if key in value:
                return _as_float(value[key], label=label)
        if "per_seed" in value and isinstance(value["per_seed"], Mapping):
            scores = [
                _metric_from_candidate(seed_value, label=label)
                for seed_value in value["per_seed"].values()
            ]
            if scores:
                return sum(scores) / len(scores)
    return _as_float(value, label=label)


def _normalise_lr_scores(scores: Any) -> list[tuple[float, float]]:
    if isinstance(scores, Mapping):
        return [
            (_as_float(rate, label="learning rate"), _metric_from_candidate(value, label="mIoU"))
            for rate, value in scores.items()
        ]
    result = []
    for index, row in enumerate(scores):
        if isinstance(row, Mapping):
            rate_key = next(
                (key for key in ("learning_rate", "lr", "sft_lr") if key in row),
                None,
            )
            if rate_key is None:
                raise ValueError(f"learning-rate candidate {index} lacks learning_rate/lr")
            score_key = next(
                (key for key in ("mIoU", "miou", "mean_iou", "dev_miou", "score") if key in row),
                None,
            )
            if score_key is None:
                raise ValueError(f"learning-rate candidate {index} lacks an mIoU score")
            result.append(
                (
                    _as_float(row[rate_key], label="learning rate"),
                    _as_float(row[score_key], label="mIoU"),
                )
            )
        else:
            try:
                rate, score = row
            except (TypeError, ValueError) as exc:
                raise ValueError("learning-rate candidates must be pairs or mappings") from exc
            result.append(
                (_as_float(rate, label="learning rate"), _as_float(score, label="mIoU"))
            )
    return result


def _select_max_with_tie(
    candidates: Sequence[tuple[float, float]],
    *,
    tolerance: float,
    label: str,
) -> float:
    if not candidates:
        raise ValueError(f"no {label} candidates supplied")
    if tolerance < 0:
        raise ValueError("tie tolerance must be nonnegative")
    for parameter, score in candidates:
        if parameter <= 0 or not math.isfinite(parameter):
            raise ValueError(f"{label} must be positive and finite: {parameter!r}")
        if not math.isfinite(score):
            raise ValueError(f"{label} score must be finite: {score!r}")
    best_score = max(score for _, score in candidates)
    eligible = [
        parameter for parameter, score in candidates if best_score - score < tolerance
    ]
    return min(eligible)


def select_learning_rate(
    dev_miou_by_lr: Mapping[Any, Any] | Iterable[Any],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> float:
    """Select the best dev LR, preferring the lower LR within a strict gap."""

    return _select_max_with_tie(
        _normalise_lr_scores(dev_miou_by_lr),
        tolerance=tie_tolerance,
        label="learning rate",
    )


select_lr = select_learning_rate


_R_KEYS = {"r", "r_opd", "r-opd", "reset"}
_E_KEYS = {"e", "e_opd", "e-opd", "early"}


def _arm_score(value: Any, *, label: str, queued_images: Any = None) -> float:
    if queued_images is not None and isinstance(value, (str, Path)):
        return aggregate_records(value, queued_images)["mIoU"]
    if queued_images is not None and isinstance(value, Iterable) and not isinstance(value, Mapping):
        materialized = list(value)
        if materialized and isinstance(materialized[0], Mapping) and REQUIRED_RECORD_FIELDS.issubset(materialized[0]):
            return aggregate_records(materialized, queued_images)["mIoU"]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{label} has no scores")
        # Lists of numeric per-seed scores are averaged before the R/E average.
        return sum(_metric_from_candidate(item, label=label) for item in value) / len(value)
    return _metric_from_candidate(value, label=label)


def _extract_re_scores(
    value: Any,
    *,
    label: str,
    queued_images: Any = None,
) -> tuple[float, float]:
    if isinstance(value, Mapping):
        arms: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _R_KEYS:
                arms["R"] = item
            elif normalized in _E_KEYS:
                arms["E"] = item
        if "R" in arms and "E" in arms:
            return (
                _arm_score(arms["R"], label=f"{label} R", queued_images=queued_images),
                _arm_score(arms["E"], label=f"{label} E", queued_images=queued_images),
            )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return (
            _arm_score(value[0], label=f"{label} R", queued_images=queued_images),
            _arm_score(value[1], label=f"{label} E", queued_images=queued_images),
        )
    raise ValueError(f"lambda candidate {label!r} must provide both R and E dev mIoU")


def _normalise_lambda_scores(
    candidates: Any,
    *,
    queued_images: Any = None,
) -> list[tuple[float, float]]:
    if isinstance(candidates, Mapping):
        result = []
        for parameter, value in candidates.items():
            r_score, e_score = _extract_re_scores(
                value,
                label=str(parameter),
                queued_images=queued_images,
            )
            result.append(
                (
                    _as_float(parameter, label="lambda"),
                    (r_score + e_score) / 2.0,
                )
            )
        return result

    grouped: dict[float, dict[str, list[Any]]] = defaultdict(lambda: {"R": [], "E": []})
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise ValueError("lambda candidates must be mappings or a lambda-to-R/E mapping")
        lambda_key = next(
            (key for key in ("lambda_opd", "lambda", "opd_lambda") if key in row),
            None,
        )
        arm_key = next((key for key in ("arm", "system") if key in row), None)
        score_key = next(
            (key for key in ("mIoU", "miou", "mean_iou", "dev_miou", "score") if key in row),
            None,
        )
        if lambda_key is None or arm_key is None or score_key is None:
            raise ValueError(f"lambda candidate {index} lacks lambda, arm, or mIoU")
        arm = str(row[arm_key]).lower()
        normalized_arm = "R" if arm in _R_KEYS else "E" if arm in _E_KEYS else None
        if normalized_arm is None:
            continue
        grouped[_as_float(row[lambda_key], label="lambda")][normalized_arm].append(row[score_key])
    result = []
    for parameter, arm_values in grouped.items():
        if not arm_values["R"] or not arm_values["E"]:
            raise ValueError(f"lambda {parameter!r} lacks R or E dev mIoU")
        r_score = sum(
            _arm_score(value, label=f"{parameter} R", queued_images=queued_images)
            for value in arm_values["R"]
        ) / len(arm_values["R"])
        e_score = sum(
            _arm_score(value, label=f"{parameter} E", queued_images=queued_images)
            for value in arm_values["E"]
        ) / len(arm_values["E"])
        result.append((parameter, (r_score + e_score) / 2.0))
    return result


def select_lambda(
    dev_r_e_miou_by_lambda: Mapping[Any, Any] | Iterable[Any],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    queued_images: Any = None,
) -> float:
    """Select lambda by average R/E dev mIoU with the lower-lambda tie rule."""

    return _select_max_with_tie(
        _normalise_lambda_scores(dev_r_e_miou_by_lambda, queued_images=queued_images),
        tolerance=tie_tolerance,
        label="lambda",
    )


select_opd_lambda = select_lambda


__all__ = [
    "ACC_THRESHOLD",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_DRAWS_PER_TRAJECTORY",
    "DEFAULT_TIE_TOLERANCE",
    "REQUIRED_RECORD_FIELDS",
    "aggregate_evaluation",
    "aggregate_formal_v2",
    "aggregate_records",
    "aggregate_systems",
    "assert_comparable_keys",
    "assert_same_keys",
    "load_bbox_records",
    "paired_bootstrap",
    "paired_image_bootstrap",
    "read_bbox_jsonl",
    "select_lambda",
    "select_learning_rate",
    "select_lr",
    "select_opd_lambda",
]
