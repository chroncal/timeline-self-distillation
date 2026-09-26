"""Render the post-run report for the formal MM-GCoT v2 evaluation.

The final evaluator writes one ``result.json`` for each final cohort.  This
module consumes those summaries without looking at raw records, changing a
metric, or selecting a threshold after the run.  The only values recomputed
here are per-image paired differences used for an explicitly illustrative
plot and fixed-rule R-SFT examples.

Typical use::

    python -m mmgcot_timeline_training.formal_report_v2 \
        --final-evaluation outputs/.../final_evaluation \
        --output-dir outputs/.../formal_report_v2
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import time
from typing import Any


COHORTS = ("independent48", "test200_retest")
EXPECTED_INTERPRETATION = {
    "independent48": "independent_confirmation",
    "test200_retest": "diagnostic_selection_conditioned_retest",
}
COHORT_DESCRIPTION = {
    "independent48": "independent confirmation",
    "test200_retest": "diagnostic selection-conditioned retest",
}
ARM_LABELS = {
    "bbox_sft": "SFT",
    "r_opd": "R",
    "e_opd": "E",
}
COMPARISONS = (
    ("R-SFT", "r_opd", "bbox_sft"),
    ("E-SFT", "e_opd", "bbox_sft"),
    ("R-E", "r_opd", "e_opd"),
)
EXAMPLE_COUNT = 3
DEFAULT_REPORT_NAME = "formal_report_v2.md"
DEFAULT_PLOT_NAME = "paired_image_deltas.png"
DEFAULT_WAIT_SECONDS = 60.0
DEFAULT_POLL_SECONDS = 5.0

_ARM_ALIASES = {
    "bbox_sft": "bbox_sft",
    "bbox-sft": "bbox_sft",
    "sft": "bbox_sft",
    "r_opd": "r_opd",
    "r-opd": "r_opd",
    "r": "r_opd",
    "reset": "r_opd",
    "e_opd": "e_opd",
    "e-opd": "e_opd",
    "e": "e_opd",
    "early": "e_opd",
}


def _stable_key(value: Any) -> tuple[str, str]:
    """Sort JSON scalar identifiers without comparing unlike Python types."""

    return type(value).__name__, repr(value)


def _seed_sort_key(value: Any) -> tuple[int, float | str, str]:
    text = str(value)
    try:
        return 0, float(text), text
    except ValueError:
        return 1, text, text


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got boolean {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite: {value!r}")
    return number


def _bounded_metric(value: Any, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]: {value!r}")
    return number


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _metric_value(value: Mapping[str, Any], *, label: str) -> float:
    for key in ("mIoU", "miou"):
        if key in value:
            return _bounded_metric(value[key], label=label)
    raise ValueError(f"{label} is missing mIoU")


def _ci95(value: Mapping[str, Any], *, label: str) -> tuple[float, float]:
    raw = value.get("ci95", value.get("ci_95"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"{label} must contain a two-element ci95")
    lower = _finite_number(raw[0], label=f"{label}[0]")
    upper = _finite_number(raw[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} has lower bound above upper bound")
    return lower, upper


def _image_key(sample_id: Any, image_id: Any, *, label: str) -> tuple[Any, Any]:
    key = (sample_id, image_id)
    try:
        hash(key[0])
        hash(key[1])
    except TypeError as exc:
        raise ValueError(f"{label} has an unhashable image identifier") from exc
    return key


def _queue(result: Mapping[str, Any], *, label: str) -> list[tuple[Any, Any]]:
    raw_queue = result.get("queued_images")
    if not isinstance(raw_queue, Sequence) or isinstance(raw_queue, (str, bytes)):
        raise ValueError(f"{label}.queued_images must be a non-empty list")
    queue: list[tuple[Any, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for index, row in enumerate(raw_queue):
        item = _mapping(row, label=f"{label}.queued_images[{index}]")
        if "sample_id" not in item or "image_id" not in item:
            raise ValueError(f"{label}.queued_images[{index}] lacks sample_id or image_id")
        key = _image_key(
            item["sample_id"], item["image_id"],
            label=f"{label}.queued_images[{index}]",
        )
        if key in seen:
            raise ValueError(f"{label}.queued_images contains duplicate image {key!r}")
        seen.add(key)
        queue.append(key)
    if not queue:
        raise ValueError(f"{label}.queued_images must not be empty")
    return queue


def _per_image_scores(
    seed_report: Mapping[str, Any],
    *,
    queue: Sequence[tuple[Any, Any]],
    label: str,
) -> dict[tuple[Any, Any], float]:
    raw_rows = seed_report.get("per_image")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError(f"{label}.per_image must be a list")
    scores: dict[tuple[Any, Any], float] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _mapping(raw_row, label=f"{label}.per_image[{index}]")
        if "sample_id" not in row or "image_id" not in row:
            raise ValueError(f"{label}.per_image[{index}] lacks image identifiers")
        key = _image_key(
            row["sample_id"], row["image_id"],
            label=f"{label}.per_image[{index}]",
        )
        if key in scores:
            raise ValueError(f"{label}.per_image contains duplicate image {key!r}")
        scores[key] = _metric_value(row, label=f"{label}.per_image[{index}].mIoU")
    expected = set(queue)
    actual = set(scores)
    if actual != expected:
        missing = sorted(expected - actual, key=_stable_key)
        extra = sorted(actual - expected, key=_stable_key)
        raise ValueError(
            f"{label}.per_image does not match queued_images; missing={missing!r}, extra={extra!r}"
        )
    return scores


def _frame_counts(seed_report: Mapping[str, Any], *, label: str) -> dict[str, int]:
    counts = _mapping(seed_report.get("frame_counts"), label=f"{label}.frame_counts")
    result: dict[str, int] = {}
    for key in ("expected", "observed", "valid_completed"):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}.frame_counts.{key} must be an integer")
        if value < 0:
            raise ValueError(f"{label}.frame_counts.{key} must be nonnegative")
        result[key] = value
    if result["expected"] < 1:
        raise ValueError(f"{label}.frame_counts.expected must be positive")
    if result["observed"] > result["expected"]:
        raise ValueError(f"{label}.frame_counts.observed exceeds expected")
    if result["valid_completed"] > result["observed"]:
        raise ValueError(f"{label}.frame_counts.valid_completed exceeds observed")
    return result


def _validate_systems(result: Mapping[str, Any], *, label: str) -> tuple[list[Any], list[tuple[Any, Any]]]:
    systems = _mapping(result.get("systems"), label=f"{label}.systems")
    missing_arms = [arm for arm in ARM_LABELS if arm not in systems]
    if missing_arms:
        raise ValueError(f"{label}.systems is missing required arms: {missing_arms}")

    queue = _queue(result, label=label)
    expected_seeds: set[Any] | None = None
    for arm in ARM_LABELS:
        system = _mapping(systems[arm], label=f"{label}.systems.{arm}")
        _metric_value(system, label=f"{label}.systems.{arm}.mIoU")
        per_seed = system.get("per_seed", system.get("by_seed"))
        per_seed = _mapping(per_seed, label=f"{label}.systems.{arm}.per_seed")
        if not per_seed:
            raise ValueError(f"{label}.systems.{arm}.per_seed must not be empty")
        seed_set = set(per_seed)
        if expected_seeds is None:
            expected_seeds = seed_set
        elif seed_set != expected_seeds:
            raise ValueError(f"{label}.systems seed sets differ between arms")
        for seed, raw_seed_report in per_seed.items():
            seed_report = _mapping(
                raw_seed_report,
                label=f"{label}.systems.{arm}.per_seed[{seed!r}]",
            )
            seed_label = f"{label}.systems.{arm}.per_seed[{seed!r}]"
            _metric_value(seed_report, label=f"{seed_label}.mIoU")
            _per_image_scores(seed_report, queue=queue, label=seed_label)
            _frame_counts(seed_report, label=seed_label)
    return sorted(expected_seeds or (), key=_seed_sort_key), queue


def _normalise_arm(value: Any) -> str | None:
    text = str(value).strip().lower().replace(" ", "_")
    return _ARM_ALIASES.get(text)


def _pair_matches(label: Any, candidate: str, baseline: str) -> bool:
    text = str(label).strip().lower()
    exact = {
        f"{candidate}-{baseline}",
        f"{ARM_LABELS[candidate]}-{ARM_LABELS[baseline]}".lower(),
    }
    if text in exact:
        return True
    parts = [part for part in text.replace("/", "-").replace(":", "-").split("-") if part]
    if len(parts) == 2:
        return _normalise_arm(parts[0]) == candidate and _normalise_arm(parts[1]) == baseline
    return False


def _find_pair(
    paired: Mapping[str, Any],
    *,
    candidate: str,
    baseline: str,
    label: str,
) -> Mapping[str, Any]:
    direct_key = f"{candidate}-{baseline}"
    if direct_key in paired:
        return _mapping(paired[direct_key], label=f"{label}.paired.{direct_key}")
    for pair_label, raw_pair in paired.items():
        pair = _mapping(raw_pair, label=f"{label}.paired.{pair_label}")
        declared_candidate = _normalise_arm(pair.get("candidate")) if "candidate" in pair else None
        declared_baseline = _normalise_arm(pair.get("baseline")) if "baseline" in pair else None
        if (declared_candidate, declared_baseline) == (candidate, baseline) or _pair_matches(
            pair_label, candidate, baseline
        ):
            return pair
    raise ValueError(f"{label}.paired is missing {candidate}-{baseline}")


def _validate_pair(
    result: Mapping[str, Any],
    *,
    candidate: str,
    baseline: str,
    seeds: Sequence[Any],
    label: str,
) -> Mapping[str, Any]:
    paired = result.get("paired", result.get("paired_deltas"))
    paired = _mapping(paired, label=f"{label}.paired")
    pair = _find_pair(paired, candidate=candidate, baseline=baseline, label=label)
    overall = _mapping(pair.get("mIoU"), label=f"{label}.{candidate}-{baseline}.mIoU")
    _finite_number(overall.get("delta"), label=f"{label}.{candidate}-{baseline}.mIoU.delta")
    _ci95(overall, label=f"{label}.{candidate}-{baseline}.mIoU.ci95")
    n_images = overall.get("n_images")
    if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 1:
        raise ValueError(f"{label}.{candidate}-{baseline}.mIoU.n_images must be positive")
    replicates = overall.get("replicates", pair.get("bootstrap_replicates"))
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError(f"{label}.{candidate}-{baseline}.mIoU.replicates must be positive")

    per_seed = _mapping(
        pair.get("per_seed", pair.get("by_seed")),
        label=f"{label}.{candidate}-{baseline}.per_seed",
    )
    if set(per_seed) != set(seeds):
        raise ValueError(f"{label}.{candidate}-{baseline}.per_seed does not match system seeds")
    for seed in seeds:
        seed_pair = _mapping(
            per_seed[seed],
            label=f"{label}.{candidate}-{baseline}.per_seed[{seed!r}]",
        )
        seed_miou = _mapping(
            seed_pair.get("mIoU"),
            label=f"{label}.{candidate}-{baseline}.per_seed[{seed!r}].mIoU",
        )
        _finite_number(
            seed_miou.get("delta"),
            label=f"{label}.{candidate}-{baseline}.per_seed[{seed!r}].mIoU.delta",
        )
        _ci95(
            seed_miou,
            label=f"{label}.{candidate}-{baseline}.per_seed[{seed!r}].mIoU.ci95",
        )
    return pair


def validate_result(result: Mapping[str, Any], *, cohort: str, source: Path | None = None) -> None:
    """Validate the final evaluator schema needed by the report."""

    label = str(source) if source is not None else cohort
    if result.get("schema_version") not in (None, "formal_v2"):
        raise ValueError(f"{label} has unsupported schema_version {result.get('schema_version')!r}")
    if result.get("cohort") != cohort:
        raise ValueError(f"{label}.cohort must be {cohort!r}")
    interpretation = result.get("interpretation")
    if interpretation != EXPECTED_INTERPRETATION[cohort]:
        raise ValueError(
            f"{label}.interpretation must be {EXPECTED_INTERPRETATION[cohort]!r}, got {interpretation!r}"
        )
    seeds, queue = _validate_systems(result, label=label)
    top_n_images = result.get("n_images")
    if isinstance(top_n_images, bool) or not isinstance(top_n_images, int) or top_n_images != len(queue):
        raise ValueError(f"{label}.n_images must equal the queued image count")
    for pair_label, candidate, baseline in COMPARISONS:
        _validate_pair(
            result,
            candidate=candidate,
            baseline=baseline,
            seeds=seeds,
            label=label,
        )


def result_paths(final_evaluation: str | Path) -> dict[str, Path]:
    """Return the two immutable post-run result paths."""

    root = Path(final_evaluation)
    return {cohort: root / cohort / "result.json" for cohort in COHORTS}


def wait_for_results(
    final_evaluation: str | Path,
    *,
    timeout_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict[str, Path]:
    """Wait for both cohort results, then fail with every missing path."""

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be nonnegative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    paths = result_paths(final_evaluation)
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = {cohort: path for cohort, path in paths.items() if not path.is_file()}
        if not missing:
            return paths
        remaining = deadline - time.monotonic()
        if timeout_seconds == 0 or remaining <= 0:
            missing_text = ", ".join(str(path) for path in missing.values())
            raise FileNotFoundError(
                "formal v2 evaluation results are absent after waiting; missing: " + missing_text
            )
        time.sleep(min(poll_seconds, remaining, 60.0))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_result(path: str | Path, *, cohort: str) -> dict[str, Any]:
    """Load and validate one final evaluator result."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        result = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}") from exc
    if not isinstance(result, Mapping):
        raise ValueError(f"{source} must contain a JSON object")
    result_dict = dict(result)
    validate_result(result_dict, cohort=cohort, source=source)
    return result_dict


def load_results(final_evaluation: str | Path) -> dict[str, dict[str, Any]]:
    """Load both final cohort results after their presence has been checked."""

    paths = result_paths(final_evaluation)
    return {
        cohort: load_result(path, cohort=cohort)
        for cohort, path in paths.items()
    }


def _system_seed_reports(result: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    systems = _mapping(result["systems"], label="result.systems")
    system = _mapping(systems[arm], label=f"result.systems.{arm}")
    return _mapping(system.get("per_seed", system.get("by_seed")), label=f"result.systems.{arm}.per_seed")


def paired_image_deltas(
    result: Mapping[str, Any], *, candidate: str = "r_opd", baseline: str = "bbox_sft"
) -> list[dict[str, Any]]:
    """Compute stored per-image candidate-minus-baseline means for illustration.

    The formal overall paired bootstrap is already present in ``result.json``.
    This helper averages the stored per-image mIoU over formal seeds only to
    provide the image-level points used by the plot and examples.
    """

    queue = _queue(result, label="result")
    candidate_reports = _system_seed_reports(result, candidate)
    baseline_reports = _system_seed_reports(result, baseline)
    if set(candidate_reports) != set(baseline_reports):
        raise ValueError("candidate and baseline seed sets differ")
    seeds = sorted(candidate_reports, key=_seed_sort_key)
    candidate_scores = {
        seed: _per_image_scores(
            _mapping(candidate_reports[seed], label=f"result.systems.{candidate}.per_seed[{seed!r}]"),
            queue=queue,
            label=f"result.systems.{candidate}.per_seed[{seed!r}]",
        )
        for seed in seeds
    }
    baseline_scores = {
        seed: _per_image_scores(
            _mapping(baseline_reports[seed], label=f"result.systems.{baseline}.per_seed[{seed!r}]"),
            queue=queue,
            label=f"result.systems.{baseline}.per_seed[{seed!r}]",
        )
        for seed in seeds
    }
    result_rows: list[dict[str, Any]] = []
    for key in queue:
        candidate_mean = sum(candidate_scores[seed][key] for seed in seeds) / len(seeds)
        baseline_mean = sum(baseline_scores[seed][key] for seed in seeds) / len(seeds)
        result_rows.append(
            {
                "sample_id": key[0],
                "image_id": key[1],
                "candidate_mIoU": candidate_mean,
                "baseline_mIoU": baseline_mean,
                "delta": candidate_mean - baseline_mean,
            }
        )
    return result_rows


def fixed_r_sft_examples(
    result: Mapping[str, Any], *, count: int = EXAMPLE_COUNT
) -> dict[str, list[dict[str, Any]]]:
    """Select fixed-count strict-positive and strict-negative examples.

    Selection is descriptive only: three largest positive deltas and three
    smallest negative deltas, with stable image-identifier tie breaks.  Zero
    deltas are omitted and no metric row is filtered or thresholded.
    """

    if count < 1:
        raise ValueError("count must be positive")
    rows = paired_image_deltas(result)
    positive = [row for row in rows if row["delta"] > 0.0]
    negative = [row for row in rows if row["delta"] < 0.0]
    positive.sort(key=lambda row: (-row["delta"], _stable_key(row["sample_id"]), _stable_key(row["image_id"])))
    negative.sort(key=lambda row: (row["delta"], _stable_key(row["sample_id"]), _stable_key(row["image_id"])))
    return {"positive": positive[:count], "negative": negative[:count]}


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite_number(value, label="value")
    return f"{number:.{digits}f}"


def _fmt_ci(value: Mapping[str, Any]) -> str:
    lower, upper = _ci95(value, label="bootstrap")
    return f"[{lower:.4f}, {upper:.4f}]"


def _fmt_id(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pair_for_report(result: Mapping[str, Any], candidate: str, baseline: str) -> Mapping[str, Any]:
    paired = _mapping(result.get("paired", result.get("paired_deltas")), label="result.paired")
    return _find_pair(
        paired,
        candidate=candidate,
        baseline=baseline,
        label=str(result.get("cohort", "result")),
    )


def render_report(
    results: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path] | None = None,
    plot_reference: str = DEFAULT_PLOT_NAME,
) -> str:
    """Render concise Markdown from validated final evaluator summaries."""

    for cohort in COHORTS:
        if cohort not in results:
            raise ValueError(f"missing result for cohort {cohort}")
        validate_result(results[cohort], cohort=cohort)

    seed_union: list[Any] = []
    for cohort in COHORTS:
        systems = _mapping(results[cohort]["systems"], label=f"{cohort}.systems")
        bbox_system = _mapping(systems["bbox_sft"], label=f"{cohort}.systems.bbox_sft")
        per_seed = _mapping(
            bbox_system.get("per_seed", bbox_system.get("by_seed")),
            label=f"{cohort}.systems.bbox_sft.per_seed",
        )
        for seed in sorted(per_seed, key=_seed_sort_key):
            if seed not in seed_union:
                seed_union.append(seed)
    seed_union.sort(key=_seed_sort_key)

    lines = [
        "# MM-GCoT formal v2 post-run report",
        "",
        "The primary metric is image-equal mIoU from the formal v2 `result.json` summaries: four bbox draws are averaged within trajectory, trajectories within image, and images within the reported cohort. The values and bootstrap intervals below are read from the evaluator summaries; the plot and examples use stored per-image rows only for illustration.",
        "",
        "## Primary question",
        "",
    ]
    independent_pair = _pair_for_report(results["independent48"], "r_opd", "bbox_sft")
    independent_miou = _mapping(independent_pair["mIoU"], label="independent48.R-SFT.mIoU")
    primary_delta = float(independent_miou["delta"])
    direction = "higher" if primary_delta > 0 else "lower" if primary_delta < 0 else "equal"
    lines.extend([
        f"On the independent 48 images, R-OPD has {direction} average mIoU than bbox-SFT "
        f"(R−SFT Δ={_fmt(primary_delta)}, image-bootstrap 95% CI {_fmt_ci(independent_miou)}). "
        "The per-seed deltas below show training variation separately from the image interval.",
        "",
        "The comparison measures transfer under the fixed, noisy v3p5 target description and the terminal Query adapter. "
        "Incorrect or ambiguous descriptions were retained. A gain alone does not establish same-entity boundary correction, "
        "nor does it identify text-attention dilution as the mechanism. Test-200 was used to select the teacher condition "
        "and is reported only as a selection-conditioned retest.",
        "",
        "## Cohort scope",
        "",
        "The cohorts are reported separately and are never pooled. `independent48` is the independent confirmation cohort. `test200_retest` is a diagnostic selection-conditioned retest and is not a second independent confirmation.",
        "",
        "| Cohort | Interpretation | Images | Seeds | Selection hash | Result source |",
        "|---|---|---:|---:|---|---|",
    ])
    for cohort in COHORTS:
        result = results[cohort]
        selection_hash = str(result.get("selection_sha256", "not recorded"))
        selection_hash = selection_hash[:12] + ("…" if len(selection_hash) > 12 else "")
        systems = _mapping(result["systems"], label=f"{cohort}.systems")
        system = _mapping(systems["bbox_sft"], label=f"{cohort}.systems.bbox_sft")
        per_seed = _mapping(system.get("per_seed", system.get("by_seed")), label="per_seed")
        source = str(source_paths[cohort]) if source_paths and cohort in source_paths else "result.json"
        lines.append(
            f"| `{cohort}` | {COHORT_DESCRIPTION[cohort]} (`{result['interpretation']}`) | "
            f"{result['n_images']} | {len(per_seed)} | `{selection_hash}` | `{source}` |"
        )

    lines.extend(
        [
            "",
            "## Per-arm mIoU",
            "",
            "The system overall is the evaluator-provided cohort mIoU. Seed columns are the evaluator-provided per-seed mIoU and are not reselected or averaged into a replacement metric.",
            "",
            "| Cohort | Arm | Overall mIoU | " + " | ".join(f"Seed {seed} mIoU" for seed in seed_union) + " |",
            "|---|---|---:|" + "---:|" * len(seed_union),
        ]
    )
    for cohort in COHORTS:
        result = results[cohort]
        systems = _mapping(result["systems"], label=f"{cohort}.systems")
        for arm, arm_label in ARM_LABELS.items():
            system = _mapping(systems[arm], label=f"{cohort}.systems.{arm}")
            per_seed = _mapping(system.get("per_seed", system.get("by_seed")), label="per_seed")
            cells = []
            for seed in seed_union:
                if seed not in per_seed:
                    cells.append("NA")
                else:
                    seed_report = _mapping(per_seed[seed], label="seed report")
                    cells.append(_fmt(_metric_value(seed_report, label="seed mIoU")))
            lines.append(
                f"| `{cohort}` | {arm_label} (`{arm}`) | {_fmt(_metric_value(system, label='system mIoU'))} | "
                + " | ".join(cells)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Validity and completion",
            "",
            "`observed` counts emitted frames. `valid_completed` counts frames with both `valid=true` and `completed=true`; missing, invalid, or incomplete frames remain visible in these denominators and contribute according to the formal evaluator schema.",
            "",
            "| Cohort | Arm | Seed | Observed / expected | Valid & completed / expected |",
            "|---|---|---|---:|---:|",
        ]
    )
    for cohort in COHORTS:
        systems = _mapping(results[cohort]["systems"], label=f"{cohort}.systems")
        for arm, arm_label in ARM_LABELS.items():
            system = _mapping(systems[arm], label=f"{cohort}.systems.{arm}")
            per_seed = _mapping(system.get("per_seed", system.get("by_seed")), label="per_seed")
            for seed in sorted(per_seed, key=_seed_sort_key):
                seed_report = _mapping(per_seed[seed], label="seed report")
                counts = _frame_counts(seed_report, label="seed report")
                expected = counts["expected"]
                lines.append(
                    f"| `{cohort}` | {arm_label} | `{seed}` | "
                    f"{counts['observed']} / {expected} ({counts['observed'] / expected:.1%}) | "
                    f"{counts['valid_completed']} / {expected} ({counts['valid_completed'] / expected:.1%}) |"
                )

    lines.extend(
        [
            "",
            "## Paired image bootstrap for mIoU",
            "",
            "Each delta is candidate minus baseline. The intervals are the evaluator-provided 95% percentile bootstrap intervals with image as the resampling unit; no post-run threshold is applied.",
            "",
            "| Cohort | Pair | Overall ΔmIoU | 95% CI | Images | Replicates |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for cohort in COHORTS:
        result = results[cohort]
        for pair_label, candidate, baseline in COMPARISONS:
            pair = _pair_for_report(result, candidate, baseline)
            overall = _mapping(pair["mIoU"], label=f"{cohort}.{pair_label}.mIoU")
            reps = overall.get("replicates", pair.get("bootstrap_replicates"))
            lines.append(
                f"| `{cohort}` | {pair_label} (`{ARM_LABELS[candidate]}−{ARM_LABELS[baseline]}`) | "
                f"{_fmt(overall['delta'])} | {_fmt_ci(overall)} | {overall['n_images']} | {reps} |"
            )

    lines.extend(
        [
            "",
            "### Per-seed paired deltas",
            "",
            "| Cohort | Pair | Seed | ΔmIoU | 95% CI |",
            "|---|---|---|---:|---|",
        ]
    )
    for cohort in COHORTS:
        result = results[cohort]
        for pair_label, candidate, baseline in COMPARISONS:
            pair = _pair_for_report(result, candidate, baseline)
            per_seed = _mapping(pair["per_seed"], label=f"{cohort}.{pair_label}.per_seed")
            for seed in sorted(per_seed, key=_seed_sort_key):
                seed_pair = _mapping(per_seed[seed], label="seed pair")
                miou = _mapping(seed_pair["mIoU"], label="seed pair mIoU")
                lines.append(
                    f"| `{cohort}` | {pair_label} | `{seed}` | {_fmt(miou['delta'])} | {_fmt_ci(miou)} |"
                )

    lines.extend(
        [
            "",
            "## R-SFT image examples (illustration only)",
            "",
            f"Fixed rule, frozen for this report: average each stored image mIoU across formal seeds, then select up to {EXAMPLE_COUNT} strict positive deltas with the largest values and up to {EXAMPLE_COUNT} strict negative deltas with the smallest values. Ties are broken by `sample_id`, then `image_id`; zero deltas are omitted. These rows do not alter any metric, denominator, or conclusion.",
        ]
    )
    for cohort in COHORTS:
        examples = fixed_r_sft_examples(results[cohort])
        lines.extend(
            [
                "",
                f"### `{cohort}`",
                "",
                "| Sign | sample_id | image_id | R mIoU | SFT mIoU | R−SFT ΔmIoU |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for sign in ("positive", "negative"):
            if not examples[sign]:
                lines.append(f"| {sign} | none under fixed rule |  |  |  |  |")
                continue
            for row in examples[sign]:
                lines.append(
                    f"| {sign} | `{_fmt_id(row['sample_id'])}` | `{_fmt_id(row['image_id'])}` | "
                    f"{_fmt(row['candidate_mIoU'])} | {_fmt(row['baseline_mIoU'])} | {_fmt(row['delta'])} |"
                )

    lines.extend(
        [
            "",
            "## Figure",
            "",
            f"![Paired image-level deltas]({plot_reference})",
            "",
            "The figure shows every queued image's stored per-image R-SFT, E-SFT, and R-E delta, averaged over formal seeds for display and ordered only for visualization. It is not a selection gate.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paired_image_delta_plot(
    results: Mapping[str, Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Write the required PNG plot for all queued paired image deltas."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_by_pair = {
        (cohort, pair_label): paired_image_deltas(
            results[cohort], candidate=candidate, baseline=baseline
        )
        for cohort in COHORTS
        for pair_label, candidate, baseline in COMPARISONS
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write the paired image-delta PNG") from exc

    figure, axes = plt.subplots(
        len(COHORTS),
        len(COMPARISONS),
        figsize=(14.0, 7.0),
        squeeze=False,
        sharey="row",
        constrained_layout=True,
    )
    for row_index, cohort in enumerate(COHORTS):
        for column_index, (pair_label, candidate, baseline) in enumerate(COMPARISONS):
            axis = axes[row_index][column_index]
            rows = sorted(rows_by_pair[(cohort, pair_label)], key=lambda row: row["delta"])
            values = [row["delta"] for row in rows]
            colors = [
                "#4472c4" if value < 0 else "#c0504d" if value > 0 else "#808080"
                for value in values
            ]
            axis.bar(range(len(values)), values, color=colors, width=1.0, linewidth=0)
            axis.axhline(0.0, color="#333333", linewidth=0.9)
            axis.set_title(f"{cohort} / {pair_label}\n(n={len(values)} images)")
            axis.set_xlabel("queued images sorted by Δ")
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks([])
        axes[row_index][0].set_ylabel("image mIoU delta")
    figure.suptitle("Formal v2 paired image-level deltas")
    figure.savefig(output, dpi=180, format="png")
    plt.close(figure)
    if not output.is_file():
        raise RuntimeError(f"matplotlib did not create the requested plot: {output}")
    return output


def generate_report(
    final_evaluation: str | Path,
    output_dir: str | Path,
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    report_name: str = DEFAULT_REPORT_NAME,
    plot_name: str = DEFAULT_PLOT_NAME,
) -> dict[str, Path]:
    """Wait for both results and write the Markdown report plus PNG plot."""

    paths = wait_for_results(
        final_evaluation,
        timeout_seconds=wait_seconds,
        poll_seconds=poll_seconds,
    )
    results = {
        cohort: load_result(path, cohort=cohort)
        for cohort, path in paths.items()
    }
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_name)
    if not report_path.is_absolute():
        report_path = output_root / report_path
    plot_path = Path(plot_name)
    if not plot_path.is_absolute():
        plot_path = output_root / plot_path
    if plot_path.suffix.lower() != ".png":
        raise ValueError("plot_name must have a .png suffix")
    write_paired_image_delta_plot(results, plot_path)
    plot_reference = Path(os.path.relpath(plot_path, report_path.parent)).as_posix()
    report_text = render_report(
        results,
        source_paths=paths,
        plot_reference=plot_reference,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return {"report": report_path, "plot": plot_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-evaluation",
        "--input-root",
        "--input-dir",
        dest="final_evaluation",
        type=Path,
        default=Path("final_evaluation"),
        help="directory containing independent48/result.json and test200_retest/result.json",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("formal_report_v2"),
        help="directory for the Markdown report and paired image-delta PNG",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="maximum time to wait for both result files before failing",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="poll interval while waiting for result files",
    )
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    parser.add_argument("--plot-name", default=DEFAULT_PLOT_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate_report(
        args.final_evaluation,
        args.output_dir,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
        report_name=args.report_name,
        plot_name=args.plot_name,
    )
    print(f"FORMAL_REPORT_WRITTEN {outputs['report']}")
    print(f"FORMAL_REPORT_PLOT_WRITTEN {outputs['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
