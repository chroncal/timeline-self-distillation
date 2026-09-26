"""Validate, adjudicate, and summarize two-stage target-bridge v2 reviews."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from mmgcot_timeline_training.bridge_v2 import (
    EXTRACTOR_CONTENT_ERROR_LABELS,
    STUDENT_SELECTION_LABELS,
    TARGET_REFERENCE_REVIEW_LABELS,
    bridge_pass,
)


FIELDS = {
    "student_target_selection_error": set(STUDENT_SELECTION_LABELS),
    "extractor_content_error": set(EXTRACTOR_CONTENT_ERROR_LABELS),
    "target_reference_review_label": set(TARGET_REFERENCE_REVIEW_LABELS),
}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _review_index(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    rows = _read(path)
    result = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate case in {path}: {case_id}")
        for field, allowed in FIELDS.items():
            if row.get(field) not in allowed:
                raise ValueError(f"invalid {field} in {path}: {row.get(field)!r}")
        result[case_id] = row
    if set(result) != expected:
        raise ValueError(f"case coverage mismatch in {path}")
    return result


def summarize(mapping_path: Path, packet_paths: list[Path], reviewer_a: Path,
              reviewer_b: Path, adjudicated: Path,
              schema_version: str = "mmgcot_target_bridge_v2_review_summary") -> dict[str, Any]:
    mapping = _read(mapping_path)
    expected = {row["case_id"] for row in mapping}
    map_by_case = {row["case_id"]: row for row in mapping}
    packet_by_sample = {}
    for path in packet_paths:
        for row in _read(path):
            packet_by_sample[row["sample_id"]] = row
    a = _review_index(reviewer_a, expected)
    b = _review_index(reviewer_b, expected)
    final = _review_index(adjudicated, expected)
    agreements = {
        field: sum(a[case][field] == b[case][field] for case in expected) / len(expected)
        for field in FIELDS
    }
    results: dict[str, Any] = {}
    cohorts: dict[str, list[str]] = defaultdict(list)
    for case_id, meta in map_by_case.items():
        cohorts[meta["cohort"]].append(case_id)
    for cohort, cases in sorted(cohorts.items()):
        student = Counter(final[case]["student_target_selection_error"] for case in cases)
        content = Counter(final[case]["extractor_content_error"] for case in cases)
        target = Counter(final[case]["target_reference_review_label"] for case in cases)
        valid_parse = sum(
            packet_by_sample[map_by_case[case]["sample_id"]]["bridge_parse_status"] == "valid"
            for case in cases
        )
        gate_input = {
            "cases": len(cases), "valid_parse": valid_parse,
            "extractor_content_error_counts": dict(content),
            "student_selection_counts": dict(student),
        }
        passed, reasons = bridge_pass(gate_input)
        results[cohort] = {
            **gate_input,
            "target_reference_counts": dict(target),
            "pass": passed,
            "failure_reasons": reasons,
            "by_task": {},
            "by_split": {},
        }
        for grouping in ("task_type", "split"):
            values = results[cohort]["by_task" if grouping == "task_type" else "by_split"]
            for value in sorted({map_by_case[case][grouping] for case in cases}):
                subset = [case for case in cases if map_by_case[case][grouping] == value]
                values[value] = {
                    "cases": len(subset),
                    "student_target_selection_error": dict(Counter(
                        final[case]["student_target_selection_error"] for case in subset
                    )),
                    "extractor_content_error": dict(Counter(
                        final[case]["extractor_content_error"] for case in subset
                    )),
                    "target_reference_review_label": dict(Counter(
                        final[case]["target_reference_review_label"] for case in subset
                    )),
                }
        task_failures = []
        for task, task_values in results[cohort]["by_task"].items():
            task_cases = task_values["cases"]
            task_samples = [case for case in cases if map_by_case[case]["task_type"] == task]
            task_valid = sum(
                packet_by_sample[map_by_case[case]["sample_id"]]["bridge_parse_status"] == "valid"
                for case in task_samples
            )
            task_pass, task_reasons = bridge_pass({
                "cases": task_cases,
                "valid_parse": task_valid,
                "extractor_content_error_counts": task_values["extractor_content_error"],
                "student_selection_counts": task_values["student_target_selection_error"],
            })
            task_values["pass"] = task_pass
            task_values["failure_reasons"] = task_reasons
            if not task_pass:
                task_failures.extend(f"{task}: {reason}" for reason in task_reasons)
        if task_failures:
            results[cohort]["pass"] = False
            results[cohort]["failure_reasons"].extend(task_failures)
    return {
        "schema_version": schema_version,
        "cases": len(expected),
        "reviewer_raw_agreement": agreements,
        "cohorts": results,
        "all_cohorts_pass": all(value["pass"] for value in results.values()),
        "gate_excludes_student_gt_correctness": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--packet", type=Path, action="append", required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema-version", default="mmgcot_target_bridge_v2_review_summary"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = summarize(
        args.mapping, args.packet, args.reviewer_a, args.reviewer_b, args.adjudicated,
        args.schema_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"all_cohorts_pass": summary["all_cohorts_pass"]}))


if __name__ == "__main__":
    main()
