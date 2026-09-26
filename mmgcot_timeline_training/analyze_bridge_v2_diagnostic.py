"""Aggregate the frozen one-trajectory bridge-v2 L0/L/E/R diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any


PANELS = {
    "A_random": ("A", "random", None, ("L0", "L", "E", "R")),
    "A_greedy": ("A", "greedy", None, ("L0", "L", "E", "R")),
    "B_prefix1": ("B", "random", 1, ("E", "L", "R")),
    "B_prefix2": ("B", "random", 2, ("E", "L", "R")),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _ci(values: list[float], seed: int, replicates: int = 10000) -> list[float]:
    if not values:
        return [None, None]  # type: ignore[list-item]
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[round(0.025 * (replicates - 1))], means[round(0.975 * (replicates - 1))]]


def analyze(manifest: Path, records_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(manifest)
    sample_ids = [row["sample_id"] for row in rows]
    by_sample: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in sample_ids}
    for path in sorted(records_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            if record.get("sample_id") in by_sample:
                by_sample[record["sample_id"]].append(record)
    outputs: dict[str, Any] = {}
    for panel_index, (name, (stage, mode, prefix, arms)) in enumerate(PANELS.items()):
        image_rows = []
        comparable = 0
        for sample_id in sample_ids:
            arm_values: dict[str, list[float]] = defaultdict(list)
            for record in by_sample[sample_id]:
                if record.get("type") != "bbox" or record.get("stage") != stage:
                    continue
                if record.get("mode") != mode or record.get("prefix_coordinates") != prefix:
                    continue
                if record.get("arm") in arms:
                    value = float(record["iou"])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite IoU")
                    arm_values[record["arm"]].append(value)
            expected_draws = 1 if mode == "greedy" else 4
            is_comparable = all(len(arm_values[arm]) == expected_draws for arm in arms)
            comparable += int(is_comparable)
            means = {
                arm: (sum(arm_values[arm]) / expected_draws if len(arm_values[arm]) == expected_draws else 0.0)
                for arm in arms
            }
            image_rows.append({"sample_id": sample_id, "comparable": is_comparable, "arm_mean_iou": means})
        contrasts = {}
        pairs = (("E", "L"), ("R", "L")) if stage == "B" else (
            ("E", "L"), ("R", "L"), ("E", "R"), ("L", "L0")
        )
        for left, right in pairs:
            values = [row["arm_mean_iou"][left] - row["arm_mean_iou"][right] for row in image_rows]
            contrasts[f"{left}-{right}"] = {
                "mean": sum(values) / len(values),
                "ci_95": _ci(values, 20260922 + panel_index * 100 + len(contrasts)),
                "improved": sum(value > 0 for value in values),
                "tied": sum(value == 0 for value in values),
                "harmed": sum(value < 0 for value in values),
            }
        outputs[name] = {
            "n_images": len(image_rows),
            "comparable_images": comparable,
            "coverage": comparable / len(image_rows),
            "arm_mean_iou": {
                arm: sum(row["arm_mean_iou"][arm] for row in image_rows) / len(image_rows)
                for arm in arms
            },
            "contrasts": contrasts,
            "per_image": image_rows,
        }
    r_values = [outputs[name]["contrasts"]["R-L"]["mean"]
                for name in ("A_random", "B_prefix1", "B_prefix2")]
    gate_reasons = []
    if not all(value > 0 for value in r_values):
        gate_reasons.append("R-L must be positive for A_random, B_prefix1, and B_prefix2")
    if min(outputs[name]["coverage"] for name in ("B_prefix1", "B_prefix2")) < 0.90:
        gate_reasons.append("both prefix panels require at least 90% complete-image coverage")
    return {
        "schema_version": "mmgcot_bridge_v2_geometry_analysis_v1",
        "manifest": str(manifest.absolute()),
        "images": len(rows),
        "panels": outputs,
        "opd_calibration_gate_pass": not gate_reasons,
        "gate_reasons": gate_reasons,
        "gate_definition": (
            "R-L point estimate > 0 on A random and both fixed student-prefix panels; "
            "both prefix panels >= 90% coverage. E-L is a reported control, not a gate."
        ),
    }


def render(summary: dict[str, Any]) -> str:
    lines = ["# Bridge-v2 small L/R/E diagnostic", "", "| Panel | Coverage | L | E | R | E-L | R-L |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, panel in summary["panels"].items():
        arms = panel["arm_mean_iou"]
        lines.append(
            f"| {name} | {panel['coverage']:.1%} | {arms.get('L', 0):.4f} | "
            f"{arms.get('E', 0):.4f} | {arms.get('R', 0):.4f} | "
            f"{panel['contrasts'].get('E-L', {}).get('mean', 0):+.4f} | "
            f"{panel['contrasts'].get('R-L', {}).get('mean', 0):+.4f} |"
        )
    lines += ["", f"OPD calibration gate: **{'PASS' if summary['opd_calibration_gate_pass'] else 'FAIL'}**."]
    if summary["gate_reasons"]:
        lines += ["", *[f"- {reason}" for reason in summary["gate_reasons"]]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    summary = analyze(args.manifest, args.records_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    (args.output_dir / "REPORT.md").write_text(render(summary))
    print(json.dumps({"gate": summary["opd_calibration_gate_pass"]}))


if __name__ == "__main__":
    main()
