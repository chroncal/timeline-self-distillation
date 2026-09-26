"""Run sealed final evaluation only after all nine formal checkpoints freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training.formal_analysis import aggregate_records, aggregate_systems
from mmgcot_timeline_training.formal_schedule_v2 import (
    CONFIG, ROOT, _read_jsonl, _run_group, _write_once)
from mmgcot_timeline_training.prepare_v2 import record_path


def _frozen_checkpoints(formal_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    receipt = formal_root / "formal" / "frozen_checkpoints.json"
    frozen = json.loads(receipt.read_text())
    paths: dict[str, Path] = {}
    common_sources = None
    for seed in config["training"]["formal_seeds"]:
        for arm in config["arms"]:
            name = f"{arm}_seed{seed}"
            item = frozen["checkpoints"][name]
            path = Path(item["path"])
            if file_hash(path) != item["sha256"]:
                raise RuntimeError(f"frozen checkpoint changed: {path}")
            run_receipt = json.loads((formal_root / "formal" / name / "config.json").read_text())
            sources = run_receipt["reproducibility"]["source_sha256"]
            if common_sources is None:
                common_sources = sources
            elif sources != common_sources:
                raise RuntimeError("formal runs used different source code")
            paths[name] = path
    if len(paths) != 9:
        raise RuntimeError("formal checkpoint count differs from frozen 3x3 matrix")
    if file_hash(formal_root / "calibration/selection.json") != frozen["selection_sha256"]:
        raise RuntimeError("calibration selection changed after checkpoint freeze")
    for name, digest in common_sources.items():
        if file_hash(ROOT / name) != digest:
            raise RuntimeError(f"evaluation code differs from formal training: {name}")
    return paths


def prepare_eval(args: argparse.Namespace, config: dict[str, Any],
                 cohort: str, selection: Path) -> Path:
    output = args.output / "prepared" / cohort
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["selection_sha256"] != file_hash(selection) or protocol["trajectories_per_image"] != 3:
            raise RuntimeError("existing final cohort preparation protocol differs")
    py = str(Path(args.python).absolute())
    jobs = []
    for shard, device in enumerate(args.devices):
        command = [py, "-m", "mmgcot_timeline_training.prepare_v2",
                   "--selection", str(selection), "--output-dir", str(output),
                   "--device", str(device), "--shard-index", str(shard),
                   "--num-shards", str(len(args.devices)), "--trajectories", "3", "--resume"]
        name = f"{cohort}_prepare_shard{shard}"
        jobs.append((name, command, args.output / "logs" / f"{name}.log",
                     output / f"shard{shard}.receipt.json"))
    _run_group(jobs, len(args.devices))
    for row in _read_jsonl(selection):
        for ti in range(3):
            if not record_path(output, row["sample_id"], ti).is_file():
                raise RuntimeError(f"missing final evaluation trajectory {row['sample_id']} t{ti}")
    return output


def _evaluate_cohort(args: argparse.Namespace, config: dict[str, Any],
                     cohort: str, selection: Path, prepared: Path,
                     checkpoints: dict[str, Path]) -> None:
    py = str(Path(args.python).absolute())
    output = args.output / cohort
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    baselines = [("base_l", 20260921, None), ("base_r", 20260921, None)]
    systems = baselines + [(arm, seed, checkpoints[f"{arm}_seed{seed}"])
                           for seed in config["training"]["formal_seeds"]
                           for arm in config["arms"]]
    for index, (arm, seed, checkpoint) in enumerate(systems):
        label = f"{arm}_seed{seed}"
        destination = output / f"{label}.jsonl"
        command = [py, "-m", "mmgcot_timeline_training.formal_train_v2",
                   "--mode", "eval", "--selection", str(selection),
                   "--prepared", str(prepared), "--output", str(destination),
                   "--device", str(args.devices[index % len(args.devices)]),
                   "--arm", arm, "--seed", str(seed)]
        if checkpoint is not None:
            command += ["--checkpoint", str(checkpoint)]
        jobs.append((label, command, args.output / "logs" / f"{cohort}_{label}.log", None))
    _run_group(jobs, len(args.devices))
    rows = _read_jsonl(selection)
    queue = [(row["sample_id"], row["image_id"]) for row in rows]
    expected_lines = len(rows) * 3 * 5
    for label, *_ in jobs:
        path = output / f"{label}.jsonl"
        if sum(1 for _ in path.open()) != expected_lines:
            raise RuntimeError(f"final evaluation frame count differs: {path}")
    comparisons = {}
    for arm in config["arms"]:
        comparisons[arm] = [*(_read_jsonl(output / f"{arm}_seed{seed}.jsonl")
                              for seed in config["training"]["formal_seeds"])]
    # Flatten each arm's three-seed records without treating them as images.
    grouped = {arm: [row for seed_rows in seed_groups for row in seed_rows]
               for arm, seed_groups in comparisons.items()}
    report = aggregate_systems(grouped, queue, mode="sample",
                               comparisons=[("r_opd", "bbox_sft"),
                                            ("e_opd", "bbox_sft"),
                                            ("r_opd", "e_opd")],
                               bootstrap_replicates=10_000)
    report["cohort"] = cohort
    report["selection_sha256"] = file_hash(selection)
    report["interpretation"] = ("independent_confirmation" if cohort == "independent48"
                                else "diagnostic_selection_conditioned_retest")
    report["baselines"] = {arm: aggregate_records(output / f"{arm}_seed20260921.jsonl",
                                                  queue, mode="sample")
                           for arm in ("base_l", "base_r")}
    _write_once(output / "result.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = json.loads(CONFIG.read_text())
    checkpoints = _frozen_checkpoints(args.formal_root, config)
    print("NINE_CHECKPOINTS_VERIFIED", flush=True)
    data = config["data_manifest"]
    for cohort, key in (("independent48", "independent_confirmation"),
                        ("test200_retest", "diagnostic_selection_test200")):
        selection = Path(data[key])
        if file_hash(selection) != data[f"{key}_sha256"]:
            raise RuntimeError(f"{cohort} selection changed")
        prepared = prepare_eval(args, config, cohort, selection)
        _evaluate_cohort(args, config, cohort, selection, prepared, checkpoints)
        print(f"FINAL_EVAL_COMPLETE {cohort}", flush=True)
