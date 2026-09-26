"""Run the frozen v2 calibration and nine formal arms without score peeking."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training.formal_analysis import (
    aggregate_records, select_lambda, select_learning_rate)
from mmgcot_timeline_training.prepare_v2 import record_path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mmgcot_formal_v2.json"


def _write_once(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise RuntimeError(f"existing frozen selection differs: {path}")
        return
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2,
                               allow_nan=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def preflight(config: dict[str, Any], prepared_root: Path) -> tuple[Path, Path]:
    data = config["data_manifest"]
    train, dev = Path(data["train"]), Path(data["dev"])
    for name, path in (("train", train), ("dev", dev),
                       ("independent_confirmation", Path(data["independent_confirmation"])),
                       ("diagnostic_selection_test200", Path(data["diagnostic_selection_test200"]))):
        expected = data[f"{name}_sha256"]
        if file_hash(path) != expected:
            raise RuntimeError(f"frozen {name} hash changed")
    for name, selection, expected_trajectories in (("train", train, 1), ("dev", dev, 1)):
        protocol_path = prepared_root / name / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        if protocol["selection_sha256"] != file_hash(selection):
            raise RuntimeError(f"prepared {name} manifest differs")
        if protocol["trajectories_per_image"] != expected_trajectories:
            raise RuntimeError(f"prepared {name} trajectory count differs")
        rows = _read_jsonl(selection)
        for row in rows:
            record = record_path(prepared_root / name, row["sample_id"], 0)
            if not record.is_file():
                raise RuntimeError(f"missing frozen {name} trajectory: {record}")
        for shard in range(4):
            if not (prepared_root / name / f"shard{shard}.receipt.json").is_file():
                raise RuntimeError(f"missing {name} shard receipt {shard}")
    return train, dev


def freeze_prepared_manifest(output: Path, prepared_root: Path,
                            train: Path, dev: Path) -> None:
    model = Path(json.loads(CONFIG.read_text())["model"])
    manifest: dict[str, Any] = {"schema": "formal_v2_prepared_manifest",
        "preparation_code_sha256": file_hash(ROOT / "mmgcot_timeline_training/prepare_v2.py"),
        "model_config_sha256": file_hash(model / "config.json"),
        "model_weights_sha256": file_hash(next(model.glob("model.safetensors-*.safetensors"))),
        "tokenizer_sha256": file_hash(model / "tokenizer.json")}
    for name, selection in (("train", train), ("dev", dev)):
        manifest[name] = [{"sample_id": row["sample_id"],
                           "record_sha256": file_hash(record_path(
                               prepared_root / name, row["sample_id"], 0))}
                          for row in _read_jsonl(selection)]
        manifest[f"{name}_protocol_sha256"] = file_hash(prepared_root / name / "protocol.json")
    _write_once(output / "prepared_manifest.json", manifest)


def _run_one(name: str, command: list[str], log_path: Path,
             completion: Path | None = None) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if completion is not None:
        if completion.exists():
            receipt = json.loads(completion.read_text())
            if "mmgcot_timeline_training.formal_train_v2" in command:
                if not Path(receipt["final_checkpoint"]).is_file():
                    raise RuntimeError(f"completed run missing checkpoint: {completion}")
                config_path = completion.parent / "config.json"
                saved = json.loads(config_path.read_text())
                for field, cli_key, conversion in (
                        ("arm", "--arm", str), ("seed", "--seed", int),
                        ("steps", "--steps", int), ("effective_batch", "--batch", int),
                        ("lr", "--lr", float), ("lambda_opd", "--lambda-opd", float)):
                    if saved[field] != conversion(command[command.index(cli_key)+1]):
                        raise RuntimeError(f"completed run {name} differs on {field}")
                for source, digest in saved["reproducibility"]["source_sha256"].items():
                    if file_hash(ROOT / source) != digest:
                        raise RuntimeError(f"completed run {name} used different source: {source}")
            elif "mmgcot_timeline_training.prepare_v2" in command:
                if receipt["selection_sha256"] != file_hash(command[command.index("--selection")+1]):
                    raise RuntimeError(f"completed preparation selection changed: {completion}")
            else:
                raise RuntimeError(f"unknown completion receipt for {name}")
            return f"SKIP {name}: {completion} exists"
        if "mmgcot_timeline_training.formal_train_v2" in command:
            run_dir = Path(command[command.index("--output")+1])
            if (run_dir / "config.json").exists():
                checkpoints = sorted(run_dir.glob("step_*.pt"))
                if checkpoints:
                    command = [*command, "--resume", str(checkpoints[-1])]
                # A run interrupted before its first checkpoint restarts from
                # step zero with its original raw logs left append-only.
        elif "mmgcot_timeline_training.prepare_v2" not in command:
            raise RuntimeError(f"unknown completion job: {name}")
    evaluation_output = Path(command[command.index("--output")+1]) if completion is None else None
    if evaluation_output is not None and evaluation_output.exists():
        receipt = evaluation_output.with_suffix(".complete.json")
        if not receipt.exists():
            raise RuntimeError(f"partial evaluation output needs review: {evaluation_output}")
        saved = json.loads(receipt.read_text())
        if saved["output_sha256"] != file_hash(evaluation_output):
            raise RuntimeError(f"completed evaluation output hash changed: {evaluation_output}")
        if saved["selection_sha256"] != file_hash(command[command.index("--selection")+1]):
            raise RuntimeError(f"completed evaluation selection changed: {evaluation_output}")
        if saved["arm"] != command[command.index("--arm")+1] or saved["seed"] != int(
                command[command.index("--seed")+1]):
            raise RuntimeError(f"completed evaluation arm/seed changed: {evaluation_output}")
        if saved["prepared_protocol_sha256"] != file_hash(
                Path(command[command.index("--prepared")+1]) / "protocol.json"):
            raise RuntimeError(f"completed evaluation trajectory protocol changed: {evaluation_output}")
        if "--checkpoint" in command and saved["checkpoint_sha256"] != file_hash(
                command[command.index("--checkpoint")+1]):
            raise RuntimeError(f"completed evaluation checkpoint changed: {evaluation_output}")
        return f"SKIP {name}: completed evaluation exists"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}: {log_path}")
    if completion is not None and not completion.exists():
        raise RuntimeError(f"{name} returned without completion receipt")
    if evaluation_output is not None and not evaluation_output.with_suffix(".complete.json").exists():
        raise RuntimeError(f"{name} returned without evaluation completion receipt")
    return f"DONE {name}"


def _run_group(jobs: list[tuple[str, list[str], Path, Path | None]], workers: int) -> None:
    # Keep one serial lane per GPU. A generic pool can start job i+workers
    # when an unrelated GPU finishes and accidentally overlap it with job i.
    def lane(items: list[tuple[str, list[str], Path, Path | None]]) -> list[str]:
        return [_run_one(*item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(lane, jobs[index::workers]): index
                      for index in range(min(workers, len(jobs)))}
        for future in as_completed(future_map):
            for outcome in future.result():
                print(outcome, flush=True)


def _train_command(py: str, selection: Path, prepared: Path, output: Path,
                   *, device: int, arm: str, seed: int, steps: int,
                   lr: float, lam: float, batch: int) -> list[str]:
    return [py, "-m", "mmgcot_timeline_training.formal_train_v2",
            "--mode", "train", "--selection", str(selection),
            "--prepared", str(prepared), "--output", str(output),
            "--cache-root", str(prepared / "prefix_cache"),
            "--device", str(device), "--arm", arm, "--seed", str(seed),
            "--steps", str(steps), "--batch", str(batch), "--lr", str(lr),
            "--lambda-opd", str(lam)]


def _eval_command(py: str, selection: Path, prepared: Path, output: Path,
                  *, device: int, arm: str, seed: int, checkpoint: Path) -> list[str]:
    return [py, "-m", "mmgcot_timeline_training.formal_train_v2",
            "--mode", "eval", "--selection", str(selection),
            "--prepared", str(prepared), "--output", str(output),
            "--cache-root", str(prepared / "prefix_cache"),
            "--device", str(device), "--arm", arm, "--seed", str(seed),
            "--checkpoint", str(checkpoint)]


def calibrate(args: argparse.Namespace, config: dict[str, Any],
              train: Path, dev: Path) -> dict[str, Any]:
    root = args.output / "calibration"
    root.mkdir(parents=True, exist_ok=True)
    py = str(Path(args.python).absolute())
    devices = args.devices
    rows = _read_jsonl(dev)
    queue = [(row["sample_id"], row["image_id"]) for row in rows]
    seed = int(config["calibration"]["seed"])
    steps = int(config["calibration"]["steps"])
    batch = int(config["training"]["effective_batch"])
    train_prepared, dev_prepared = args.prepared / "train", args.prepared / "dev"

    lr_candidates = config["calibration"]["sft_lr_candidates"]
    jobs = []
    for index, lr in enumerate(lr_candidates):
        label = f"sft_lr{lr:g}"
        run = root / label
        jobs.append((label, _train_command(py, train, train_prepared, run,
                    device=devices[index % len(devices)], arm="bbox_sft", seed=seed,
                    steps=steps, lr=lr, lam=0.0, batch=batch), root / "logs" / f"{label}.log",
                    run / "complete.json"))
    _run_group(jobs, len(devices))
    jobs = []
    for index, lr in enumerate(lr_candidates):
        label = f"sft_lr{lr:g}"
        run = root / label
        output = run / "dev_eval.jsonl"
        jobs.append((label, _eval_command(py, dev, dev_prepared, output,
                    device=devices[index % len(devices)], arm="bbox_sft", seed=seed,
                    checkpoint=run / f"step_{steps:04d}.pt"),
                    root / "logs" / f"{label}_eval.log", None))
    _run_group(jobs, len(devices))
    lr_scores = {float(lr): aggregate_records(root / f"sft_lr{lr:g}" / "dev_eval.jsonl",
                                                queue, mode="sample")["mIoU"]
                 for lr in lr_candidates}
    selected_lr = select_learning_rate(lr_scores)
    _write_once(root / "lr_selection.json", {"scores": {str(k): v for k,v in lr_scores.items()},
                 "selected_lr": selected_lr, "dev_sha256": file_hash(dev), "rule": "max; <0.002 tie -> lower"})

    lambda_candidates = config["calibration"]["common_lambda_candidates"]
    combinations = [(arm, lam) for lam in lambda_candidates for arm in ("r_opd", "e_opd")]
    jobs = []
    for index, (arm, lam) in enumerate(combinations):
        label = f"{arm}_lambda{lam:g}"
        run = root / label
        jobs.append((label, _train_command(py, train, train_prepared, run,
                    device=devices[index % len(devices)], arm=arm, seed=seed,
                    steps=steps, lr=selected_lr, lam=lam, batch=batch), root / "logs" / f"{label}.log",
                    run / "complete.json"))
    _run_group(jobs, len(devices))
    jobs = []
    for index, (arm, lam) in enumerate(combinations):
        label = f"{arm}_lambda{lam:g}"
        run = root / label
        jobs.append((label, _eval_command(py, dev, dev_prepared, run / "dev_eval.jsonl",
                    device=devices[index % len(devices)], arm=arm, seed=seed,
                    checkpoint=run / f"step_{steps:04d}.pt"),
                    root / "logs" / f"{label}_eval.log", None))
    _run_group(jobs, len(devices))
    scores = {float(lam): {
        arm: aggregate_records(root / f"{arm}_lambda{lam:g}" / "dev_eval.jsonl",
                               queue, mode="sample")["mIoU"]
        for arm in ("r_opd", "e_opd")}
        for lam in lambda_candidates}
    selected_lambda = select_lambda(scores)
    result = {"selected_lr": selected_lr, "selected_lambda": selected_lambda,
              "lr_scores": {str(k): v for k,v in lr_scores.items()},
              "lambda_scores": {str(k): v for k,v in scores.items()},
              "dev_sha256": file_hash(dev), "config_sha256": file_hash(CONFIG),
              "rule": "R/E average; max; <0.002 tie -> lower lambda"}
    _write_once(root / "selection.json", result)
    return result


def formal(args: argparse.Namespace, config: dict[str, Any],
           train: Path, selection: dict[str, Any]) -> None:
    py = str(Path(args.python).absolute())
    devices = args.devices
    runs = args.output / "formal"
    runs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in config["training"]["formal_seeds"]:
        for arm in config["arms"]:
            index = len(jobs)
            label = f"{arm}_seed{seed}"
            run = runs / label
            jobs.append((label, _train_command(py, train, args.prepared / "train", run,
                device=devices[index % len(devices)], arm=arm, seed=seed,
                steps=config["training"]["formal_steps"],
                lr=selection["selected_lr"], lam=selection["selected_lambda"] if arm != "bbox_sft" else 0.0,
                batch=int(config["training"]["effective_batch"])),
                runs / "logs" / f"{label}.log", run / "complete.json"))
    _run_group(jobs, len(devices))
    checkpoints = {job[0]: str(runs / job[0] / "step_0200.pt") for job in jobs}
    for path in checkpoints.values():
        if not Path(path).is_file():
            raise RuntimeError(f"missing formal checkpoint {path}")
    _write_once(runs / "frozen_checkpoints.json", {
        "selection_sha256": file_hash(args.output / "calibration" / "selection.json"),
        "checkpoints": {name: {"path": path, "sha256": file_hash(path)}
                        for name,path in checkpoints.items()},
        "no_independent_evaluation_prior": True})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "calibrate", "formal", "all"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    configuration = json.loads(CONFIG.read_text())
    train_selection, dev_selection = preflight(configuration, parsed.prepared)
    freeze_prepared_manifest(parsed.output, parsed.prepared, train_selection, dev_selection)
    print("PREFLIGHT_OK", flush=True)
    if parsed.stage in ("calibrate", "all"):
        chosen = calibrate(parsed, configuration, train_selection, dev_selection)
        print("CALIBRATION " + json.dumps(chosen), flush=True)
    else:
        chosen = json.loads((parsed.output / "calibration/selection.json").read_text()) if parsed.stage == "formal" else None
    if parsed.stage in ("formal", "all"):
        assert chosen is not None
        formal(parsed, configuration, train_selection, chosen)
        print("FORMAL_CHECKPOINTS_FROZEN", flush=True)
