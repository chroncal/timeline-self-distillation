"""Run the no-SFT R/E OPD ablation and evaluate fixed step-200 checkpoints."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training.formal_analysis import aggregate_records, aggregate_systems


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
DATA = Path("/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1")
PREP = ROOT / "outputs/research_experiments/mmgcot_timeline_training/formal_v2_prepared"
FORMAL = ROOT / "outputs/research_experiments/mmgcot_timeline_training/formal_v2_run"
OUT = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2"
SEEDS = (20260921, 20260922, 20260923)
ARMS = ("r_opd_pure", "e_opd_pure")
LAMBDA = 0.1
LR = 1e-4
STEPS = 200
BATCH = 16


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), check=False)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {log_path}")


def train(arm: str, seed: int, device: int) -> Path:
    destination = OUT / "formal" / f"{arm}_seed{seed}"
    checkpoint = destination / f"step_{STEPS:04d}.pt"
    completed = destination / "complete.json"
    if completed.exists():
        receipt = json.loads(completed.read_text())
        config = json.loads((destination / "config.json").read_text())
        if not checkpoint.is_file() or receipt["step"] != STEPS:
            raise RuntimeError(f"completed training artifact invalid: {destination}")
        if config["pure_trainer_sha256"] != file_hash(ROOT / "mmgcot_timeline_training/formal_pure_opd.py"):
            raise RuntimeError("pure trainer changed since completed run")
        print(f"SKIP {arm} seed={seed}: complete", flush=True)
        return checkpoint
    command = [str(PY), "-m", "mmgcot_timeline_training.formal_pure_opd",
               "--mode", "train", "--selection", str(DATA / "train_frozen.jsonl"),
               "--prepared", str(PREP / "train"), "--output", str(destination),
               "--cache-root", str(PREP / "train/prefix_cache"),
               "--device", str(device), "--arm", arm, "--seed", str(seed),
               "--steps", str(STEPS), "--batch", str(BATCH), "--lr", str(LR),
               "--lambda-opd", str(LAMBDA)]
    checkpoints = sorted(destination.glob("step_*.pt"))
    if checkpoints:
        command += ["--resume", str(checkpoints[-1])]
    run(command, OUT / "logs" / f"train_{arm}_seed{seed}.log")
    if not completed.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"training returned without completion: {destination}")
    print(f"DONE {arm} seed={seed}", flush=True)
    return checkpoint


def evaluate(arm: str, seed: int, device: int, cohort: str,
             selection: Path, prepared: Path, checkpoint: Path) -> Path:
    destination = OUT / "evaluation" / cohort / f"{arm}_seed{seed}.jsonl"
    completion = destination.with_suffix(".complete.json")
    if completion.exists():
        receipt = json.loads(completion.read_text())
        if (not destination.is_file() or receipt["output_sha256"] != file_hash(destination)
                or receipt["checkpoint_sha256"] != file_hash(checkpoint)
                or receipt["selection_sha256"] != file_hash(selection)):
            raise RuntimeError(f"completed evaluation artifact invalid: {destination}")
        print(f"SKIP EVAL {cohort} {arm} seed={seed}", flush=True)
        return destination
    if destination.exists():
        raise RuntimeError(f"partial evaluation needs review: {destination}")
    command = [str(PY), "-m", "mmgcot_timeline_training.formal_pure_opd",
               "--mode", "eval", "--selection", str(selection),
               "--prepared", str(prepared), "--output", str(destination),
               "--cache-root", str(prepared / "prefix_cache"),
               "--device", str(device), "--arm", arm, "--seed", str(seed),
               "--checkpoint", str(checkpoint)]
    run(command, OUT / "logs" / f"eval_{cohort}_{arm}_seed{seed}.log")
    if not completion.is_file():
        raise RuntimeError(f"evaluation returned without completion: {destination}")
    print(f"DONE EVAL {cohort} {arm} seed={seed}", flush=True)
    return destination


def lane(device: int, jobs: list[tuple[str, int]]) -> None:
    for arm, seed in jobs:
        train(arm, seed, device)


def eval_lane(device: int, jobs: list[tuple[str, int]], cohort: str,
              selection: Path, prepared: Path) -> None:
    for arm, seed in jobs:
        checkpoint = OUT / "formal" / f"{arm}_seed{seed}" / f"step_{STEPS:04d}.pt"
        evaluate(arm, seed, device, cohort, selection, prepared, checkpoint)


def parallel(worker, cohort: str | None = None, selection: Path | None = None,
             prepared: Path | None = None) -> None:
    # Stable GPU lanes; no two jobs may share a GPU concurrently.
    lanes = {
        0: [("r_opd_pure", SEEDS[0]), ("r_opd_pure", SEEDS[2])],
        1: [("e_opd_pure", SEEDS[0]), ("e_opd_pure", SEEDS[2])],
        2: [("r_opd_pure", SEEDS[1])],
        3: [("e_opd_pure", SEEDS[1])],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        if cohort is None:
            futures = [pool.submit(worker, device, jobs) for device, jobs in lanes.items()]
        else:
            futures = [pool.submit(worker, device, jobs, cohort, selection, prepared)
                       for device, jobs in lanes.items()]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def freeze_checkpoints() -> None:
    entries = {}
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint = OUT / "formal" / f"{arm}_seed{seed}" / "step_0200.pt"
            entries[f"{arm}_seed{seed}"] = {"path": str(checkpoint), "sha256": file_hash(checkpoint)}
    write_once(OUT / "formal/frozen_checkpoints.json", {
        "schema": "pure_opd_v1", "objective": "reverse_kl_only",
        "sft_coefficient": 0.0, "lambda_opd": LAMBDA, "lr": LR,
        "steps": STEPS, "effective_batch": BATCH,
        "train_selection_sha256": file_hash(DATA / "train_frozen.jsonl"),
        "pure_trainer_sha256": file_hash(ROOT / "mmgcot_timeline_training/formal_pure_opd.py"),
        "checkpoints": entries,
    })


def wait_for_prepared(cohort: str) -> Path:
    prepared = FORMAL / "final_evaluation/prepared" / cohort
    expected = file_hash(DATA / f"{cohort}.jsonl") if (DATA / f"{cohort}.jsonl").exists() else None
    while True:
        protocol = prepared / "protocol.json"
        receipts = list(prepared.glob("shard*.receipt.json")) if prepared.exists() else []
        if protocol.exists() and len(receipts) == 4:
            meta = json.loads(protocol.read_text())
            if meta["trajectories_per_image"] != 3:
                raise RuntimeError(f"final cohort trajectory count differs: {cohort}")
            if expected and meta["selection_sha256"] != expected:
                raise RuntimeError(f"final cohort selection differs: {cohort}")
            return prepared
        print(f"WAIT PREPARED {cohort}", flush=True)
        time.sleep(60)


def summarize(cohort: str, selection: Path) -> None:
    queue = [(row["sample_id"], row["image_id"])
             for row in (json.loads(line) for line in selection.read_text().splitlines())]
    reports = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = OUT / "evaluation" / cohort / f"{arm}_seed{seed}.jsonl"
            reports[f"{arm}_seed{seed}"] = aggregate_records(path, queue, mode="sample")
    # The primary comparison against SFT and mixed OPD uses the original
    # frozen evaluation frames when available; wait for those below.
    if cohort != "dev":
        formal_cohort = FORMAL / "final_evaluation" / cohort
        required = [formal_cohort / f"{arm}_seed{seed}.jsonl"
                    for seed in SEEDS for arm in ("bbox_sft", "r_opd", "e_opd")]
        required.append(formal_cohort / "base_l_seed20260921.jsonl")
        while not all(p.with_suffix(".complete.json").is_file() for p in required):
            print(f"WAIT ORIGINAL EVALUATION {cohort}", flush=True)
            time.sleep(60)
        systems = {}
        for arm in ARMS:
            systems[arm] = [json.loads(line)
                            for seed in SEEDS
                            for line in (OUT / "evaluation" / cohort /
                                         f"{arm}_seed{seed}.jsonl").read_text().splitlines()]
        for arm in ("bbox_sft", "r_opd", "e_opd"):
            systems[arm] = [json.loads(line)
                            for seed in SEEDS
                            for line in (formal_cohort / f"{arm}_seed{seed}.jsonl").read_text().splitlines()]
        baseline_rows = [json.loads(line) for line in
                         (formal_cohort / "base_l_seed20260921.jsonl").read_text().splitlines()]
        systems["base_l"] = [{**row, "seed": seed} for seed in SEEDS for row in baseline_rows]
        paired = aggregate_systems(systems, queue, mode="sample",
                                   comparisons=[("r_opd_pure", "base_l"),
                                                ("e_opd_pure", "base_l"),
                                                ("r_opd_pure", "bbox_sft"),
                                                ("e_opd_pure", "bbox_sft"),
                                                ("r_opd_pure", "r_opd"),
                                                ("e_opd_pure", "e_opd"),
                                                ("r_opd_pure", "e_opd_pure")],
                                   bootstrap_replicates=10_000)
        reports["paired"] = paired
    reports["cohort"] = cohort
    reports["selection_sha256"] = file_hash(selection)
    write_once(OUT / "evaluation" / cohort / "result.json", reports)
    print(f"REPORT COMPLETE {cohort}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "configs/mmgcot_formal_v2.json").read_text())["data_manifest"]
    for name in ("train", "dev", "independent_confirmation", "diagnostic_selection_test200"):
        if file_hash(Path(manifest[name])) != manifest[f"{name}_sha256"]:
            raise RuntimeError(f"frozen {name} selection changed")
    if file_hash(PREP / "train/protocol.json") != json.loads(
            (FORMAL / "formal/r_opd_seed20260921/config.json").read_text()
            )["prepared_protocol_sha256"]:
        raise RuntimeError("frozen training trajectory protocol changed")
    print("PURE OPD TRAINING START", flush=True)
    parallel(lane)
    freeze_checkpoints()
    print("SIX PURE OPD CHECKPOINTS FROZEN", flush=True)
    dev = DATA / "dev_frozen.jsonl"
    parallel(eval_lane, "dev", dev, PREP / "dev")
    summarize("dev", dev)
    for cohort, filename in (("independent48", "independent48_frozen.jsonl"),
                             ("test200_retest", "test200_frozen.jsonl")):
        selection = DATA / filename
        if not selection.exists():
            # Manifest filenames are authoritative in the frozen v2 config.
            manifest = json.loads((ROOT / "configs/mmgcot_formal_v2.json").read_text())["data_manifest"]
            key = "independent_confirmation" if cohort == "independent48" else "diagnostic_selection_test200"
            selection = Path(manifest[key])
        prepared = wait_for_prepared(cohort)
        if json.loads((prepared / "protocol.json").read_text())["selection_sha256"] != file_hash(selection):
            raise RuntimeError(f"prepared selection changed: {cohort}")
        parallel(eval_lane, cohort, selection, prepared)
        summarize(cohort, selection)
    print("PURE OPD V1 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
