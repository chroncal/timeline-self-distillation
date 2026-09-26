"""Read-only dev check of the fixed step-50 pure-OPD checkpoints."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training.formal_analysis import aggregate_systems


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/research_experiments/mmgcot_timeline_training"
PURE = BASE / "pure_opd_v2"
OUTPUT = PURE / "interim_dev_step50"
SELECTION = Path("/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1/dev_frozen.jsonl")
PREPARED = BASE / "formal_v2_prepared/dev"
BASELINE = BASE / "formal_v2_run/formal/interim_dev_readonly/base_l_seed20260921.jsonl"
SEED = 20260921


def evaluate(arm: str, device: int) -> Path:
    checkpoint = PURE / "formal" / f"{arm}_seed{SEED}" / "step_0050.pt"
    output = OUTPUT / f"{arm}_seed{SEED}.jsonl"
    completion = output.with_suffix(".complete.json")
    if completion.exists():
        receipt = json.loads(completion.read_text())
        if receipt["checkpoint_sha256"] != file_hash(checkpoint) or receipt["output_sha256"] != file_hash(output):
            raise RuntimeError(f"existing interim evaluation changed: {output}")
        return output
    if output.exists():
        raise RuntimeError(f"partial interim evaluation: {output}")
    command = [str(ROOT / ".venv/bin/python"), "-m", "mmgcot_timeline_training.formal_pure_opd",
               "--mode", "eval", "--selection", str(SELECTION), "--prepared", str(PREPARED),
               "--output", str(output), "--cache-root", str(PREPARED / "prefix_cache"),
               "--device", str(device), "--arm", arm, "--seed", str(SEED),
               "--checkpoint", str(checkpoint)]
    with (OUTPUT / f"{arm}.log").open("a") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), check=False)
    if result.returncode or not completion.exists():
        raise RuntimeError(f"interim evaluation failed: {arm}")
    print(f"EVAL_DONE {arm}", flush=True)
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arms = ("r_opd_pure", "e_opd_pure")
    while True:
        checkpoints = [PURE / "formal" / f"{arm}_seed{SEED}" / "step_0050.pt" for arm in arms]
        if all(p.exists() for p in checkpoints):
            break
        print("WAIT_STEP50", flush=True)
        time.sleep(30)
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["step"] != 50 or payload["config"]["sft_coefficient"] != 0.0:
            raise RuntimeError(f"interim checkpoint contract mismatch: {checkpoint}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        r = pool.submit(evaluate, arms[0], 6)
        e = pool.submit(evaluate, arms[1], 7)
        outputs = {arms[0]: r.result(), arms[1]: e.result()}
    queue = [(row["sample_id"], row["image_id"])
             for row in (json.loads(line) for line in SELECTION.read_text().splitlines())]
    report = aggregate_systems({"base_l": BASELINE, **outputs}, queue, mode="sample",
                               comparisons=[("r_opd_pure", "base_l"),
                                            ("e_opd_pure", "base_l"),
                                            ("r_opd_pure", "e_opd_pure")],
                               bootstrap_replicates=10_000)
    report["interpretation"] = "exploratory_dev_step50_not_checkpoint_selection"
    report["checkpoint_sha256"] = {arm: file_hash(checkpoint) for arm, checkpoint in zip(arms, checkpoints)}
    destination = OUTPUT / "result.json"
    if destination.exists():
        raise RuntimeError(f"interim report already exists: {destination}")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("INTERIM_STEP50_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
