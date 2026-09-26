"""Resume-safe six-run B/C matrix, then paired common-trajectory evaluation.

Run inside tmux after the one-step compatibility/gradient smoke. This script
never selects checkpoints based on dev metrics: only step 200 is evaluated.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from mmgcot_diagnostic.protocol import file_hash


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1")
OUTPUT = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1"
PREP = ROOT / "outputs/research_experiments/mmgcot_timeline_training/formal_v2_prepared"
PRIOR = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2"
FORMAL = ROOT / "outputs/research_experiments/mmgcot_timeline_training/formal_v2_run"
PYTHON = ROOT / ".venv/bin/python"
SEEDS = (20260921, 20260922, 20260923)
DEVICES = (4, 5, 6, 7)
DATA_CONFIG = json.loads((ROOT / "configs/mmgcot_formal_v2.json").read_text())["data_manifest"]


def _run(command: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONNOUSERSITE": "1"})
    with log.open("a") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"job failed with code {result.returncode}; inspect {log}")


def _idle_devices():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True)
    memory = [int(line.strip()) for line in result.stdout.splitlines()]
    return len(memory) >= 8 and all(memory[index] < 500 for index in DEVICES)


def _original_formal_active():
    return subprocess.run(["tmux", "has-session", "-t", "mmgcot_v2_final_eval_0924"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          check=False).returncode == 0


def _wait_for_stage():
    smoke = OUTPUT / "smoke"
    while not (smoke / "a_compatible.json").exists() or \
          not (smoke / "bf16_continuation_audit.json").exists() or not all(
        (smoke / scope / "complete.json").exists() for scope in ("terminal_q", "last_two_qvo")
    ):
        if (smoke / "failed.txt").exists():
            raise RuntimeError(f"smoke failed: {(smoke / 'failed.txt').read_text().strip()}")
        print(f"WAIT_SMOKE {time.time()}", flush=True)
        time.sleep(60)
    while _original_formal_active() or not _idle_devices():
        print(f"WAIT_ORIGINAL_FORMAL_AND_IDLE_GPU {time.time()} old_active={_original_formal_active()} "
              f"idle={_idle_devices()}", flush=True)
        time.sleep(60)


def _selection(cohort: str) -> tuple[Path, Path]:
    if cohort == "dev":
        return Path(DATA_CONFIG["dev"]), PREP / "dev"
    key = "independent_confirmation" if cohort == "independent48" else "diagnostic_selection_test200"
    if cohort == "independent48":
        prepared = FORMAL / "final_evaluation/prepared" / cohort
    else:
        old = FORMAL / "final_evaluation/prepared" / cohort
        # A partially prepared old Test-200 cohort is immutable per record and
        # prepare_v2 --resume can finish its missing shards after B/C training.
        prepared = old if (old / "protocol.json").is_file() else OUTPUT / "prepared" / cohort
    return Path(DATA_CONFIG[key]), prepared


def _check_data():
    protocol = json.loads((ROOT / "configs/mmgcot_pure_e_decision_v1.json").read_text())
    if (protocol["seeds"] != list(SEEDS) or protocol["formal_steps"] != 200 or
            protocol["effective_batch"] != 16 or protocol["optimizer"]["lr"] != 1e-4 or
            protocol["opd_coefficient"] != 0.1):
        raise RuntimeError("matrix runner and frozen protocol differ")
    for name in ("train", "dev", "independent_confirmation", "diagnostic_selection_test200"):
        if file_hash(Path(DATA_CONFIG[name])) != DATA_CONFIG[f"{name}_sha256"]:
            raise RuntimeError(f"frozen {name} manifest hash changed")
    for cohort in ("dev", "independent48"):
        selection, prepared = _selection(cohort)
        protocol = json.loads((prepared / "protocol.json").read_text())
        expected_t = 1 if cohort == "dev" else 3
        if protocol["selection_sha256"] != file_hash(selection) or protocol["trajectories_per_image"] != expected_t:
            raise RuntimeError(f"frozen {cohort} trajectories differ")
    if not (OUTPUT / "smoke/a_compatible.json").is_file():
        raise RuntimeError("A compatibility audit missing")


def _prepare_test200():
    from mmgcot_timeline_training.prepare_v2 import record_path

    selection, prepared = _selection("test200_retest")
    protocol_path = prepared / "protocol.json"
    ready = protocol_path.is_file() and len(list(prepared.glob("shard*.receipt.json"))) == 4
    if not ready:
        jobs = []
        for shard in range(4):
            receipt = prepared / f"shard{shard}.receipt.json"
            if receipt.exists():
                if json.loads(receipt.read_text())["selection_sha256"] != file_hash(selection):
                    raise RuntimeError("completed Test-200 shard used different selection")
                continue
            command = [str(PYTHON), "-m", "mmgcot_timeline_training.prepare_v2",
                       "--selection", str(selection), "--output-dir", str(prepared),
                       "--device", str(DEVICES[shard]), "--shard-index", str(shard),
                       "--num-shards", "4", "--trajectories", "3", "--resume"]
            jobs.append((command, OUTPUT / "logs" / f"prepare_test200_shard{shard}.log"))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_run, command, log) for command, log in jobs]
            for future in futures:
                future.result()
    protocol = json.loads((prepared / "protocol.json").read_text())
    if protocol["selection_sha256"] != file_hash(selection) or protocol["trajectories_per_image"] != 3:
        raise RuntimeError("Test-200 preparation protocol differs")
    for row in (json.loads(line) for line in selection.read_text().splitlines()):
        for ti in range(3):
            if not record_path(prepared, row["sample_id"], ti).is_file():
                raise RuntimeError("Test-200 frozen trajectory missing")
    print("TEST200_PREPARED", prepared, flush=True)


def _eval_a_job(cohort: str, seed: int, device: int):
    old = PRIOR / "evaluation" / cohort / f"e_opd_pure_seed{seed}.jsonl"
    if old.with_suffix(".complete.json").is_file():
        print(f"REUSE_A_EVAL {cohort} {seed}", flush=True)
        return
    selection, prepared = _selection(cohort)
    checkpoint = PRIOR / "formal" / f"e_opd_pure_seed{seed}" / "step_0200.pt"
    output = OUTPUT / "evaluation" / cohort / f"a_numeric_seed{seed}.jsonl"
    completion = output.with_suffix(".complete.json")
    if completion.is_file():
        meta = json.loads(completion.read_text())
        if meta["output_sha256"] != file_hash(output) or meta["checkpoint_sha256"] != file_hash(checkpoint):
            raise RuntimeError("A evaluation receipt differs")
        return
    if output.exists():
        output.rename(output.with_name(output.name + f".interrupted.{time.time_ns()}"))
    command = [str(PYTHON), "-m", "mmgcot_timeline_training.formal_pure_opd",
               "--mode", "eval", "--selection", str(selection), "--prepared", str(prepared),
               "--cache-root", str(prepared / "prefix_cache"), "--output", str(output),
               "--device", str(device), "--arm", "e_opd_pure", "--seed", str(seed),
               "--checkpoint", str(checkpoint)]
    _run(command, OUTPUT / "logs" / f"eval_{cohort}_a_seed{seed}.log")
    if not completion.is_file():
        raise RuntimeError("A evaluation exited without completion")
    print(f"EVAL_A_DONE {cohort} {seed}", flush=True)


def _train_job(scope: str, seed: int, device: int):
    output = OUTPUT / "formal" / f"{scope}_seed{seed}"
    checkpoint = output / "step_0200.pt"
    completion = output / "complete.json"
    if completion.is_file():
        if json.loads(completion.read_text())["step"] != 200 or not checkpoint.is_file():
            raise RuntimeError(f"invalid completed run {output}")
        print(f"REUSE_TRAIN {scope} {seed}", flush=True)
        return
    partial = sorted(output.glob("step_*.pt")) if output.exists() else []
    if output.exists() and not partial:
        archive = output.with_name(output.name + f".interrupted.{time.time_ns()}")
        output.rename(archive)
    resume = ["--resume", str(partial[-1])] if partial else []
    command = [str(PYTHON), "-m", "mmgcot_timeline_training.formal_pure_e_decision",
               "--mode", "train", "--supervision-scope", "coordinate_decision",
               "--adapter-scope", scope, "--device", str(device), "--seed", str(seed),
               "--selection", str(DATA_CONFIG["train"]), "--prepared", str(PREP / "train"),
               "--cache-root", str(PREP / "train/prefix_cache"), "--output", str(output),
               "--steps", "200", "--batch", "16", "--lr", "1e-4", "--lambda-opd", "0.1",
               "--checkpoint-every", "50", *resume]
    _run(command, OUTPUT / "logs" / f"train_{scope}_seed{seed}.log")
    if not completion.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"training exited without frozen checkpoint: {output}")
    print(f"TRAIN_DONE {scope} {seed}", flush=True)


def _eval_job(cohort: str, scope: str, seed: int, device: int):
    selection, prepared = _selection(cohort)
    checkpoint = OUTPUT / "formal" / f"{scope}_seed{seed}" / "step_0200.pt"
    output = OUTPUT / "evaluation" / cohort / f"{scope}_seed{seed}.jsonl"
    completion = output.with_suffix(".complete.json")
    if completion.is_file():
        meta = json.loads(completion.read_text())
        if meta["checkpoint_sha256"] != file_hash(checkpoint) or meta["output_sha256"] != file_hash(output):
            raise RuntimeError(f"existing evaluation changed: {output}")
        print(f"REUSE_EVAL {cohort} {scope} {seed}", flush=True)
        return
    if output.exists():
        # Keep interrupted raw records for audit and rerun the fixed protocol.
        archive = output.with_name(output.name + f".interrupted.{time.time_ns()}")
        output.rename(archive)
    command = [str(PYTHON), "-m", "mmgcot_timeline_training.formal_pure_e_decision",
               "--mode", "eval", "--supervision-scope", "coordinate_decision",
               "--adapter-scope", scope, "--device", str(device), "--seed", str(seed),
               "--selection", str(selection), "--prepared", str(prepared),
               "--cache-root", str(prepared / "prefix_cache"), "--output", str(output),
               "--checkpoint", str(checkpoint)]
    _run(command, OUTPUT / "logs" / f"eval_{cohort}_{scope}_seed{seed}.log")
    if not completion.is_file():
        raise RuntimeError(f"evaluation exited without completion: {output}")
    print(f"EVAL_DONE {cohort} {scope} {seed}", flush=True)


def _parallel_jobs(jobs):
    # One process per physical GPU. As soon as one job finishes, that GPU
    # receives the next job; no different students share a model process.
    from queue import Queue, Empty
    from threading import Event

    queue = Queue()
    failure = Event()
    for job in jobs:
        queue.put(job)

    def worker(device):
        while not failure.is_set():
            try:
                function, parameters = queue.get_nowait()
            except Empty:
                return
            try:
                function(*parameters, device)
            except Exception:
                failure.set()
                raise
            finally:
                queue.task_done()

    with ThreadPoolExecutor(max_workers=len(DEVICES)) as pool:
        futures = [pool.submit(worker, device) for device in DEVICES]
        for future in futures:
            future.result()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _wait_for_stage()
    _check_data()
    jobs = [(_train_job, (scope, seed)) for scope in ("terminal_q", "last_two_qvo") for seed in SEEDS]
    _parallel_jobs(jobs)
    entries = {}
    for scope in ("terminal_q", "last_two_qvo"):
        for seed in SEEDS:
            checkpoint = OUTPUT / "formal" / f"{scope}_seed{seed}" / "step_0200.pt"
            entries[f"{scope}_seed{seed}"] = {"path": str(checkpoint), "sha256": file_hash(checkpoint)}
    frozen = OUTPUT / "formal/frozen_checkpoints.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["checkpoints"] != entries:
            raise RuntimeError("frozen B/C checkpoint set changed")
    else:
        frozen.write_text(json.dumps({"schema": "pure_e_decision_v1", "checkpoints": entries}, indent=2) + "\n")
    print("SIX_CHECKPOINTS_FROZEN", flush=True)
    for cohort in ("dev", "independent48", "test200_retest"):
        if cohort == "test200_retest":
            _prepare_test200()
        jobs = [(_eval_a_job, (cohort, seed)) for seed in SEEDS]
        jobs += [(_eval_job, (cohort, scope, seed))
                for scope in ("terminal_q", "last_two_qvo") for seed in SEEDS]
        _parallel_jobs(jobs)
        command = [str(PYTHON), "-m", "mmgcot_timeline_training.e_decision_report",
                   "--cohort", cohort, "--output-root", str(OUTPUT)]
        _run(command, OUTPUT / "logs" / f"report_{cohort}.log")
        print(f"REPORT_DONE {cohort}", flush=True)
    (OUTPUT / "complete.json").write_text(json.dumps({"cohorts": ["dev", "independent48", "test200_retest"],
        "checkpoint_receipt_sha256": file_hash(frozen)}, indent=2) + "\n")
    print("PURE_E_DECISION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
