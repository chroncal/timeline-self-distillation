"""Launch several deterministic MM-GCoT workers per physical GPU.

Each worker owns a disjoint modulo shard and loads an independent frozen model
replica.  This preserves the existing per-trajectory RNG/cache semantics while
filling GPUs that are severely under-utilized by a single latency-bound worker.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mmgcot_diagnostic.protocol import file_hash, frozen_protocol


def assignments(gpus: list[str], workers_per_gpu: int) -> list[tuple[int, str, int]]:
    """Return ``(shard, gpu, replica)`` in stable round-robin order."""
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU IDs must be non-empty and unique")
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    return [
        (replica * len(gpus) + gpu_index, gpu, replica)
        for replica in range(workers_per_gpu)
        for gpu_index, gpu in enumerate(gpus)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    args = parser.parse_args()

    gpus = args.gpus.split(",")
    try:
        layout = assignments(gpus, args.workers_per_gpu)
    except ValueError as exc:
        parser.error(str(exc))
    available = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    memory = {index.strip(): int(used.strip()) for index, used in (line.split(",") for line in available.splitlines())}
    unknown = [gpu for gpu in gpus if gpu not in memory]
    if unknown:
        raise RuntimeError(f"unknown GPU IDs: {unknown}")
    if any(memory[gpu] >= 500 for gpu in gpus):
        raise RuntimeError(f"requested GPU is occupied: {memory}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Workers compare this file byte-for-structure with ``frozen_protocol``.
    # Keep execution-only packing metadata in launch.json below.
    protocol = frozen_protocol()
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")

    snapshots = args.output_dir / "source_snapshot"
    snapshots.mkdir()
    root = Path(__file__).resolve().parents[1]
    dependencies = [
        *Path(__file__).resolve().parent.glob("*.py"),
        root / "live_kv_probe_prototype/run_hf_fork.py",
        root / "reasoning_checkpoints/extractor.py",
    ]
    for path in dependencies:
        target = snapshots / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    receipt = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_hash(args.manifest),
        "execution_layout": "independent_model_replicas",
        "scientific_protocol_unchanged": True,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "num_shards": len(layout),
        "started_at": time.time(),
        "workers": [],
    }
    workers: list[tuple[subprocess.Popen, object]] = []
    for shard, gpu, replica in layout:
        command = [
            sys.executable,
            "-u",
            "-m",
            "mmgcot_diagnostic.run",
            "--manifest",
            str(args.manifest.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
            "--device",
            gpu,
            "--shard-index",
            str(shard),
            "--num-shards",
            str(len(layout)),
        ]
        log = (args.output_dir / f"shard{shard}.log").open("x")
        environment = dict(
            os.environ,
            PYTHONNOUSERSITE="1",
            PYTHONHASHSEED="20260920",
            CUDA_VISIBLE_DEVICES=gpu,
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        workers.append((process, log))
        receipt["workers"].append(
            {"shard": shard, "gpu": gpu, "replica": replica, "pid": process.pid, "command": command}
        )
        print(f"START shard={shard} gpu={gpu} replica={replica} pid={process.pid}", flush=True)
    (args.output_dir / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")

    running = set(range(len(workers)))
    while running:
        for index in list(running):
            process, log = workers[index]
            code = process.poll()
            if code is None:
                continue
            log.close()
            running.remove(index)
            receipt["workers"][index]["exit_code"] = code
            print(f"EXIT shard={receipt['workers'][index]['shard']} code={code}", flush=True)
            if code != 0:
                for other in running:
                    workers[other][0].terminate()
                for other in running:
                    receipt["workers"][other]["exit_code"] = workers[other][0].wait()
                    workers[other][1].close()
                running.clear()
                break
        if running:
            time.sleep(2)
    receipt["finished_at"] = time.time()
    (args.output_dir / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if any(worker.get("exit_code") != 0 for worker in receipt["workers"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
