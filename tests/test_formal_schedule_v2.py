"""Scheduler invariants that prevent expensive, incomparable formal jobs."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training import formal_schedule_v2 as schedule
from mmgcot_timeline_training.prepare_v2 import record_path


def _selection(path: Path, sample_id: str) -> dict:
    row = {"sample_id": sample_id, "image_id": sample_id}
    path.write_text(json.dumps(row) + "\n")
    return row


def test_preflight_requires_every_frozen_record_and_receipt(tmp_path: Path) -> None:
    paths = {}
    config = {"data_manifest": {}}
    for name in ("train", "dev", "independent_confirmation", "diagnostic_selection_test200"):
        path = tmp_path / f"{name}.jsonl"
        row = _selection(path, name)
        paths[name] = (path, row)
        config["data_manifest"][name] = str(path)
        config["data_manifest"][f"{name}_sha256"] = file_hash(path)
    prepared = tmp_path / "prepared"
    for name in ("train", "dev"):
        path, row = paths[name]
        base = prepared / name
        base.mkdir(parents=True)
        (base / "protocol.json").write_text(json.dumps({
            "selection_sha256": file_hash(path), "trajectories_per_image": 1}))
        target = record_path(base, row["sample_id"], 0)
        target.parent.mkdir()
        target.write_text("{}")
        for shard in range(4):
            (base / f"shard{shard}.receipt.json").write_text("{}")
    assert schedule.preflight(config, prepared) == (paths["train"][0], paths["dev"][0])
    (prepared / "dev" / "shard3.receipt.json").unlink()
    with pytest.raises(RuntimeError, match="missing dev shard receipt"):
        schedule.preflight(config, prepared)
    (prepared / "dev" / "shard3.receipt.json").write_text("{}")
    record_path(prepared / "train", "train", 0).unlink()
    with pytest.raises(RuntimeError, match="missing frozen train trajectory"):
        schedule.preflight(config, prepared)


def test_gpu_lanes_never_overlap_when_other_gpu_finishes_first(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    active: set[int] = set()
    lock = threading.Lock()
    seen: list[str] = []

    def fake_run(name: str, command: list[str], log_path: Path,
                 completion: Path | None = None) -> str:
        device = int(command[command.index("--device") + 1])
        with lock:
            assert device not in active
            active.add(device)
        time.sleep(0.01 if device == 4 else 0.03)
        with lock:
            active.remove(device)
            seen.append(name)
        return name

    monkeypatch.setattr(schedule, "_run_one", fake_run)
    jobs = [(f"job{i}", ["--device", str(4+i%2)], tmp_path / f"{i}.log", None)
            for i in range(8)]
    schedule._run_group(jobs, workers=2)
    assert set(seen) == {f"job{i}" for i in range(8)}


def test_partial_evaluation_is_not_silently_skipped(tmp_path: Path) -> None:
    output = tmp_path / "eval.jsonl"
    output.write_text("partial\n")
    command = ["python", "--output", str(output)]
    with pytest.raises(RuntimeError, match="partial evaluation"):
        schedule._run_one("eval", command, tmp_path / "eval.log")


def test_prepare_job_accepts_output_dir_and_completion_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text('{}\n')
    output = tmp_path / "prepared"
    receipt = output / "shard0.receipt.json"
    command = ["python", "-m", "mmgcot_timeline_training.prepare_v2",
               "--selection", str(selection), "--output-dir", str(output), "--resume"]

    def fake_run(*args, **kwargs):
        output.mkdir()
        receipt.write_text(json.dumps({"selection_sha256": file_hash(selection)}))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(schedule.subprocess, "run", fake_run)
    assert schedule._run_one("prepare", command, tmp_path / "prepare.log", receipt) == "DONE prepare"
