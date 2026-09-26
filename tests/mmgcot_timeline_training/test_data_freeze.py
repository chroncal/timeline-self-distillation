from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest

from mmgcot_timeline_training.data_freeze import (
    DEV_PER_TASK,
    TASKS,
    TRAIN_PER_TASK,
    assign_balanced_images,
    choose_review_rows,
    freeze,
    split_train_dev,
)


def row(task: str, image: int, source: int | None = None) -> dict:
    return {
        "task_type": task,
        "image_id": str(image),
        "source_id": str(image if source is None else source),
        "source_row_index": image if source is None else source,
        "source_occurrence": 1,
    }


def test_balanced_assignment_is_disjoint_and_stable() -> None:
    candidates = {
        "attribute": [row("attribute", i) for i in range(7)],
        "object": [row("object", i) for i in range(3, 10)],
    }
    first = assign_balanced_images(candidates, per_task=5)
    second = assign_balanced_images({task: list(reversed(rows)) for task, rows in candidates.items()}, per_task=5)
    assert first == second
    assert {task: len(first[task]) for task in TASKS} == {"attribute": 5, "object": 5}
    ids = [item["image_id"] for task in TASKS for item in first[task]]
    assert len(ids) == len(set(ids)) == 10


def test_balanced_assignment_fails_closed_when_quota_is_impossible() -> None:
    candidates = {"attribute": [row("attribute", 1)], "object": [row("object", 1)]}
    with pytest.raises(RuntimeError, match="cannot form"):
        assign_balanced_images(candidates, per_task=1)


def test_split_and_blind_review_quotas() -> None:
    assignments = {
        task: [row(task, image + offset) for image in range(TRAIN_PER_TASK + DEV_PER_TASK)]
        for task, offset in (("attribute", 0), ("object", 1000))
    }
    train_raw, dev_raw = split_train_dev(assignments)

    def public(split: str, item: dict) -> dict:
        return {
            **item,
            "sample_id": f"{split}:{item['task_type']}:{item['image_id']}",
            "split": split,
            "image_path": str(Path("/tmp") / f"{item['image_id']}.jpg"),
            "image_sha256": "0" * 64,
            "question": "q",
            "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
        }

    train = [public("train", item) for item in train_raw]
    dev = [public("dev", item) for item in dev_raw]
    assert len(train) == 2 * TRAIN_PER_TASK and len(dev) == 2 * DEV_PER_TASK
    assert not ({x["image_id"] for x in train} & {x["image_id"] for x in dev})
    review = choose_review_rows(train, dev)
    assert len(review) == 50
    counts = {(split, task): 0 for split in ("train", "dev") for task in TASKS}
    for item in review:
        counts[(item["split"], item["task_type"])] += 1
        assert item["target_description"] is None
        assert item["review_label"] is None
    assert counts == {
        ("train", "attribute"): 20,
        ("train", "object"): 20,
        ("dev", "attribute"): 5,
        ("dev", "object"): 5,
    }


def test_freeze_refuses_to_overwrite_before_reading_inputs(tmp_path: Path) -> None:
    existing = tmp_path / "already-frozen"
    existing.mkdir()
    args = Namespace(
        output_dir=existing,
        raw_root=tmp_path / "missing-raw",
        frozen_root=tmp_path / "missing-frozen",
        timeline_outputs=tmp_path / "missing-outputs",
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze(args)
