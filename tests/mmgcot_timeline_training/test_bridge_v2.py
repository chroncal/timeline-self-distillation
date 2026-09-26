from __future__ import annotations

import json

from mmgcot_timeline_training.bridge_v2 import (
    bbox_suffix_v2,
    bridge_pass,
    parse_bridge,
    token_ids_sha256,
)
from mmgcot_timeline_training.freeze_bridge_confirmation import freeze


def test_parse_bridge_keeps_answer_and_entity_distinct() -> None:
    text = 'canvas"</task_answer>\n<target_entity>"the backpack beside the man"</target_entity>'
    parsed = parse_bridge(text, completed=True)
    assert parsed["task_answer"] == "canvas"
    assert parsed["target_entity_reference"] == "the backpack beside the man"
    assert parsed["bridge_parse_status"] == "valid"


def test_bbox_suffix_uses_entity_but_not_task_answer_field() -> None:
    suffix = bbox_suffix_v2("What material is the bag?", "the bag beside the man")
    assert "the bag beside the man" in suffix
    assert "task_answer" not in suffix


def test_parse_bridge_does_not_repair_invalid_generation() -> None:
    parsed = parse_bridge('canvas', completed=True)
    assert parsed["bridge_parse_status"] == "format_or_incomplete"
    assert parsed["target_entity_reference"] == ""


def test_token_hash_is_stable_and_order_sensitive() -> None:
    assert token_ids_sha256([1, 2]) == token_ids_sha256([1, 2])
    assert token_ids_sha256([1, 2]) != token_ids_sha256([2, 1])


def test_bridge_gate_does_not_use_student_gt_correctness() -> None:
    summary = {
        "cases": 20,
        "valid_parse": 20,
        "extractor_content_error_counts": {"none": 19, "answer_value_not_entity": 1},
        "student_selection_counts": {"different_entity": 20},
    }
    assert bridge_pass(summary) == (True, [])


def test_confirmation_freeze_excludes_prior_review_and_balances(tmp_path) -> None:
    def rows(split: str):
        result = []
        for task in ("attribute", "object"):
            for index in range(8):
                result.append({
                    "sample_id": f"{split}:{task}:{index}", "split": split,
                    "task_type": task, "image_id": f"{split}-{task}-{index}",
                    "image_path": "/tmp/x", "image_sha256": "a" * 64,
                    "question": "q", "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
                })
        return result
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    prior, output = tmp_path / "prior.jsonl", tmp_path / "out.jsonl"
    train.write_text("".join(json.dumps(x) + "\n" for x in rows("train")))
    dev.write_text("".join(json.dumps(x) + "\n" for x in rows("dev")))
    excluded = {"sample_id": "train:attribute:0"}
    prior.write_text(json.dumps(excluded) + "\n")
    freeze(train, dev, prior, output)
    frozen = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(frozen) == 20
    assert excluded["sample_id"] not in {row["sample_id"] for row in frozen}
    counts = {(split, task): 0 for split in ("train", "dev") for task in ("attribute", "object")}
    for row in frozen:
        counts[(row["split"], row["task_type"])] += 1
    assert set(counts.values()) == {5}
