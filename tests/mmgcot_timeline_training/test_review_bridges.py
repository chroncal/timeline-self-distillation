from __future__ import annotations

import json

import pytest

from mmgcot_timeline_training.generate_review_bridges import merge


def test_merge_packet_contains_only_blind_review_fields(tmp_path) -> None:
    selection = tmp_path / "selection.jsonl"
    row = {
        "sample_id": "train:attribute:1:1", "split": "train", "image_id": "1",
        "task_type": "attribute", "image_path": "/tmp/1.jpg", "image_sha256": "a" * 64,
        "question": "q", "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
    }
    selection.write_text(json.dumps(row) + "\n")
    output = tmp_path / "out"
    (output / "records").mkdir(parents=True)
    from mmgcot_timeline_training.generate_review_bridges import _record_path
    record = {**row, "target_description": "red cup", "target_description_status": "usable"}
    _record_path(output, row["sample_id"]).write_text(json.dumps(record) + "\n")
    args = type("Args", (), {"selection": selection, "output_dir": output})()
    merge(args)
    packet = json.loads((output / "blind_review_packet.jsonl").read_text())
    assert packet["target_description"] == "red cup"
    assert packet["review_label"] is None
    assert not ({"iou", "bbox_prediction", "teacher_probability", "training_arm"} & set(packet))


def test_merge_refuses_incomplete_records(tmp_path) -> None:
    selection = tmp_path / "selection.jsonl"
    selection.write_text(json.dumps({"sample_id": "missing"}) + "\n")
    output = tmp_path / "out"
    output.mkdir()
    args = type("Args", (), {"selection": selection, "output_dir": output})()
    with pytest.raises(RuntimeError, match="missing bridge record"):
        merge(args)
