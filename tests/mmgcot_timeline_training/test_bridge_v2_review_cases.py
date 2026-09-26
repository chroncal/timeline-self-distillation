from __future__ import annotations

import json

from PIL import Image

from mmgcot_timeline_training.build_bridge_v2_review_cases import build


def test_review_builder_separates_student_and_extractor_stages(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    packet = tmp_path / "packet.jsonl"
    packet.write_text(json.dumps({
        "sample_id": "s", "split": "train", "task_type": "attribute",
        "image_path": str(image), "question": "What color is the cup?",
        "ground_truth_bbox": [0.1, 0.1, 0.8, 0.8],
        "reasoning_text": "The selected cup is red.", "reasoning_finish": "stop",
        "task_answer": "red", "target_entity_reference": "the selected cup",
        "extractor_serialization_error": "none",
        "student_target_selection_error_labels": ["none", "selected_different_entity"],
        "extractor_content_error_labels": ["none", "answer_value_not_entity"],
        "target_reference_review_labels": ["same_target", "different_target"],
    }) + "\n")
    output = tmp_path / "review"
    build([packet], output)
    mapping = json.loads((output / "private_mapping.jsonl").read_text())
    stage1 = json.loads(open(mapping["stage1_path"]).read())
    stage2 = json.loads(open(mapping["stage2_path"]).read())
    assert "frozen_reasoning" in stage1 and "target_entity_reference" not in stage1
    assert stage2["target_entity_reference"] == "the selected cup"
    assert "frozen_reasoning" not in stage2


def test_review_builder_accepts_v4_without_task_answer(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "white").save(image)
    packet = tmp_path / "packet.jsonl"
    packet.write_text(json.dumps({
        "sample_id": "v4", "split": "dev", "task_type": "attribute",
        "image_path": str(image), "question": "What color is the shirt?",
        "ground_truth_bbox": [0.1, 0.1, 0.8, 0.8],
        "reasoning_text": "The shirt is black.", "reasoning_finish": "stop",
        "target_source": "question_subject", "query_entity": "the shirt",
        "candidate_entity": "the shirt", "target_entity_reference": "the shirt",
        "bridge_parse_status": "valid", "extractor_serialization_error": "none",
        "student_target_selection_error_labels": ["none"],
        "extractor_content_error_labels": ["none"],
        "target_reference_review_labels": ["same_target"],
    }) + "\n")
    output = tmp_path / "review"
    build([packet], output)
    mapping = json.loads((output / "private_mapping.jsonl").read_text())
    stage2 = json.loads(open(mapping["stage2_path"]).read())
    assert "task_answer" not in stage2
    assert stage2["query_entity"] == "the shirt"
    assert stage2["candidate_entity"] == "the shirt"
