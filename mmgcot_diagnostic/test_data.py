"""CPU-only tests for MM-GCoT data adaptation and selection invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmgcot_diagnostic.data import (
    ImageRecord,
    SEED,
    _select_materialized,
    _public_row,
    _pair_train_variants,
    _make_train_candidate,
    cot_coordinate_consistent,
    infer_train_task_type,
    is_valid_xyxy,
    last_cot_bbox,
    model_input,
    parse_four_numbers,
    stable_order,
    xywh_to_xyxy,
)


def test_train_pair_provenance_uses_cot_row_not_preceding_answer_row(tmp_path):
    question = "What is the color of the cup?"
    normal = {
        "id": "1", "image": "vg/VG_100K/1.jpg",
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}\nAnswer the question using a single word or phrase."},
            {"from": "gpt", "value": "red"},
        ],
    }
    cot = {
        "id": "1_cot", "image": "vg/VG_100K/1.jpg",
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}\nAnswer the question step by step, ultimately using a single word or phrase as the answer."},
            {"from": "gpt", "value": "Step 1: The cup is at [0.1, 0.2, 0.3, 0.4]\nFinal Answer: red"},
        ],
    }
    pairs, stats = _pair_train_variants(
        [normal, cot], source_path=tmp_path / "train.json", exclusions=[]
    )
    assert stats["valid_variant_pairs"] == 1
    _, cot_row, _, cot_row_index = pairs[0]
    candidate = _make_train_candidate(
        cot_row,
        source_path=tmp_path / "train.json",
        source_row_index=cot_row_index,
        source_occurrence=cot_row_index + 1,
    )
    assert candidate["source_id"] == "1_cot"
    assert candidate["source_row_index"] == 1
    assert candidate["source_occurrence"] == 2


def test_xywh_boundary_coordinates_are_valid_and_converted_without_clipping():
    assert xywh_to_xyxy("[0, 0, 1, 1]") == [0.0, 0.0, 1.0, 1.0]
    assert xywh_to_xyxy([0.25, 0.5, 0.75, 0.5]) == [0.25, 0.5, 1.0, 1.0]
    assert is_valid_xyxy([0.0, 0.0, 1.0, 1.0])
    assert is_valid_xyxy([0.0, 0.0, 0.001, 0.001])


def test_bad_boxes_are_rejected_instead_of_repaired():
    for value in ([0.9, 0.1, 0.2, 0.2], [0.1, 0.9, 0.2, 0.2], [0, 0, 0, 1], [0, 0, 1, 0]):
        with pytest.raises(ValueError):
            xywh_to_xyxy(value)
    with pytest.raises(ValueError):
        parse_four_numbers("[0, 0, NaN, 1]")


def test_last_cot_box_and_coordinate_must_match():
    cot = "Step 1: the scene is at [0.000, 0.000, 1.000, 1.000]\nStep 2: the target is at [0.200, 0.300, 0.400, 0.500]"
    assert cot_coordinate_consistent(cot, "[0.2,0.3,0.4,0.5]")
    assert last_cot_bbox(cot) == [0.2, 0.3, 0.6, 0.8]
    assert not cot_coordinate_consistent(cot, "[0.2,0.3,0.4,0.49]")


def test_inference_firewall_only_exposes_image_and_question():
    row = {
        "image_path": "/tmp/1.jpg",
        "question": "What color?",
        "reference_cot": "SECRET",
        "reference_answer": "red",
        "ground_truth_bbox": [0.1, 0.2, 0.3, 0.4],
    }
    assert model_input(row) == {"image_path": "/tmp/1.jpg", "question": "What color?"}
    assert "SECRET" not in json.dumps(model_input(row))


def test_seeded_order_is_stable_and_uses_identity():
    rows = [
        {"image_id": "a", "source_id": "1"},
        {"image_id": "b", "source_id": "2"},
        {"image_id": "c", "source_id": "3"},
    ]
    assert stable_order(rows, seed=SEED, namespace="x") == stable_order(rows, seed=SEED, namespace="x")
    assert stable_order(rows, seed=SEED, namespace="x") != stable_order(rows, seed=SEED + 1, namespace="x")


def test_task_heuristic_separates_high_confidence_attribute_and_object_questions():
    assert infer_train_task_type("What is the color of the hat?") == "attribute"
    assert infer_train_task_type("What is the material of the table?") == "attribute"
    assert infer_train_task_type("What is behind the person?") == "object"
    assert infer_train_task_type("Is the ball red?") is None


def test_unique_image_exclusion_across_splits_without_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    class FakeImage:
        def __init__(self, image_id: str):
            self.image_id = image_id
            self.image_path = str(tmp_path / f"{image_id}.jpg")
            self.sha256 = image_id * 64

    def fake_materialize_image(*, image_id, image_ref, image_dir, local_index):
        calls.append(image_id)
        return FakeImage(image_id)

    monkeypatch.setattr("mmgcot_diagnostic.data.materialize_image", fake_materialize_image)
    rows = {
        "attribute": [
            {"image_id": "shared", "source_id": "a1", "task_type": "attribute"},
            {"image_id": "a-only", "source_id": "a2", "task_type": "attribute"},
        ],
        "object": [
            {"image_id": "shared", "source_id": "o1", "task_type": "object"},
            {"image_id": "o-only", "source_id": "o2", "task_type": "object"},
        ],
    }
    for task_rows in rows.values():
        for row_index, row in enumerate(task_rows):
            row.update(
                {
                    "source_file": "/source.json",
                    "image_ref": f"vg/VG_100K/{row['image_id']}.jpg",
                    "question": "q",
                    "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
                    "reference_cot": "c",
                    "reference_answer": "a",
                    "source_row_index": row_index,
                    "source_occurrence": row_index + 1,
                }
            )
    exclusions: list[dict] = []
    selected, counts, _ = _select_materialized(
        candidates_by_task=rows,
        task_targets={"attribute": 1, "object": 1},
        split="formal",
        seed=SEED,
        used_image_ids={"shared"},
        image_dir=tmp_path,
        local_index={},
        exclusions=exclusions,
    )
    assert {row["image_id"] for row in selected} == {"a-only", "o-only"}
    assert counts == {"attribute": 1, "object": 1}
    assert "shared" not in {row["image_id"] for row in selected}
    assert calls.count("shared") == 0


def test_selection_reports_shortfall_without_fabricating_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeImage:
        def __init__(self, image_id: str):
            self.image_id = image_id
            self.image_path = str(tmp_path / f"{image_id}.jpg")
            self.sha256 = "0" * 64

    monkeypatch.setattr(
        "mmgcot_diagnostic.data.materialize_image",
        lambda **kwargs: FakeImage(kwargs["image_id"]),
    )
    candidate = {
        "image_id": "only",
        "source_id": "1",
        "task_type": "attribute",
        "source_file": "/source.json",
        "image_ref": "vg/VG_100K/only.jpg",
        "question": "q",
        "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
        "reference_cot": "c",
        "reference_answer": "a",
        "source_row_index": 0,
        "source_occurrence": 1,
    }
    exclusions: list[dict] = []
    selected, counts, _ = _select_materialized(
        candidates_by_task={"attribute": [candidate], "object": []},
        task_targets={"attribute": 2, "object": 2},
        split="pilot",
        seed=SEED,
        used_image_ids=set(),
        image_dir=tmp_path,
        local_index={},
        exclusions=exclusions,
    )
    assert len(selected) == 1
    assert counts == {"attribute": 1, "object": 0}


def test_duplicate_official_source_ids_get_unique_occurrence_qualified_sample_ids():
    image = ImageRecord(
        image_id="2326982",
        image_path="/tmp/2326982.jpg",
        source_kind="test",
        source_path=None,
        source_url="https://example.invalid/2326982.jpg",
        source_image_ref="vg/VG_100K/2326982.jpg",
        sha256="0" * 64,
        size_bytes=1,
        width=1,
        height=1,
    )
    common = {
        "task_type": "attribute",
        "source_id": "425",
        "source_file": "/tmp/attributes.json",
        "image_id": "2326982",
        "question": "What color?",
        "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
        "reference_cot": "Step 1: target is at [0.1,0.1,0.1,0.1]",
        "reference_answer": "red",
    }
    first = _public_row({**common, "source_row_index": 0, "source_occurrence": 1}, image, split="formal")
    second = _public_row({**common, "source_row_index": 1, "source_occurrence": 2}, image, split="formal")
    assert first["source_id"] == second["source_id"] == "425"
    assert first["sample_id"] == "formal:attribute:425:1"
    assert second["sample_id"] == "formal:attribute:425:2"
    assert first["sample_id"] != second["sample_id"]
