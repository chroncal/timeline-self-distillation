"""CPU synthetic tests for the independent MM-GCoT statistics module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmgcot_diagnostic.analysis import ValidationError, analyze, write_outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest() -> list[dict]:
    return [
        {
            "sample_id": "s1",
            "image_id": "image-1",
            "task_type": "object",
            "question": "what is it?",
            "ground_truth_bbox": [0.0, 0.0, 1.0, 1.0],
            "cohort": "pilot",
        },
        {
            "sample_id": "s2",
            "image_id": "image-1",
            "task_type": "object",
            "question": "what is it?",
            "ground_truth_bbox": [0.0, 0.0, 1.0, 1.0],
            "cohort": "pilot",
        },
        {
            "sample_id": "s3",
            "image_id": "image-2",
            "task_type": "attribute",
            "question": "what color?",
            "ground_truth_bbox": [0.0, 0.0, 1.0, 1.0],
            "cohort": "formal",
        },
    ]


def _trajectory(sample_id: str, image_id: str, index: int, *, complete: bool = True) -> list[dict]:
    base = {
        "sample_id": sample_id,
        "image_id": image_id,
        "trajectory_index": index,
        "task_type": "object" if sample_id != "s3" else "attribute",
    }
    if complete:
        return [
            dict(
                base,
                type="trajectory",
                finish_reason="stop",
                entity="the target",
                entity_status="usable",
                reasoning_length=100,
            )
        ]
    return [
        dict(
            base,
            type="trajectory",
            finish_reason="length",
            entity="",
            entity_status="not_generated",
            reasoning_length=4096,
        ),
        dict(base, type="failure", reason="reasoning_length"),
        dict(base, type="trajectory_done", inference_complete=False),
    ]


def _bbox(
    base: dict,
    *,
    stage: str,
    arm: str,
    mode: str,
    draw: int,
    value: float,
    prefix: int | None = None,
) -> dict:
    # Full-height boxes make the requested IoU exactly the fractional width.
    width = max(1, int(round(value * 1000)))
    return dict(
        base,
        type="bbox",
        stage=stage,
        arm=arm,
        mode=mode,
        draw=draw,
        seed=draw + 1,
        prefix_coordinates=prefix,
        bbox=[0, 0, width, 1000],
        valid=True,
        iou=width / 1000.0,
        completed=True,
        text=f"0,0,{width},1000]}}</answer>",
        token_ids=[1, 2, 3],
    )


def _complete_outputs(sample_id: str, image_id: str, trajectory_index: int, *, scale: float) -> list[dict]:
    base = {
        "sample_id": sample_id,
        "image_id": image_id,
        "trajectory_index": trajectory_index,
        "task_type": "object" if sample_id != "s3" else "attribute",
    }
    rows: list[dict] = []
    # Deliberately use identical values within a draw panel: the test then
    # isolates draw -> trajectory -> image weighting from sampling variance.
    a_values = {"E": 0.8 * scale, "L": 0.2 * scale, "R": 0.4 * scale, "L0": 0.1 * scale}
    for arm in ("L0", "L", "E", "R"):
        for draw in range(4):
            rows.append(_bbox(base, stage="A", arm=arm, mode="random", draw=draw, value=a_values[arm]))
        rows.append(_bbox(base, stage="A", arm=arm, mode="greedy", draw=0, value=a_values[arm]))
    for prefix, values in (
        (1, {"E": 0.8 * scale, "L": 0.2 * scale, "R": 0.4 * scale}),
        (2, {"E": 0.6 * scale, "L": 0.3 * scale, "R": 0.5 * scale}),
    ):
        for arm in ("E", "L", "R"):
            for draw in range(4):
                rows.append(
                    _bbox(
                        base,
                        stage="B",
                        arm=arm,
                        mode="random",
                        draw=draw,
                        value=values[arm],
                        prefix=prefix,
                    )
                )
    rows.append(dict(base, type="trajectory_done", inference_complete=True))
    return rows


def _records(tmp_path: Path) -> Path:
    records = tmp_path / "run" / "records"
    records.mkdir(parents=True)
    for sample_id, image_id in (("s1", "image-1"), ("s2", "image-1"), ("s3", "image-2")):
        for trajectory_index in range(3):
            rows = _trajectory(sample_id, image_id, trajectory_index)
            if sample_id == "s3" and trajectory_index in (1, 2):
                rows = _trajectory(sample_id, image_id, trajectory_index, complete=False)
            else:
                rows += _complete_outputs(
                    sample_id,
                    image_id,
                    trajectory_index,
                    scale=1.0 if sample_id != "s3" else 0.75,
                )
            _write_jsonl(records / f"{sample_id}_t{trajectory_index}.jsonl", rows)
    # A distribution file in the run directory must never be scanned.
    distributions = records.parent / "distributions"
    distributions.mkdir()
    (distributions / "large.jsonl.gz").write_bytes(b"not a records file")
    return records


def test_image_equal_and_missing_denominator(tmp_path: Path) -> None:
    records = _records(tmp_path)
    summary = analyze(_manifest(), records, bootstrap_replicates=40)

    panel = summary["aggregates"]["A"]["random"]
    all_cohort = panel["all_cohort"]
    comparable = panel["comparable"]
    # image-1 has two samples and image-2 has one sample with one complete
    # trajectory.  The final mean is the mean of two image-level values, not a
    # raw sample/trajectory mean.
    assert all_cohort["n_images"] == 2
    assert all_cohort["image_values"]["image-1"]["E"] == pytest.approx(0.8)
    assert all_cohort["image_values"]["image-2"]["E"] == pytest.approx(0.2)
    assert all_cohort["contrasts"]["E-L"]["mean"] == pytest.approx(
        ((0.8 - 0.2) + (0.6 / 3.0 - 0.15 / 3.0)) / 2.0
    )
    assert comparable["n_images"] == 2
    assert comparable["contrasts"]["E-L"]["mean"] == pytest.approx((0.6 + 0.45) / 2.0)
    assert panel["coverage"]["missing_cells"] == 2 * 16
    assert panel["coverage"]["complete_trajectories"] == 7
    assert summary["coverage"]["partial"] is False
    assert summary["coverage"]["workflow_has_missing_or_unavailable_branches"] is True
    assert summary["reviews"]["available"] is False

    b1 = summary["aggregates"]["B"]["prefix1"]
    b2 = summary["aggregates"]["B"]["prefix2"]
    assert b1["definition"]["prefix_coordinates"] == 1
    assert b2["definition"]["prefix_coordinates"] == 2
    assert b1["all_cohort"]["contrasts"]["E-L"]["n_images"] == 2
    assert b2["comparable"]["n_images"] == 2


def test_invalid_iou_zero_and_duplicate_are_rejected(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    base = {"sample_id": "s", "image_id": "i", "trajectory_index": 0, "task_type": "object"}
    manifest = [
        {
            **base,
            "question": "q",
            "ground_truth_bbox": [0.0, 0.0, 1.0, 1.0],
        }
    ]
    rows = _trajectory("s", "i", 0)
    bad = _bbox(base, stage="A", arm="L0", mode="random", draw=0, value=0.8)
    bad["valid"] = False
    bad["bbox"] = None
    bad["iou"] = 0.1
    rows.extend([bad, dict(base, type="trajectory_done", inference_complete=False)])
    _write_jsonl(records / "s_t0.jsonl", rows)
    with pytest.raises(ValidationError, match="valid=false requires iou=0"):
        analyze(manifest, records, bootstrap_replicates=2)

    bad["iou"] = 0.0
    bad["valid"] = True
    bad["bbox"] = [0, 0, 800, 1000]
    rows = _trajectory("s", "i", 0) + [bad, dict(base, type="trajectory_done", inference_complete=False)]
    # The trajectory journal has a duplicate terminal row in addition to the
    # duplicate output below; construct a clean duplicate-output journal.
    rows = _trajectory("s", "i", 0) + [
        _bbox(base, stage="A", arm="L0", mode="random", draw=0, value=0.8),
        _bbox(base, stage="A", arm="L0", mode="random", draw=0, value=0.8),
        dict(base, type="trajectory_done", inference_complete=False),
    ]
    _write_jsonl(records / "s_t0.jsonl", rows)
    with pytest.raises(ValidationError, match="duplicate bbox output key"):
        analyze(manifest, records, bootstrap_replicates=2)


def test_prefix_unavailable_is_partial_and_not_comparable(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    base = {"sample_id": "s", "image_id": "i", "trajectory_index": 0, "task_type": "object"}
    manifest = [
        {
            "sample_id": "s",
            "image_id": "i",
            "task_type": "object",
            "question": "q",
            "ground_truth_bbox": [0.0, 0.0, 1.0, 1.0],
        }
    ]
    rows = _trajectory("s", "i", 0)
    rows += [
        _bbox(base, stage="A", arm="L", mode="random", draw=0, value=0.8),
        dict(base, type="prefix_unavailable", prefix_coordinates=1),
        dict(base, type="trajectory_done", inference_complete=True),
    ]
    _write_jsonl(records / "s_t0.jsonl", rows)
    summary = analyze(manifest, records, bootstrap_replicates=5)
    b1 = summary["aggregates"]["B"]["prefix1"]
    assert b1["coverage"]["prefix_unavailable_cells"] == 12
    assert b1["comparable"]["n_images"] == 0
    assert summary["coverage"]["partial"] is True  # two expected trajectories were never run
    assert summary["coverage"]["workflow_has_missing_or_unavailable_branches"] is True


def test_reversed_invalid_box_is_retained_and_can_source_b(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    base = {"sample_id": "s", "image_id": "i", "trajectory_index": 0, "task_type": "object"}
    manifest = [{"sample_id": "s", "image_id": "i", "task_type": "object",
                 "question": "q", "ground_truth_bbox": [0.0, 0.0, 1.0, 1.0]}]
    rows = _trajectory("s", "i", 0) + _complete_outputs("s", "i", 0, scale=1.0)
    source = next(row for row in rows if row.get("type") == "bbox" and row.get("stage") == "A"
                  and row.get("arm") == "L" and row.get("mode") == "random" and row.get("draw") == 0)
    source.update(bbox=[87, 458, 796, 105], valid=False, iou=0.0,
                  text="87,458,796,105]}</answer>", completed=True)
    _write_jsonl(records / "s_t0.jsonl", rows)
    summary = analyze(manifest, records, bootstrap_replicates=3)
    assert summary["aggregates"]["A"]["random"]["coverage"]["invalid_bbox_cells"] == 1
    assert summary["aggregates"]["B"]["prefix1"]["comparable"]["n_images"] == 1


def test_outputs_include_report_summary_and_plot(tmp_path: Path) -> None:
    summary = analyze(_manifest(), _records(tmp_path), bootstrap_replicates=3)
    output_dir = tmp_path / "analysis"
    paths = write_outputs(summary, output_dir)
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "REPORT.md").is_file()
    assert Path(paths["paired_differences_plot"]).is_file()
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "blind reviews" in report
    assert "不能声称" in report


def test_correct_unique_subset_uses_only_blind_reviewed_trajectories(tmp_path: Path) -> None:
    records = _records(tmp_path)
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(reviews, [{"sample_id": "s1", "image_id": "image-1",
                            "trajectory_index": 0, "status": "correct_unique"}])
    summary = analyze(_manifest(), records, reviews=reviews, bootstrap_replicates=5)
    subset = summary["identity_confirmed_subset"]
    assert subset["n_correct_unique_trajectories"] == 1
    population = subset["A"]["random"]["all_cohort"]
    assert population["n_images"] == 1
    assert population["contrasts"]["E-L"]["mean"] == pytest.approx(0.6)
