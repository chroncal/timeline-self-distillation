from __future__ import annotations

import json

import pytest

from mmgcot_timeline_training.formal_analysis import (
    aggregate_formal_v2,
    aggregate_records,
    assert_comparable_keys,
    paired_image_bootstrap,
    read_bbox_jsonl,
    select_lambda,
    select_learning_rate,
)


def _frame(
    sample_id: str,
    image_id: str,
    trajectory_index: int,
    draw: int,
    iou: float | None,
    *,
    arm: str = "A",
    seed: int = 7,
    valid: bool = True,
    completed: bool = True,
    mode: str = "sample",
) -> dict:
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "trajectory_index": trajectory_index,
        "arm": arm,
        "seed": seed,
        "mode": mode,
        "draw": draw,
        "iou": iou,
        "valid": valid,
        "completed": completed,
    }


def _system_rows(arm: str, values: dict[tuple[str, int, int], list[float | None]]) -> list[dict]:
    rows = []
    for (sample_id, trajectory, _unused), draws in values.items():
        image_id = {"s1": "i1", "s2": "i2"}[sample_id]
        for draw, value in enumerate(draws):
            rows.append(
                _frame(
                    sample_id,
                    image_id,
                    trajectory,
                    draw,
                    value,
                    arm=arm,
                    valid=value is not None,
                    completed=value is not None,
                )
            )
    return rows


def test_jsonl_aggregation_uses_image_equal_hierarchy_and_strict_accuracy(tmp_path) -> None:
    rows = []
    # i1: trajectory means .50 and .60, hence image mIoU .55.  The invalid
    # fourth draw in trajectory 1 contributes zero, rather than being dropped.
    rows.extend(
        _system_rows(
            "A",
            {
                ("s1", 0, 0): [0.4, 0.6, 0.4, 0.6],
                ("s1", 1, 0): [0.8, 0.8, 0.8, None],
                ("s2", 0, 0): [0.5, 0.5, 0.5, 0.5],
                ("s2", 1, 0): [0.5, 0.5, 0.5, 0.5],
            },
        )
    )
    # The evaluator writes four stochastic frames as mode="sample".  An
    # unrelated greedy record must not enter this panel's denominator.
    rows.append(_frame("s1", "i1", 0, 0, 1.0, mode="greedy"))
    path = tmp_path / "bbox.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = aggregate_records(
        read_bbox_jsonl(path),
        queued_images=[("s1", "i1"), ("s2", "i2"), ("s3", "i3")],
        mode="sample",
    )

    # (.55 + .50 + .00) / 3; the queued, completely absent i3 remains in the
    # denominator.  For Acc@0.5, i1 has five passing boxes out of eight,
    # whereas i2's boxes are exactly .5 and do not pass.  Thresholding i1's
    # mean IoU instead would incorrectly produce 1/3.
    assert report["mIoU"] == pytest.approx(1.05 / 3)
    assert report["Acc@0.5"] == pytest.approx(5 / 24)
    assert report["mode"] == "sample"
    assert report["per_seed"][7]["per_image"][0]["mIoU"] == pytest.approx(0.55)
    assert report["per_seed"][7]["per_image"][0]["Acc@0.5"] == pytest.approx(5 / 8)
    assert report["per_seed"][7]["frame_counts"] == {
        "expected": 24,
        "observed": 16,
        "valid_completed": 15,
    }


def test_comparable_systems_report_seed_deltas_and_ten_thousand_image_bootstrap() -> None:
    baseline = _system_rows(
        "SFT",
        {
            ("s1", 0, 0): [0.4, 0.4, 0.4, 0.4],
            ("s1", 1, 0): [0.4, 0.4, 0.4, 0.4],
            ("s2", 0, 0): [0.8, 0.8, 0.8, 0.8],
            ("s2", 1, 0): [0.8, 0.8, 0.8, 0.8],
        },
    )
    candidate = _system_rows(
        "R",
        {
            ("s1", 0, 0): [0.5, 0.5, 0.5, 0.5],
            ("s1", 1, 0): [0.5, 0.5, 0.5, 0.5],
            ("s2", 0, 0): [0.9, 0.9, 0.9, 0.9],
            ("s2", 1, 0): [0.9, 0.9, 0.9, 0.9],
        },
    )
    report = aggregate_formal_v2(
        baseline + candidate,
        queued_images=[("s1", "i1"), ("s2", "i2")],
    )

    pair = report["paired"]["R-SFT"]
    assert pair["per_seed"][7]["delta"] == pytest.approx(0.1)
    assert pair["delta_mIoU"] == pytest.approx(0.1)
    assert pair["mIoU"]["replicates"] == 10_000
    assert pair["mIoU"]["n_images"] == 2
    assert pair["mIoU"]["ci95"] == pytest.approx([0.1, 0.1])
    assert report["systems"]["SFT"]["Acc@0.5"] == pytest.approx(0.5)
    assert report["systems"]["R"]["Acc@0.5"] == pytest.approx(0.5)


def test_paired_accuracy_uses_box_hits_even_when_image_mean_is_below_threshold() -> None:
    baseline = [_frame("s", "i", 0, draw, 0.49, arm="SFT") for draw in range(4)]
    candidate = [
        _frame("s", "i", 0, draw, iou, arm="R")
        for draw, iou in enumerate([0.9, 0.3, 0.3, 0.3])
    ]

    report = aggregate_formal_v2(baseline + candidate, [("s", "i")], mode="sample")

    assert report["systems"]["R"]["mIoU"] == pytest.approx(0.45)
    assert report["systems"]["R"]["Acc@0.5"] == pytest.approx(0.25)
    assert report["paired"]["R-SFT"]["per_seed"][7]["delta_Acc@0.5"] == pytest.approx(0.25)
    assert report["paired"]["R-SFT"]["Acc@0.5"]["ci95"] == pytest.approx([0.25, 0.25])


def test_key_mismatch_is_rejected_before_pairing() -> None:
    left = [_frame("s", "i", 0, draw, 0.4, arm="left") for draw in range(4)]
    right = [_frame("s", "i", 0, draw, 0.4, arm="right") for draw in range(3)]
    with pytest.raises(AssertionError, match="comparable keys differ"):
        assert_comparable_keys({"left": left, "right": right})


def test_image_bootstrap_preserves_image_units() -> None:
    baseline = {"i1": 0.0, "i2": 1.0}
    candidate = {"i1": 0.5, "i2": 1.0}
    result = paired_image_bootstrap(baseline, candidate, seed=19)
    assert result["delta"] == pytest.approx(0.25)
    assert result["n_images"] == 2
    assert result["replicates"] == 10_000

    # The same image-level scores yield the same paired distribution no matter
    # how those scores would have been assembled from draws or trajectories.
    assert result == paired_image_bootstrap(baseline, candidate, seed=19)


def test_calibration_selectors_apply_strict_point_zero_zero_two_ties() -> None:
    # The 0.0001 LR is within, but the 0.001 LR is exactly at, the strict gap.
    assert select_learning_rate({1e-4: 0.800, 3e-4: 0.799, 1e-3: 0.798}) == pytest.approx(1e-4)
    assert select_learning_rate({1e-4: 0.800, 3e-4: 0.798}) == pytest.approx(1e-4)

    # R/E are averaged before selection: .80 and .79 average .795, while
    # .799/.789 average .794, which is within .002 and therefore picks .1.
    assert select_lambda(
        {
            0.1: {"R": 0.800, "E": 0.790},
            0.3: {"R": 0.799, "E": 0.789},
            1.0: {"R": 0.700, "E": 0.700},
        }
    ) == pytest.approx(0.1)
    assert select_lambda(
        {
            0.1: {"R": 0.800, "E": 0.790},
            0.3: {"R": 0.798, "E": 0.788},
        }
    ) == pytest.approx(0.1)
