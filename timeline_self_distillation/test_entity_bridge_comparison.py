"""CPU contract tests for the frozen entity-bridge comparison runner."""

from __future__ import annotations

import pytest

import timeline_self_distillation.run_entity_bridge_comparison as comparison


def _pilot_row(sample_id: str, seeds: list[int], *, old_entity: str = "saved target") -> dict:
    return {
        "sample_id": sample_id,
        "entity": old_entity,
        "conditions": {
            "early_span1": {"draws": [{"seed": seed} for seed in seeds]},
        },
    }


def _synthetic_record(sample_id: str, old_mean: float, new_mean: float, new_failed: bool) -> dict:
    conditions = {}
    for arm in comparison.ARM_NAMES:
        value = new_mean if arm.startswith("new__") else old_mean
        failed = bool(new_failed and arm.startswith("new__"))
        conditions[arm] = {
            "entity": "" if failed else "target",
            "entity_failed": failed,
            "entity_audit": {"usable_for_probe": not failed},
            "draws": [
                {
                    "draw": draw,
                    "seed": 100 + draw,
                    "token_ids": [1, 2],
                    "bbox": None if failed else [0, 0, 1, 1],
                    "parse_valid": not failed,
                    "numeric_valid": not failed,
                    "iou": 0.0 if failed else value,
                    "hit_05": bool(value >= 0.5 and not failed),
                }
                for draw in range(4)
            ],
        }
    return {"sample_id": sample_id, "conditions": conditions}


def test_draw_seed_mapping_uses_pilot_record_seeds_not_micro_eval_formula():
    row = _pilot_row("row-3", [7101, 7107, 7113, 7119])

    assert comparison.pilot_draw_seeds(row) == [7101, 7107, 7113, 7119]
    assert comparison.pilot_draw_seed(row, 2) == 7113
    assert comparison.pilot_draw_seed(row, 2) != comparison.micro_eval_seed(260600564, 3, 2)


def test_failed_new_bridge_is_zero_scored_without_dropping_sample():
    rows = [
        _synthetic_record("row-0", old_mean=0.4, new_mean=0.8, new_failed=False),
        _synthetic_record("row-1", old_mean=0.2, new_mean=0.0, new_failed=True),
    ]

    summary = comparison.summarize_records(rows)

    assert summary["sample_count"] == 2
    assert summary["draw_count"] == 2 * 6 * 4
    assert summary["arms"]["new__late_entity"]["parse_valid"] == 4
    assert summary["arms"]["new__late_entity"]["mean_iou"] == pytest.approx(0.4)
    assert [row["sample_id"] for row in summary["per_sample"]] == ["row-0", "row-1"]


def test_summary_uses_all_samples_and_draws_without_best_draw_selection():
    rows = [
        _synthetic_record("row-0", old_mean=0.2, new_mean=0.4, new_failed=False),
        _synthetic_record("row-1", old_mean=0.6, new_mean=0.8, new_failed=False),
    ]

    summary = comparison.summarize_records(rows)

    assert summary["arms"]["old__late_entity"]["mean_iou"] == pytest.approx(0.4)
    assert summary["arms"]["new__late_entity"]["mean_iou"] == pytest.approx(0.6)
    assert summary["arms"]["new__late_entity"]["draw_count"] == 8
    assert len(summary["per_sample"]) == 2


def test_new_bridge_entity_is_used_only_by_new_arms_old_arms_keep_saved_entity():
    row = _pilot_row("row-0", [1, 2, 3, 4], old_entity="pilot entity")
    bridge = {
        "entity": "instance bridge",
        "entity_result": {"text": 'instance bridge"'},
        "entity_audit": {"usable_for_probe": True},
    }

    arms = comparison.entity_arm_inputs(row, bridge)

    assert arms["old__late_entity"]["entity"] == "pilot entity"
    assert arms["old__early_step0"]["entity"] == "pilot entity"
    assert arms["old__early_span1"]["entity"] == "pilot entity"
    assert arms["new__late_entity"]["entity"] == "instance bridge"
    assert arms["new__early_step0"]["entity"] == "instance bridge"
    assert arms["new__early_span1"]["entity"] == "instance bridge"


def test_new_bridge_failure_keeps_entity_empty_and_marks_failure():
    row = _pilot_row("row-0", [1, 2, 3, 4])
    bridge = {"entity": "", "entity_result": {}, "entity_audit": {"status": "failed"}}

    arms = comparison.entity_arm_inputs(row, bridge)

    assert all(arms[name]["entity"] == "" for name in comparison.NEW_ARMS)
    assert all(arms[name]["entity_failed"] for name in comparison.NEW_ARMS)
    assert all(not arms[name]["entity_failed"] for name in comparison.OLD_ARMS)
