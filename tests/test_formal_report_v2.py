from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmgcot_timeline_training.formal_report_v2 import (
    fixed_r_sft_examples,
    generate_report,
    main,
    validate_result,
)


SEEDS = ("11", "22")
QUEUE = [
    {"sample_id": "s1", "image_id": "i1"},
    {"sample_id": "s2", "image_id": "i2"},
    {"sample_id": "s3", "image_id": "i3"},
    {"sample_id": "s4", "image_id": "i4"},
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _system(values_by_seed: dict[str, list[float]], *, incomplete: bool = False) -> dict:
    per_seed = {}
    for seed, values in values_by_seed.items():
        per_seed[seed] = {
            "mIoU": _mean(values),
            "miou": _mean(values),
            "n_images": len(QUEUE),
            "per_image": [
                {
                    "sample_id": item["sample_id"],
                    "image_id": item["image_id"],
                    "mIoU": value,
                    "Acc@0.5": 0.5,
                }
                for item, value in zip(QUEUE, values)
            ],
            "frame_counts": {
                "expected": 16,
                "observed": 14 if incomplete and seed == "22" else 16,
                "valid_completed": 12 if incomplete and seed == "22" else 16,
            },
        }
    all_values = [value for values in values_by_seed.values() for value in values]
    return {
        "mIoU": _mean(all_values),
        "miou": _mean(all_values),
        "Acc@0.5": 0.5,
        "n_images": len(QUEUE),
        "n_seeds": len(SEEDS),
        "per_seed": per_seed,
        "by_seed": per_seed,
    }


def _pair(candidate: dict[str, list[float]], baseline: dict[str, list[float]]) -> dict:
    deltas = {
        seed: _mean(candidate[seed]) - _mean(baseline[seed])
        for seed in SEEDS
    }
    overall_delta = _mean(list(deltas.values()))
    return {
        "candidate": "r_opd",
        "baseline": "bbox_sft",
        "mIoU": {
            "delta": overall_delta,
            "candidate_minus_baseline": overall_delta,
            "ci95": [overall_delta - 0.05, overall_delta + 0.05],
            "n_images": len(QUEUE),
            "replicates": 100,
        },
        "per_seed": {
            seed: {
                "mIoU": {
                    "delta": deltas[seed],
                    "ci95": [deltas[seed] - 0.02, deltas[seed] + 0.02],
                }
            }
            for seed in SEEDS
        },
        "bootstrap_replicates": 100,
    }


def _generic_pair(candidate_name: str, baseline_name: str, delta: float) -> dict:
    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "mIoU": {
            "delta": delta,
            "ci95": [delta - 0.03, delta + 0.03],
            "n_images": len(QUEUE),
            "replicates": 100,
        },
        "per_seed": {
            seed: {"mIoU": {"delta": delta, "ci95": [delta - 0.01, delta + 0.01]}}
            for seed in SEEDS
        },
    }


def _result(cohort: str, interpretation: str, *, incomplete: bool = False) -> dict:
    sft = {
        "11": [0.20, 0.40, 0.60, 0.80],
        "22": [0.30, 0.30, 0.50, 0.70],
    }
    r = {
        "11": [0.50, 0.10, 0.90, 0.70],
        "22": [0.60, 0.20, 0.80, 0.60],
    }
    e = {
        "11": [0.30, 0.50, 0.70, 0.90],
        "22": [0.40, 0.40, 0.60, 0.80],
    }
    return {
        "schema_version": "formal_v2",
        "cohort": cohort,
        "interpretation": interpretation,
        "selection_sha256": "a" * 64,
        "queued_images": QUEUE,
        "n_images": len(QUEUE),
        "draws_per_trajectory": 4,
        "bootstrap_replicates": 100,
        "systems": {
            "bbox_sft": _system(sft, incomplete=incomplete),
            "r_opd": _system(r, incomplete=incomplete),
            "e_opd": _system(e),
        },
        "paired": {
            "r_opd-bbox_sft": _pair(r, sft),
            "e_opd-bbox_sft": _generic_pair("e_opd", "bbox_sft", 0.05),
            "r_opd-e_opd": _generic_pair("r_opd", "e_opd", -0.02),
        },
    }


def _write_inputs(root: Path) -> None:
    for cohort, interpretation in (
        ("independent48", "independent_confirmation"),
        ("test200_retest", "diagnostic_selection_conditioned_retest"),
    ):
        path = root / cohort / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_result(cohort, interpretation, incomplete=cohort == "test200_retest")),
            encoding="utf-8",
        )


def test_generate_report_writes_markdown_png_and_preserves_cohort_scope(tmp_path: Path) -> None:
    final_evaluation = tmp_path / "final_evaluation"
    _write_inputs(final_evaluation)

    outputs = generate_report(
        final_evaluation,
        tmp_path / "formal_report",
        wait_seconds=0,
    )

    report = outputs["report"].read_text(encoding="utf-8")
    assert outputs["report"].name == "formal_report_v2.md"
    assert outputs["plot"].name == "paired_image_deltas.png"
    assert outputs["plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "independent48" in report
    assert "independent confirmation" in report
    assert "test200_retest" in report
    assert "diagnostic selection-conditioned retest" in report
    assert "not a second independent confirmation" in report
    assert "Seed 11 mIoU" in report
    assert "Observed / expected" in report
    assert "Valid & completed / expected" in report
    assert "R-SFT" in report
    assert "95% CI" in report
    assert "illustration only" in report


def test_fixed_examples_use_all_images_with_deterministic_sign_rule() -> None:
    result = _result("independent48", "independent_confirmation")

    examples = fixed_r_sft_examples(result)

    assert [(row["sample_id"], row["image_id"]) for row in examples["positive"]] == [
        ("s1", "i1"),
        ("s3", "i3"),
    ]
    assert [(row["sample_id"], row["image_id"]) for row in examples["negative"]] == [
        ("s2", "i2"),
        ("s4", "i4"),
    ]
    assert examples["positive"][0]["delta"] == pytest.approx(0.3)
    assert examples["negative"][0]["delta"] == pytest.approx(-0.2)


def test_cli_fails_when_either_result_is_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="independent48/result.json"):
        main(
            [
                "--final-evaluation",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "report"),
                "--wait-seconds",
                "0",
            ]
        )


def test_validation_does_not_accept_selection_conditioned_result_as_independent() -> None:
    result = _result("independent48", "diagnostic_selection_conditioned_retest")

    with pytest.raises(ValueError, match="interpretation"):
        validate_result(result, cohort="independent48")

