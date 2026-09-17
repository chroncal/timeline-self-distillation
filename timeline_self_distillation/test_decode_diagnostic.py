"""CPU contract tests for the bounded checkpoint decode diagnostic."""

from __future__ import annotations

import math

import pytest

import timeline_self_distillation.run_decode_diagnostic as diagnostic


def _record(
    sample_id: str,
    sample_order: int,
    model: str,
    mode: str,
    draw: int,
    iou: float,
    *,
    invalid: bool = False,
) -> dict:
    return {
        "sample_id": sample_id,
        "sample_order": sample_order,
        "model": model,
        "mode": mode,
        "draw": draw,
        "seed": diagnostic.micro_seed(diagnostic.SEED, sample_order, draw),
        "ids": [1, 2] if not invalid else [],
        "text": "1,2,3,4]}</answer>" if not invalid else "",
        "pieces": ["1", ","],
        "parse_valid": not invalid,
        "numeric_valid": not invalid,
        "iou": 0.0 if invalid else iou,
        "acc_05": bool(iou >= 0.5) and not invalid,
    }


def test_micro_seed_uses_fixed_record_order_not_row_number():
    assert diagnostic.micro_seed(diagnostic.SEED, 2, 3) == diagnostic.SEED + 9_000_000 + 2_000 + 3
    assert diagnostic.micro_seed(diagnostic.SEED, 2, 3) != diagnostic.micro_seed(diagnostic.SEED, 3, 3)


def test_score_bbox_invalid_is_zero_without_nonfinite_result():
    result = diagnostic.score_bbox({"text": "not a legal bbox", "parse_valid": False}, [0, 0, 10, 10])

    assert result["parse_valid"] is False
    assert result["bbox"] is None
    assert result["iou"] == 0.0
    assert result["acc_05"] is False
    assert math.isfinite(result["iou"])


def test_summary_keeps_all_five_samples_and_uses_random_80_as_primary():
    records = []
    sample_ids = ("row-0", "row-1", "row-3", "row-4", "row-5")
    for sample_order, sample_id in enumerate(sample_ids):
        for model in diagnostic.MODELS:
            records.append(_record(sample_id, sample_order, model, "greedy", 0, 0.1))
            for draw in range(diagnostic.RANDOM_DRAWS):
                records.append(
                    _record(
                        sample_id,
                        sample_order,
                        model,
                        "random",
                        draw,
                        0.8 if draw == 0 else 0.2,
                        invalid=(sample_order == 4 and draw == 15),
                    )
                )

    summary = diagnostic.summarize_outputs(records)

    assert summary["sample_count"] == 5
    assert summary["total_count"] == 255
    for model in diagnostic.MODELS:
        metrics = summary["models"][model]
        assert metrics["random_count"] == 80
        assert metrics["greedy_count"] == 5
        assert metrics["random"]["invalid_count"] == 1
        assert metrics["random"]["sample_count"] == 5
        assert metrics["random"]["draw_count"] == 80
        assert metrics["random"]["mean_iou"] == pytest.approx(0.235)
        assert set(metrics["random"]["per_sample"]) == set(sample_ids)
        assert metrics["primary_metric"] == "random_draws_mean_iou"


def test_summary_rejects_missing_sample_or_nonfinite_iou():
    with pytest.raises(ValueError, match="expected 255"):
        diagnostic.summarize_outputs([])

    record = _record("row-0", 0, "base", "random", 0, float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        diagnostic.summarize_outputs([record])


class _Tokenizer:
    def decode(self, ids, **_kwargs):
        return "".join({1: "4", 2: ",", 3: "7"}[int(value)] for value in ids)


def test_pairs_use_same_seed_and_attach_both_metric_dicts():
    base = _record("row-0", 0, "base", "random", 0, 0.1)
    reverse = _record("row-0", 0, "reverse", "random", 0, 0.2)

    class _Metrics:
        @staticmethod
        def first_numeric_difference(base_text, checkpoint_text):
            return {"different": base_text != checkpoint_text}

        @staticmethod
        def first_token_divergence(base_ids, checkpoint_ids, tokenizer):
            return {"different": base_ids != checkpoint_ids, "tokenizer_type": type(tokenizer).__name__}

    pairs = diagnostic.build_pairs([base], [reverse], _Tokenizer(), metrics=_Metrics)

    assert len(pairs) == 1
    assert pairs[0]["seed"] == base["seed"] == reverse["seed"]
    assert pairs[0]["checkpoint_model"] == "reverse"
    assert pairs[0]["first_numeric_difference"] == {"different": False}
    assert pairs[0]["first_token_divergence"]["different"] is False


def test_pairs_reject_seed_or_prefix_mismatch():
    base = _record("row-0", 0, "base", "random", 0, 0.1)
    reverse = _record("row-0", 0, "reverse", "random", 0, 0.2)
    reverse["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        diagnostic.build_pairs([base], [reverse], _Tokenizer(), metrics=object())
