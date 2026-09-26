from __future__ import annotations

from copy import deepcopy

import pytest

from mmgcot_diagnostic.protocol import BOX_OPEN
from mmgcot_timeline_training import bridge_v3
from mmgcot_timeline_training.formal_contexts import (
    ProvenanceError,
    V3P5_VERSION,
    bbox_suffix_v3p5,
    build_formal_context,
    compute_early_offset,
    inference_input,
    parse_v3p5,
    split_native_opening,
    token_ids_sha256,
    trajectory_provenance_hash,
    validate_frozen_trajectory,
)


class CharTokenizer:
    """A reversible tokenizer that makes the native final-token invariant clear."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in ids)


def _record(*, description: str = "the red car", completed: bool = True) -> dict:
    reasoning = [11, 22, 33, 44, 55, 66, 77, 88]
    bridge_text = f'red"</task_answer>\n<target_entity>"{description}"</target_entity>'
    parsed = bridge_v3.parse_bridge(bridge_text, completed=completed)
    result = {
        "sample_id": "train:object:1",
        "image_id": "image-1",
        "image_path": "/tmp/image-1.jpg",
        "image_sha256": "image-bytes-hash",
        "question": "What is the color of the car?",
        "trajectory_index": 0,
        "prompt_token_ids": [101, 102],
        "rendered_prompt": "rendered prompt",
        "reasoning_token_ids": reasoning,
        "reasoning_token_ids_sha256": token_ids_sha256(reasoning),
        "reasoning_finish": "stop",
        "bridge_version": V3P5_VERSION,
        "bridge_generation": {
            "text": bridge_text,
            "completed": completed,
            "token_ids": [201, 202],
        },
        "bridge_parse_status": parsed["bridge_parse_status"],
        "task_answer": parsed["task_answer"],
        "task_answer_status": parsed["task_answer_status"],
        "target_entity_reference": parsed["target_entity_reference"],
        "target_entity_reference_status": parsed["target_entity_reference_status"],
        "ground_truth_bbox": [0.1, 0.2, 0.4, 0.5],
    }
    result["source_record_sha256"] = trajectory_provenance_hash(result)
    return result


def test_v3p5_parser_and_suffix_are_frozen() -> None:
    text = 'blue"</task_answer>\n<target_entity>"the fence"</target_entity>'
    parsed = parse_v3p5(text, completed=True)

    assert parsed.usable
    assert parsed.task_answer == "blue"
    assert parsed.target_entity_reference == "the fence"
    assert bbox_suffix_v3p5("What color?", parsed.target_description) == bridge_v3.bbox_suffix_v3(
        "What color?", "the fence"
    )

    invalid = parse_v3p5("blue", completed=True)
    assert not invalid.usable
    assert invalid.target_entity_reference == ""


def test_context_builds_exact_l_r_e_prefixes_and_early_offset() -> None:
    record = _record()
    context = build_formal_context(record, CharTokenizer(), sentence_boundaries=[1, 3, 4])

    assert context.early_offset == 1
    assert context.L.reasoning_prefix_ids == tuple(record["reasoning_token_ids"])
    assert context.R.reasoning_prefix_ids == ()
    assert context.E.reasoning_prefix_ids == (11,)
    assert context.L.continuation_token_ids == (
        *record["reasoning_token_ids"],
        *context.suffix_body_token_ids,
    )
    assert context.R.context_token_ids == tuple(record["prompt_token_ids"]) + context.suffix_body_token_ids
    assert context.opening_token_id == ord("[")
    assert context.L.opening_prefixed_token_ids[-1] == context.opening_token_id
    assert context.provenance.reasoning_token_ids_sha256 == record["reasoning_token_ids_sha256"]


def test_early_offset_matches_protocol_fallback_rule() -> None:
    assert compute_early_offset([1, 2, 3, 4, 5], []) == 1
    assert compute_early_offset([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]) == 2


def test_missing_or_unresolved_description_uses_unconditioned_suffix() -> None:
    record = _record(description="UNRESOLVED")
    context = build_formal_context(record, CharTokenizer())
    expected = bbox_suffix_v3p5(record["question"], None)

    assert context.fallback_used
    assert not context.has_description
    assert context.suffix_text == expected
    assert "the red car" not in context.suffix_text


def test_native_opening_split_asserts_tokenizer_contract() -> None:
    suffix = bbox_suffix_v3p5("What color?", "the fence")
    split = split_native_opening(CharTokenizer(), suffix)

    assert split.suffix_token_ids == tuple(ord(character) for character in suffix)
    assert split.suffix_token_ids == split.body_token_ids + (split.opening_token_id,)
    assert split.opening_token_ids[-1] == split.opening_token_id
    body, opening = split
    assert body == split.body_token_ids
    assert opening == split.opening_token_id

    class MismatchingTokenizer(CharTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if text == BOX_OPEN:
                return [*super().encode(text)[:-1], 999]
            return super().encode(text)

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return super().decode([ord("[") if token == 999 else token for token in ids])

    with pytest.raises(AssertionError, match="final token"):
        split_native_opening(MismatchingTokenizer(), suffix)


def test_provenance_validation_rejects_drift_and_ignores_gt() -> None:
    record = _record()
    expected = {
        "sample_id": record["sample_id"],
        "image_id": record["image_id"],
        "image_sha256": record["image_sha256"],
        "question": record["question"],
        "trajectory_index": 0,
    }
    trajectory = validate_frozen_trajectory(record, expected)
    assert trajectory.inference_input() == {
        "image_path": record["image_path"],
        "question": record["question"],
    }
    assert set(inference_input(record)) == {"image_path", "question"}
    assert trajectory_provenance_hash(record) == trajectory_provenance_hash(
        {**record, "ground_truth_bbox": [0.8, 0.8, 0.9, 0.9]}
    )

    changed = deepcopy(record)
    changed["reasoning_token_ids"][0] += 1
    with pytest.raises(ProvenanceError, match="reasoning token hash"):
        validate_frozen_trajectory(changed)

    changed = deepcopy(record)
    changed["question"] = "a changed question"
    with pytest.raises(ProvenanceError, match="provenance hash"):
        validate_frozen_trajectory(changed)

    wrong_expected = {**expected, "image_id": "different-image"}
    with pytest.raises(ProvenanceError, match="image_id"):
        validate_frozen_trajectory(record, wrong_expected)
