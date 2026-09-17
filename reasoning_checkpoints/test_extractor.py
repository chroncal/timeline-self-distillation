from __future__ import annotations

import pytest

from reasoning_checkpoints.extractor import extract_reasoning_checkpoints, split_reasoning_close


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids: list[int], **_: object) -> str:
        return "".join(chr(token_id) for token_id in ids)


def test_split_and_extract_preserve_original_ids() -> None:
    tokenizer = CharacterTokenizer()
    generated = tokenizer.encode("First.\nSecond; done</think>")
    reasoning_ids, close_ids = split_reasoning_close(tokenizer, generated)
    checkpoints = extract_reasoning_checkpoints(
        tokenizer,
        reasoning_ids,
        sample_id="row-1",
        final_bbox=[1, 2, 3, 4],
        bbox_iou=0.6,
        bbox_hit_gt=True,
    )
    assert tokenizer.decode(reasoning_ids) == "First.\nSecond; done"
    assert tokenizer.decode(close_ids) == "</think>"
    assert checkpoints[0]["checkpoint_kind"] == "reasoning_start"
    assert checkpoints[-1]["checkpoint_kind"] == "reasoning_end"
    assert checkpoints[-1]["prefix_token_ids"] == reasoning_ids
    assert checkpoints[-1]["remaining_reasoning_text"] == ""
    assert all(record["prefix_token_ids"] == reasoning_ids[: record["token_offset"]] for record in checkpoints)


def test_punctuation_newline_run_is_one_checkpoint() -> None:
    tokenizer = CharacterTokenizer()
    ids = tokenizer.encode("One.\nTwo.")
    checkpoints = extract_reasoning_checkpoints(
        tokenizer,
        ids,
        sample_id="x",
        final_bbox=None,
        bbox_iou=None,
        bbox_hit_gt=None,
    )
    offsets = [record["token_offset"] for record in checkpoints]
    assert offsets == [0, 5, len(ids)]


def _checkpoint_deltas(checkpoints: list[dict[str, object]]) -> list[str]:
    deltas: list[str] = []
    previous = ""
    for checkpoint in checkpoints[1:]:
        prefix = str(checkpoint["prefix_text"])
        deltas.append(prefix[len(previous) :])
        previous = prefix
    return deltas


def _extract_text(text: str) -> list[dict[str, object]]:
    tokenizer = CharacterTokenizer()
    return extract_reasoning_checkpoints(
        tokenizer,
        tokenizer.encode(text),
        sample_id="fixture",
        final_bbox=None,
        bbox_iou=None,
        bbox_hit_gt=None,
    )


def test_decimal_points_are_not_sentence_boundaries() -> None:
    checkpoints = _extract_text("Use 0.15 to 0.16.\nDone.\n")
    assert _checkpoint_deltas(checkpoints) == ["Use 0.15 to 0.16.\n", "Done.\n"]


def test_numbered_structure_opener_is_merged_into_its_content() -> None:
    checkpoints = _extract_text("1. Analyze image.\n2. Locate target.\n")
    assert _checkpoint_deltas(checkpoints) == ["1. Analyze image.\n", "2. Locate target.\n"]


def test_marker_only_line_is_merged_but_short_sentences_survive() -> None:
    checkpoints = _extract_text("-\nActual text.\nNo.\nYes.\n")
    assert _checkpoint_deltas(checkpoints) == ["-\nActual text.\n", "No.\n", "Yes.\n"]


def test_ellipsis_is_not_split_into_tiny_checkpoints() -> None:
    checkpoints = _extract_text("Maybe... and then continue.\nDone.\n")
    assert _checkpoint_deltas(checkpoints) == ["Maybe... and then continue.\n", "Done.\n"]


def test_single_letter_sentence_is_not_mistaken_for_a_list_marker() -> None:
    checkpoints = _extract_text("I.\nDone.\n")
    assert _checkpoint_deltas(checkpoints) == ["I.\n", "Done.\n"]


def test_abbreviation_and_domain_periods_are_not_boundaries() -> None:
    checkpoints = _extract_text("See e.g. https://example.com/item. Continue.\n")
    assert _checkpoint_deltas(checkpoints) == [
        "See e.g. https://example.com/item. ",
        "Continue.\n",
    ]


def test_punctuation_inside_inline_code_does_not_split_code_span() -> None:
    checkpoints = _extract_text("Use `foo.bar` here.\nDone.\n")
    assert _checkpoint_deltas(checkpoints) == ["Use `foo.bar` here.\n", "Done.\n"]


def test_quoted_sentence_boundary_is_recorded_after_closing_quote() -> None:
    checkpoints = _extract_text('He said "stop here." Then left.\n')
    assert _checkpoint_deltas(checkpoints) == ['He said "stop here." ', "Then left.\n"]


def test_boundary_is_never_invented_inside_one_token() -> None:
    class WholeTextTokenizer:
        def decode(self, ids: list[int], **_: object) -> str:
            return "One. Two." if ids else ""

    checkpoints = extract_reasoning_checkpoints(
        WholeTextTokenizer(),
        [42],
        sample_id="one-token",
        final_bbox=None,
        bbox_iou=None,
        bbox_hit_gt=None,
    )
    assert [record["token_offset"] for record in checkpoints] == [0, 1]


def test_empty_reasoning_still_has_start_and_end_sentinels() -> None:
    checkpoints = _extract_text("")
    assert [record["checkpoint_kind"] for record in checkpoints] == ["reasoning_start", "reasoning_end"]
    assert [record["token_offset"] for record in checkpoints] == [0, 0]
    assert [record["reasoning_progress"] for record in checkpoints] == [0.0, 1.0]


def test_reasoning_close_must_be_unique_and_terminal() -> None:
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="single terminal"):
        split_reasoning_close(tokenizer, tokenizer.encode("First</think>Second</think>"))


def test_reasoning_close_uses_last_exact_prefix_when_decode_is_zero_width() -> None:
    class ZeroWidthTokenizer:
        texts = {
            (): "",
            (1,): "reason",
            (1, 2): "reason",
            (3,): "</think>",
            (2, 3): "</think>",
            (1, 2, 3): "reason</think>",
        }

        def decode(self, ids: list[int], **_: object) -> str:
            return self.texts[tuple(ids)]

    reasoning_ids, close_ids = split_reasoning_close(ZeroWidthTokenizer(), [1, 2, 3])
    assert reasoning_ids == [1, 2]
    assert close_ids == [3]


def test_non_monotonic_prefix_decode_is_not_used_as_text_boundary() -> None:
    class NonMonotonicTokenizer:
        texts = {(): "", (1,): "X.", (1, 2): "A.B."}

        def decode(self, ids: list[int], **_: object) -> str:
            return self.texts[tuple(ids)]

    checkpoints = extract_reasoning_checkpoints(
        NonMonotonicTokenizer(),
        [1, 2],
        sample_id="non-monotonic",
        final_bbox=None,
        bbox_iou=None,
        bbox_hit_gt=None,
    )
    assert [record["token_offset"] for record in checkpoints] == [0, 2]


def test_consecutive_terminal_punctuation_is_one_boundary() -> None:
    checkpoints = _extract_text("What?! Really!!!\n")
    assert _checkpoint_deltas(checkpoints) == ["What?! ", "Really!!!\n"]


def test_boundary_waits_for_unicode_closing_quote() -> None:
    checkpoints = _extract_text("“Stop。” Next.\n")
    assert _checkpoint_deltas(checkpoints) == ["“Stop。” ", "Next.\n"]


def test_arabic_and_indic_terminal_punctuation() -> None:
    checkpoints = _extract_text("مرحبا؟ نعم۔ फिर।\n")
    assert _checkpoint_deltas(checkpoints) == ["مرحبا؟ ", "نعم۔ ", "फिर।\n"]
