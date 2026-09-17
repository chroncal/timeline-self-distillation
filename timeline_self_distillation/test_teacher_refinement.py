"""CPU tests for the second-round teacher refinement planning helpers.

The runner's model/cache path is intentionally not exercised here.  These
tests cover the deterministic text-to-token contract that decides which
reasoning values are scrubbed and where the pre-coordinate cache is anchored.
"""

from __future__ import annotations

import pytest

from timeline_self_distillation.run_teacher_refinement import (
    pre_coordinate_anchor,
    scrub_and_sham,
    source_index_from_sample_id,
)


class _CharTokenizer:
    """One-character tokenizer with an explicit one-token ``?`` placeholder."""

    def __init__(self, text: str) -> None:
        self._pieces = list(text)
        self._pieces.append("?")
        self._placeholder_id = len(self._pieces) - 1

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._pieces[int(token_id)] for token_id in ids)

    def encode(self, text: str, *, add_special_tokens=False):
        del add_special_tokens
        if text != "?":
            raise AssertionError(f"fixture only encodes the placeholder, got {text!r}")
        return [self._placeholder_id]


def _tokenized(text: str) -> tuple[_CharTokenizer, list[int]]:
    tokenizer = _CharTokenizer(text)
    return tokenizer, list(range(len(text)))


def test_extended_coordinate_values_mask_only_value_digits_and_preserve_count() -> None:
    text = "Reason. x1:~470, y=300. Box [850, 100, 900, 500]. 1. list item."
    tokenizer, ids = _tokenized(text)

    scrubbed, sham, info = scrub_and_sham(tokenizer, ids)
    coordinate_positions = set(info["coordinate_token_positions"])
    changed_positions = {
        index for index, (old, new) in enumerate(zip(ids, scrubbed, strict=True)) if old != new
    }

    assert changed_positions == coordinate_positions
    assert len(scrubbed) == len(ids) == len(sham)
    assert info["masked_token_count"] == len(coordinate_positions)
    assert len(info["sham_token_positions"]) == len(coordinate_positions)
    assert all(index not in coordinate_positions for index in info["sham_token_positions"])
    assert all(not tokenizer.decode([ids[index]]).strip().isdigit() for index in info["sham_token_positions"])

    # The ``1`` in x1 and the numbered-list ``1.`` are not coordinate values.
    assert scrubbed[text.index("x1") + 1] == ids[text.index("x1") + 1]
    list_number = text.index("1. list")
    assert scrubbed[list_number] == ids[list_number]


def test_no_coordinate_values_is_a_late_noop() -> None:
    tokenizer, ids = _tokenized("Reason. x1 is a variable; item 1. is a list entry.")

    scrubbed, sham, info = scrub_and_sham(tokenizer, ids)

    assert scrubbed == ids
    assert sham == ids
    assert info["coordinate_token_positions"] == []
    assert info["sham_token_positions"] == []
    assert info["pre_coordinate_offset"] is None
    assert info["anchor_kind"] == "late_no_coordinate"


def test_pre_coordinate_anchor_is_deterministic_and_uses_last_prior_natural_boundary() -> None:
    tokenizer, ids = _tokenized("Reasoning ends. x=850 and continues.")

    first = pre_coordinate_anchor(tokenizer, ids)
    second = pre_coordinate_anchor(tokenizer, ids)

    assert first == second
    assert first["first_coordinate_token_position"] == "Reasoning ends. x=850 and continues.".index("850")
    # The extractor extends a punctuation boundary through its following
    # whitespace, so this is the final token offset before ``850``.
    assert first["pre_coordinate_offset"] == first["natural_boundary_offsets"][0] == 16
    assert first["anchor_kind"] == "last_prior_natural_boundary"


def test_pre_coordinate_without_prior_boundary_uses_step_zero() -> None:
    tokenizer, ids = _tokenized("x=850 after no complete sentence")

    info = pre_coordinate_anchor(tokenizer, ids)

    assert info["pre_coordinate_offset"] == 0
    assert info["anchor_kind"] == "step0_no_prior_natural_boundary"


def test_source_index_is_extracted_from_fixed_row_id() -> None:
    assert source_index_from_sample_id("row-17") == 17
    with pytest.raises(ValueError, match="row-N"):
        source_index_from_sample_id("sample-17")
