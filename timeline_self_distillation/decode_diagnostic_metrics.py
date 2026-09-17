"""Paired coordinate differences, including decisions to end a numeral.

All exposed token, coordinate and digit positions are one-based. These are
descriptive diagnostics, not a loss mask or a post-processing repair.
"""

from __future__ import annotations

import re
from itertools import zip_longest

from reasoning_checkpoints.extractor import decode_ids

FIELDS = ("x_min", "y_min", "x_max", "y_max")
_FOUR_VALUES = re.compile(r"\s*(\d+),(\d+),(\d+),(\d+)\]")


def first_numeric_difference(base_text: str, checkpoint_text: str) -> dict:
    """Compare raw bbox-tail fields even when their geometry is invalid.

    The text begins at the first coordinate, excluding BOX_OPEN. Field-wise
    comparison does not misalign later numbers after a numeral changes length.
    Missing a field is reported as uncomparable, never as no difference.
    """
    base, checkpoint = _FOUR_VALUES.match(base_text), _FOUR_VALUES.match(checkpoint_text)
    if base is None or checkpoint is None:
        return {
            "comparable": False,
            "different": None,
            "reason": "one_or_both_tails_lack_four_complete_coordinate_fields",
            "base_fields_complete": base is not None,
            "checkpoint_fields_complete": checkpoint is not None,
        }
    for index, (old_value, new_value) in enumerate(zip(base.groups(), checkpoint.groups(), strict=True)):
        if old_value == new_value:
            continue
        for digit, (a, b) in enumerate(zip_longest(old_value, new_value, fillvalue="<END>")):
            if a != b:
                return {
                    "comparable": True,
                    "different": True,
                    "coordinate": FIELDS[index],
                    "coordinate_index": index + 1,
                    "digit_index": digit + 1,
                    "base_value": old_value,
                    "checkpoint_value": new_value,
                    "base_char": a,
                    "checkpoint_char": b,
                    "kind": "value_length" if "<END>" in (a, b) else "digit",
                }
    return {"comparable": True, "different": False, "coordinate": None, "digit_index": None}


def first_token_divergence(base_ids, checkpoint_ids, tokenizer) -> dict:
    """Find first differing generated token under the shared-prefix coupling."""
    for index, (old_id, new_id) in enumerate(zip_longest(base_ids, checkpoint_ids)):
        if old_id == new_id:
            continue
        old_piece = "<END>" if old_id is None else decode_ids(tokenizer, [old_id])
        new_piece = "<END>" if new_id is None else decode_ids(tokenizer, [new_id])
        prefix = decode_ids(tokenizer, base_ids[:index])
        old_digit, new_digit = old_piece.isdigit(), new_piece.isdigit()
        terminators = (",", "]")
        if old_digit and new_digit:
            kind = "digit"
        elif (old_digit and new_piece.startswith(terminators)) or (new_digit and old_piece.startswith(terminators)):
            kind = "numeric_termination"
        elif old_id is None or new_id is None:
            kind = "sequence_end"
        else:
            kind = "format_or_ending"
        return {
            "different": True,
            "token_position": index + 1,
            "base_token_id": old_id,
            "checkpoint_token_id": new_id,
            "base_piece": old_piece,
            "checkpoint_piece": new_piece,
            "shared_prefix": prefix,
            "kind": kind,
        }
    return {"different": False, "token_position": None, "kind": "identical_ids"}
