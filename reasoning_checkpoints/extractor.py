"""Extract natural prefix checkpoints without reconstructing generated tokens."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

NATURAL_ENDINGS = frozenset(".!?;。！？；؟؛۔।॥．")
THINK_CLOSE = "</think>"
_TRAILING_CLOSERS = frozenset("\"'”’)]}*_`」』»›】）》")
_COMMON_ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|fig|eq|approx)\.)$",
    re.IGNORECASE,
)
_STRUCTURE_ONLY = re.compile(
    r"^\s*(?:"
    r"[-+*•·▪◦]+|"  # unordered-list marker
    r"#{1,6}|>{1,3}|"  # Markdown heading/quote marker
    r"[-+*]\s*\[[ xX]\]|"  # task-list marker
    r"\(?\d{1,4}[.)]"  # 1. / 1) / (1)
    r")\s*$"
)


def decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode literally while retaining protocol tokens and whitespace."""

    try:
        return tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=False)


def exact_decoded_prefix_offset(tokenizer: Any, token_ids: Sequence[int], text: str) -> int | None:
    """Find the latest exact decoded prefix, retaining any zero-width ids."""

    for offset in range(len(token_ids), -1, -1):
        if decode_ids(tokenizer, token_ids[:offset]) == text:
            return offset
    return None


def split_reasoning_close(tokenizer: Any, generated_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Split the terminal ``</think>`` at an exact boundary in generated ids."""

    ids = [int(token_id) for token_id in generated_ids]
    decoded = decode_ids(tokenizer, ids)
    marker_index = decoded.rfind(THINK_CLOSE)
    if (
        decoded.count(THINK_CLOSE) != 1
        or marker_index < 0
        or decoded[marker_index + len(THINK_CLOSE) :].strip()
    ):
        raise ValueError("reasoning generation does not end with a single terminal </think>")
    content_text = decoded[:marker_index]
    offset = exact_decoded_prefix_offset(tokenizer, ids, content_text)
    if offset is None:
        raise ValueError("</think> does not begin on an exact generated-token boundary")
    close_ids = ids[offset:]
    if THINK_CLOSE not in decode_ids(tokenizer, close_ids):
        raise ValueError("terminal reasoning marker was lost at the token split")
    return ids[:offset], close_ids


def _terminal_punctuation_index(decoded_prefix: str) -> int | None:
    """Locate terminal punctuation, allowing closing quotes/markup after it."""

    index = len(decoded_prefix.rstrip()) - 1
    while index >= 0 and decoded_prefix[index] in _TRAILING_CLOSERS:
        index -= 1
    if index >= 0 and decoded_prefix[index] in NATURAL_ENDINGS:
        return index
    return None


def _is_natural_boundary(decoded_prefix: str) -> bool:
    if not decoded_prefix:
        return False
    if decoded_prefix.endswith(("\n", "\r")):
        return True
    return _terminal_punctuation_index(decoded_prefix) is not None


def _is_decimal_point(full_text: str, decoded_prefix: str) -> bool:
    """Identify a period whose immediate neighbours are both digits."""

    terminal_index = _terminal_punctuation_index(decoded_prefix)
    return (
        terminal_index is not None
        and terminal_index > 0
        and terminal_index + 1 < len(full_text)
        and full_text[terminal_index] == "."
        and full_text[terminal_index - 1].isdigit()
        and full_text[terminal_index + 1].isdigit()
    )


def _is_ellipsis_point(full_text: str, decoded_prefix: str) -> bool:
    """Identify a period inside a multi-period ellipsis run."""

    terminal_index = _terminal_punctuation_index(decoded_prefix)
    if terminal_index is None or terminal_index >= len(full_text) or full_text[terminal_index] != ".":
        return False
    previous_is_dot = terminal_index > 0 and full_text[terminal_index - 1] == "."
    next_is_dot = terminal_index + 1 < len(full_text) and full_text[terminal_index + 1] == "."
    return previous_is_dot or next_is_dot


def _is_abbreviation_or_domain_point(full_text: str, decoded_prefix: str) -> bool:
    """Reject periods with strong local evidence of abbreviation/domain use."""

    terminal_index = _terminal_punctuation_index(decoded_prefix)
    if terminal_index is None or full_text[terminal_index] != ".":
        return False
    if (
        terminal_index > 0
        and terminal_index + 1 < len(full_text)
        and full_text[terminal_index - 1].isalpha()
        and full_text[terminal_index + 1].isalpha()
    ):
        return True
    left_context = full_text[: terminal_index + 1]
    if _COMMON_ABBREVIATION.search(left_context):
        return True
    return bool(re.search(r"(?:\b[A-Za-z]\.){2,}$", left_context))


def _boundary_precedes_more_punctuation_or_closer(full_text: str, decoded_prefix: str) -> bool:
    """Defer a boundary until a punctuation run or closing delimiter ends."""

    terminal_index = _terminal_punctuation_index(decoded_prefix)
    if terminal_index is None or terminal_index + 1 >= len(full_text):
        return False
    stripped_length = len(decoded_prefix.rstrip())
    if stripped_length > terminal_index + 1:
        return False
    next_character = full_text[terminal_index + 1]
    return next_character in NATURAL_ENDINGS or next_character in _TRAILING_CLOSERS


def _boundary_is_inside_protected_span(decoded_prefix: str) -> bool:
    """Reject punctuation before a quote/code/markup/bracket span closes."""

    terminal_index = _terminal_punctuation_index(decoded_prefix)
    if terminal_index is None:
        return False
    observed = decoded_prefix.rstrip()
    if len(re.findall(r'(?<!\\)"', observed)) % 2:
        return True
    if observed.count("`") % 2 or observed.count("**") % 2 or observed.count("__") % 2:
        return True
    for opening, closing in (("“", "”"), ("「", "」"), ("『", "』"), ("«", "»"), ("‹", "›")):
        if observed.count(opening) > observed.count(closing):
            return True
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if observed.count(opening) > observed.count(closing):
            return True
    return False


def _is_structure_only(segment: str) -> bool:
    """Return whether a candidate segment is only a document/list opener."""

    return bool(_STRUCTURE_ONLY.fullmatch(segment))


def natural_boundary_offsets(tokenizer: Any, reasoning_ids: Sequence[int]) -> list[int]:
    """Return token offsets at punctuation and line/structure boundaries.

    If punctuation is immediately followed by whitespace-only tokens and a
    newline, the later boundary replaces the earlier one.  This records one
    faithful boundary for the sentence/line rather than inflating the count.
    """

    ids = [int(token_id) for token_id in reasoning_ids]
    full_text = decode_ids(tokenizer, ids)
    candidates: list[int] = []
    decoded_prefixes: dict[int, str] = {0: ""}
    for offset in range(1, len(ids) + 1):
        decoded = decode_ids(tokenizer, ids[:offset])
        decoded_prefixes[offset] = decoded
        if not full_text.startswith(decoded):
            continue
        if not _is_natural_boundary(decoded):
            continue
        if decoded.endswith(("\n", "\r")):
            pass
        elif (
            _is_decimal_point(full_text, decoded)
            or _is_ellipsis_point(full_text, decoded)
            or _is_abbreviation_or_domain_point(full_text, decoded)
            or _boundary_precedes_more_punctuation_or_closer(full_text, decoded)
            or _boundary_is_inside_protected_span(decoded)
        ):
            continue
        previous = candidates[-1] if candidates else 0
        segment = decoded[len(decoded_prefixes[previous]) :]
        # ``1.``, ``-`` and similar markers introduce the following content;
        # keeping them alone creates low-information checkpoints.  Deferring
        # the boundary merges the original marker tokens into the next natural
        # unit without ever reconstructing or moving a token.
        if _is_structure_only(segment):
            continue
        if candidates:
            previous = candidates[-1]
            between = decoded[len(decoded_prefixes[previous]) :]
            if between and not between.strip():
                candidates[-1] = offset
                continue
        candidates.append(offset)
    return candidates


def extract_reasoning_checkpoints(
    tokenizer: Any,
    reasoning_ids: Sequence[int],
    *,
    sample_id: str,
    final_bbox: Sequence[float] | None,
    bbox_iou: float | None,
    bbox_hit_gt: bool | None,
) -> list[dict[str, Any]]:
    """Create the required checkpoint records entirely from original ids."""

    ids = [int(token_id) for token_id in reasoning_ids]
    if ids:
        offsets_and_kinds = [
            (
                offset,
                "reasoning_start"
                if offset == 0
                else "reasoning_end"
                if offset == len(ids)
                else "natural_boundary",
            )
            for offset in sorted({0, len(ids), *natural_boundary_offsets(tokenizer, ids)})
        ]
    else:
        # The protocol requires both sentinels even though their offsets
        # coincide for a degenerate empty reasoning trajectory.
        offsets_and_kinds = [(0, "reasoning_start"), (0, "reasoning_end")]
    full_reasoning = decode_ids(tokenizer, ids)
    records: list[dict[str, Any]] = []
    for checkpoint_index, (offset, checkpoint_kind) in enumerate(offsets_and_kinds):
        prefix_ids = ids[:offset]
        remaining_ids = ids[offset:]
        records.append(
            {
                "sample_id": str(sample_id),
                "full_original_reasoning": full_reasoning,
                "reasoning_token_count": len(ids),
                "checkpoint_index": checkpoint_index,
                "checkpoint_kind": checkpoint_kind,
                "token_offset": offset,
                "prefix_token_ids": prefix_ids,
                "prefix_text": decode_ids(tokenizer, prefix_ids),
                "remaining_reasoning_text": decode_ids(tokenizer, remaining_ids),
                "reasoning_progress": (
                    1.0 if checkpoint_kind == "reasoning_end" else 0.0 if not ids else offset / len(ids)
                ),
                "original_final_bbox": None if final_bbox is None else [float(value) for value in final_bbox],
                "original_bbox_iou": None if bbox_iou is None else float(bbox_iou),
                "original_bbox_hit_gt": None if bbox_hit_gt is None else bool(bbox_hit_gt),
            }
        )
    return records
