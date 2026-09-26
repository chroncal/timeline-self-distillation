"""Bounded, grammar-aware losses for the MM-GCoT v2 bbox continuation.

The terminal answer is a constrained text continuation.  A coordinate can be
split across several tokenizer pieces (and a tokenizer piece can contain
punctuation), so the loss code works with the raw response token sequence and
the legal vocabulary support recorded before each token.  It never derives a
loss mask from a presumed four-coordinate/four-token layout.

The public loss functions accept logits from the caller's chosen prefix
replay.  In particular, ``sft_numeric_nll`` expects logits obtained with the
ground-truth prefix, while ``reverse_kl_full_support`` expects logits obtained
with the student's sampled prefix.  Both normalize each example over its own
numeric token positions and then use a fixed effective-batch denominator.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "aggregate_per_example",
    "batch_reverse_kl_full_support",
    "batch_sft_numeric_nll",
    "build_grammar_supports",
    "extract_grammar_supports",
    "extract_legal_support",
    "gt_prefix_sft_loss",
    "legal_support_logits",
    "numeric_char_spans",
    "numeric_token_mask",
    "numeric_token_positions",
    "numeric_token_spans",
    "quantize_bbox",
    "reverse_kl_full_support",
    "sft_legal_support_numeric_nll",
    "sft_numeric_nll",
    "student_prefix_reverse_kl",
    "token_char_spans",
]


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _require_real_logits(logits: torch.Tensor, name: str) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not logits.is_floating_point() or logits.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if logits.ndim < 1 or logits.shape[-1] <= 0:
        raise ValueError(f"{name} must have a non-empty vocabulary dimension")
    if logits.numel() == 0 or not torch.isfinite(logits.detach()).all().item():
        raise FloatingPointError(f"{name} contains nonfinite logits")


def _probability_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return torch.promote_types(dtype, torch.float32)


def _finite_output(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError(f"{name} must produce a scalar tensor")
    if not torch.isfinite(value.detach()).all().item():
        raise FloatingPointError(f"nonfinite {name}")
    return value.reshape(())


def _effective_batch_size(value: int | None, default: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("effective_batch_size must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError("effective_batch_size must be a positive integer")
    return value


def _raw_token_ids(token_ids: Sequence[int] | torch.Tensor, *, allow_empty: bool = True) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError("token IDs must be one-dimensional")
        if token_ids.dtype == torch.bool or token_ids.is_floating_point() or token_ids.is_complex():
            raise TypeError("token IDs must use an integer dtype")
        values = [int(value) for value in token_ids.detach().cpu().tolist()]
    else:
        try:
            values = list(token_ids)
        except TypeError as exc:
            raise TypeError("token IDs must be a one-dimensional integer sequence") from exc
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("token IDs must contain integers")
        values = [int(value) for value in values]
    if not allow_empty and not values:
        raise ValueError("token IDs must be non-empty")
    return values


def _support_ids(
    support_ids: Sequence[int] | torch.Tensor,
    *,
    vocab_size: int,
    device: torch.device,
    allow_empty: bool,
    name: str = "support_ids",
) -> torch.Tensor:
    if isinstance(support_ids, torch.Tensor):
        if support_ids.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if support_ids.dtype == torch.bool or support_ids.is_floating_point() or support_ids.is_complex():
            raise TypeError(f"{name} must contain integer token IDs")
        values = support_ids.to(device=device, dtype=torch.long)
    else:
        try:
            raw = list(support_ids)
        except TypeError as exc:
            raise TypeError(f"{name} must be a one-dimensional integer sequence") from exc
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
            raise TypeError(f"{name} must contain integer token IDs")
        values = torch.as_tensor([int(value) for value in raw], dtype=torch.long, device=device)
    if values.numel() == 0 and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if values.numel() and (int(values.min().item()) < 0 or int(values.max().item()) >= vocab_size):
        raise ValueError(f"{name} contains an ID outside [0, {vocab_size})")
    if values.numel() != torch.unique(values).numel():
        raise ValueError(f"{name} contains duplicate token IDs")
    return values


def _token_char_spans_from_pieces(pieces: Sequence[str]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        if not isinstance(piece, str):
            raise TypeError("decoded token pieces must be strings")
        stop = cursor + len(piece)
        spans.append((cursor, stop))
        cursor = stop
    return tuple(spans)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def token_char_spans(
    response_token_ids: Sequence[int] | torch.Tensor,
    tokenizer: Any,
) -> tuple[tuple[int, int], ...]:
    """Return exact character spans for raw response IDs.

    The full tokenizer decode must equal the concatenation of one-token
    decodes.  If a tokenizer cannot provide that additive mapping, silently
    inventing token offsets would change the supervised positions, so the
    function fails closed.
    """

    ids = _raw_token_ids(response_token_ids)
    pieces = [_decode(tokenizer, [token_id]) for token_id in ids]
    full = _decode(tokenizer, ids)
    if "".join(pieces) != full:
        raise ValueError("response token offsets are not additive")
    return _token_char_spans_from_pieces(pieces)


def numeric_char_spans(response_text: str, *, numeric_end: int | None = None) -> tuple[tuple[int, int], ...]:
    """Return half-open character spans occupied by bbox digits.

    By default scanning stops at the first closing bracket, matching the
    ``BOX_REGEX`` continuation in :mod:`mmgcot_diagnostic.protocol` and
    preventing a later unrelated number from entering the loss mask.
    """

    if not isinstance(response_text, str):
        raise TypeError("response_text must be a string")
    if numeric_end is None:
        numeric_end = response_text.find("]")
        if numeric_end < 0:
            numeric_end = len(response_text)
    if isinstance(numeric_end, bool) or not isinstance(numeric_end, Integral):
        raise TypeError("numeric_end must be an integer")
    numeric_end = int(numeric_end)
    if not 0 <= numeric_end <= len(response_text):
        raise ValueError("numeric_end is outside response_text")
    return tuple((match.start(), match.end()) for match in re.finditer(r"[0-9]+", response_text[:numeric_end]))


def numeric_token_mask(
    response_token_ids: Sequence[int] | torch.Tensor,
    tokenizer: Any | None = None,
    *,
    token_spans: Sequence[tuple[int, int]] | None = None,
    numeric_spans: Sequence[tuple[int, int]] | None = None,
    response_text: str | None = None,
) -> list[bool]:
    """Mark raw response tokens whose character spans overlap bbox digits.

    ``numeric_spans`` and ``token_spans`` use half-open character offsets.  If
    they are omitted, the spans are derived from the exact decoded response.
    A token such as ``"12,"`` is therefore marked numeric because its span
    overlaps digits, while the implementation still scores the single raw
    token exactly once.  No coordinate-to-token alignment is assumed.
    """

    ids = _raw_token_ids(response_token_ids)
    if response_text is None:
        if tokenizer is None:
            raise TypeError("tokenizer or response_text is required")
        response_text = _decode(tokenizer, ids)
    elif not isinstance(response_text, str):
        raise TypeError("response_text must be a string")

    if tokenizer is not None:
        decoded = _decode(tokenizer, ids)
        if decoded != response_text:
            raise ValueError("response_text does not match raw response token IDs")
    if token_spans is None:
        if tokenizer is None:
            raise TypeError("tokenizer is required when token_spans is omitted")
        token_spans = token_char_spans(ids, tokenizer)
    else:
        token_spans = tuple(tuple(span) for span in token_spans)
        if len(token_spans) != len(ids):
            raise ValueError("token_spans must have one span per raw response token")
    if numeric_spans is None:
        numeric_spans = numeric_char_spans(response_text)
    else:
        numeric_spans = tuple(tuple(span) for span in numeric_spans)

    def _validate_span(span: tuple[int, int], name: str) -> tuple[int, int]:
        if len(span) != 2 or any(isinstance(value, bool) or not isinstance(value, Integral) for value in span):
            raise TypeError(f"{name} must contain integer (start, stop) spans")
        start, stop = (int(value) for value in span)
        if not 0 <= start <= stop <= len(response_text):
            raise ValueError(f"{name} contains an invalid character span")
        return start, stop

    token_spans = tuple(_validate_span(span, "token_spans") for span in token_spans)
    numeric_spans = tuple(_validate_span(span, "numeric_spans") for span in numeric_spans)
    return [
        any(max(token_start, number_start) < min(token_stop, number_stop)
            for number_start, number_stop in numeric_spans)
        for token_start, token_stop in token_spans
    ]


def numeric_token_positions(
    response_token_ids: Sequence[int] | torch.Tensor,
    tokenizer: Any | None = None,
    **kwargs: Any,
) -> tuple[int, ...]:
    """Return the raw token positions selected by :func:`numeric_token_mask`."""

    return tuple(index for index, selected in enumerate(numeric_token_mask(response_token_ids, tokenizer, **kwargs)) if selected)


def numeric_token_spans(
    response_token_ids: Sequence[int] | torch.Tensor,
    tokenizer: Any,
    **kwargs: Any,
) -> tuple[tuple[int, int], ...]:
    """Return token character spans selected as numeric bbox spans."""

    ids = _raw_token_ids(response_token_ids)
    spans = token_char_spans(ids, tokenizer)
    mask = numeric_token_mask(ids, tokenizer, token_spans=spans, **kwargs)
    return tuple(span for span, selected in zip(spans, mask, strict=True) if selected)


def extract_grammar_supports(
    grammar: Any,
    vocab_size: int,
    raw_response_token_ids: Sequence[int] | torch.Tensor,
    *,
    xgrammar_module: Any | None = None,
    device: torch.device | str = "cpu",
) -> list[list[int]]:
    """Replay a grammar and record legal IDs before each raw response token.

    The dependency on ``xgrammar`` is lazy so the loss module remains usable in
    CPU-only unit tests.  Passing ``xgrammar_module`` also makes the exact
    support extraction testable with a small grammar double.
    """

    ids = _raw_token_ids(raw_response_token_ids, allow_empty=False)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral) or int(vocab_size) <= 0:
        raise ValueError("vocab_size must be a positive integer")
    vocab_size = int(vocab_size)
    if xgrammar_module is None:
        try:
            import xgrammar as xgrammar_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("xgrammar is required to extract grammar supports") from exc

    matcher = xgrammar_module.GrammarMatcher(grammar, terminate_without_stop_token=True)
    bitmask = xgrammar_module.allocate_token_bitmask(1, vocab_size)
    scores = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
    supports: list[list[int]] = []
    for position, token_id in enumerate(ids):
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"raw response token ID {token_id} is outside the vocabulary")
        scores.zero_()
        xgrammar_module.reset_token_bitmask(bitmask)
        if matcher.fill_next_token_bitmask(bitmask):
            xgrammar_module.apply_token_bitmask_inplace(
                scores, bitmask.to(scores.device), vocab_size=vocab_size
            )
        support = torch.isfinite(scores[0]).nonzero(as_tuple=False).flatten().tolist()
        if not support:
            raise RuntimeError(f"grammar produced an empty support at response position {position}")
        if token_id not in support:
            raise RuntimeError(f"raw response token {token_id} is outside grammar support at position {position}")
        supports.append([int(value) for value in support])
        if not matcher.accept_token(token_id):
            raise RuntimeError(f"grammar rejected raw response token at position {position}")
        if matcher.is_completed() and position != len(ids) - 1:
            raise RuntimeError("raw response has tokens after grammar completion")
    if not matcher.is_completed():
        raise RuntimeError("raw response did not complete the grammar")
    return supports


# The name used by the existing timeline diagnostic is useful to callers that
# want the same replay contract without importing the diagnostic runner.
build_grammar_supports = extract_grammar_supports


def quantize_bbox(gt_xyxy: Sequence[float]) -> tuple[int, int, int, int]:
    """Quantize normalized ``xyxy`` GT to grammar integers using half-up.

    Coordinates are required to lie in ``[0, 1]`` and describe a positive
    extent.  ``floor(1000*x + 0.5)`` is used deliberately because Python's
    ``round`` implements ties-to-even.
    """

    try:
        values = list(gt_xyxy)
    except TypeError as exc:
        raise TypeError("gt_xyxy must be a four-value sequence") from exc
    if len(values) != 4:
        raise ValueError(f"gt_xyxy must contain four values, got {values!r}")
    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"gt_xyxy contains a nonnumeric value: {value!r}") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"gt_xyxy contains a nonfinite or out-of-range value: {value!r}")
        converted.append(number)
    if converted[0] >= converted[2] or converted[1] >= converted[3]:
        raise ValueError(f"gt_xyxy must have positive extent: {converted!r}")
    quantized = tuple(int(math.floor(1000.0 * value + 0.5)) for value in converted)
    if quantized[0] >= quantized[2] or quantized[1] >= quantized[3]:
        raise ValueError(f"quantized gt_xyxy has non-positive extent: {quantized!r}")
    return quantized


def legal_support_logits(
    logits: torch.Tensor,
    support_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Select one row's legal grammar support without changing its graph."""

    _require_real_logits(logits, "logits")
    support = _support_ids(
        support_ids,
        vocab_size=logits.shape[-1],
        device=logits.device,
        allow_empty=False,
    )
    return logits.index_select(-1, support)


extract_legal_support = legal_support_logits


def _sequence_logits(logits: torch.Tensor, name: str) -> torch.Tensor:
    _require_real_logits(logits, name)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [response_tokens, vocabulary]")
    return logits


def _sequence_targets(target_token_ids: Sequence[int] | torch.Tensor, length: int, vocab_size: int) -> list[int]:
    if isinstance(target_token_ids, torch.Tensor):
        if target_token_ids.ndim != 1:
            raise ValueError("target_token_ids must be one-dimensional")
        if target_token_ids.dtype == torch.bool or target_token_ids.is_floating_point() or target_token_ids.is_complex():
            raise TypeError("target_token_ids must contain integer token IDs")
        values = [int(value) for value in target_token_ids.detach().cpu().tolist()]
    else:
        try:
            raw = list(target_token_ids)
        except TypeError as exc:
            raise TypeError("target_token_ids must be a one-dimensional integer sequence") from exc
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
            raise TypeError("target_token_ids must contain integer token IDs")
        values = [int(value) for value in raw]
    if len(values) != length:
        raise ValueError(f"target_token_ids has length {len(values)}, expected {length}")
    if any(value < 0 or value >= vocab_size for value in values):
        raise ValueError("target_token_ids contains an ID outside the vocabulary")
    return values


def _sequence_mask(mask: Sequence[bool] | torch.Tensor, length: int, *, name: str) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        if mask.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if mask.is_complex() or not (mask.dtype == torch.bool or mask.is_floating_point() or mask.dtype in _INTEGER_DTYPES):
            raise TypeError(f"{name} must be boolean or real numeric")
        values = mask.detach()
    else:
        try:
            values = torch.as_tensor(mask)
        except Exception as exc:
            raise TypeError(f"{name} must be a one-dimensional boolean sequence") from exc
        if values.is_complex() or not (values.dtype == torch.bool or values.is_floating_point() or values.dtype in _INTEGER_DTYPES):
            raise TypeError(f"{name} must be boolean or real numeric")
    if values.ndim != 1 or values.numel() != length:
        raise ValueError(f"{name} must have length {length}")
    if values.is_floating_point() and not torch.isfinite(values).all().item():
        raise FloatingPointError(f"{name} contains nonfinite values")
    return values.to(dtype=torch.bool)


def _sequence_supports(
    support_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
    length: int,
    *,
    vocab_size: int,
    device: torch.device,
    allow_empty: bool,
) -> list[torch.Tensor]:
    if isinstance(support_ids, torch.Tensor):
        if support_ids.ndim != 2 or support_ids.shape[0] != length:
            raise ValueError(f"support_ids must have shape [{length}, support]")
        rows: list[Any] = [support_ids[index] for index in range(length)]
    else:
        try:
            rows = list(support_ids)
        except TypeError as exc:
            raise TypeError("support_ids must contain one ID sequence per response token") from exc
        if len(rows) != length:
            raise ValueError(f"support_ids has {len(rows)} rows, expected {length}")
    return [
        _support_ids(
            row,
            vocab_size=vocab_size,
            device=device,
            allow_empty=allow_empty,
            name=f"support_ids[{index}]",
        )
        for index, row in enumerate(rows)
    ]


def _single_sft_mean(
    logits: torch.Tensor,
    target_token_ids: Sequence[int] | torch.Tensor,
    support_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
    numeric_mask: Sequence[bool] | torch.Tensor,
) -> torch.Tensor:
    rows = _sequence_logits(logits, "model_logits")
    targets = _sequence_targets(target_token_ids, rows.shape[0], rows.shape[-1])
    supports = _sequence_supports(
        support_ids,
        rows.shape[0],
        vocab_size=rows.shape[-1],
        device=rows.device,
        allow_empty=False,
    )
    mask = _sequence_mask(numeric_mask, rows.shape[0], name="numeric_mask")
    compute_dtype = _probability_dtype(rows)
    work = rows.to(dtype=compute_dtype)
    losses: list[torch.Tensor] = []
    for position, (target, support) in enumerate(zip(targets, supports, strict=True)):
        location = torch.where(support == target)[0]
        if location.numel() != 1:
            raise ValueError(f"GT token {target} is outside legal grammar support at position {position}")
        if bool(mask[position].item()):
            log_probs = F.log_softmax(work[position].index_select(0, support), dim=-1)
            value = -log_probs[location[0]]
            if not torch.isfinite(value.detach()).all().item():
                raise FloatingPointError(f"nonfinite SFT NLL at response position {position}")
            losses.append(value)
    if not losses:
        return work.sum() * 0.0
    result = torch.stack(losses).sum() / len(losses)
    return _finite_output(result, "SFT NLL")


def _single_reverse_kl_mean(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    support_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
    numeric_mask: Sequence[bool] | torch.Tensor,
) -> torch.Tensor:
    student = _sequence_logits(student_logits, "student_logits")
    teacher = _sequence_logits(teacher_logits, "teacher_logits")
    if student.shape != teacher.shape:
        raise ValueError(
            f"student_logits and teacher_logits must have the same shape, got {tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    supports = _sequence_supports(
        support_ids,
        student.shape[0],
        vocab_size=student.shape[-1],
        device=student.device,
        allow_empty=True,
    )
    mask = _sequence_mask(numeric_mask, student.shape[0], name="numeric_mask")
    compute_dtype = _probability_dtype(student, teacher)
    student_work = student.to(dtype=compute_dtype)
    teacher_work = teacher.detach().to(device=student.device, dtype=compute_dtype)
    losses: list[torch.Tensor] = []
    for position, support in enumerate(supports):
        if not bool(mask[position].item()):
            continue
        if support.numel() == 0:
            raise ValueError(f"active numeric position {position} has empty grammar support")
        selected_student = student_work[position].index_select(0, support)
        selected_teacher = teacher_work[position].index_select(0, support)
        log_p = F.log_softmax(selected_student, dim=-1)
        log_q = F.log_softmax(selected_teacher, dim=-1)
        value = (log_p.exp() * (log_p - log_q)).sum()
        if not torch.isfinite(value.detach()).all().item():
            raise FloatingPointError(f"nonfinite reverse KL at response position {position}")
        losses.append(value)
    if not losses:
        return student_work.sum() * 0.0
    result = torch.stack(losses).sum() / len(losses)
    return _finite_output(result, "reverse KL")


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.ndim == 0
    return isinstance(value, (bool, Integral, Real))


def _as_batch_logits(value: torch.Tensor | Sequence[torch.Tensor | None], *, name: str, allow_none: bool) -> list[torch.Tensor | None]:
    if isinstance(value, torch.Tensor):
        _require_real_logits(value, name)
        if value.ndim == 2:
            return [value]
        if value.ndim == 3 and value.shape[0] > 0:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError(f"{name} must have shape [T,V] or [B,T,V]")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a tensor or a sequence of tensors") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one example")
    result: list[torch.Tensor | None] = []
    for index, item in enumerate(values):
        if item is None and allow_none:
            result.append(None)
            continue
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"{name}[{index}] must be a tensor")
        _require_real_logits(item, f"{name}[{index}]")
        if item.ndim != 2:
            raise ValueError(f"{name}[{index}] must have shape [T,V]")
        result.append(item)
    return result


def _position_logits(value: Any, *, name: str) -> torch.Tensor | None:
    """Stack a caller's ``[V]`` logits emitted one response position at a time."""

    if isinstance(value, torch.Tensor) or isinstance(value, (str, bytes)):
        return None
    try:
        values = list(value)
    except TypeError:
        return None
    if not values or not all(isinstance(item, torch.Tensor) and item.ndim == 1 for item in values):
        return None
    try:
        stacked = torch.stack(values, dim=0)
    except RuntimeError as exc:
        raise ValueError(f"{name} position logits must have a common vocabulary size") from exc
    _require_real_logits(stacked, name)
    return stacked


def _split_batch_metadata(
    value: Any,
    batch_size: int,
    lengths: Sequence[int],
    *,
    kind: str,
    name: str,
) -> list[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, torch.Tensor):
        if kind == "support":
            if batch_size == 1 and value.ndim == 2:
                return [value]
            if value.ndim == 3 and value.shape[0] == batch_size:
                return [value[index] for index in range(batch_size)]
        else:
            if batch_size == 1 and value.ndim == 1:
                return [value]
            if value.ndim == 2 and value.shape[0] == batch_size:
                return [value[index] for index in range(batch_size)]
        raise ValueError(f"{name} has an incompatible tensor shape for batch size {batch_size}")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence with one item per example") from exc

    if batch_size == 1:
        length = lengths[0]
        if kind == "support":
            if len(values) == 1 and isinstance(values[0], (Sequence, torch.Tensor)):
                first = values[0]
                first_length = len(first) if not isinstance(first, torch.Tensor) else first.shape[0]
                if first_length == length:
                    return [first]
            if len(values) == length and all(not _is_scalar_like(item) for item in values):
                return [values]
            if length == 1 and all(_is_scalar_like(item) for item in values):
                return [[values]]
        else:
            if len(values) == length and all(_is_scalar_like(item) for item in values):
                return [values]
            if len(values) == 1 and isinstance(values[0], (Sequence, torch.Tensor)):
                first = values[0]
                first_length = len(first) if not isinstance(first, torch.Tensor) else first.shape[0]
                if first_length == length:
                    return [first]
            if length == 1 and len(values) == 1 and _is_scalar_like(values[0]):
                return [values]
        raise ValueError(f"{name} does not describe one sequence of length {length}")

    if len(values) != batch_size:
        raise ValueError(f"{name} has {len(values)} examples, expected {batch_size}")
    return values


def aggregate_per_example(
    per_example_losses: Sequence[torch.Tensor | float | None] | torch.Tensor,
    effective_batch_size: int | None = None,
    *,
    active_mask: Sequence[bool] | torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum per-example means and divide by a fixed batch denominator.

    ``None`` entries and entries disabled by ``active_mask`` are exact zeros,
    but still occupy the denominator.  This is the OPD-missing contract.
    """

    if isinstance(per_example_losses, torch.Tensor):
        if per_example_losses.ndim == 0:
            values: list[Any] = [per_example_losses]
        else:
            values = [per_example_losses[index] for index in range(per_example_losses.shape[0])]
    else:
        try:
            values = list(per_example_losses)
        except TypeError as exc:
            raise TypeError("per_example_losses must be a sequence") from exc
    if not values:
        raise ValueError("per_example_losses must contain at least one example")
    denominator = _effective_batch_size(effective_batch_size, len(values))

    if active_mask is None:
        active = [value is not None for value in values]
    else:
        mask_tensor = torch.as_tensor(active_mask)
        if mask_tensor.ndim != 1 or mask_tensor.numel() != len(values):
            raise ValueError("active_mask must have one entry per example")
        if mask_tensor.is_complex() or not (
            mask_tensor.dtype == torch.bool
            or mask_tensor.is_floating_point()
            or mask_tensor.dtype in _INTEGER_DTYPES
        ):
            raise TypeError("active_mask must be boolean or real numeric")
        if mask_tensor.is_floating_point() and not torch.isfinite(mask_tensor).all().item():
            raise FloatingPointError("active_mask contains nonfinite values")
        active = [bool(value) for value in mask_tensor.to(dtype=torch.bool).tolist()]

    anchor = reference if isinstance(reference, torch.Tensor) else None
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("each per-example loss must be scalar")
            _require_real_logits(value.reshape(1), "per_example_loss")
            if anchor is None:
                anchor = value
    if anchor is None:
        anchor = torch.tensor(0.0, dtype=torch.float32)

    terms: list[torch.Tensor] = []
    for is_active, value in zip(active, values, strict=True):
        if not is_active or value is None:
            terms.append(anchor.sum() * 0.0)
            continue
        if isinstance(value, torch.Tensor):
            term = value.reshape(()).to(device=anchor.device, dtype=_probability_dtype(anchor, value))
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("per-example losses must be finite scalars") from exc
            if not math.isfinite(scalar):
                raise FloatingPointError("per-example loss is nonfinite")
            term = torch.as_tensor(scalar, device=anchor.device, dtype=_probability_dtype(anchor))
        if not torch.isfinite(term.detach()).all().item():
            raise FloatingPointError("per-example loss is nonfinite")
        terms.append(term)
    result = torch.stack(terms).sum() / denominator
    return _finite_output(result, "batch loss")


def batch_sft_numeric_nll(
    model_logits: torch.Tensor | Sequence[torch.Tensor],
    target_token_ids: Any,
    support_ids: Any,
    numeric_mask: Any,
    *,
    effective_batch_size: int | None = None,
) -> torch.Tensor:
    """Compute per-example normalized GT-prefix SFT contributions."""

    logits = _as_batch_logits(model_logits, name="model_logits", allow_none=False)
    batch_size = len(logits)
    lengths = [int(item.shape[0]) for item in logits if item is not None]
    if len(lengths) != batch_size:
        raise ValueError("SFT logits cannot contain missing examples")
    targets = _split_batch_metadata(target_token_ids, batch_size, lengths, kind="target", name="target_token_ids")
    supports = _split_batch_metadata(support_ids, batch_size, lengths, kind="support", name="support_ids")
    masks = _split_batch_metadata(numeric_mask, batch_size, lengths, kind="mask", name="numeric_mask")
    per_example: list[torch.Tensor] = []
    for index, item in enumerate(logits):
        if targets[index] is None or supports[index] is None or masks[index] is None:
            raise ValueError(f"SFT metadata is missing for example {index}")
        per_example.append(_single_sft_mean(item, targets[index], supports[index], masks[index]))  # type: ignore[arg-type]
    return aggregate_per_example(per_example, effective_batch_size, reference=logits[0])


def sft_numeric_nll(
    model_logits: torch.Tensor | Sequence[torch.Tensor],
    target_token_ids: Any,
    support_ids: Any,
    numeric_mask: Any,
    *,
    effective_batch_size: int | None = None,
) -> torch.Tensor:
    """Legal-support numeric NLL for logits produced by GT-prefix forcing."""

    position_logits = _position_logits(model_logits, name="model_logits")
    if position_logits is not None:
        value = _single_sft_mean(position_logits, target_token_ids, support_ids, numeric_mask)
        return _finite_output(value / _effective_batch_size(effective_batch_size, 1), "SFT NLL")
    if isinstance(model_logits, torch.Tensor) and model_logits.ndim <= 2:
        rows = _sequence_logits(model_logits, "model_logits")
        value = _single_sft_mean(rows, target_token_ids, support_ids, numeric_mask)
        return _finite_output(value / _effective_batch_size(effective_batch_size, 1), "SFT NLL")
    return batch_sft_numeric_nll(
        model_logits,
        target_token_ids,
        support_ids,
        numeric_mask,
        effective_batch_size=effective_batch_size,
    )


def batch_reverse_kl_full_support(
    student_logits: torch.Tensor | Sequence[torch.Tensor | None],
    teacher_logits: torch.Tensor | Sequence[torch.Tensor | None] | None,
    support_ids: Any,
    numeric_mask: Any,
    *,
    effective_batch_size: int | None = None,
    opd_present: Sequence[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute full-support reverse KL with fixed-denominator OPD masking."""

    students = _as_batch_logits(student_logits, name="student_logits", allow_none=True)
    batch_size = len(students)
    lengths = [int(item.shape[0]) if item is not None else 0 for item in students]
    if teacher_logits is None:
        teachers: list[torch.Tensor | None] = [None] * batch_size
    else:
        teachers = _as_batch_logits(teacher_logits, name="teacher_logits", allow_none=True)
        if len(teachers) != batch_size:
            raise ValueError("student_logits and teacher_logits have different batch sizes")
    for index, (student, teacher) in enumerate(zip(students, teachers, strict=True)):
        if student is not None and lengths[index] == 0:
            lengths[index] = int(student.shape[0])
        if teacher is not None and student is not None and teacher.shape != student.shape:
            raise ValueError(f"student/teacher shape mismatch at example {index}")
        if teacher is not None and student is None:
            raise ValueError(f"teacher logits are present but student logits are missing at example {index}")
    if any(length <= 0 for length in lengths if length != 0) or all(length == 0 for length in lengths):
        raise ValueError("at least one student example must contain response logits")

    support_batch = _split_batch_metadata(support_ids, batch_size, lengths, kind="support", name="support_ids")
    mask_batch = _split_batch_metadata(numeric_mask, batch_size, lengths, kind="mask", name="numeric_mask")
    if opd_present is None:
        present = [teacher is not None for teacher in teachers]
    else:
        present_tensor = torch.as_tensor(opd_present)
        if present_tensor.ndim != 1 or present_tensor.numel() != batch_size:
            raise ValueError("opd_present must have one entry per example")
        if present_tensor.is_complex() or not (
            present_tensor.dtype == torch.bool
            or present_tensor.is_floating_point()
            or present_tensor.dtype in _INTEGER_DTYPES
        ):
            raise TypeError("opd_present must be boolean or real numeric")
        if present_tensor.is_floating_point() and not torch.isfinite(present_tensor).all().item():
            raise FloatingPointError("opd_present contains nonfinite values")
        present = [bool(value) for value in present_tensor.to(dtype=torch.bool).tolist()]

    per_example: list[torch.Tensor | None] = []
    reference = next((item for item in students if item is not None), None)
    for index, (student, teacher) in enumerate(zip(students, teachers, strict=True)):
        if not present[index]:
            per_example.append(None)
            continue
        if student is None or teacher is None:
            raise ValueError(f"OPD example {index} is marked present but logits are missing")
        if support_batch[index] is None or mask_batch[index] is None:
            raise ValueError(f"OPD metadata is missing for example {index}")
        per_example.append(_single_reverse_kl_mean(student, teacher, support_batch[index], mask_batch[index]))  # type: ignore[arg-type]
    return aggregate_per_example(per_example, effective_batch_size, active_mask=present, reference=reference)


def reverse_kl_full_support(
    student_logits: torch.Tensor | Sequence[torch.Tensor | None],
    teacher_logits: torch.Tensor | Sequence[torch.Tensor | None] | None,
    support_ids: Any,
    numeric_mask: Any,
    *,
    effective_batch_size: int | None = None,
    opd_present: Sequence[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``KL(p_student || q_teacher)`` on every legal support token."""

    position_student = _position_logits(student_logits, name="student_logits")
    position_teacher = _position_logits(teacher_logits, name="teacher_logits")
    if position_student is not None or position_teacher is not None:
        if position_student is None or position_teacher is None:
            raise ValueError("student and teacher position logits must be supplied together")
        value = _single_reverse_kl_mean(position_student, position_teacher, support_ids, numeric_mask)
        return _finite_output(value / _effective_batch_size(effective_batch_size, 1), "reverse KL")
    if (
        isinstance(student_logits, torch.Tensor)
        and student_logits.ndim <= 2
        and isinstance(teacher_logits, torch.Tensor)
    ):
        value = _single_reverse_kl_mean(student_logits, teacher_logits, support_ids, numeric_mask)
        return _finite_output(value / _effective_batch_size(effective_batch_size, 1), "reverse KL")
    return batch_reverse_kl_full_support(
        student_logits,
        teacher_logits,
        support_ids,
        numeric_mask,
        effective_batch_size=effective_batch_size,
        opd_present=opd_present,
    )


# Descriptive aliases keep the two prefix contracts visible at call sites.
gt_prefix_sft_loss = sft_numeric_nll
sft_legal_support_numeric_nll = sft_numeric_nll
student_prefix_reverse_kl = reverse_kl_full_support
