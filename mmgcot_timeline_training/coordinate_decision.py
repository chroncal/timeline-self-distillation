"""Grammar-state coordinate decisions, independent of the sampled action."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Decision:
    coordinate_index: int | None
    decision_type: str
    supervise: bool


@lru_cache(maxsize=4096)
def _decode_piece(tokenizer, token: int) -> str:
    return tokenizer.decode([token], skip_special_tokens=False)


def classify(prefix: str, candidates: list[str]) -> Decision:
    """Classify the *next* token using the legal support before sampling it.

    The bbox opening has already supplied ``{"bbox":[``.  Tokens which
    contain both a closing bracket and further markup are handled by their
    first character, with no tokenizer-specific delimiter IDs.
    """
    index = prefix.count(",")
    if "]" in prefix or index > 3:
        return Decision(None, "format", False)
    starts_digit = any(bool(t) and t[0] in "0123456789" for t in candidates)
    starts_end = any(bool(t) and t[0] in ",]" for t in candidates)
    if starts_digit and starts_end:
        kind = "digit_or_end"
    elif starts_digit:
        kind = "digit"
    elif starts_end:
        kind = "forced_separator"
    else:
        kind = "format"
    return Decision(index if kind != "format" else None, kind, starts_digit)


def annotate(token_ids: list[int], support_ids: list[list[int]], tokenizer) -> dict:
    from mmgcot_timeline_training.formal_losses import numeric_token_mask

    prefix = ""
    decisions: list[Decision] = []
    numeric = numeric_token_mask(token_ids, tokenizer)
    for token, support in zip(token_ids, support_ids, strict=True):
        token_text = _decode_piece(tokenizer, token)
        candidate_texts = [_decode_piece(tokenizer, v) for v in support]
        decisions.append(classify(prefix, candidate_texts))
        prefix += token_text
    mask = [d.supervise for d in decisions]
    if any(n and not m for n, m in zip(numeric, mask, strict=True)):
        raise RuntimeError("numeric position absent from grammar-state decision mask")
    return {"numeric_mask": numeric, "coordinate_decision_mask": mask,
            "coordinate_index": [d.coordinate_index for d in decisions],
            "decision_type": [d.decision_type for d in decisions]}


def reverse_kl_on_support(student_logits, teacher_logits):
    """Full-support KL(student || frozen teacher), before position weighting."""
    import torch

    student_log = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    return (student_log.exp() * (student_log - teacher_log)).sum()
