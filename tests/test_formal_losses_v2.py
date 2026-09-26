from __future__ import annotations

import math

import pytest
import torch

from mmgcot_timeline_training.formal_losses import (
    extract_grammar_supports,
    extract_legal_support,
    numeric_token_mask,
    quantize_bbox,
    reverse_kl_full_support,
    sft_numeric_nll,
)


def test_quantize_bbox_uses_normalized_half_up_rounding() -> None:
    assert quantize_bbox([0.0005, 0.0015, 0.9995, 1.0]) == (1, 2, 1000, 1000)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.001, 1.001])
def test_quantize_bbox_rejects_nonfinite_or_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError):
        quantize_bbox([0.1, 0.1, value, 0.9])


class _Tokenizer:
    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in ids)


def test_numeric_mask_uses_raw_token_spans_without_coordinate_token_assumption() -> None:
    tokenizer = _Tokenizer({1: "12,", 2: "34", 3: ",", 4: "5", 5: "]}</answer>"})

    assert numeric_token_mask([1, 2, 3, 4, 5], tokenizer) == [True, True, False, True, False]


class _FakeMask:
    def __init__(self) -> None:
        self.allowed: list[int] = []

    def to(self, _device):
        return self


class _FakeXgrammar:
    class GrammarMatcher:
        def __init__(self, grammar, *, terminate_without_stop_token):
            assert terminate_without_stop_token is True
            self.grammar = grammar
            self.position = 0

        def fill_next_token_bitmask(self, mask: _FakeMask) -> bool:
            mask.allowed = list(self.grammar[self.position])
            return True

        def accept_token(self, token_id: int) -> bool:
            if token_id not in self.grammar[self.position]:
                return False
            self.position += 1
            return True

        def is_completed(self) -> bool:
            return self.position == len(self.grammar)

    @staticmethod
    def allocate_token_bitmask(_batch: int, _vocab_size: int) -> _FakeMask:
        return _FakeMask()

    @staticmethod
    def reset_token_bitmask(_mask: _FakeMask) -> None:
        return None

    @staticmethod
    def apply_token_bitmask_inplace(scores: torch.Tensor, mask: _FakeMask, *, vocab_size: int) -> None:
        allowed = set(mask.allowed)
        for token_id in range(vocab_size):
            if token_id not in allowed:
                scores[0, token_id] = -torch.inf


def test_grammar_supports_are_recorded_before_each_raw_token() -> None:
    supports = extract_grammar_supports(
        [[1, 2], [3]],
        4,
        [2, 3],
        xgrammar_module=_FakeXgrammar,
    )

    assert supports == [[1, 2], [3]]


def test_sft_scores_only_numeric_gt_tokens_on_legal_support() -> None:
    logits = torch.tensor(
        [[0.2, -0.3, 0.7, -0.1, 0.5],
         [5.0, -4.0, 3.0, -2.0, 1.0],
         [-0.4, 0.1, 0.8, -0.2, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = [2, 3, 4]
    supports = [[0, 2], [1, 3], [0, 2, 4]]
    numeric = [True, False, True]

    loss = sft_numeric_nll(logits, targets, supports, numeric)
    expected = (
        -torch.log_softmax(logits[0, [0, 2]], dim=0)[1]
        - torch.log_softmax(logits[2, [0, 2, 4]], dim=0)[2]
    ) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))
    assert torch.equal(logits.grad[0, [1, 3, 4]], torch.zeros(3, dtype=logits.dtype))


def test_sft_rejects_a_gt_token_outside_grammar_support() -> None:
    logits = torch.zeros(1, 4)

    with pytest.raises(ValueError, match="outside legal grammar support"):
        sft_numeric_nll(logits, [3], [[0, 1]], [True])


def test_batch_sft_normalizes_each_example_before_fixed_denominator() -> None:
    logits = torch.tensor(
        [[[0.2, -0.3, 0.7, -0.1], [0.5, 0.1, -0.4, 0.2]],
         [[-0.2, 0.4, 0.6, -0.5], [3.0, -2.0, 1.0, 0.0]],
         [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = [[0, 2], [1, 0], [3, 3]]
    supports = [[[0, 1, 2, 3], [0, 1, 2, 3]]] * 3
    numeric = [[True, True], [True, False], [False, False]]

    loss = sft_numeric_nll(logits, targets, supports, numeric, effective_batch_size=3)
    first = sum(-torch.log_softmax(logits[0, position], dim=0)[target] for position, target in enumerate(targets[0])) / 2
    second = -torch.log_softmax(logits[1, 0], dim=0)[targets[1][0]]
    torch.testing.assert_close(loss, (first + second) / 3)


def test_extract_legal_support_preserves_selected_logits_and_gradients() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)

    selected = extract_legal_support(logits, [3, 1])
    torch.testing.assert_close(selected, torch.tensor([[0.4, 0.2]]))
    selected.sum().backward()
    assert torch.equal(logits.grad, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))


def test_reverse_kl_uses_full_legal_support_and_detaches_teacher() -> None:
    student = torch.tensor(
        [[0.4, -0.2, 0.8, 7.0, -9.0],
         [1.0, 2.0, -1.0, 0.5, -2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[-0.1, 0.6, 0.2, -8.0, 8.0],
         [0.1, -0.2, 0.3, 0.4, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    supports = [[0, 1, 2], [0, 1, 2, 3, 4]]

    loss = reverse_kl_full_support(student, teacher, supports, [True, False])
    student_logp = torch.log_softmax(student[0, [0, 1, 2]], dim=0)
    teacher_logp = torch.log_softmax(teacher[0, [0, 1, 2]], dim=0)
    expected = (student_logp.exp() * (student_logp - teacher_logp)).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.equal(student.grad[1], torch.zeros_like(student.grad[1]))
    assert torch.equal(student.grad[0, [3, 4]], torch.zeros(2, dtype=student.dtype))


def test_reverse_kl_batch_keeps_opd_missing_zero_in_fixed_denominator() -> None:
    students = [
        torch.tensor([[0.4, -0.2, 0.8]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[5.0, -4.0, 1.0]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0.1, 0.7, -0.3]], dtype=torch.float64, requires_grad=True),
    ]
    teachers = [
        torch.tensor([[0.2, 0.1, -0.5]], dtype=torch.float64),
        None,
        torch.tensor([[-0.1, 0.6, 0.2]], dtype=torch.float64),
    ]
    supports = [[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]]
    numeric = [[True], [True], [True]]

    loss = reverse_kl_full_support(
        students,
        teachers,
        supports,
        numeric,
        effective_batch_size=3,
    )
    expected_terms = []
    for student, teacher in (zip(students, teachers)):
        if teacher is None:
            continue
        logp = torch.log_softmax(student[0], dim=0)
        logq = torch.log_softmax(teacher[0], dim=0)
        expected_terms.append((logp.exp() * (logp - logq)).sum())
    torch.testing.assert_close(loss, sum(expected_terms) / 3)
    loss.backward()
    assert students[1].grad is None or torch.equal(students[1].grad, torch.zeros_like(students[1].grad))


def test_losses_fail_closed_on_nonfinite_logits() -> None:
    with pytest.raises(FloatingPointError, match="nonfinite logits"):
        sft_numeric_nll(torch.tensor([[0.0, math.nan]]), [0], [[0]], [True])

    with pytest.raises(FloatingPointError, match="nonfinite logits"):
        reverse_kl_full_support(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, math.inf]]),
            [[0, 1]],
            [True],
        )
