"""CPU witnesses for the timeline self-distillation loss reference.

These tests deliberately use tiny tensors.  They check the mathematical and
autograd contract of the reference functions, not a model or a training run.
"""

from __future__ import annotations

import pytest
import torch
from loss_reference import coordinate_wasserstein_1, masked_reverse_kl


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a two-token coordinate mask and a shared four-way support."""

    coordinate_mask = torch.tensor([[True, False]])
    support_mask = torch.tensor([[[True, True, True, False], [True, True, True, False]]])
    return coordinate_mask, support_mask


def test_masked_reverse_kl_reasoning_logit_has_exactly_zero_gradient() -> None:
    student_logits = torch.tensor(
        [[[0.3, -0.2, 0.1, 2.0], [1.1, -0.4, 0.7, -3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[[0.1, 0.2, -0.3, 4.0], [0.2, 0.4, -0.2, -4.0]]], dtype=torch.float64
    )
    coordinate_mask, support_mask = _masks()

    loss = masked_reverse_kl(student_logits, teacher_logits, coordinate_mask, support_mask)
    loss.backward()

    assert torch.equal(student_logits.grad[0, 1], torch.zeros_like(student_logits.grad[0, 1]))


def test_masked_reverse_kl_detaches_teacher_and_keeps_valid_student_gradient() -> None:
    student_logits = torch.tensor([[[0.7, -0.6, 0.1, 0.0]]], dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    coordinate_mask = torch.tensor([[True]])
    support_mask = torch.tensor([[[True, True, True, False]]])

    masked_reverse_kl(student_logits, teacher_logits, coordinate_mask, support_mask).backward()

    assert teacher_logits.grad is None
    assert torch.count_nonzero(student_logits.grad[..., :3]) > 0


def test_masked_reverse_kl_common_support_outside_is_zero_and_finite() -> None:
    student_logits = torch.tensor(
        [[[0.5, -0.1, 0.8, 1000.0, -1000.0]]], dtype=torch.float64, requires_grad=True
    )
    teacher_logits = torch.tensor([[[0.4, 0.2, -0.7, -1000.0, 1000.0]]], dtype=torch.float64)
    coordinate_mask = torch.tensor([[True]])
    support_mask = torch.tensor([[[True, True, True, False, False]]])

    loss = masked_reverse_kl(student_logits, teacher_logits, coordinate_mask, support_mask)
    assert torch.isfinite(loss)
    loss.backward()

    assert torch.isfinite(student_logits.grad).all()
    assert torch.equal(student_logits.grad[..., 3:], torch.zeros_like(student_logits.grad[..., 3:]))


def test_masked_reverse_kl_active_coordinate_with_empty_support_raises() -> None:
    student_logits = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.zeros_like(student_logits)
    coordinate_mask = torch.tensor([[True, False]])
    support_mask = torch.tensor(
        [[[False, False, False], [True, True, True]]], dtype=torch.bool
    )

    with pytest.raises(ValueError, match="active coordinate row has empty common support"):
        masked_reverse_kl(student_logits, teacher_logits, coordinate_mask, support_mask)


def test_masked_reverse_kl_empty_coordinate_sample_stays_in_batch_denominator() -> None:
    student_logits = torch.tensor(
        [
            [[0.2, -0.4, 0.1], [0.6, 0.2, -0.1]],
            [[-0.5, 0.7, 0.3], [0.1, -0.2, 0.8]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [
            [[0.0, 0.4, -0.2], [0.3, -0.1, 0.2]],
            [[0.1, -0.3, 0.6], [-0.2, 0.8, 0.0]],
        ],
        dtype=torch.float64,
    )
    coordinate_mask = torch.tensor([[False, False], [True, True]])
    support_mask = torch.ones_like(student_logits, dtype=torch.bool)

    actual = masked_reverse_kl(student_logits, teacher_logits, coordinate_mask, support_mask)
    valid_only = masked_reverse_kl(
        student_logits[1:].detach().clone().requires_grad_(),
        teacher_logits[1:],
        coordinate_mask[1:],
        support_mask[1:],
    )

    torch.testing.assert_close(actual, valid_only / 2)


def test_coordinate_wasserstein_1_point_masses_use_grid_distance_and_detach_q() -> None:
    p = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    grid = torch.tensor([0.0, 0.5, 2.0, 5.0], dtype=torch.float64)

    distance = coordinate_wasserstein_1(p, q, grid)
    assert distance.item() == pytest.approx(4.5)
    distance.backward()
    assert q.grad is None


def test_coordinate_wasserstein_1_requires_ordered_numeric_bins() -> None:
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        coordinate_wasserstein_1(p, q, torch.tensor([1.0, 0.0]))


def test_masked_reverse_kl_matches_finite_difference_token_gradient() -> None:
    student_logits = torch.tensor([[0.3, -0.7, 0.4]], dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.tensor([[0.2, 0.1, -0.5]], dtype=torch.float64)
    coordinate_mask = torch.tensor([True])
    support_mask = torch.tensor([[True, True, True]])

    loss = masked_reverse_kl(student_logits, teacher_logits, coordinate_mask, support_mask)
    (gradient,) = torch.autograd.grad(loss, student_logits)

    epsilon = 1e-6
    finite_difference = []
    for index in range(student_logits.shape[-1]):
        plus = student_logits.detach().clone()
        minus = student_logits.detach().clone()
        plus[..., index] += epsilon
        minus[..., index] -= epsilon
        plus_loss = masked_reverse_kl(plus, teacher_logits, coordinate_mask, support_mask)
        minus_loss = masked_reverse_kl(minus, teacher_logits, coordinate_mask, support_mask)
        finite_difference.append(((plus_loss - minus_loss) / (2.0 * epsilon)).item())

    torch.testing.assert_close(gradient, torch.tensor([finite_difference], dtype=torch.float64), atol=1e-7, rtol=1e-5)
