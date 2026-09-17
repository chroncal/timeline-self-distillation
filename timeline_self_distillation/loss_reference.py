"""Small, model-free CPU reference losses for timeline self-distillation.

The timeline protocol freezes the reasoning backbone and updates only the
final bounding-box adapter.  These functions therefore expose only the
student/teacher distributions needed at selected coordinate positions; they
do not accept rollout log-probabilities and never create a rollout gradient.

The reference is intentionally dense and easy to audit.  It is not a
replacement for a distributed vocabulary-parallel kernel and makes no claim
about running a real model or a real training job.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

__all__ = ["coordinate_wasserstein_1", "masked_reverse_kl"]

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _probability_compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Use fp32 arithmetic for low-precision logits, preserving fp64 checks."""

    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return torch.promote_types(dtype, torch.float32)


def _require_floating(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype")


def _broadcast_bool_mask(
    mask: torch.Tensor,
    shape: torch.Size,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert a binary/numeric mask and broadcast it to ``shape``."""

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=device)
    else:
        mask = mask.to(device=device)
    if mask.is_complex():
        raise TypeError(f"{name} must be boolean or real numeric")
    if not (mask.dtype == torch.bool or mask.is_floating_point() or mask.dtype in _INTEGER_DTYPES):
        raise TypeError(f"{name} must be boolean or real numeric")
    try:
        mask = torch.broadcast_to(mask != 0, shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} with shape {tuple(mask.shape)} cannot broadcast to {tuple(shape)}") from exc
    return mask


def masked_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    coordinate_mask: torch.Tensor,
    support_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return a coordinate-masked, batch-mean reverse KL reference loss.

    Args:
        student_logits: Floating logits with shape ``[B, ..., V]``.  The final
            dimension is the vocabulary/support dimension.
        teacher_logits: Same shape as ``student_logits``.  It is detached
            before any arithmetic, so this function never sends gradients to
            a teacher/inference backbone.
        coordinate_mask: Boolean or numeric mask broadcastable to
            ``student_logits.shape[:-1]``.  The nonzero positions are the
            final-coordinate positions to supervise; all other positions have
            exactly zero loss and gradient.
        support_mask: Boolean or numeric mask broadcastable to the full
            logits shape.  Each row is normalized *only* over this common
            support.  Consequently logits outside the support have exactly
            zero gradient, and no ``0 * inf`` term is formed.
        temperature: Positive finite temperature used to scale both logits;
            it defaults to ``1.0``.  No rollout distribution is involved.

    Returns:
        A scalar.  For each batch sample, selected token KLs are divided by
        that sample's number of selected coordinate positions.  An empty
        sample contributes zero, while the final mean still divides by the
        full batch size (including empty samples).

    Notes:
        The per-row quantity is ``KL(p_student || p_teacher)`` on the common
        support, where both distributions are softmaxes of the corresponding
        logits divided by ``temperature``.  Renormalizing on the support is
        what makes the outside-support gradient exactly zero; merely masking
        a full-vocabulary KL after softmax would still couple every logit via
        the normalizer.
    """

    _require_floating(student_logits, "student_logits")
    _require_floating(teacher_logits, "teacher_logits")
    if student_logits.ndim < 2:
        raise ValueError("logits must have shape [batch, ..., vocabulary]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits and teacher_logits must have the same shape, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise TypeError("temperature must be a positive finite scalar") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite scalar")

    batch_size = student_logits.shape[0]
    # The empty-batch mean is undefined mathematically.  Returning a zero
    # connected to student_logits is the least surprising autograd witness and
    # keeps this helper safe for a fully filtered microbatch.
    if batch_size == 0:
        return student_logits.sum() * 0.0

    leading_shape = student_logits.shape[:-1]
    coordinate_mask = _broadcast_bool_mask(
        coordinate_mask,
        leading_shape,
        device=student_logits.device,
        name="coordinate_mask",
    )
    support_mask = _broadcast_bool_mask(
        support_mask,
        student_logits.shape,
        device=student_logits.device,
        name="support_mask",
    )

    compute_dtype = _probability_compute_dtype(student_logits, teacher_logits)
    student = student_logits.to(dtype=compute_dtype)
    # Detach before moving/casting: a caller can safely pass a teacher tensor
    # with requires_grad=True and still gets no teacher graph.
    teacher = teacher_logits.detach().to(device=student_logits.device, dtype=compute_dtype)

    # log_softmax(all -inf) is NaN.  For a row with an empty support, use a
    # finite fallback normalization and then zero the row via the original
    # support mask.  This keeps the backward pass finite while retaining a
    # zero gradient for the empty-support row.
    support_nonempty = support_mask.any(dim=-1)
    # An active coordinate without a common support is a malformed teacher /
    # student contract, not an unlabelled coordinate.  Fail closed so the
    # training caller cannot silently report a zero loss for that coordinate.
    active_empty_support = coordinate_mask & ~support_nonempty
    if active_empty_support.any().item():
        raise ValueError("an active coordinate row has empty common support")

    safe_support = support_mask | ~support_nonempty.unsqueeze(-1)
    student_masked = student.masked_fill(~safe_support, -torch.inf)
    teacher_masked = teacher.masked_fill(~safe_support, -torch.inf)
    # A row with no common support must not feed arbitrary (possibly
    # non-finite) logits into log_softmax.  The selected branch has no
    # gradient for that row, and the zero fallback gives it a finite dummy
    # normalization before its KL terms are masked to zero below.
    has_support = support_nonempty.unsqueeze(-1)
    student_masked = torch.where(has_support, student_masked, torch.zeros_like(student_masked))
    teacher_masked = torch.where(has_support, teacher_masked, torch.zeros_like(teacher_masked))
    student_log_probs = F.log_softmax(student_masked / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_masked / temperature, dim=-1)

    # Do not form p * (log p - log q) at masked positions: p is zero there,
    # while log probabilities are -inf.  Selecting finite placeholders first
    # avoids the indeterminate 0*inf expression in both forward and backward.
    student_probs = student_log_probs.exp()
    finite_student_log_probs = torch.where(support_mask, student_log_probs, torch.zeros_like(student_log_probs))
    finite_teacher_log_probs = torch.where(support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
    finite_student_probs = torch.where(support_mask, student_probs, torch.zeros_like(student_probs))
    token_terms = finite_student_probs * (finite_student_log_probs - finite_teacher_log_probs)
    token_kl = token_terms.sum(dim=-1)

    coordinate_weights = coordinate_mask.to(dtype=compute_dtype)
    weighted_token_kl = token_kl * coordinate_weights
    reduce_dims = tuple(range(1, token_kl.ndim))
    if reduce_dims:
        selected_sum = weighted_token_kl.sum(dim=reduce_dims)
        selected_count = coordinate_weights.sum(dim=reduce_dims)
    else:
        # This covers a [B, V] logits tensor: each batch row is one selected
        # coordinate, represented by a [B] mask.
        selected_sum = weighted_token_kl
        selected_count = coordinate_weights

    per_sample = torch.where(
        selected_count > 0,
        selected_sum / selected_count.clamp_min(1.0),
        torch.zeros_like(selected_sum),
    )
    return per_sample.mean()


def coordinate_wasserstein_1(
    p: torch.Tensor,
    q: torch.Tensor,
    grid: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Compute discrete 1-Wasserstein distance along ordered coordinate bins.

    ``p`` and ``q`` are probability masses with shape ``[..., K]`` and the
    final axis corresponds to the *ordered numeric coordinate bins* ``grid``.
    The implementation uses the one-dimensional CDF identity

    ``W1(p, q) = sum_i |CDF_p[i] - CDF_q[i]| * (grid[i+1] - grid[i])``.

    ``q`` is detached before computation, so only ``p`` can receive a
    gradient.  ``grid`` must be finite and strictly increasing.  Token IDs
    are not suitable as ``grid`` values: IDs are arbitrary vocabulary labels,
    not an ordered numeric geometry.  A digit-level diagnostic may map each
    digit to its numeric value with explicitly declared place-value weights,
    but that is only a digit/place-value surrogate; it is not the W1 distance
    between complete bounding boxes and must not be reported as one (or
    implicitly replaced by a sum of per-digit distances).

    The function expects nonnegative, finite, normalized probability masses;
    it deliberately does not apply a softmax or silently renormalize inputs.
    This keeps the reference faithful to the caller's declared distribution.

    Returns:
        A tensor with shape ``p.shape[:-1]`` (a scalar for one distribution).
    """

    _require_floating(p, "p")
    _require_floating(q, "q")
    if p.ndim < 1:
        raise ValueError("p and q must have a final bin dimension")
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {tuple(p.shape)} and {tuple(q.shape)}")

    if isinstance(grid, torch.Tensor):
        grid_tensor = grid.to(device=p.device)
    else:
        try:
            grid_tensor = torch.as_tensor(grid, device=p.device)
        except Exception as exc:  # torch raises several type-specific errors here.
            raise TypeError("grid must be a one-dimensional numeric tensor/sequence") from exc
    if grid_tensor.ndim != 1 or grid_tensor.numel() != p.shape[-1]:
        raise ValueError(
            f"grid must be one-dimensional with {p.shape[-1]} bins, got shape {tuple(grid_tensor.shape)}"
        )
    if grid_tensor.dtype == torch.bool or grid_tensor.is_complex() or not (
        grid_tensor.is_floating_point() or grid_tensor.dtype in _INTEGER_DTYPES
    ):
        raise TypeError("grid must contain real numeric bin coordinates, not booleans or strings")

    compute_dtype = _probability_compute_dtype(p, q)
    p_work = p.to(dtype=compute_dtype)
    q_work = q.detach().to(device=p.device, dtype=compute_dtype)
    grid_work = grid_tensor.to(dtype=compute_dtype)

    # Validation is intentionally detached: it guards the reference contract
    # without introducing non-differentiable checks into p's graph.
    p_check = p_work.detach()
    q_check = q_work.detach()
    if not torch.isfinite(p_check).all().item() or not torch.isfinite(q_check).all().item():
        raise ValueError("p and q must be finite")
    if (p_check < 0).any().item() or (q_check < 0).any().item():
        raise ValueError("p and q must be nonnegative probability masses")
    p_mass = p_check.sum(dim=-1)
    q_mass = q_check.sum(dim=-1)
    if not torch.allclose(p_mass, torch.ones_like(p_mass), rtol=1e-5, atol=1e-6):
        raise ValueError("p must be normalized along its final dimension")
    if not torch.allclose(q_mass, torch.ones_like(q_mass), rtol=1e-5, atol=1e-6):
        raise ValueError("q must be normalized along its final dimension")

    if not torch.isfinite(grid_work).all().item():
        raise ValueError("grid must contain finite numeric bins")
    if grid_work.numel() > 1 and not (grid_work.diff() > 0).all().item():
        raise ValueError("grid must be strictly increasing numeric bins")

    if grid_work.numel() == 1:
        return p_work[..., 0] * 0.0

    cdf_delta = torch.cumsum(p_work - q_work, dim=-1)[..., :-1]
    widths = grid_work.diff()
    return (cdf_delta.abs() * widths).sum(dim=-1)


def _cpu_witness() -> dict[str, float]:
    """Run a tiny model-free witness when this file is executed directly."""

    student = torch.tensor([[0.3, -0.7, 0.4]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[0.2, 0.1, -0.5]], dtype=torch.float64)
    loss = masked_reverse_kl(
        student,
        teacher,
        coordinate_mask=torch.tensor([True]),
        support_mask=torch.tensor([[True, True, True]]),
    )
    (student_gradient,) = torch.autograd.grad(loss, student)
    distance = coordinate_wasserstein_1(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64),
    )
    return {
        "masked_reverse_kl": float(loss.detach()),
        "student_gradient_l1": float(student_gradient.abs().sum()),
        "coordinate_wasserstein_1": float(distance),
    }


if __name__ == "__main__":
    print(_cpu_witness())
