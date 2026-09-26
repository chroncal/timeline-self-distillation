# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
    *,
    output_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype = torch.float32,
    center_logits: bool = False,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    output_dtype:
        Optional output dtype. The default preserves the historical behavior
        of returning the same dtype as ``logits``. JSD uses ``float32`` here
        so probabilities and the aggregated tail do not inherit bf16 rounding.
    compute_dtype:
        Dtype used for the log-sum-exp and gathered logits. The historical
        default is fp32; fp64 callers can opt into fp64 reference arithmetic.
    center_logits:
        Center before both gather and logsumexp. JSD enables this to avoid
        cancellation from subtracting a large rounded log-normalizer. The
        default keeps historical partial-KL arithmetic unchanged.

    Returns:
        [B, T, K] tensor with dtype ``output_dtype`` (or ``logits.dtype``).
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    output_dtype = logits.dtype if output_dtype is None else output_dtype
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=output_dtype, device=logits.device)

    out = torch.empty((N, K), dtype=output_dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits = flat_logits[s:e].to(compute_dtype)
        if center_logits:
            chunk_logits = chunk_logits - chunk_logits.amax(dim=-1, keepdim=True).detach()
        log_z = torch.logsumexp(chunk_logits, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(output_dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


_TAIL_ROUNDOFF_TOLERANCE = 1e-6


def _tail_probability(mass: torch.Tensor, *, name: str) -> torch.Tensor:
    """Return ``1 - mass`` while only correcting demonstrable fp roundoff.

    A top-k probability mass is allowed to exceed one by a very small amount
    because teacher log-probabilities are serialized in fp32 and then summed.
    The tolerance covers the observed serialization roundoff (about 1.6e-7),
    while a materially invalid distribution fails closed instead of being
    silently projected onto the simplex.
    """
    if not torch.isfinite(mass).all():
        raise ValueError(f"{name} top-k probability mass is non-finite")
    if (mass < -_TAIL_ROUNDOFF_TOLERANCE).any():
        raise ValueError(f"{name} top-k probability mass is negative")
    over = mass - 1.0
    if (over > _TAIL_ROUNDOFF_TOLERANCE).any():
        raise ValueError(
            f"{name} top-k probability mass exceeds one: "
            f"max={mass.detach().max().item():.10g}, tolerance={_TAIL_ROUNDOFF_TOLERANCE}"
        )
    # Only this lower clamp is intentional: it fixes tiny negative tails from
    # fp32 subtraction, while the validation above rejects invalid mass.
    return (1.0 - mass).clamp_min(0.0)


def _safe_log_probability(probabilities: torch.Tensor) -> torch.Tensor:
    """Log probabilities without forming ``0 * -inf`` in the JSD terms."""
    tiny = torch.finfo(probabilities.dtype).tiny
    return probabilities.clamp_min(tiny).log()


def _jsd_from_topk_and_tail(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute symmetric natural-log JSD on a shared sparse support.

    The final dimension contains the teacher/student top-k probabilities and
    one already-aggregated tail probability. Teacher probabilities are
    expected to be detached; the mixture deliberately remains differentiable
    through ``student_probs``.
    """
    mixture = 0.5 * (teacher_probs + student_probs)
    teacher_log = _safe_log_probability(teacher_probs)
    student_log = _safe_log_probability(student_probs)
    mixture_log = _safe_log_probability(mixture)
    teacher_kl = (teacher_probs * (teacher_log - mixture_log)).sum(dim=-1)
    student_kl = (student_probs * (student_log - mixture_log)).sum(dim=-1)
    return 0.5 * (teacher_kl + student_kl)


def compute_jsd_topk_tail(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sparse symmetric JSD using teacher top-k plus one tail bucket.

    ``P`` is the detached teacher top-k probability vector followed by
    ``1 - sum(P_topk)``. ``Q`` gathers student probabilities at those exact
    IDs and appends the corresponding student tail. No full-vocabulary teacher
    tensor is materialized, and the chunked gather path keeps the student
    projection bounded by the configured token chunk size.

    ``active_mask`` is an optional packed ``[1, T]`` mask. Error-span OPD
    supplies it so prompt and unselected rows (whose top-k IDs are repeated
    placeholders) are replaced by a harmless one-tail distribution before any
    gather. With no mask, every supplied row is treated as active for callers
    using the native kernel directly.
    """
    del data_format  # The FSDP logits contract is identical for both formats.
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    # Match the historical forward-KL sequence-parallel slicing exactly.
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    if active_mask is None:
        active = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
    else:
        active = active_mask.to(device=student_logits.device, dtype=torch.bool)
        if active.shape != student_logits.shape[:2]:
            raise ValueError(
                "JSD active_mask must match the packed student-logit shape, "
                f"got {active.shape} and {student_logits.shape[:2]}"
            )

    topk = teacher_topk_ids.shape[-1]
    vocab_size = student_logits.shape[-1]
    if topk > vocab_size:
        raise ValueError(f"JSD top-k ({topk}) cannot exceed student vocabulary size ({vocab_size})")

    # Inactive rows can contain repeated zero IDs and arbitrary placeholder
    # log-probs. Replace both with a valid support before gathering, then set
    # their probabilities to the one tail bucket. This prevents placeholders
    # from being interpreted as repeated real categories or invalid mass.
    fallback_ids = torch.arange(topk, device=student_logits.device, dtype=teacher_topk_ids.dtype)
    fallback_ids = fallback_ids.view(1, 1, topk).expand_as(teacher_topk_ids)
    active_entries = active.unsqueeze(-1)
    teacher_topk_ids = torch.where(active_entries, teacher_topk_ids, fallback_ids)
    invalid_ids = (teacher_topk_ids < 0) | (teacher_topk_ids >= vocab_size)
    if (invalid_ids & active_entries).any():
        raise ValueError("teacher top-k IDs contain an out-of-range active token")
    if topk > 1:
        sorted_ids = torch.sort(teacher_topk_ids, dim=-1).values
        duplicate_ids = sorted_ids[..., 1:] == sorted_ids[..., :-1]
        if (duplicate_ids.any(dim=-1) & active).any():
            raise ValueError("teacher top-k IDs contain duplicate active tokens")
    teacher_topk_log_probs = teacher_topk_log_probs.detach()
    teacher_topk_log_probs = torch.where(
        active_entries,
        teacher_topk_log_probs,
        torch.full_like(teacher_topk_log_probs, float("-inf")),
    )

    loss_config: DistillationLossConfig = config.distillation_loss
    chunk_size = getattr(loss_config, "chunked_topk_chunk_size", 4096)
    # Keep sparse gather for the configured chunked path. The eager fallback is
    # retained for compatibility with short-context callers that explicitly
    # disable chunking, but both branches perform the sparse arithmetic in fp32
    # (or fp64 for a fp64 reference input).
    compute_dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    teacher_topk_log_probs = teacher_topk_log_probs.to(compute_dtype)
    if getattr(loss_config, "use_chunked_topk", False):
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=chunk_size,
            output_dtype=compute_dtype,
            compute_dtype=compute_dtype,
            center_logits=True,
        )
    else:
        student_topk_log_probs = torch.gather(
            F.log_softmax(student_logits.to(compute_dtype), dim=-1),
            dim=-1,
            index=teacher_topk_ids,
        )

    student_topk_probs = student_topk_log_probs.exp()
    teacher_topk_probs = teacher_topk_log_probs.to(compute_dtype).exp()
    # Do not let an inactive row's gathered placeholder probability affect its
    # tail or its autograd graph.
    student_topk_probs = torch.where(active_entries, student_topk_probs, torch.zeros_like(student_topk_probs))
    teacher_topk_probs = torch.where(active_entries, teacher_topk_probs, torch.zeros_like(teacher_topk_probs))

    student_mass = student_topk_probs.sum(dim=-1)
    teacher_mass = teacher_topk_probs.sum(dim=-1)
    student_tail = _tail_probability(student_mass, name="student")
    teacher_tail = _tail_probability(teacher_mass, name="teacher")
    student_probs = torch.cat([student_topk_probs, student_tail.unsqueeze(-1)], dim=-1)
    teacher_probs = torch.cat([teacher_topk_probs, teacher_tail.unsqueeze(-1)], dim=-1).detach()
    distillation_losses = _jsd_from_topk_and_tail(teacher_probs=teacher_probs, student_probs=student_probs)
    # The inactive support was made exactly one-tail above. Keep this explicit
    # so future changes to the sparse formula cannot leak placeholder loss.
    distillation_losses = torch.where(active, distillation_losses, torch.zeros_like(distillation_losses))

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
