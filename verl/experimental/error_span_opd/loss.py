"""Masked aggregation of verl's existing top-k forward-KL kernel outputs."""

from __future__ import annotations

import torch

from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding


@register_distillation_loss(DistillationLossSettings(names=["error_span_forward_kl_topk"], use_topk=True))
def error_span_forward_kl_topk(config, distillation_config, model_output, data):
    # Projection and sparse KL are computed by verl.trainer.distillation.fsdp.losses.
    # Keep the official forward_kl_topk convention of clamping negative partial
    # KL values to zero; do not renormalize the teacher's retained probabilities.
    raw_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    losses = raw_losses.clamp_min(0.0)
    mask = data["response_mask"]
    if mask.is_nested:
        mask = mask.to_padded_tensor(False)
    mask = mask.bool()
    count = data["error_span_active_sequences"].reshape(-1)
    active = int(count[0].item())
    if active <= 0 or not torch.equal(count, torch.full_like(count, active)):
        raise ValueError("optimizer batch must have a consistent positive active-span count")
    if config.loss_agg_mode != "seq-mean-token-mean":
        raise ValueError("error span OPD requires equal active trajectories after within-span averaging")
    # The complete rollout batch is one PPO minibatch. Its inactive rows are
    # retained for auditing but must not dilute the active-sequence denominator.
    tu.assign_non_tensor(data, global_batch_size=active)
    metrics = {}
    if mask.any():
        if not torch.isfinite(raw_losses[mask]).all():
            raise ValueError("non-finite distillation loss on selected span")
        metrics["distillation/negative_partial_kl_fraction"] = float((raw_losses[mask] < 0).float().mean().detach())
        metrics["distillation/zero_loss_fraction"] = float((losses[mask] == 0).float().mean().detach())
        for key in ("student_mass", "teacher_mass"):
            values = no_padding_2_padding(model_output[key], data)[mask]
            metrics[f"distillation/{key}"] = float(values.mean().detach())
        metrics["distillation/mean_active_span_tokens"] = float(mask.sum().detach())
    return losses, metrics


@register_distillation_loss(DistillationLossSettings(names=["error_span_jsd_topk_tail"], use_topk=True))
def error_span_jsd_topk_tail(config, distillation_config, model_output, data):
    """Aggregate native sparse top-k-plus-tail JSD over selected error spans.

    The native FSDP kernel already masks inactive packed rows before forming
    the sparse support. Keep the response mask here as a second boundary at
    the padded response layout: only selected tokens are returned to the
    aggregator, and inactive placeholder rows cannot dilute the active
    trajectory denominator.

    Unlike ``error_span_forward_kl_topk``, JSD is a true divergence on a
    normalized shared support, so its values are returned without the legacy
    partial-KL ``clamp_min(0)`` operation.
    """
    del distillation_config
    raw_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    mask = data["response_mask"]
    if mask.is_nested:
        mask = mask.to_padded_tensor(False)
    mask = mask.bool().to(device=raw_losses.device)
    if raw_losses.shape != mask.shape:
        raise ValueError(f"JSD losses and response mask shapes differ: {raw_losses.shape} vs {mask.shape}")

    count = data["error_span_active_sequences"].reshape(-1)
    active = int(count[0].item())
    if active <= 0 or not torch.equal(count, torch.full_like(count, active)):
        raise ValueError("optimizer batch must have a consistent positive active-span count")
    if config.loss_agg_mode != "seq-mean-token-mean":
        raise ValueError("error span OPD requires equal active trajectories after within-span averaging")
    # The complete rollout batch is one PPO minibatch. Its inactive rows are
    # retained for auditing but must not dilute the active-sequence denominator.
    tu.assign_non_tensor(data, global_batch_size=active)

    # Make masking explicit at this custom-loss boundary. The native kernel
    # receives the packed mask too, so repeated IDs in inactive placeholders
    # are never interpreted as real support categories.
    losses = torch.where(mask, raw_losses, torch.zeros_like(raw_losses))
    metrics = {}
    if mask.any():
        if not torch.isfinite(raw_losses[mask]).all():
            raise ValueError("non-finite JSD loss on selected span")
        metrics["distillation/zero_loss_fraction"] = float((raw_losses[mask] == 0).float().mean().detach())
        for key in ("student_mass", "teacher_mass"):
            values = no_padding_2_padding(model_output[key], data)[mask]
            metrics[f"distillation/{key}"] = float(values.mean().detach())
        metrics["distillation/mean_active_span_tokens"] = float(mask.sum().detach())
    return losses, metrics
