# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Hybrid Standard OPD and routed suffix teacher-forcing losses."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import (
    DistillationLossSettings,
    compute_forward_kl_topk,
    register_distillation_loss,
)
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig
from verl.workers.utils.padding import no_padding_2_padding

STANDARD_REPAIR_KIND = 0
LOCALIZATION_REPAIR_KIND = 1
REFERENT_REPAIR_KIND = 2


def _masked_sequence_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average tokens within each non-empty sample, then average samples."""
    token_counts = mask.sum(dim=-1)
    sample_mask = token_counts > 0
    if not sample_mask.any():
        return losses.new_zeros(())
    sample_losses = (losses * mask).sum(dim=-1) / token_counts.clamp_min(1)
    return sample_losses[sample_mask].mean()


@register_distillation_loss(
    DistillationLossSettings(
        names=["routed_forward_kl_ce"],
        use_topk=True,
        requires_token_logprobs=True,
    )
)  # type: ignore[arg-type]
def compute_routed_forward_kl_ce(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    """Use native top-k forward-KL OPD on standard samples and suffix CE on repairs.

    ``response_mask`` is the label mask. Routed samples must set it to zero on
    the preserved Student prefix and one only on the accepted Teacher suffix.
    ``repair_kind`` is a per-sample integer: 0 standard, 1 localization, and 2
    referent. Teacher log-probabilities are deliberately ignored for routed
    tokens; their loss is the target-token negative Student log-probability.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    standard_losses, standard_metrics = compute_forward_kl_topk(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    response_mask = response_mask.bool()

    repair_kind = tu.get(data, "repair_kind")
    if repair_kind is None:
        repair_kind = torch.zeros(student_log_probs.shape[0], dtype=torch.long, device=student_log_probs.device)
    elif not isinstance(repair_kind, torch.Tensor):
        repair_kind = torch.as_tensor(repair_kind, dtype=torch.long, device=student_log_probs.device)
    repair_kind = repair_kind.to(device=student_log_probs.device, dtype=torch.long).reshape(-1)

    if student_log_probs.shape != standard_losses.shape or student_log_probs.shape != response_mask.shape:
        raise ValueError(
            "Expected Student log-probs, Standard OPD losses, and response mask shapes to match, got "
            f"{student_log_probs.shape}, {standard_losses.shape}, and {response_mask.shape}."
        )
    if repair_kind.shape[0] != student_log_probs.shape[0]:
        raise ValueError("repair_kind must contain exactly one code per sample")
    if not torch.isin(repair_kind, torch.tensor([0, 1, 2], device=repair_kind.device)).all():
        raise ValueError("repair_kind codes must be 0 (standard), 1 (localization), or 2 (referent)")

    suffix_ce = -student_log_probs
    routed_sample = repair_kind.ne(STANDARD_REPAIR_KIND).unsqueeze(-1)
    losses = torch.where(routed_sample, suffix_ce, standard_losses)

    standard_mask = response_mask & repair_kind.eq(STANDARD_REPAIR_KIND).unsqueeze(-1)
    localization_mask = response_mask & repair_kind.eq(LOCALIZATION_REPAIR_KIND).unsqueeze(-1)
    referent_mask = response_mask & repair_kind.eq(REFERENT_REPAIR_KIND).unsqueeze(-1)
    metrics = {
        **standard_metrics,
        "routed/standard_opd_loss": Metric(AggregationType.MEAN, _masked_sequence_mean(standard_losses, standard_mask)),
        "routed/localization_loss": Metric(AggregationType.MEAN, _masked_sequence_mean(suffix_ce, localization_mask)),
        "routed/referent_loss": Metric(AggregationType.MEAN, _masked_sequence_mean(suffix_ce, referent_mask)),
    }
    return losses, metrics
