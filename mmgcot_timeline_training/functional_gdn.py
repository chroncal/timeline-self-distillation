"""Graph-preserving one-token cached continuation for Qwen3.5 GDN layers.

Patch only the local model instance within one scoring call. Frozen prefix
cache is copied first; no installed Transformers module is modified.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen


def _assign_recurrent(layer, value, **_):
    # Native cache copy_ rounds the FP32 recurrence to the prefix-cache dtype
    # (BF16 in the actual model). Preserve that forward semantic while keeping
    # a differentiable cast back to the preceding bbox token.
    layer.recurrent_states = value.to(layer.recurrent_states.dtype)
    return layer.recurrent_states


@contextmanager
def functional_gdn(model, cache):
    text_model = getattr(model.model, "language_model", model.model)
    layers = text_model.layers
    originals = []
    for index, block in enumerate(layers):
        if block.layer_type != "linear_attention":
            continue
        gdn = block.linear_attn
        layer = cache.layers[index]
        if not layer.has_previous_state:
            raise RuntimeError(f"GDN cache {index} has no frozen prefix state")
        originals.append((gdn, gdn.causal_conv1d_update, gdn.recurrent_gated_delta_rule))
        layer.update_recurrent_state = MethodType(_assign_recurrent, layer)

        def conv_update(hidden, conv_state, weight, bias, activation, *, layer=layer):
            if activation != "silu":
                raise RuntimeError(f"unexpected GDN activation: {activation}")
            combined = torch.cat((conv_state, hidden), dim=-1).to(weight.dtype)
            layer.conv_states = combined[..., -conv_state.shape[-1]:]
            result = F.conv1d(combined, weight.unsqueeze(1), bias,
                              groups=combined.shape[1])
            return F.silu(result[..., -hidden.shape[-1]:]).to(hidden.dtype)

        gdn.causal_conv1d_update = conv_update
        gdn.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
    try:
        yield
    finally:
        for gdn, conv, recurrent in originals:
            gdn.causal_conv1d_update = conv
            gdn.recurrent_gated_delta_rule = recurrent
