"""Terminal-layer query-only LoRA for cached Qwen3.5 OPD scoring.

The module deliberately has a narrow boundary: it can wrap only the final
full-attention ``q_proj`` of a Qwen3.5 CausalLM or ConditionalGeneration
model.  It does not modify K, V, or the per-head attention gate, so a no-grad
prefix cache remains independent of the adapter.  Its tokenwise scorer returns
only the support logits required by the OPD loss and never infers or inserts a
boundary token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

__all__ = [
    "TerminalQueryLoRA",
    "install_terminal_query_lora",
    "score_detached_cache_tokenwise",
    "terminal_parameter_whitelist",
]


class TerminalQueryLoRA(nn.Module):
    """A zero-increment FP32 rank-``r`` update to q_proj's query slice.

    Keeping LoRA factors in FP32 gives AdamW FP32 parameters/gradients for a
    BF16 Qwen backbone.  The computed update is cast to q_proj's output dtype
    immediately before it is added, so the frozen model's attention interface
    still receives its native BF16 (or FP32) activations.
    """

    def __init__(self, base: nn.Linear, *, num_heads: int, head_dim: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        expected_out_features = num_heads * 2 * head_dim
        if base.out_features != expected_out_features:
            raise ValueError(
                "Qwen3.5 q_proj must pack [query, gate] within every head: "
                f"expected {expected_out_features} outputs, got {base.out_features}"
            )
        self.base = base
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.enabled = True
        factory_kwargs = {"device": base.weight.device, "dtype": torch.float32}
        self.down = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(rank, num_heads * head_dim, bias=False, **factory_kwargs)
        # Standard LoRA zero increment: A is random but B is zero, so the
        # wrapped model is initially bit-identical while B receives a gradient.
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        if not self.enabled:
            return base_output

        query_delta = self.up(self.down(hidden_states.to(dtype=torch.float32))).view(
            *hidden_states.shape[:-1], self.num_heads, self.head_dim
        )
        # Qwen3.5 views q_proj as [..., heads, 2 * head_dim] before chunking
        # the final dimension into query and gate.  Stack within each head,
        # rather than treating the global first half as all query values.
        packed_delta = torch.stack((query_delta, torch.zeros_like(query_delta)), dim=-2).reshape(
            *hidden_states.shape[:-1], self.num_heads * 2 * self.head_dim
        )
        return base_output + packed_delta.to(dtype=base_output.dtype)


def _terminal_qwen_components(model: nn.Module) -> tuple[nn.Module, nn.Module, object]:
    """Get final text layer, its attention module, and text config for both HF wrappers."""
    try:
        outer_model = model.model
        if hasattr(outer_model, "language_model"):
            # Qwen3_5ForConditionalGeneration / multimodal wrapper.
            text_model = outer_model.language_model
            text_config = model.config.text_config
        else:
            # Qwen3_5ForCausalLM / text-only wrapper.
            text_model = outer_model
            text_config = model.config
        final_layer = text_model.layers[-1]
        attention = final_layer.self_attn
    except (AttributeError, IndexError) as error:
        raise TypeError(
            "model must be Qwen3.5 CausalLM or ConditionalGeneration with a final text self_attn layer"
        ) from error
    return final_layer, attention, text_config


def install_terminal_query_lora(model: nn.Module, *, rank: int = 8) -> TerminalQueryLoRA:
    """Freeze ``model`` and install a query-only adapter on its final full-attn layer."""
    final_layer, attention, text_config = _terminal_qwen_components(model)
    if getattr(final_layer, "layer_type", None) != "full_attention":
        raise ValueError("terminal adapter requires the final Qwen3.5 layer to be full_attention")
    if isinstance(attention.q_proj, TerminalQueryLoRA):
        raise ValueError("terminal query LoRA is already installed")
    if not isinstance(attention.q_proj, nn.Linear):
        raise TypeError("terminal self-attention q_proj must be nn.Linear")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = TerminalQueryLoRA(
        attention.q_proj,
        num_heads=text_config.num_attention_heads,
        head_dim=text_config.head_dim,
        rank=rank,
    )
    attention.q_proj = adapter
    return adapter


def terminal_parameter_whitelist(model: nn.Module, adapter: TerminalQueryLoRA) -> Mapping[str, nn.Parameter]:
    """Return, and verify, the only two parameters an OPD optimizer may update."""
    whitelist: dict[str, nn.Parameter] = {
        "q_lora_down.weight": adapter.down.weight,
        "q_lora_up.weight": adapter.up.weight,
    }
    expected_ids = {id(parameter) for parameter in whitelist.values()}
    actual_trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual_trainable_ids != expected_ids:
        raise RuntimeError("model trainables are not exactly the terminal q-only LoRA whitelist")
    return whitelist


def _one_dimensional_token_ids(
    token_ids: torch.Tensor | Sequence[int], *, name: str, device: torch.device, vocab_size: int
) -> torch.Tensor:
    """Validate a single sequence of vocabulary IDs without silently casting."""
    ids = torch.as_tensor(token_ids, device=device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional token-ID sequence")
    if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
        raise TypeError(f"{name} must contain integer token IDs")
    ids = ids.to(dtype=torch.long)
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise ValueError(f"{name} contains an ID outside [0, {vocab_size})")
    return ids


def _require_detached_cache(cache: object) -> None:
    """Fail before an unsafe graph-bearing cache is mutated by the scorer."""
    try:
        layers = cache.layers
    except AttributeError as error:
        raise TypeError("cache must expose Qwen/transformers cache.layers") from error
    for layer_index, layer in enumerate(layers):
        for attribute, value in vars(layer).items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise ValueError(f"cache layer {layer_index}.{attribute} must be detached before scoring")


def score_detached_cache_tokenwise(
    model: nn.Module,
    adapter: TerminalQueryLoRA,
    cache: object,
    token_ids: torch.Tensor | Sequence[int],
    support_token_ids: torch.Tensor | Sequence[int],
) -> tuple[object, torch.Tensor]:
    """Advance one cached token per forward and return only requested support logits.

    ``cache`` must already represent the caller-selected boundary; this helper
    never inserts an opening/closing/bbox token and deliberately does not infer
    any grammar boundary.  Every supplied token is advanced with a distinct
    ``[B=1, T=1]`` forward, which is required by Qwen3.5 GDN recurrence.

    All model parameters except ``adapter`` must be frozen (verified through
    :func:`terminal_parameter_whitelist`).  With the adapter restricted to the
    final full-attention query slice, the lower GDN state and full-attention
    K/V stay detached while the selected logits retain a phi gradient.  The
    adapter's current ``enabled`` value is honored; callers can use it to score
    base or adapted distributions without replacing the wrapper.  The cache is
    mutated in place, so callers must give this function a branch-owned cache,
    never the master cache.  For ConditionalGeneration, callers must also set
    ``model.model.rope_deltas`` to the value associated with that cache before
    calling; this helper does not snapshot or restore M-RoPE state.

    Args:
        token_ids: One-dimensional, non-empty IDs to consume from the supplied
            boundary.  The returned row ``i`` is the next-token logits after
            consuming ``token_ids[i]``.
        support_token_ids: One-dimensional, non-empty IDs to select from the
            full vocabulary.  No vocabulary masking or re-normalization occurs.

    Returns:
        The updated cache and selected logits with shape ``[1, T, K]``.
    """
    terminal_parameter_whitelist(model, adapter)
    _require_detached_cache(cache)
    try:
        device = next(model.parameters()).device
        _, _, text_config = _terminal_qwen_components(model)
        vocab_size = int(text_config.vocab_size)
    except (AttributeError, StopIteration) as error:
        raise TypeError("model must expose parameters and config.vocab_size") from error
    tokens = _one_dimensional_token_ids(token_ids, name="token_ids", device=device, vocab_size=vocab_size)
    support = _one_dimensional_token_ids(
        support_token_ids, name="support_token_ids", device=device, vocab_size=vocab_size
    )

    selected_logits: list[torch.Tensor] = []
    for token_id in tokens:
        output = model(
            input_ids=token_id.reshape(1, 1),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = output.past_key_values
        _require_detached_cache(cache)
        selected_logits.append(output.logits[:, -1, :].index_select(-1, support))
    return cache, torch.stack(selected_logits, dim=1)
