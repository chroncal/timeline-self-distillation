#!/usr/bin/env python3
"""CPU witnesses for Qwen3.5 GDN cache semantics and a q-only safe boundary.

Run with the project environment:

    PYTHONNOUSERSITE=1 .venv/bin/python timeline_self_distillation/gdn_cache_witness.py

The script constructs only random, tiny models.  It deliberately hides CUDA before
importing torch and never moves a tensor or module off CPU.

It checks three implementation facts in transformers 5.5.3:

1. A GDN cache must be continued one token at a time: a cached two-token call
   does not consume the cached recurrent state, whereas two cached single-token
   calls agree with a full uncached sequence.
2. A gradient-bearing continuation through a GDN cache is unsafe: the static,
   in-place cache update produces an autograd version-counter error even after a
   no-grad prefill.
3. The proposed restricted boundary is safe in this tiny model: all base
   parameters are frozen; phi is a LoRA update to *only* the query half of the
   final full-attention q_proj; prefix cache construction is no-grad and phi is
   off there.  The lower GDN and all cached K/V therefore remain detached, and
   gradients reach only phi through final full attention, frozen MLP and lm_head.

This is an implementation witness, not a full-model/FSDP or multimodal parity
test.  It intentionally does not modify the production model or training code.
"""

from __future__ import annotations

import os

# This must precede importing torch.  It changes this short-lived test process
# only, and makes an accidental CUDA allocation impossible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
)

SEED = 20_260_916
ATOL = 1e-6
RTOL = 1e-6


def tiny_config(*, layer_types: list[str]) -> Qwen3_5TextConfig:
    """A CPU-sized configuration with a valid GDN and a valid full-attn layer."""
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=layer_types,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype="float32",
    )


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def assert_cpu(module: nn.Module) -> None:
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())


def gdn_recurrence_witness() -> None:
    """Show the multi-token GDN cache bug and the one-token recurrence parity."""
    torch.manual_seed(SEED)
    config = tiny_config(layer_types=["linear_attention"])
    gdn = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    assert_cpu(gdn)
    prefix = torch.randn(1, 5, config.hidden_size)
    suffix = torch.randn(1, 2, config.hidden_size)

    with torch.no_grad():
        full_suffix = gdn(torch.cat((prefix, suffix), dim=1))[:, prefix.shape[1] :]

        multi_cache = DynamicCache(config=config)
        _ = gdn(prefix, cache_params=multi_cache)
        multi_suffix = gdn(suffix, cache_params=multi_cache)

        single_cache = DynamicCache(config=config)
        _ = gdn(prefix, cache_params=single_cache)
        single_suffix = torch.cat(
            (
                gdn(suffix[:, :1], cache_params=single_cache),
                gdn(suffix[:, 1:], cache_params=single_cache),
            ),
            dim=1,
        )

    multi_error = max_abs(multi_suffix, full_suffix)
    single_error = max_abs(single_suffix, full_suffix)
    assert not torch.allclose(multi_suffix, full_suffix, atol=1e-5, rtol=1e-5)
    assert torch.allclose(single_suffix, full_suffix, atol=ATOL, rtol=RTOL)
    print(
        "GDN recurrence: "
        f"multi_vs_full_max_abs={multi_error:.9g}; "
        f"single_vs_full_max_abs={single_error:.9g}; "
        f"single_tolerance=atol={ATOL:g},rtol={RTOL:g} PASS"
    )


def gdn_autograd_failure_witness() -> None:
    """Record the expected in-place-cache autograd rejection for one decode token."""
    torch.manual_seed(SEED)
    config = tiny_config(layer_types=["linear_attention"])
    gdn = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    assert_cpu(gdn)
    prefix = torch.randn(1, 5, config.hidden_size)
    suffix = torch.randn(1, 1, config.hidden_size, requires_grad=True)
    cache = DynamicCache(config=config)

    # This is the tempting but unsafe "detached main cache + train suffix" path.
    with torch.no_grad():
        _ = gdn(prefix, cache_params=cache)
    out = gdn(suffix, cache_params=cache)
    assert cache.layers[0].conv_states.requires_grad
    assert cache.layers[0].recurrent_states.requires_grad

    try:
        out.square().mean().backward()
    except RuntimeError as error:
        message = str(error).lower()
        assert "inplace" in message and "modified" in message, str(error)
        print("GDN autograd: expected inplace-cache RuntimeError PASS")
        return
    raise AssertionError("expected autograd to reject the in-place GDN cache update")


class QueryOnlyLoRA(nn.Module):
    """Add phi only to q_proj's per-head query slice, never its gate slice.

    Qwen3.5 does ``q_proj(...).view(..., n_heads, 2 * head_dim)`` then
    chunks each head into ``[query, gate]``.  Thus the packed layout is
    [head0 query, head0 gate, head1 query, head1 gate, ...], not one global
    first-half query block followed by one global gate block.
    """

    def __init__(self, base: nn.Linear, num_heads: int, head_dim: int, rank: int = 2):
        super().__init__()
        self.base = base
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.enabled = False
        self.position_gate: torch.Tensor | None = None
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, num_heads * head_dim, bias=False)
        # Nonzero factors make both phi gradients observable in the witness.
        nn.init.normal_(self.down.weight, std=0.05)
        nn.init.normal_(self.up.weight, std=0.05)

    def set_position_gate(self, gate: torch.Tensor | None) -> None:
        if gate is not None:
            assert gate.ndim == 1
        self.position_gate = gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        if not self.enabled:
            return base_output

        query_delta = self.up(self.down(hidden_states)).view(
            *hidden_states.shape[:-1], self.num_heads, self.head_dim
        )
        if self.position_gate is not None:
            assert self.position_gate.numel() == hidden_states.shape[1]
            query_delta = query_delta * self.position_gate.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ).view(1, -1, 1, 1)

        # Stack query/gate inside each head, then restore q_proj's flat output.
        # The zero second slice is the invariant that phi cannot touch the gate.
        packed_delta = torch.stack((query_delta, torch.zeros_like(query_delta)), dim=-2).reshape(
            *hidden_states.shape[:-1], self.num_heads * 2 * self.head_dim
        )
        return base_output + packed_delta


def make_safe_model() -> tuple[Qwen3_5ForCausalLM, QueryOnlyLoRA]:
    """Make [GDN, full-attn] with every base parameter frozen."""
    torch.manual_seed(SEED)
    config = tiny_config(layer_types=["linear_attention", "full_attention"])
    model = Qwen3_5ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    final_attention = model.model.layers[-1].self_attn
    adapter = QueryOnlyLoRA(
        final_attention.q_proj,
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
    )
    final_attention.q_proj = adapter
    assert_cpu(model)
    return model, adapter


def assert_cache_is_detached(cache: DynamicCache) -> None:
    """Layer 0 is GDN; layer 1 is full-attention K/V in this tiny topology."""
    linear = cache.layers[0]
    full = cache.layers[1]
    assert not linear.conv_states.requires_grad
    assert not linear.recurrent_states.requires_grad
    assert not full.keys.requires_grad
    assert not full.values.requires_grad


def q_slice_witness(adapter: QueryOnlyLoRA, model: Qwen3_5ForCausalLM) -> None:
    """Numerically verify the non-interleaving-safe q/gate packing."""
    with torch.no_grad():
        hidden = model.model.embed_tokens(torch.tensor([[3, 7]]))
        base = adapter.base(hidden)
        adapter.enabled = True
        adapter.set_position_gate(torch.ones(2))
        delta = (adapter(hidden) - base).view(1, 2, adapter.num_heads, 2, adapter.head_dim)
        assert float(delta[..., 1, :].abs().max()) == 0.0
        assert float(delta[..., 0, :].abs().max()) > 0.0
    print("q-only packing: gate_delta_max_abs=0; query_delta_nonzero PASS")


def cached_suffix(
    model: Qwen3_5ForCausalLM,
    adapter: QueryOnlyLoRA,
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, DynamicCache]:
    """No-grad adapter-off prefill, followed by recurrent one-token suffix calls."""
    adapter.enabled = False
    adapter.set_position_gate(None)
    with torch.no_grad():
        prefill = model(prefix, use_cache=True, logits_to_keep=1, return_dict=True)
    cache = prefill.past_key_values
    assert isinstance(cache, DynamicCache)
    assert_cache_is_detached(cache)

    adapter.enabled = True
    logits = []
    for index in range(suffix.shape[1]):
        # A single cached token is essential because layer 0 is GDN.
        adapter.set_position_gate(torch.ones(1))
        output = model(
            suffix[:, index : index + 1],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        logits.append(output.logits)
    stacked_logits = torch.cat(logits, dim=1)
    loss = sum(stacked_logits[0, index, int(target)] for index, target in enumerate(targets))
    assert stacked_logits.requires_grad
    # phi appears only after layer 0; its gradient cannot make either cache live.
    assert_cache_is_detached(cache)
    return stacked_logits, loss, cache


def assert_base_grads_none(model: Qwen3_5ForCausalLM, adapter: QueryOnlyLoRA) -> None:
    phi_ids = {id(adapter.down.weight), id(adapter.up.weight)}
    for parameter in model.parameters():
        if id(parameter) in phi_ids:
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad
            assert parameter.grad is None


def safe_q_only_witness() -> None:
    """Validate cached q-only phi gradients for one and two suffix tokens."""
    model, adapter = make_safe_model()
    prefix = torch.tensor([[3, 7, 11, 13, 17]])
    suffix = torch.tensor([[19, 23]])
    targets = torch.tensor([31, 37])
    q_slice_witness(adapter, model)

    # One-token branch: this is the smallest grad-aware cached decode test.
    one_logits, one_loss, _ = cached_suffix(model, adapter, prefix, suffix[:, :1], targets[:1])
    one_loss.backward()
    one_down_grad = float(adapter.down.weight.grad.abs().max())
    one_up_grad = float(adapter.up.weight.grad.abs().max())
    assert one_down_grad > 0.0 and one_up_grad > 0.0
    assert_base_grads_none(model, adapter)
    print(
        "safe q-only cached suffix=1: "
        f"loss={float(one_loss.detach()):.9g}; "
        f"phi_grad_max=[{one_down_grad:.9g}, {one_up_grad:.9g}] PASS"
    )

    # Two-token branch and an uncached, per-position-gated control on exactly
    # the same input/targets.  Prefix positions receive zero phi in the control.
    model.zero_grad(set_to_none=True)
    cached_logits, cached_loss, _ = cached_suffix(model, adapter, prefix, suffix, targets)
    cached_loss.backward()
    cached_down_grad = adapter.down.weight.grad.detach().clone()
    cached_up_grad = adapter.up.weight.grad.detach().clone()
    assert_base_grads_none(model, adapter)

    model.zero_grad(set_to_none=True)
    adapter.enabled = True
    adapter.set_position_gate(torch.tensor([0.0] * prefix.shape[1] + [1.0] * suffix.shape[1]))
    full = model(torch.cat((prefix, suffix), dim=1), use_cache=False, logits_to_keep=0, return_dict=True)
    full_logits = full.logits[:, prefix.shape[1] :]
    full_loss = sum(full_logits[0, index, int(target)] for index, target in enumerate(targets))
    full_loss.backward()
    assert_base_grads_none(model, adapter)

    logit_error = max_abs(cached_logits.detach(), full_logits.detach())
    down_grad_error = max_abs(cached_down_grad, adapter.down.weight.grad)
    up_grad_error = max_abs(cached_up_grad, adapter.up.weight.grad)
    assert torch.allclose(cached_logits.detach(), full_logits.detach(), atol=ATOL, rtol=RTOL)
    assert torch.allclose(cached_down_grad, adapter.down.weight.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(cached_up_grad, adapter.up.weight.grad, atol=ATOL, rtol=RTOL)
    print(
        "safe q-only cached suffix=2: "
        f"cached_vs_full_logit_max_abs={logit_error:.9g}; "
        f"phi_grad_max_abs=[{down_grad_error:.9g}, {up_grad_error:.9g}] PASS"
    )

    # K/V (and GDN state) are invariant to a q-only update.  This separate
    # no-grad comparison makes the claimed cache boundary directly observable.
    def collect_cache(adapter_on: bool) -> DynamicCache:
        adapter.enabled = False
        adapter.set_position_gate(None)
        with torch.no_grad():
            prefill = model(prefix, use_cache=True, logits_to_keep=1, return_dict=True)
            cache = prefill.past_key_values
            adapter.enabled = adapter_on
            for index in range(suffix.shape[1]):
                adapter.set_position_gate(torch.ones(1))
                _ = model(
                    suffix[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
        assert_cache_is_detached(cache)
        return cache

    disabled_cache = collect_cache(adapter_on=False)
    enabled_cache = collect_cache(adapter_on=True)
    assert torch.equal(disabled_cache.layers[0].conv_states, enabled_cache.layers[0].conv_states)
    assert torch.equal(disabled_cache.layers[0].recurrent_states, enabled_cache.layers[0].recurrent_states)
    assert torch.equal(disabled_cache.layers[1].keys, enabled_cache.layers[1].keys)
    assert torch.equal(disabled_cache.layers[1].values, enabled_cache.layers[1].values)
    print("safe q-only cache: lower GDN state and final full-attn K/V invariant PASS")


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; seed={SEED}")
    gdn_recurrence_witness()
    gdn_autograd_failure_witness()
    safe_q_only_witness()
    print("ALL CPU GDN/cache witnesses PASS")


if __name__ == "__main__":
    main()
