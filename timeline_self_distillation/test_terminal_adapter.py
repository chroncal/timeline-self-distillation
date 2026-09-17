"""CPU integration tests for the terminal query-only Qwen3.5 adapter."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from timeline_self_distillation.terminal_adapter import (
    install_terminal_query_lora,
    score_detached_cache_tokenwise,
    terminal_parameter_whitelist,
)

SEED = 20_260_916


def _tiny_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(SEED)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        dtype="float32",
    )
    return Qwen3_5ForCausalLM(config).eval()


def test_install_is_zero_increment_and_exposes_only_phi_trainables() -> None:
    model = _tiny_model()
    tokens = torch.tensor([[3, 7, 11, 13, 17]])
    with torch.no_grad():
        base_logits = model(tokens, use_cache=False, return_dict=True).logits

    adapter = install_terminal_query_lora(model, rank=8)
    whitelist = terminal_parameter_whitelist(model, adapter)
    assert set(whitelist) == {"q_lora_down.weight", "q_lora_up.weight"}
    assert {id(parameter) for parameter in model.parameters() if parameter.requires_grad} == {
        id(parameter) for parameter in whitelist.values()
    }

    with torch.no_grad():
        adapter.enabled = False
        off_logits = model(tokens, use_cache=False, return_dict=True).logits
        adapter.enabled = True
        zero_logits = model(tokens, use_cache=False, return_dict=True).logits
    torch.testing.assert_close(off_logits, base_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(zero_logits, base_logits, rtol=0.0, atol=0.0)


def test_enabled_adapter_changes_per_head_query_but_not_gate_with_fp32_factors() -> None:
    model = _tiny_model()
    adapter = install_terminal_query_lora(model, rank=8)
    assert adapter.down.weight.dtype == adapter.base.weight.dtype
    assert adapter.up.weight.dtype == adapter.base.weight.dtype
    with torch.no_grad():
        torch.nn.init.normal_(adapter.up.weight, std=0.05)
        hidden = model.model.embed_tokens(torch.tensor([[3, 7]]))
        delta = (adapter(hidden) - adapter.base(hidden)).view(1, 2, 2, 2, 8)
    assert torch.count_nonzero(delta[..., 0, :]) > 0
    assert torch.count_nonzero(delta[..., 1, :]) == 0


def test_tokenwise_detached_cache_scorer_matches_full_and_trains_only_phi() -> None:
    model = _tiny_model()
    adapter = install_terminal_query_lora(model, rank=8)
    # The previous test covers the zero-increment initialization.  Make phi
    # nonzero here so cached/full parity is not a trivial zero-adapter case.
    with torch.no_grad():
        torch.nn.init.normal_(adapter.up.weight, std=0.05)
    prefix = torch.tensor([[3, 7, 11, 13, 17]])
    suffix = torch.tensor([19, 23])
    support = torch.tensor([2, 31, 37])

    adapter.enabled = False
    with torch.no_grad():
        prefill = model(prefix, use_cache=True, logits_to_keep=1, return_dict=True)
    adapter.enabled = True
    cache, cached_support_logits = score_detached_cache_tokenwise(
        model, adapter, prefill.past_key_values, suffix, support
    )

    assert cached_support_logits.shape == (1, 2, 3)
    assert cached_support_logits.requires_grad
    assert cache.get_seq_length() == prefix.shape[1] + suffix.numel()
    assert not cache.layers[0].conv_states.requires_grad
    assert not cache.layers[0].recurrent_states.requires_grad
    assert not cache.layers[1].keys.requires_grad
    assert not cache.layers[1].values.requires_grad

    cached_loss = cached_support_logits[0, -1].sum()
    cached_loss.backward()
    cached_down_grad = adapter.down.weight.grad.detach().clone()
    cached_up_grad = adapter.up.weight.grad.detach().clone()
    assert torch.count_nonzero(cached_down_grad) > 0
    assert torch.count_nonzero(cached_up_grad) > 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

    model.zero_grad(set_to_none=True)
    full = model(torch.cat((prefix, suffix.unsqueeze(0)), dim=1), use_cache=False, return_dict=True).logits
    full_support_logits = full[:, prefix.shape[1] :, support]
    torch.testing.assert_close(cached_support_logits.detach(), full_support_logits.detach(), atol=1e-6, rtol=1e-6)
    full_support_logits[0, -1].sum().backward()
    torch.testing.assert_close(adapter.down.weight.grad, cached_down_grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(adapter.up.weight.grad, cached_up_grad, atol=1e-6, rtol=1e-6)


def test_q_only_adapter_leaves_gdn_state_and_terminal_kv_cache_unchanged() -> None:
    model = _tiny_model()
    adapter = install_terminal_query_lora(model, rank=8)
    with torch.no_grad():
        torch.nn.init.normal_(adapter.up.weight, std=0.05)
    prefix = torch.tensor([[3, 7, 11, 13, 17]])
    suffix = torch.tensor([19, 23])

    def cache_after_suffix(adapter_on: bool):
        adapter.enabled = False
        with torch.no_grad():
            cache = model(prefix, use_cache=True, logits_to_keep=1, return_dict=True).past_key_values
            adapter.enabled = adapter_on
            for token in suffix:
                _ = model(
                    token.reshape(1, 1),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
        return cache

    disabled_cache = cache_after_suffix(adapter_on=False)
    enabled_cache = cache_after_suffix(adapter_on=True)
    assert torch.equal(disabled_cache.layers[0].conv_states, enabled_cache.layers[0].conv_states)
    assert torch.equal(disabled_cache.layers[0].recurrent_states, enabled_cache.layers[0].recurrent_states)
    assert torch.equal(disabled_cache.layers[1].keys, enabled_cache.layers[1].keys)
    assert torch.equal(disabled_cache.layers[1].values, enabled_cache.layers[1].values)


def test_adapter_factors_follow_a_bfloat16_qwen_projection() -> None:
    model = _tiny_model().to(dtype=torch.bfloat16)
    adapter = install_terminal_query_lora(model, rank=8)
    assert adapter.down.weight.dtype is torch.float32
    assert adapter.up.weight.dtype is torch.float32
    with torch.no_grad():
        logits = model(torch.tensor([[3, 7, 11, 13, 17]]), use_cache=False, return_dict=True).logits
    assert logits.dtype is torch.bfloat16


def test_install_supports_qwen_conditional_generation_language_model_path() -> None:
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        dtype="float32",
    )
    model = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=text_config,
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 16,
                "num_position_embeddings": 16,
            },
        )
    ).eval()
    adapter = install_terminal_query_lora(model, rank=8)
    assert model.model.language_model.layers[-1].self_attn.q_proj is adapter
    assert set(terminal_parameter_whitelist(model, adapter)) == {"q_lora_down.weight", "q_lora_up.weight"}
