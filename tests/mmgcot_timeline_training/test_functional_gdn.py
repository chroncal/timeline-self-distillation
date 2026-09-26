"""Reference checks for multi-token gradient-preserving cached continuation."""

import torch

from mmgcot_timeline_training import formal_train_v2 as base
from mmgcot_timeline_training.expanded_bbox_adapter import LinearLoRA
from mmgcot_timeline_training.functional_gdn import functional_gdn


def _tiny_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["full_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    model.requires_grad_(False)
    adapter = LinearLoRA(model.model.layers[0].self_attn.o_proj, rank=2)
    model.model.layers[0].self_attn.o_proj = adapter
    return model, adapter


def test_functional_cache_matches_native_logits_and_preserves_cross_token_gradient():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    base.torch = torch
    model, adapter = _tiny_model()
    adapter.enabled = False
    with torch.no_grad():
        frozen = model(input_ids=torch.tensor([[2, 3, 4, 5, 6]]), use_cache=True).past_key_values
    adapter.up.weight.data.normal_(std=0.01)
    adapter.enabled = True

    native = base._cache_to(frozen, "cpu")
    with torch.no_grad():
        model(input_ids=torch.tensor([[7]]), past_key_values=native, use_cache=True)
        reference = model(input_ids=torch.tensor([[8]]), past_key_values=native, use_cache=True).logits

    differentiable = base._cache_to(frozen, "cpu")
    with functional_gdn(model, differentiable):
        model(input_ids=torch.tensor([[7]]), past_key_values=differentiable, use_cache=True)
        candidate = model(input_ids=torch.tensor([[8]]), past_key_values=differentiable,
                          use_cache=True).logits
        torch.testing.assert_close(candidate, reference, atol=1e-6, rtol=1e-6)
        candidate[0, -1, 10].backward()
    assert adapter.down.weight.grad is not None
    assert torch.isfinite(adapter.down.weight.grad).all()

    # The first token's adapter must influence the later output through its
    # cached K/V and GDN history even when the adapter is off on token two.
    first_only = base._cache_to(frozen, "cpu")
    adapter.down.weight.grad = None
    with functional_gdn(model, first_only):
        adapter.enabled = True
        model(input_ids=torch.tensor([[7]]), past_key_values=first_only, use_cache=True)
        adapter.enabled = False
        late = model(input_ids=torch.tensor([[8]]), past_key_values=first_only,
                     use_cache=True).logits
        late[0, -1, 10].backward()
    assert adapter.down.weight.grad is not None
    assert adapter.down.weight.grad.abs().max() > 0
