from types import SimpleNamespace

import torch
from torch import nn

from mmgcot_timeline_training.expanded_bbox_adapter import check_whitelist, install_extra
from timeline_self_distillation.terminal_adapter import install_terminal_query_lora


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1024, 4096, bias=False)
        self.v_proj = nn.Linear(1024, 512, bias=False)
        self.o_proj = nn.Linear(2048, 1024, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_type = "full_attention"
        self.self_attn = _Attention()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        layers = nn.ModuleList([nn.Identity() for _ in range(24)])
        layers[19] = _Layer()
        layers[23] = _Layer()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = layers
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_attention_heads=8, head_dim=256))


def test_expanded_adapter_whitelist_zero_increment_and_q_gate():
    model = _Model()
    terminal = install_terminal_query_lora(model, rank=8)
    adapters = install_extra(model, terminal)
    assert len(adapters) == 6
    params = check_whitelist(model, adapters)
    assert len(params) == 12
    assert sum(p.numel() for p in params.values()) == 122880
    x = torch.randn(1, 1, 1024)
    for layer_id in (19, 23):
        q = adapters[f"{layer_id}.q"]
        frozen = q.base(x)
        q.enabled = True
        torch.testing.assert_close(q(x), frozen, atol=0, rtol=0)
        q.up.weight.data.normal_(std=0.01)
        difference = (q(x) - frozen).view(1, 1, 8, 2, 256)
        assert difference[..., 0, :].abs().max() > 0
        assert torch.count_nonzero(difference[..., 1, :]) == 0
