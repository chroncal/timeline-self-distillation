"""Last-two-full-attention Q/V/O bbox-only adapters."""

from __future__ import annotations

import torch
from torch import nn

from timeline_self_distillation.terminal_adapter import TerminalQueryLoRA


class LinearLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = base
        self.enabled = False
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.enabled:
            result = result + self.up(self.down(x.float())).to(result.dtype)
        return result


def install_extra(model: nn.Module, terminal_q: TerminalQueryLoRA) -> dict[str, nn.Module]:
    layers = model.model.language_model.layers
    config = model.config.text_config
    adapters: dict[str, nn.Module] = {"23.q": terminal_q}
    if len(layers) != 24 or layers[19].layer_type != "full_attention" or layers[23].layer_type != "full_attention":
        raise RuntimeError("expected Qwen3.5-0.8B full attention layers 19 and 23")
    for layer_id in (19, 23):
        attn = layers[layer_id].self_attn
        if layer_id == 19:
            q = TerminalQueryLoRA(attn.q_proj, num_heads=config.num_attention_heads,
                                  head_dim=config.head_dim, rank=8)
            attn.q_proj = q
            adapters["19.q"] = q
        for part in ("v", "o"):
            name = f"{part}_proj"
            adapter = LinearLoRA(getattr(attn, name))
            setattr(attn, name, adapter)
            adapters[f"{layer_id}.{part}"] = adapter
    for adapter in adapters.values():
        adapter.enabled = False
    check_whitelist(model, adapters)
    return adapters


def check_whitelist(model: nn.Module, adapters: dict[str, nn.Module]) -> dict[str, nn.Parameter]:
    params = {f"{name}.{part}": getattr(adapter, part).weight
              for name, adapter in adapters.items() for part in ("down", "up")}
    if len(params) != 12 or sum(p.numel() for p in params.values()) != 122880:
        raise RuntimeError("expanded adapter parameter count changed")
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != {id(p) for p in params.values()}:
        raise RuntimeError("trainable parameter whitelist changed")
    return params
