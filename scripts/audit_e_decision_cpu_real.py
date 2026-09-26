"""Run real Qwen3.5-0.8B BF16 cache/gradient checks on CPU while GPUs are busy."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

from mmgcot_diagnostic.protocol import MODEL, file_hash
from mmgcot_timeline_training import formal_train_v2 as base
from mmgcot_timeline_training.expanded_bbox_adapter import check_whitelist, install_extra
from mmgcot_timeline_training.formal_cache_io import load_cache_bundle
from mmgcot_timeline_training.functional_gdn import functional_gdn
from timeline_self_distillation.terminal_adapter import install_terminal_query_lora


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "outputs/research_experiments/mmgcot_timeline_training/formal_v2_prepared/train/prefix_cache"
OUTPUT = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/cpu_real_audit.json"


def advance(model, cache, token):
    result = model(input_ids=torch.tensor([[token]], dtype=torch.long),
                   past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
    return result.past_key_values, result.logits[:, -1, :]


def main():
    torch.set_num_threads(4)
    base.torch = torch
    cache_path = sorted(CACHE_ROOT.glob("*/metadata.json"))[0].parent
    bundle = load_cache_bundle(cache_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    q23 = install_terminal_query_lora(model, rank=8)
    adapters = install_extra(model, q23)
    check_whitelist(model, adapters)
    model.model.rope_deltas = bundle["rope_deltas"].clone()
    for adapter in adapters.values():
        adapter.enabled = False
    with torch.no_grad():
        _, off = advance(model, base._cache_to(bundle["L"], "cpu"), bundle["opening_id"])
        for adapter in adapters.values():
            adapter.enabled = True
        _, zero = advance(model, base._cache_to(bundle["L"], "cpu"), bundle["opening_id"])
    if not torch.equal(off, zero):
        raise RuntimeError("real BF16 zero-increment adapter changes first bbox logits")
    for adapter in adapters.values():
        adapter.up.weight.data.normal_(std=0.001)
    native = base._cache_to(bundle["L"], "cpu")
    graph = base._cache_to(bundle["L"], "cpu")
    tokens = [bundle["opening_id"], 19, 24]
    with torch.no_grad():
        native_logits = []
        for token in tokens:
            native, logits = advance(model, native, token)
            native_logits.append(logits.detach().float())
        graph_logits = []
        with functional_gdn(model, graph):
            for token in tokens:
                graph, logits = advance(model, graph, token)
                graph_logits.append(logits.detach().float())
    differences = [float((a - b).abs().max()) for a, b in zip(native_logits, graph_logits)]
    if max(differences) > 0.02:
        raise RuntimeError(f"real BF16 functional/native logits disagree: {differences}")
    graph = base._cache_to(bundle["L"], "cpu")
    for adapter in adapters.values():
        adapter.enabled = True
    with functional_gdn(model, graph):
        graph, _ = advance(model, graph, tokens[0])
        graph, _ = advance(model, graph, tokens[1])
        for adapter in adapters.values():
            adapter.enabled = False
        graph, final_logits = advance(model, graph, tokens[2])
        final_logits[0, 10].float().backward()
    q19_down = adapters["19.q"].down.weight.grad
    if q19_down is None or not torch.isfinite(q19_down).all() or q19_down.abs().max() <= 0:
        raise RuntimeError("real BF16 later loss does not reach earlier Q19 adapter")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "cache_path": str(cache_path), "cache_metadata_sha256": file_hash(cache_path / "metadata.json"),
        "model_config_sha256": file_hash(Path(MODEL) / "config.json"),
        "zero_increment_exact": True, "native_vs_functional_max_logit_diff": differences,
        "cross_token_q19_down_gradient_max": float(q19_down.abs().max()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }, indent=2) + "\n")
    print("CPU_REAL_BF16_AUDIT_PASS", differences, flush=True)


if __name__ == "__main__":
    main()
