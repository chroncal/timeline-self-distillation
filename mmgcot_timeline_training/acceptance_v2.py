"""Real-model cache and token-boundary acceptance check for formal v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mmgcot_diagnostic.protocol import MODEL, file_hash


def _fingerprint(cache, torch) -> str:
    sha = hashlib.sha256()
    for layer in cache.layers:
        for name, value in sorted(vars(layer).items()):
            if isinstance(value, torch.Tensor):
                sha.update(name.encode())
                sha.update(str(value.dtype).encode())
                sha.update(str(tuple(value.shape)).encode())
                sha.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return sha.hexdigest()


def main(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import xgrammar as xgr
    import mmgcot_timeline_training.formal_train_v2 as formal
    from mmgcot_timeline_training.formal_cache_io import load_cache_bundle

    formal.torch = torch
    formal.xgr = xgr
    record = json.loads(args.record.read_text())
    engine = formal.FormalEngine(args.model, cache_root=args.cache_root)
    state = engine.build_state(record)
    if state["opening_id"] != 8631:
        raise RuntimeError("checked tokenizer opening boundary differs")
    original_fingerprint = {arm: _fingerprint(state[arm], torch) for arm in ("L", "R", "E")}
    cache_paths = list(args.cache_root.iterdir())
    if len(cache_paths) != 1:
        raise RuntimeError("acceptance root must contain one frozen bundle")
    loaded = load_cache_bundle(cache_paths[0], expected_hashes=state["hashes"] if "hashes" in state else None)
    prefix = engine.tokenizer.encode("273,508,", add_special_tokens=False)
    if not prefix:
        raise RuntimeError("empty acceptance coordinate prefix")
    positions_checked = 0
    with torch.no_grad():
        for arm in ("L", "R", "E"):
            for position, token in enumerate([state["opening_id"], *prefix]):
                if position == 0:
                    a = engine._branch(state, arm)
                    b = engine._branch(loaded, arm)
                engine.adapter.enabled = False
                engine._set_rope(state)
                a, logits_a = engine.base.advance(engine.model, a, [token])
                engine._set_rope(state)
                b, logits_b = engine.base.advance(engine.model, b, [token])
                if not torch.equal(logits_a, logits_b):
                    raise RuntimeError(f"BF16 cache reload changed {arm} logits at {position}")
                positions_checked += 1
        engine.adapter.enabled = False
        engine._set_rope(state)
        off_cache = engine._branch(state, "L")
        _, off_logits = engine.base.advance(engine.model, off_cache, [state["opening_id"]])
        engine.adapter.enabled = True
        engine._set_rope(state)
        on_cache = engine._branch(state, "L")
        _, on_logits = engine.base.advance(engine.model, on_cache, [state["opening_id"]])
        if not torch.equal(off_logits, on_logits):
            raise RuntimeError("zero adapter changed same-context opening logits")
    sample = engine.sample_bbox(state, branch="L", seed=20260921, greedy=True)
    teacher = engine.score_teacher(state, "R", sample)
    if sum(item is not None for item in teacher) != sum(sample["numeric_mask"]):
        raise RuntimeError("teacher numeric position alignment changed")
    if {arm: _fingerprint(state[arm], torch) for arm in ("L", "R", "E")} != original_fingerprint:
        raise RuntimeError("master frozen cache was mutated by a branch")
    result = {"record_sha256": file_hash(args.record),
              "cache_metadata_sha256": file_hash(cache_paths[0] / "metadata.json"),
              "opening_id": state["opening_id"],
              "positions_compared": positions_checked,
              "cache_reload_logits_exact": True,
              "zero_adapter_same_context_logits_exact": True,
              "teacher_student_raw_prefix_aligned": True,
              "master_cache_unchanged": True,
              "sample_completed": sample["completed"],
              "sample_valid": sample["valid"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("FORMAL_V2_ACCEPTANCE_OK " + json.dumps(result), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
