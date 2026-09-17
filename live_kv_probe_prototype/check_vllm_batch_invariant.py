#!/usr/bin/env python3
"""Minimal Qwen3.5/GDN vLLM batch-invariance initialization witness."""

from __future__ import annotations


def main() -> None:
    import torch
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    print("VERSIONS", torch.__version__, vllm.__version__, flush=True)
    args = AsyncEngineArgs(
        model="/mnt/sda/sujingyang/models/Qwen3.5-0.8B",
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        max_model_len=18433,
        max_num_batched_tokens=24576,
        max_num_seqs=32,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1},
        seed=260600564,
    )
    engine = AsyncLLM.from_engine_args(args)
    print("ENGINE_STARTED", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
