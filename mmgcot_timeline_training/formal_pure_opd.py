"""Pure R/E OPD ablation on the frozen MM-GCoT v2 trajectories.

This module intentionally leaves formal_train_v2.py unchanged so the original
nine-run experiment and its source-hash checks remain reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from mmgcot_diagnostic.protocol import MODEL, file_hash, stable_seed
from mmgcot_timeline_training import formal_train_v2 as base


def train(args: argparse.Namespace) -> None:
    from timeline_self_distillation.terminal_adapter import terminal_parameter_whitelist

    rows = base._jsonl(args.selection)
    prepared = json.loads((args.prepared / "protocol.json").read_text())
    if prepared["selection_sha256"] != file_hash(args.selection):
        raise RuntimeError("prepared trajectory selection differs from training selection")
    if prepared["trajectories_per_image"] != 1:
        raise RuntimeError("training requires one fixed trajectory per image")
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "formal_pure_opd_v1", "arm": args.arm, "objective": "reverse_kl_only",
        "sft_coefficient": 0.0, "seed": args.seed, "steps": args.steps,
        "effective_batch": args.batch, "checkpoint_every": args.checkpoint_every,
        "lr": args.lr, "lambda_opd": args.lambda_opd,
        "selection_sha256": file_hash(args.selection),
        "prepared_protocol_sha256": file_hash(args.prepared / "protocol.json"),
        "model": args.model, "rank": 8, "optimizer": "AdamW",
        "cache_root": str(args.cache_root.absolute()) if args.cache_root else None,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
        "grad_clip": 1.0, "semantic_review_policy": "report_only",
        "reproducibility": base._reproducibility_receipt(args.model),
        "pure_trainer_sha256": file_hash(Path(__file__)),
    }
    receipt_path = out / "config.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != receipt:
            raise RuntimeError("resume configuration changed")
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        (out / "launch_command.json").write_text(
            json.dumps([sys.executable, *sys.argv], ensure_ascii=False, indent=2) + "\n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = base.FormalEngine(args.model, cache_root=args.cache_root)
    optimizer = torch.optim.AdamW(engine.parameters, lr=args.lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0)
    order = base._order(args.seed, len(rows), args.steps, args.batch)
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != receipt or checkpoint["sample_order_sha256"] != order_hash:
            raise RuntimeError("resume checkpoint configuration or sample order differs")
        engine.adapter.down.weight.data.copy_(checkpoint["down"].to("cuda"))
        engine.adapter.up.weight.data.copy_(checkpoint["up"].to("cuda"))
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        start_step = checkpoint["step"]
        if start_step >= args.steps:
            raise RuntimeError("resume checkpoint already at final step")
    base._append(out / "attempts.jsonl", {
        "started_at_unix": time.time(), "resume_checkpoint": str(args.resume) if args.resume else None,
        "start_step": start_step, "command": [sys.executable, *sys.argv],
        "config_sha256": file_hash(receipt_path),
    })

    for step in range(start_step, args.steps):
        begun = time.time()
        optimizer.zero_grad(set_to_none=True)
        kl_sum = 0.0
        missing = invalid = 0
        for slot in range(args.batch):
            row = rows[order[step * args.batch + slot]]
            frozen = base._find_record(args.prepared, row, 0)
            state = engine.build_state(frozen)
            if not state["has_target"]:
                missing += 1
                continue
            seed = stable_seed(args.seed, step, slot, row["sample_id"], "bbox_rollout")
            sample = engine.sample_bbox(state, branch="L", seed=seed)
            invalid += int(not sample["valid"])
            teacher_branch = "R" if args.arm == "r_opd_pure" else "E"
            teacher = engine.score_teacher(state, teacher_branch, sample)
            kl = engine.backward_opd(state, sample, teacher, args.lambda_opd, args.batch)
            kl_sum += kl
            base._append(out / "rollouts.jsonl", {
                "step": step + 1, "slot": slot, "sample_id": row["sample_id"],
                "arm": args.arm, "seed": seed, "token_ids": sample["token_ids"],
                "support_ids": sample["support_ids"], "numeric_mask": sample["numeric_mask"],
                "teacher_support_logits": [x.cpu().tolist() if x is not None else None for x in teacher],
                "completed": sample["completed"], "valid": sample["valid"],
                "bbox": sample["bbox"], "parse_error": sample["parse_error"],
                "early_offset": state["early_offset"], "has_target": True,
            })
        if any(p.grad is not None for p in engine.model.parameters() if not p.requires_grad):
            raise RuntimeError("frozen backbone acquired gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(engine.parameters, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        terminal_parameter_whitelist(engine.model, engine.adapter)
        base._append(out / "steps.jsonl", {
            "step": step + 1, "sft": 0.0, "kl": kl_sum / args.batch,
            "weighted_kl": args.lambda_opd * kl_sum / args.batch,
            "loss": args.lambda_opd * kl_sum / args.batch,
            "grad_norm": grad_norm, "missing_target": missing,
            "invalid_rollout": invalid, "seconds": time.time() - begun,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        })
        print(f"STEP {args.arm} seed={args.seed} {step + 1}/{args.steps} "
              f"kl={kl_sum / args.batch:.5f} time={time.time() - begun:.1f}s", flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            base._atomic_checkpoint(out / f"step_{step + 1:04d}.pt", {
                "config": receipt, "step": step + 1,
                "down": engine.adapter.down.weight.detach().cpu(),
                "up": engine.adapter.up.weight.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "sample_order_sha256": order_hash,
            })
    (out / "complete.json").write_text(json.dumps({
        "step": args.steps, "config_sha256": file_hash(receipt_path),
        "selection_sha256": file_hash(args.selection),
        "final_checkpoint": str(out / f"step_{args.steps:04d}.pt"),
    }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--arm", choices=("r_opd_pure", "e_opd_pure"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-opd", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 1 or args.lambda_opd <= 0:
        parser.error("invalid training configuration")
    if args.mode == "eval" and not args.checkpoint:
        parser.error("evaluation requires --checkpoint")
    if args.mode == "train" and (args.checkpoint or args.limit):
        parser.error("checkpoint and limit are evaluation-only")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # CUDA_VISIBLE_DEVICES must be set before importing torch/xgrammar: the
    # latter may initialize CUDA during import on this environment.
    import torch
    import xgrammar as xgr
    # FormalEngine's methods resolve these modules from formal_train_v2 globals.
    base.torch = torch
    base.xgr = xgr
    train(parsed) if parsed.mode == "train" else base.evaluate(parsed)
