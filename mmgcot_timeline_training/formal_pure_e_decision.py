"""Pure E-OPD B/C experiment: coordinate decisions and last-two-layer QVO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from mmgcot_diagnostic.protocol import MODEL, file_hash, iou, stable_seed
from mmgcot_timeline_training import formal_train_v2 as base
from mmgcot_timeline_training.coordinate_decision import annotate, reverse_kl_on_support


class DecisionEngine(base.FormalEngine):
    def __init__(self, model_path: str, *, supervision_scope: str,
                 adapter_scope: str, cache_root: Path | None):
        from mmgcot_timeline_training.expanded_bbox_adapter import install_extra

        super().__init__(model_path, cache_root=cache_root)
        self.supervision_scope = supervision_scope
        self.adapter_scope = adapter_scope
        self.adapters = ({"23.q": self.adapter} if adapter_scope == "terminal_q"
                         else install_extra(self.model, self.adapter))
        self.parameters = [parameter for adapter in self.adapters.values()
                           for parameter in (adapter.down.weight, adapter.up.weight)]
        self.set_enabled(False)
        self.whitelist()

    def whitelist(self):
        if self.adapter_scope == "terminal_q":
            from timeline_self_distillation.terminal_adapter import terminal_parameter_whitelist
            return terminal_parameter_whitelist(self.model, self.adapter)
        from mmgcot_timeline_training.expanded_bbox_adapter import check_whitelist
        return check_whitelist(self.model, self.adapters)

    def set_enabled(self, value: bool):
        for adapter in self.adapters.values():
            adapter.enabled = value

    def build_state(self, record):
        self.set_enabled(False)
        return super().build_state(record)

    def sample_bbox(self, state, *, branch, seed, greedy=False, forced_prefix_ids=()):
        self.set_enabled(branch == "L")
        rollout = super().sample_bbox(state, branch=branch, seed=seed, greedy=greedy,
                                      forced_prefix_ids=forced_prefix_ids)
        ann = annotate(rollout["token_ids"], rollout["support_ids"], self.tokenizer)
        if rollout["numeric_mask"] != ann["numeric_mask"]:
            raise RuntimeError("new position annotation changed baseline numeric mask")
        rollout.update(ann)
        return rollout

    def score_decision_teacher(self, state, rollout):
        if self.supervision_scope == "numeric":
            return super().score_teacher(state, "E", rollout)
        self.set_enabled(False)
        self._set_rope(state)
        cache = self._branch(state, "E")
        matcher = xgr.GrammarMatcher(self.base.box_grammar, terminate_without_stop_token=True)
        current = state["opening_id"]
        results = []
        with torch.no_grad():
            for token, support_ids, supervise in zip(rollout["token_ids"], rollout["support_ids"],
                                                       rollout["coordinate_decision_mask"], strict=True):
                cache, logits = self.base.advance(self.model, cache, [current])
                if self._support(logits, matcher).cpu().tolist() != support_ids:
                    raise RuntimeError("teacher grammar support differs from student")
                support = torch.tensor(support_ids, device="cuda", dtype=torch.long)
                results.append(logits[0, support].float().detach().cpu() if supervise else None)
                if not matcher.accept_token(int(token)):
                    raise RuntimeError("teacher rejected student raw token")
                current = token
        return results

    def backward_decision(self, state, rollout, teacher, coefficient, batch):
        if self.supervision_scope == "numeric":
            value = super().backward_opd(state, rollout, teacher, coefficient, batch)
            return {"numeric_kl": value, "ending_kl": 0.0, "total_kl": value,
                    "per_decision_kl": value}
        self.set_enabled(True)
        self._set_rope(state)
        cache = self._branch(state, "L")
        count = sum(rollout["numeric_mask"])
        if count < 1:
            raise RuntimeError("rollout has no sampled numeric token")
        current = state["opening_id"]
        numeric_kl = stop_kl = 0.0
        terms = []
        use_graph = self.adapter_scope == "last_two_qvo"
        if use_graph:
            from mmgcot_timeline_training.functional_gdn import functional_gdn
        context = functional_gdn(self.model, cache) if use_graph else _null_context()
        with context:
            for token, support_ids, is_numeric, supervise, teacher_logits in zip(
                rollout["token_ids"], rollout["support_ids"], rollout["numeric_mask"],
                rollout["coordinate_decision_mask"], teacher, strict=True
            ):
                # B's terminal Query has no effect on later K/V or GDN state;
                # C must keep the complete bbox computation graph until backward.
                with torch.set_grad_enabled(use_graph or supervise):
                    cache, logits = self.base.advance(self.model, cache, [current])
                    if supervise:
                        if teacher_logits is None:
                            raise RuntimeError("missing teacher distribution")
                        support = torch.tensor(support_ids, device="cuda", dtype=torch.long)
                        value = reverse_kl_on_support(
                            logits[0, support], teacher_logits.to("cuda")) / count
                        if not torch.isfinite(value):
                            raise FloatingPointError("nonfinite decision KL")
                        if is_numeric:
                            numeric_kl += float(value.detach())
                        else:
                            stop_kl += float(value.detach())
                        if use_graph:
                            terms.append(value)
                        else:
                            (coefficient * value / batch).backward()
                current = token
            if use_graph and terms:
                (coefficient * torch.stack(terms).sum() / batch).backward()
        return {"numeric_kl": numeric_kl, "ending_kl": stop_kl,
                "total_kl": numeric_kl + stop_kl,
                "per_decision_kl": (numeric_kl + stop_kl) * count / max(1, len(terms) if use_graph
                                  else sum(rollout["coordinate_decision_mask"]))}


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def _adapter_weights(engine):
    return {name: {part: getattr(adapter, part).weight.detach().cpu().clone()
                   for part in ("down", "up")}
            for name, adapter in engine.adapters.items()}


def _restore_adapter(engine, weights):
    if set(weights) != set(engine.adapters):
        raise RuntimeError("checkpoint adapter block list changed")
    for name, adapter in engine.adapters.items():
        for part in ("down", "up"):
            getattr(adapter, part).weight.data.copy_(weights[name][part].to("cuda"))


def _cache_fingerprint(cache):
    digest = hashlib.sha256()
    for layer in cache.layers:
        for name, value in sorted(vars(layer).items()):
            if isinstance(value, torch.Tensor):
                digest.update(name.encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit(args):
    """Real BF16 continuation parity and master-cache immutability smoke."""
    if args.adapter_scope != "last_two_qvo":
        raise RuntimeError("BF16 functional continuation audit is for expanded C adapter")
    from mmgcot_timeline_training.functional_gdn import functional_gdn

    rows = base._jsonl(args.selection)
    row = rows[base._order(args.seed, len(rows), 1, 1)[0]]
    frozen = base._find_record(args.prepared, row, 0)
    engine = DecisionEngine(args.model, supervision_scope=args.supervision_scope,
                            adapter_scope=args.adapter_scope, cache_root=args.cache_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["config"]["adapter_scope"] != args.adapter_scope:
        raise RuntimeError("audit checkpoint adapter differs")
    state = engine.build_state(frozen)
    engine._set_rope(state)
    engine.set_enabled(False)
    with torch.no_grad():
        _, original_opening = engine.base.advance(
            engine.model, engine._branch(state, "L"), [state["opening_id"]])
        engine.set_enabled(True)
        _, zero_opening = engine.base.advance(
            engine.model, engine._branch(state, "L"), [state["opening_id"]])
    if not torch.equal(original_opening, zero_opening):
        raise RuntimeError("zero-increment expanded adapter differs from frozen model")
    _restore_adapter(engine, checkpoint["adapter_weights"])
    before = {arm: _cache_fingerprint(state[arm]) for arm in ("L", "E")}
    if not state["has_target"]:
        raise RuntimeError("audit sample lacks target description")
    sample = engine.sample_bbox(state, branch="L", seed=stable_seed(args.seed, "audit"))
    teacher = engine.score_decision_teacher(state, sample)
    if len(teacher) != len(sample["token_ids"]):
        raise RuntimeError("teacher/student positions differ")
    native = engine._branch(state, "L")
    differentiable = engine._branch(state, "L")
    engine.set_enabled(True)
    engine._set_rope(state)
    inputs = [state["opening_id"], *sample["token_ids"][:4]]
    with torch.no_grad():
        original_logits = []
        for token in inputs:
            native, logits = engine.base.advance(engine.model, native, [token])
            original_logits.append(logits.detach().float().cpu())
        with functional_gdn(engine.model, differentiable):
            functional_logits = []
            for token in inputs:
                differentiable, logits = engine.base.advance(engine.model, differentiable, [token])
                functional_logits.append(logits.detach().float().cpu())
    deltas = [float((a - b).abs().max()) for a, b in zip(original_logits, functional_logits)]
    if not all(math.isfinite(x) for x in deltas) or max(deltas) > 0.02:
        raise RuntimeError(f"real BF16 native/functional continuation mismatch: {deltas}")
    after = {arm: _cache_fingerprint(state[arm]) for arm in ("L", "E")}
    if before != after:
        raise RuntimeError("master frozen cache changed during B/C scoring")
    result = {"sample_id": row["sample_id"], "checkpoint_sha256": file_hash(args.checkpoint),
              "zero_increment_initial_logits_identical": True,
              "max_logit_differences": deltas, "master_cache_fingerprints": before,
              "same_student_teacher_raw_token_prefix": True,
              "decision_positions": sum(sample["coordinate_decision_mask"]),
              "numeric_positions": sum(sample["numeric_mask"])}
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("BF16_CONTINUATION_AUDIT_PASS", result["max_logit_differences"], flush=True)


def _protect_recovery_logs(out: Path, start: int, resume: bool):
    if (out / "complete.json").exists():
        raise RuntimeError("run is already complete")
    for name in ("rollouts.jsonl", "steps.jsonl"):
        path = out / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        if not resume:
            raise RuntimeError(f"existing run log requires explicit checkpoint recovery: {path}")
        retained = []
        for line in lines:
            if int(json.loads(line)["step"]) <= start:
                retained.append(line)
        if len(retained) != len(lines):
            stamp = str(time.time_ns())
            archive = out / f"{name}.interrupted.{stamp}"
            archive.write_text("\n".join(lines) + "\n")
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text("\n".join(retained) + ("\n" if retained else ""))
            os.replace(temporary, path)


def train(args):
    rows = base._jsonl(args.selection)
    prepared = json.loads((args.prepared / "protocol.json").read_text())
    if prepared["selection_sha256"] != file_hash(args.selection) or prepared["trajectories_per_image"] != 1:
        raise RuntimeError("frozen training selection or trajectory count changed")
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "pure_e_decision_v1", "supervision_scope": args.supervision_scope,
               "adapter_scope": args.adapter_scope, "activation_scope": "bbox_only", "teacher": "E",
               "objective": "full_support_reverse_kl_only", "sft_coefficient": 0.0,
               "seed": args.seed, "steps": args.steps, "effective_batch": args.batch,
               "lr": args.lr, "lambda_opd": args.lambda_opd,
               "checkpoint_every": args.checkpoint_every, "selection_sha256": file_hash(args.selection),
               "prepared_protocol_sha256": file_hash(args.prepared / "protocol.json"),
               "model": args.model, "cache_root": str(args.cache_root.absolute()) if args.cache_root else None,
               "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "grad_clip": 1.0,
               "reproducibility": base._reproducibility_receipt(args.model),
               "protocol_sha256": file_hash(Path(__file__).resolve().parents[1] /
                                             "configs/mmgcot_pure_e_decision_v1.json"),
               "new_source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in
                                     ("formal_pure_e_decision.py", "coordinate_decision.py",
                                      "expanded_bbox_adapter.py", "functional_gdn.py")}}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != receipt:
            raise RuntimeError("resume configuration changed")
    else:
        config_path.write_text(json.dumps(receipt, indent=2) + "\n")
        (out / "launch_command.json").write_text(json.dumps([sys.executable, *sys.argv], indent=2) + "\n")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = DecisionEngine(args.model, supervision_scope=args.supervision_scope,
                            adapter_scope=args.adapter_scope, cache_root=args.cache_root)
    optimizer = torch.optim.AdamW(engine.parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    order = base._order(args.seed, len(rows), args.steps, args.batch)
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != receipt or checkpoint["sample_order_sha256"] != order_hash:
            raise RuntimeError("checkpoint configuration or sample order mismatch")
        _restore_adapter(engine, checkpoint["adapter_weights"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        start = checkpoint["step"]
        if start >= args.steps:
            raise RuntimeError("resume checkpoint is already at final step")
    _protect_recovery_logs(out, start, bool(args.resume))
    base._append(out / "attempts.jsonl", {"started_at_unix": time.time(), "start_step": start,
                                          "resume_checkpoint": str(args.resume) if args.resume else None,
                                          "command": [sys.executable, *sys.argv],
                                          "config_sha256": file_hash(config_path)})
    for step in range(start, args.steps):
        begun = time.time()
        optimizer.zero_grad(set_to_none=True)
        totals = {"numeric_kl": 0.0, "ending_kl": 0.0, "total_kl": 0.0, "per_decision_kl": 0.0}
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
            teacher = engine.score_decision_teacher(state, sample)
            values = engine.backward_decision(state, sample, teacher, args.lambda_opd, args.batch)
            for key in totals:
                totals[key] += values[key]
            base._append(out / "rollouts.jsonl", {"step": step + 1, "slot": slot,
                "sample_id": row["sample_id"], "seed": seed,
                "token_ids": sample["token_ids"], "support_ids": sample["support_ids"],
                "numeric_mask": sample["numeric_mask"],
                "coordinate_decision_mask": sample["coordinate_decision_mask"],
                "coordinate_index": sample["coordinate_index"], "decision_type": sample["decision_type"],
                "teacher_support_logits": [t.tolist() if t is not None else None for t in teacher],
                "completed": sample["completed"], "valid": sample["valid"],
                "bbox": sample["bbox"], "parse_error": sample["parse_error"],
                "early_offset": state["early_offset"], "has_target": True})
        if any(p.grad is not None for p in engine.model.parameters() if not p.requires_grad):
            raise RuntimeError("frozen backbone acquired gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(engine.parameters, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        engine.whitelist()
        step_data = {"step": step + 1, **{key: val / args.batch for key, val in totals.items()},
                     "weighted_kl": args.lambda_opd * totals["total_kl"] / args.batch,
                     "grad_norm": grad_norm, "missing_target": missing,
                     "invalid_rollout": invalid, "seconds": time.time() - begun,
                     "peak_memory_bytes": torch.cuda.max_memory_allocated()}
        base._append(out / "steps.jsonl", step_data)
        print(f"STEP {args.adapter_scope} seed={args.seed} {step + 1}/{args.steps} "
              f"kl={step_data['total_kl']:.6f} time={step_data['seconds']:.1f}s", flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            base._atomic_checkpoint(out / f"step_{step + 1:04d}.pt", {
                "config": receipt, "step": step + 1, "adapter_weights": _adapter_weights(engine),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), "sample_order_sha256": order_hash})
        if args.stop_after_step and step + 1 >= args.stop_after_step:
            return
    (out / "complete.json").write_text(json.dumps({"step": args.steps,
        "config_sha256": file_hash(config_path), "selection_sha256": file_hash(args.selection),
        "final_checkpoint": str(out / f"step_{args.steps:04d}.pt")}, indent=2) + "\n")


def evaluate(args):
    rows = base._jsonl(args.selection)
    prepared = json.loads((args.prepared / "protocol.json").read_text())
    if prepared["selection_sha256"] != file_hash(args.selection):
        raise RuntimeError("prepared evaluation selection mismatch")
    engine = DecisionEngine(args.model, supervision_scope=args.supervision_scope,
                            adapter_scope=args.adapter_scope, cache_root=args.cache_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["config"]["adapter_scope"] != args.adapter_scope or checkpoint["step"] != 200:
        raise RuntimeError("evaluation requires matching step-200 checkpoint")
    _restore_adapter(engine, checkpoint["adapter_weights"])
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            for ti in range(prepared["trajectories_per_image"]):
                frozen = base._find_record(args.prepared, row, ti)
                state = engine.build_state(frozen)
                for draw in range(5):
                    greedy = draw == 0
                    seed = stable_seed(row["sample_id"], ti, draw, "formal_eval_bbox")
                    pred = engine.sample_bbox(state, branch="L", seed=seed, greedy=greedy)
                    handle.write(json.dumps({"sample_id": row["sample_id"], "image_id": row["image_id"],
                        "task_type": row["task_type"], "trajectory_index": ti,
                        "arm": args.adapter_scope, "seed": args.seed,
                        "mode": "greedy" if greedy else "sample", "draw": 0 if greedy else draw - 1,
                        "bbox": pred["bbox"], "valid": pred["valid"], "completed": pred["completed"],
                        "parse_error": pred["parse_error"], "response_token_ids": pred["token_ids"],
                        "iou": iou(pred["bbox"], row["ground_truth_bbox"], pred["valid"]),
                        "has_target": state["has_target"], "reasoning_length": state["reasoning_length"],
                        "early_offset": state["early_offset"]}, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            print(f"EVAL {args.adapter_scope} {row['sample_id']}", flush=True)
    output.with_suffix(".complete.json").write_text(json.dumps({
        "selection_sha256": file_hash(args.selection),
        "prepared_protocol_sha256": file_hash(args.prepared / "protocol.json"),
        "checkpoint_sha256": file_hash(args.checkpoint), "seed": args.seed,
        "evaluated_images": len(rows), "trajectories_per_image": prepared["trajectories_per_image"],
        "draws_per_trajectory": {"greedy": 1, "sample": 4}, "output_sha256": file_hash(output)},
        indent=2) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train", "eval", "audit"), required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--supervision-scope", choices=("numeric", "coordinate_decision"),
                   default="coordinate_decision")
    p.add_argument("--adapter-scope", choices=("terminal_q", "last_two_qvo"), required=True)
    p.add_argument("--activation-scope", choices=("bbox_only",), default="bbox_only")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-opd", type=float, default=0.1)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--stop-after-step", type=int)
    args = p.parse_args()
    if args.batch < 1 or args.steps < 1 or args.lambda_opd <= 0:
        p.error("invalid training configuration")
    if args.mode in ("eval", "audit") and not args.checkpoint:
        p.error("evaluation and audit require --checkpoint")
    if args.mode == "train" and args.checkpoint:
        p.error("checkpoint is evaluation-only")
    if args.adapter_scope == "last_two_qvo" and args.supervision_scope != "coordinate_decision":
        p.error("expanded adapter is defined for coordinate-decision supervision only")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import xgrammar as xgr
    base.torch = torch
    base.xgr = xgr
    if parsed.mode == "train":
        train(parsed)
    elif parsed.mode == "audit":
        audit(parsed)
    else:
        evaluate(parsed)
