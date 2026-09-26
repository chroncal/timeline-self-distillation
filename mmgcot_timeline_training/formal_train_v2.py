"""Three-arm terminal-bbox training on frozen MM-GCoT v3p5 trajectories.

Inference and bridge generation live in prepare_v2.py.  This module never
feeds the reference box into an inference prompt.  The reference is consumed
only by the SFT loss and evaluation metric.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

from mmgcot_diagnostic.protocol import MODEL, file_hash, iou, parse_box, stable_seed
from mmgcot_timeline_training.prepare_v2 import record_path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _cache_to(cache: object, device: str) -> object:
    """Copy layer objects and all tensor state, preserving native dtypes."""
    result = copy.copy(cache)
    result.layers = []
    for original in cache.layers:
        layer = copy.copy(original)
        for key, value in vars(original).items():
            if isinstance(value, torch.Tensor):
                setattr(layer, key, value.detach().to(device, copy=True))
            elif isinstance(value, torch.device):
                setattr(layer, key, torch.device(device))
        result.layers.append(layer)
    return result


def _cache_bytes(cache: object) -> int:
    return sum(value.numel() * value.element_size() for layer in cache.layers
               for value in vars(layer).values() if isinstance(value, torch.Tensor))


class FormalEngine:
    def __init__(self, model_path: str, *, cache_root: Path | None = None):
        import mmgcot_diagnostic.run as diagnostic_run
        from timeline_self_distillation.terminal_adapter import (
            install_terminal_query_lora, terminal_parameter_whitelist)

        diagnostic_run.torch = torch
        diagnostic_run.xgr = xgr
        self.base = diagnostic_run.Engine(model_path)
        self.model = self.base.model
        self.tokenizer = self.base.tokenizer
        self.adapter = install_terminal_query_lora(self.model, rank=8)
        self.parameters = list(terminal_parameter_whitelist(self.model, self.adapter).values())
        if sum(p.numel() for p in self.parameters) != 24576:
            raise RuntimeError("terminal trainable parameter count changed")
        self.cached: dict[str, dict[str, Any]] = {}
        self.cache_root = cache_root
        self.cache_hashes = None
        if cache_root is not None:
            model_root = Path(model_path)
            self.cache_hashes = {
                "model_config": file_hash(model_root / "config.json"),
                "model_weights": file_hash(next(model_root.glob("model.safetensors-*.safetensors"))),
                "tokenizer": file_hash(model_root / "tokenizer.json"),
                "trainer_code": file_hash(Path(__file__)),
            }

    def _set_rope(self, state: dict[str, Any]) -> None:
        self.model.model.rope_deltas = state["rope_deltas"].to("cuda").clone()

    def _branch(self, state: dict[str, Any], arm: str) -> object:
        return _cache_to(state[arm], "cuda")

    def build_state(self, record: dict[str, Any]) -> dict[str, Any]:
        from mmgcot_diagnostic.run import sentence_offsets
        from mmgcot_timeline_training.formal_contexts import build_formal_context

        key = f"{record['sample_id']}:{record['trajectory_index']}"
        if key in self.cached:
            return self.cached[key]
        if file_hash(record["image_path"]) != record["image_sha256"]:
            raise RuntimeError(f"frozen image hash changed: {key}")
        reasoning = [int(v) for v in record["reasoning_token_ids"]]
        contract = build_formal_context(
            record, self.tokenizer, sentence_offsets(self.tokenizer, reasoning))
        suffix_hash = hashlib.sha256(contract.suffix_text.encode()).hexdigest()
        cache_hashes = {**(self.cache_hashes or {}),
                        "trajectory": contract.trajectory.trajectory_sha256,
                        "suffix": suffix_hash}
        cache_path = None
        if self.cache_root is not None:
            digest = hashlib.sha256(json.dumps(cache_hashes, sort_keys=True).encode()).hexdigest()[:24]
            cache_path = self.cache_root / digest
            if cache_path.exists():
                from mmgcot_timeline_training.formal_cache_io import load_cache_bundle

                loaded = load_cache_bundle(cache_path, expected_hashes=cache_hashes)
                if loaded["opening_id"] != contract.opening_token_id:
                    raise RuntimeError("persisted cache opening differs from tokenizer contract")
                state = {**loaded, "early_offset": contract.early_offset,
                         "reasoning_length": len(reasoning),
                         "has_target": contract.has_description,
                         "suffix_sha256": suffix_hash,
                         "cache_bytes": sum(_cache_bytes(loaded[arm]) for arm in ("L", "R", "E"))}
                self.cached[key] = state
                return state
        self.adapter.enabled = False
        with torch.no_grad():
            c0, _, prompt_ids, rendered = self.base.prefill({
                "image_path": record["image_path"], "question": record["question"]})
            if prompt_ids != record["prompt_token_ids"] or rendered != record["rendered_prompt"]:
                raise RuntimeError(f"frozen prompt drift: {key}")
            rope = self.base.rope_deltas.detach().cpu().clone()
            k = contract.early_offset
            has_target = contract.has_description
            suffix = contract.suffix_text
            suffix_ids = list(contract.suffix_token_ids)
            opening = contract.opening_token_id
            sources = {arm: list(contract.arm_prefixes[arm].reasoning_prefix_ids)
                       for arm in ("L", "R", "E")}
            states: dict[str, Any] = {}
            for arm, prefix in sources.items():
                self.base.restore_rope()
                before = self.base.fork(c0)
                if prefix:
                    before, _ = self.base.advance(self.model, before, prefix)
                before, _ = self.base.advance(self.model, before, suffix_ids[:-1])
                states[arm] = _cache_to(before, "cpu")
        state = {**states, "rope_deltas": rope, "opening_id": opening,
                 "early_offset": k, "reasoning_length": len(reasoning),
                 "has_target": bool(has_target), "suffix_sha256": suffix_hash,
                 "hashes": cache_hashes,
                 "cache_bytes": sum(_cache_bytes(value) for value in states.values())}
        if cache_path is not None:
            from mmgcot_timeline_training.formal_cache_io import save_cache_bundle

            self.cache_root.mkdir(parents=True, exist_ok=True)
            # Leave a substantial reserve on this shared filesystem. The
            # in-memory cache remains usable even when persistence is skipped.
            if shutil.disk_usage(self.cache_root).free > state["cache_bytes"] + 20 * 1024**3:
                try:
                    save_cache_bundle(cache_path, {**state, "hashes": cache_hashes})
                except OSError:
                    if not cache_path.exists():
                        raise
                    # A parallel arm may have published this exact immutable
                    # prefix while the staging directory was being written.
                    from mmgcot_timeline_training.formal_cache_io import load_cache_bundle

                    load_cache_bundle(cache_path, expected_hashes=cache_hashes)
        self.cached[key] = state
        return state

    def _support(self, logits: torch.Tensor, matcher: object) -> torch.Tensor:
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite bbox logits")
        bitmask = xgr.allocate_token_bitmask(1, self.model.config.text_config.vocab_size)
        xgr.reset_token_bitmask(bitmask)
        scores = logits.float().detach().clone()
        if matcher.fill_next_token_bitmask(bitmask):
            xgr.apply_token_bitmask_inplace(
                scores, bitmask.to(scores.device),
                vocab_size=self.model.config.text_config.vocab_size)
        support = torch.where(torch.isfinite(scores[0]))[0]
        if support.numel() == 0:
            raise RuntimeError("empty bbox grammar support")
        return support

    def _numeric_mask(self, ids: list[int]) -> list[bool]:
        from mmgcot_timeline_training.formal_losses import numeric_token_mask

        return numeric_token_mask(ids, self.tokenizer)

    def sample_bbox(self, state: dict[str, Any], *, branch: str,
                    seed: int, greedy: bool = False,
                    forced_prefix_ids: list[int] | tuple[int, ...] = ()) -> dict[str, Any]:
        self.adapter.enabled = branch == "L"
        self._set_rope(state)
        cache = self._branch(state, branch)
        matcher = xgr.GrammarMatcher(self.base.box_grammar, terminate_without_stop_token=True)
        rng = torch.Generator(device="cuda").manual_seed(seed)
        input_id = state["opening_id"]
        ids: list[int] = []
        supports: list[list[int]] = []
        with torch.no_grad():
            for position in range(48):
                cache, logits = self.base.advance(self.model, cache, [input_id])
                support = self._support(logits, matcher)
                if position < len(forced_prefix_ids):
                    token = int(forced_prefix_ids[position])
                    if not bool((support == token).any().item()):
                        raise RuntimeError("forced student prefix token outside grammar support")
                else:
                    probabilities = torch.softmax(logits[0, support].float(), dim=-1)
                    selected = int(probabilities.argmax().item()) if greedy else int(
                        torch.multinomial(probabilities, 1, generator=rng).item())
                    token = int(support[selected].item())
                if not matcher.accept_token(token):
                    raise RuntimeError("bbox grammar rejected sampled token")
                ids.append(token)
                supports.append(support.cpu().tolist())
                if matcher.is_completed():
                    break
                input_id = token
        if len(ids) < len(forced_prefix_ids):
            raise RuntimeError("forced coordinate prefix exceeded grammar response")
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        bbox, valid, error = parse_box(text, matcher.is_completed())
        return {"token_ids": ids, "support_ids": supports, "numeric_mask": self._numeric_mask(ids),
                "completed": bool(matcher.is_completed()), "text": text,
                "bbox": bbox, "valid": valid, "parse_error": error, "seed": seed}

    def score_teacher(self, state: dict[str, Any], branch: str,
                      rollout: dict[str, Any]) -> list[torch.Tensor | None]:
        if branch not in ("R", "E"):
            raise ValueError(branch)
        self.adapter.enabled = False
        self._set_rope(state)
        cache = self._branch(state, branch)
        input_id = state["opening_id"]
        result: list[torch.Tensor | None] = []
        matcher = xgr.GrammarMatcher(self.base.box_grammar, terminate_without_stop_token=True)
        with torch.no_grad():
            for token, support_ids, numeric in zip(
                    rollout["token_ids"], rollout["support_ids"], rollout["numeric_mask"], strict=True):
                cache, logits = self.base.advance(self.model, cache, [input_id])
                teacher_support = self._support(logits, matcher).cpu().tolist()
                if teacher_support != support_ids:
                    raise RuntimeError("teacher/student grammar supports differ on shared prefix")
                if numeric:
                    support = torch.tensor(support_ids, dtype=torch.long, device="cuda")
                    result.append(logits[0, support].float().detach())
                else:
                    result.append(None)
                if not matcher.accept_token(int(token)):
                    raise RuntimeError("student raw coordinate prefix rejected by teacher grammar")
                input_id = token
        return result

    def backward_sft(self, state: dict[str, Any], gt: list[float],
                     effective_batch_size: int) -> float:
        from mmgcot_timeline_training.formal_losses import quantize_bbox

        coords = quantize_bbox(gt)
        text = ",".join(map(str, coords)) + "]}</answer>"
        ids = self.base.ids(text)
        numeric = self._numeric_mask(ids)
        count = sum(numeric)
        if count == 0:
            raise RuntimeError("GT has no numeric bbox tokens")
        self.adapter.enabled = True
        self._set_rope(state)
        cache = self._branch(state, "L")
        matcher = xgr.GrammarMatcher(self.base.box_grammar, terminate_without_stop_token=True)
        current = state["opening_id"]
        total = 0.0
        for token, is_numeric in zip(ids, numeric, strict=True):
            with torch.set_grad_enabled(is_numeric):
                cache, logits = self.base.advance(self.model, cache, [current])
                support = self._support(logits, matcher)
                if is_numeric:
                    candidates = logits[0, support].float()
                    location = torch.where(support == token)[0]
                    if len(location) != 1:
                        raise RuntimeError("GT digit outside legal grammar support")
                    loss = -torch.log_softmax(candidates, dim=-1)[location[0]] / count
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite GT SFT loss")
                    (loss / effective_batch_size).backward()
                    total += float(loss.detach())
            if not matcher.accept_token(int(token)):
                raise RuntimeError("GT token sequence rejected by bbox grammar")
            current = token
        if not matcher.is_completed():
            raise RuntimeError("GT bbox serialization incomplete")
        return total

    def backward_opd(self, state: dict[str, Any], rollout: dict[str, Any],
                     teacher_logits: list[torch.Tensor | None],
                     coefficient: float, effective_batch_size: int) -> float:
        self.adapter.enabled = True
        self._set_rope(state)
        cache = self._branch(state, "L")
        current = state["opening_id"]
        count = sum(rollout["numeric_mask"])
        if count == 0:
            raise RuntimeError("sampled bbox has no numeric tokens")
        total = 0.0
        for token, support_ids, numeric, teacher in zip(
                rollout["token_ids"], rollout["support_ids"],
                rollout["numeric_mask"], teacher_logits, strict=True):
            with torch.set_grad_enabled(numeric):
                cache, logits = self.base.advance(self.model, cache, [current])
                if numeric:
                    if teacher is None:
                        raise RuntimeError("teacher position missing")
                    support = torch.tensor(support_ids, dtype=torch.long, device="cuda")
                    student_log = torch.log_softmax(logits[0, support].float(), dim=-1)
                    teacher_log = torch.log_softmax(teacher.detach().float(), dim=-1)
                    value = (student_log.exp() * (student_log - teacher_log)).sum() / count
                    if not torch.isfinite(value):
                        raise FloatingPointError("nonfinite reverse KL")
                    (coefficient * value / effective_batch_size).backward()
                    total += float(value.detach())
            current = token
        return total


def _find_record(root: Path, row: dict[str, Any], trajectory_index: int) -> dict[str, Any]:
    path = record_path(root, row["sample_id"], trajectory_index)
    if not path.exists():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    for key in ("sample_id", "image_id", "image_sha256", "question"):
        if str(record[key]) != str(row[key]):
            raise RuntimeError(f"frozen record changed on {key}: {path}")
    if record["trajectory_index"] != trajectory_index:
        raise RuntimeError("trajectory index changed")
    return record


def _order(seed: int, n: int, steps: int, batch: int) -> list[int]:
    rng = random.Random(seed)
    indices = []
    while len(indices) < steps * batch:
        epoch = list(range(n))
        rng.shuffle(epoch)
        indices.extend(epoch)
    return indices[:steps * batch]


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _reproducibility_receipt(model_path: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    model_root = Path(model_path)
    source_names = (
        "mmgcot_timeline_training/formal_train_v2.py",
        "mmgcot_timeline_training/formal_contexts.py",
        "mmgcot_timeline_training/formal_losses.py",
        "mmgcot_timeline_training/formal_cache_io.py",
        "mmgcot_diagnostic/protocol.py",
        "timeline_self_distillation/terminal_adapter.py",
        "configs/mmgcot_formal_v2.json",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                     text=True).strip()
    model_files = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    model_files += tuple(sorted(p.name for p in model_root.glob("model.safetensors-*.safetensors")))
    return {
        "git_commit": commit,
        "source_sha256": {name: file_hash(root / name) for name in source_names},
        "model_files_sha256": {name: file_hash(model_root / name) for name in model_files},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": importlib.metadata.version("transformers"),
                        "xgrammar": importlib.metadata.version("xgrammar"),
                        "cuda_runtime": torch.version.cuda,
                        "torch_num_threads": torch.get_num_threads(),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "gpu": torch.cuda.get_device_name(0)},
    }


def train(args: argparse.Namespace) -> None:
    from timeline_self_distillation.terminal_adapter import terminal_parameter_whitelist

    rows = _jsonl(args.selection)
    prepared = json.loads((args.prepared / "protocol.json").read_text())
    if prepared["selection_sha256"] != file_hash(args.selection):
        raise RuntimeError("prepared trajectory selection differs from training selection")
    if prepared["trajectories_per_image"] != 1:
        raise RuntimeError("training requires one fixed trajectory per image")
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "formal_train_v2", "arm": args.arm, "seed": args.seed,
               "steps": args.steps, "effective_batch": args.batch,
               "checkpoint_every": args.checkpoint_every,
               "lr": args.lr, "lambda_opd": args.lambda_opd,
               "selection_sha256": file_hash(args.selection),
               "prepared_protocol_sha256": file_hash(args.prepared / "protocol.json"),
               "model": args.model, "rank": 8, "optimizer": "AdamW",
               "cache_root": str(args.cache_root.absolute()) if args.cache_root else None,
               "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
               "grad_clip": 1.0, "semantic_review_policy": "report_only",
               "reproducibility": _reproducibility_receipt(args.model)}
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
    engine = FormalEngine(args.model, cache_root=args.cache_root)
    optimizer = torch.optim.AdamW(engine.parameters, lr=args.lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0)
    order = _order(args.seed, len(rows), args.steps, args.batch)
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != receipt:
            raise RuntimeError("checkpoint configuration differs")
        if checkpoint["sample_order_sha256"] != order_hash:
            raise RuntimeError("checkpoint sample order differs")
        engine.adapter.down.weight.data.copy_(checkpoint["down"].to("cuda"))
        engine.adapter.up.weight.data.copy_(checkpoint["up"].to("cuda"))
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        start_step = checkpoint["step"]
        if start_step >= args.steps:
            raise RuntimeError("resume checkpoint already at final step")
    _append(out / "attempts.jsonl", {"started_at_unix": time.time(),
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "start_step": start_step, "command": [sys.executable, *sys.argv],
        "config_sha256": file_hash(receipt_path)})
    for step in range(start_step, args.steps):
        begun = time.time()
        optimizer.zero_grad(set_to_none=True)
        sft_sum = kl_sum = 0.0
        missing = invalid = 0
        for slot in range(args.batch):
            row = rows[order[step*args.batch+slot]]
            frozen = _find_record(args.prepared, row, 0)
            state = engine.build_state(frozen)
            sft = engine.backward_sft(state, row["ground_truth_bbox"], args.batch)
            sft_sum += sft
            if not state["has_target"]:
                missing += 1
            if args.arm in ("r_opd", "e_opd") and state["has_target"]:
                seed = stable_seed(args.seed, step, slot, row["sample_id"], "bbox_rollout")
                sample = engine.sample_bbox(state, branch="L", seed=seed)
                if not sample["valid"]:
                    invalid += 1
                teacher_branch = "R" if args.arm == "r_opd" else "E"
                teacher = engine.score_teacher(state, teacher_branch, sample)
                kl = engine.backward_opd(state, sample, teacher, args.lambda_opd, args.batch)
                kl_sum += kl
                _append(out / "rollouts.jsonl", {
                    "step": step+1, "slot": slot, "sample_id": row["sample_id"],
                    "arm": args.arm, "seed": seed, "token_ids": sample["token_ids"],
                    "support_ids": sample["support_ids"], "numeric_mask": sample["numeric_mask"],
                    "teacher_support_logits": [x.cpu().tolist() if x is not None else None for x in teacher],
                    "completed": sample["completed"], "valid": sample["valid"],
                    "bbox": sample["bbox"], "parse_error": sample["parse_error"],
                    "early_offset": state["early_offset"],
                    "has_target": state["has_target"],
                })
        if any(p.grad is not None for p in engine.model.parameters() if not p.requires_grad):
            raise RuntimeError("frozen backbone acquired gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(engine.parameters, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        terminal_parameter_whitelist(engine.model, engine.adapter)
        _append(out / "steps.jsonl", {"step": step+1, "sft": sft_sum/args.batch,
               "kl": kl_sum/args.batch, "weighted_kl": args.lambda_opd*kl_sum/args.batch,
               "loss": (sft_sum+args.lambda_opd*kl_sum)/args.batch,
               "grad_norm": grad_norm, "missing_target": missing,
               "invalid_rollout": invalid, "seconds": time.time()-begun,
               "peak_memory_bytes": torch.cuda.max_memory_allocated()})
        print(f"STEP {args.arm} seed={args.seed} {step+1}/{args.steps} "
              f"sft={sft_sum/args.batch:.5f} kl={kl_sum/args.batch:.5f} "
              f"time={time.time()-begun:.1f}s", flush=True)
        if (step+1) % args.checkpoint_every == 0 or step+1 == args.steps:
            payload = {"config": receipt, "step": step+1,
                       "down": engine.adapter.down.weight.detach().cpu(),
                       "up": engine.adapter.up.weight.detach().cpu(),
                       "optimizer": optimizer.state_dict(),
                       "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state(),
                       "sample_order_sha256": order_hash}
            _atomic_checkpoint(out / f"step_{step+1:04d}.pt", payload)
        if args.stop_after_step is not None and step+1 >= args.stop_after_step:
            print(f"CONTROLLED_STOP_AFTER_STEP {step+1}", flush=True)
            return
    (out / "complete.json").write_text(json.dumps({"step": args.steps,
        "config_sha256": file_hash(receipt_path), "selection_sha256": file_hash(args.selection),
        "final_checkpoint": str(out / f"step_{args.steps:04d}.pt")}, indent=2) + "\n")


def evaluate(args: argparse.Namespace) -> None:
    rows = _jsonl(args.selection)
    prepared = json.loads((args.prepared / "protocol.json").read_text())
    if prepared["selection_sha256"] != file_hash(args.selection):
        raise RuntimeError("prepared/evaluation selection mismatch")
    engine = FormalEngine(args.model, cache_root=args.cache_root)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        engine.adapter.down.weight.data.copy_(checkpoint["down"].to("cuda"))
        engine.adapter.up.weight.data.copy_(checkpoint["up"].to("cuda"))
    result_path = args.output.absolute()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(result_path)
    branch = {"base_r": "R", "base_e": "E"}.get(args.arm, "L")
    with result_path.open("x", encoding="utf-8") as handle:
        for row in rows[:args.limit] if args.limit is not None else rows:
            for ti in range(prepared["trajectories_per_image"]):
                frozen = _find_record(args.prepared, row, ti)
                state = engine.build_state(frozen)
                for draw in range(5):
                    greedy = draw == 0
                    seed = stable_seed(row["sample_id"], ti, draw, "formal_eval_bbox")
                    pred = engine.sample_bbox(state, branch=branch, seed=seed, greedy=greedy)
                    if args.arm == "base_l":
                        # Zero adapter and enabled L have identical output.
                        pass
                    record = {"sample_id": row["sample_id"], "image_id": row["image_id"],
                              "task_type": row["task_type"], "trajectory_index": ti,
                              "arm": args.arm, "seed": args.seed, "mode": "greedy" if greedy else "sample",
                              "draw": 0 if greedy else draw-1,
                              "bbox": pred["bbox"], "valid": pred["valid"],
                              "completed": pred["completed"], "parse_error": pred["parse_error"],
                              "response_token_ids": pred["token_ids"], "iou": iou(
                                  pred["bbox"], row["ground_truth_bbox"], pred["valid"]),
                              "has_target": state["has_target"],
                              "reasoning_length": state["reasoning_length"],
                              "early_offset": state["early_offset"]}
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            print(f"EVAL {args.arm} {row['sample_id']}", flush=True)
    result_path.with_suffix(".complete.json").write_text(json.dumps({
        "selection_sha256": file_hash(args.selection),
        "prepared_protocol_sha256": file_hash(args.prepared / "protocol.json"),
        "checkpoint_sha256": file_hash(args.checkpoint) if args.checkpoint else None,
        "arm": args.arm, "seed": args.seed,
        "evaluated_images": args.limit if args.limit is not None else len(rows),
        "trajectories_per_image": prepared["trajectories_per_image"],
        "draws_per_trajectory": {"greedy": 1, "sample": 4},
        "output_sha256": file_hash(result_path)}, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--arm", choices=("bbox_sft", "r_opd", "e_opd", "base_l", "base_r", "base_e"), required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lambda-opd", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mode == "train" and args.arm.startswith("base_"):
        parser.error("base arms are evaluation-only")
    if args.mode == "eval" and args.arm not in ("base_l", "base_r", "base_e") and not args.checkpoint:
        parser.error("trained-arm evaluation requires --checkpoint")
    if args.batch < 1 or args.steps < 1 or args.lambda_opd < 0:
        parser.error("invalid training configuration")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.stop_after_step is not None and not 1 <= args.stop_after_step < args.steps:
        parser.error("--stop-after-step must be within the training run")
    if args.limit is not None and (args.mode != "eval" or args.limit < 1):
        parser.error("--limit is a positive evaluation-only smoke bound")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.device)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import xgrammar as xgr
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train(parsed) if parsed.mode == "train" else evaluate(parsed)
