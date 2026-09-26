"""Group fresh verl rollouts before adding local privileged teacher targets."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.error_span_opd.worker import ErrorSpanWorker
from verl.experimental.routed_grounding.router import xyxy_iou
from verl.utils.ray_utils import auto_await


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


@lru_cache(maxsize=4)
def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _replayed_annotations(path: str, step: int, expected_sha256: str) -> dict[tuple[str, int], dict[str, Any]]:
    source = str(Path(path).expanduser().resolve())
    if _file_sha256(source) != expected_sha256:
        raise RuntimeError("annotation replay source hash differs from frozen recovery config")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with Path(source).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) != int(step):
                continue
            if row.get("arm") != "ROSD":
                raise RuntimeError(f"annotation replay row {line_number} is not ROSD")
            key = (str(row.get("sample_id")), int(row.get("rollout_index", -1)))
            if key in rows:
                raise RuntimeError(f"duplicate annotation replay key at line {line_number}: {key!r}")
            rows[key] = row
    if not rows:
        raise RuntimeError(f"annotation replay source has no rows for step {step}")
    return rows


class ErrorSpanManager(AgentLoopManager):
    """Use standard verl generation, FSDP updates, and weight synchronization."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.agent_loop_workers_class = ray.remote(ErrorSpanWorker)
        super().__init__(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts: Any) -> Any:
        output = await super().generate_sequences(prompts)
        cfg = self.config.error_span_opd
        validate = bool(prompts.meta_info.get("validate", False))
        step = int(prompts.meta_info.get("global_steps", -1))
        arm = str(cfg.arm)
        method_variant = str(cfg.get("method_variant", "local_span"))
        is_rosd = method_variant == "rosd"
        root = Path(self.config.trainer.default_local_dir)
        pwidth = int(output.batch["prompts"].shape[1])
        rwidth = int(output.batch["responses"].shape[1])
        records, payloads = [], []
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(len(output)):
            extra = plain(prompts.non_tensor_batch["extra_info"][i])
            receipt = plain(output.non_tensor_batch["routed_grounding_receipt"][i])
            parsed = receipt["student_parse"]
            gt = list(extra["target_bbox_normalized"])
            bbox = parsed.get("bbox")
            valid = bool(parsed.get("parse_valid", False)) and bbox is not None
            iou = float(xyxy_iou(bbox, gt)) if valid else 0.0
            plen = int(output.batch["attention_mask"][i, :pwidth].sum())
            rlen = int(output.batch["attention_mask"][i, pwidth:].sum())
            response = output.batch["responses"][i, :rlen].tolist()
            prompt = output.batch["prompts"][i, pwidth - plen:].tolist()
            sample_id = str(extra["sample_id"])
            qgroup = str(plain(prompts.non_tensor_batch["uid"][i]))
            draw = len(groups[qgroup])
            groups[qgroup].append(i)
            records.append(dict(
                step=step, arm=arm, validate=validate, sample_id=sample_id,
                data_source=str(prompts.non_tensor_batch["data_source"][i]),
                image_id=extra["image_id"], rollout_index=draw, group_id=qgroup,
                expression=extra["expression"], image_path=extra["image_path"],
                ground_truth_bbox=gt, bbox=bbox, parse_valid=valid, iou=iou,
                hit=bool(valid and iou >= 0.5), prompt_token_ids=prompt,
                response_ids=response, raw_response=receipt["student_token_response"],
                protocol_response=receipt["student_protocol_response"],
                student_phase_receipts=receipt["student_phase_receipts"],
                prompt_contains_gt=False, prompt_contains_successful_reference=False,
            ))
            payloads.append(dict(
                sample_id=sample_id, expression=extra["expression"],
                image_path=extra["image_path"], image_id=extra.get("image_id"),
                source_split=extra.get("source_split", extra.get("split")),
                source_annotation_id=extra.get("source_annotation_id"),
                target_ann_id=extra.get("target_ann_id"),
                data_source=str(prompts.non_tensor_batch["data_source"][i]),
                ground_truth_bbox=gt,
                prompt_ids=prompt, response_ids=response, arm=arm, step=step,
                original_final_bbox=bbox,
            ))
        # Durably record all assigned draws before any selector or teacher call.
        append_rows(root / ("monitor_draws.jsonl" if validate else "training_draws.jsonl"), records)
        expected_policy_step = step if validate else step - 1
        for record in records:
            for phase in record["student_phase_receipts"]:
                versions = phase.get("server_extra_fields", {})
                for key in ("global_steps", "min_global_steps", "max_global_steps"):
                    if versions.get(key) != expected_policy_step:
                        raise RuntimeError(f"student rollout has stale/missing weight version: "
                                           f"{key}={versions.get(key)}, expected={expected_policy_step}")
        if validate:
            return output

        expected_n = int(self.config.actor_rollout_ref.rollout.n)
        annotation_replay_source = cfg.get("annotation_replay_source", None)
        annotation_replay_through_step = int(cfg.get("annotation_replay_through_step", 0) or 0)
        replay_annotations = bool(
            annotation_replay_source and 1 <= step <= annotation_replay_through_step
        )
        jobs, indices, reference_records = [], [], []
        for group, positions in groups.items():
            if len(positions) != expected_n:
                raise ValueError(f"group {group} has {len(positions)} draws; expected {expected_n}")
            if len({records[i]["sample_id"] for i in positions}) != 1:
                raise ValueError("a rollout group combines different questions")
            # The current single-group viability run does not consume or even
            # select a successful-reference trajectory.  Preserve selection
            # only for the historical B contract.
            first = (
                next((i for i in positions if records[i]["hit"]), None)
                if arm == "B" or is_rosd
                else None
            )
            reference = None
            reference_full = None
            if first is not None:
                raw = records[first]["raw_response"]
                if raw.count("</think>") != 1:
                    raise ValueError("successful reference lacks unique reasoning closure")
                reference = raw.split("</think>", 1)[0]
                if reference.startswith("<think>"):
                    reference = reference[len("<think>"):]
                reference_full = raw
            reference_records.append(dict(
                step=step, group_id=group, sample_id=records[positions[0]]["sample_id"],
                selected_rollout_index=None if first is None else records[first]["rollout_index"],
                reference_reasoning=reference,
                reference_response_ids=None if first is None else records[first]["response_ids"],
                rule=(
                    "first bbox success in assigned rollout order; no resampling"
                    if arm == "B" or is_rosd
                    else "successful-reference selection disabled"
                ),
            ))
            for i in positions:
                if is_rosd:
                    if first is None:
                        continue
                    payloads[i]["sample_success"] = bool(records[i]["hit"])
                    payloads[i]["reference_full_response"] = reference_full
                    payloads[i]["reference_available"] = True
                    payloads[i]["reference_response_ids"] = records[first]["response_ids"]
                    payloads[i]["reference_rollout_index"] = records[first]["rollout_index"]
                    payloads[i]["reference_sample_id"] = records[first]["sample_id"]
                    if not replay_annotations:
                        worker = self.agent_loop_workers[len(jobs) % len(self.agent_loop_workers)]
                        jobs.append(worker.annotate_error_span.remote(payloads[i]))
                    indices.append(i)
                    continue
                if not records[i]["parse_valid"] or records[i]["hit"]:
                    continue
                payloads[i]["reference_reasoning"] = reference if arm == "B" else None
                payloads[i]["reference_available"] = reference is not None
                payloads[i]["reference_response_ids"] = records[first]["response_ids"] if arm == "B" and first is not None else None
                payloads[i]["reference_rollout_index"] = records[first]["rollout_index"] if first is not None else None
                payloads[i]["reference_sample_id"] = records[first]["sample_id"] if first is not None else None
                worker = self.agent_loop_workers[len(jobs) % len(self.agent_loop_workers)]
                jobs.append(worker.annotate_error_span.remote(payloads[i]))
                indices.append(i)
        append_rows(root / "references.jsonl", reference_records)
        started = time.perf_counter()
        if replay_annotations:
            source_rows = _replayed_annotations(
                str(annotation_replay_source),
                step,
                str(cfg.get("annotation_replay_sha256", "")),
            )
            expected_keys = {
                (records[index]["sample_id"], int(records[index]["rollout_index"]))
                for index in indices
            }
            if set(source_rows) != expected_keys:
                missing = sorted(expected_keys - set(source_rows))[:5]
                extra = sorted(set(source_rows) - expected_keys)[:5]
                raise RuntimeError(
                    f"annotation replay membership differs at step {step}: missing={missing}, extra={extra}"
                )
            annotations = []
            for index in indices:
                key = (records[index]["sample_id"], int(records[index]["rollout_index"]))
                annotation = dict(source_rows[key])
                response_hash = hashlib.sha256(
                    json.dumps(records[index]["response_ids"], separators=(",", ":")).encode("ascii")
                ).hexdigest()
                if annotation.get("response_token_ids_sha256") != response_hash:
                    raise RuntimeError(f"annotation replay response hash differs for {key!r}")
                annotations.append(annotation)
        else:
            annotations = await asyncio.gather(*jobs)
        shape = (len(output), pwidth + rwidth, int(cfg.topk))
        teacher_ids = torch.zeros(shape, dtype=torch.int32)
        # Finite zero-mass placeholders avoid 0 * (-inf) outside the loss mask.
        teacher_logprobs = torch.full(shape, -100.0, dtype=torch.float32)
        mask = torch.zeros_like(output.batch["response_mask"])
        audit = []
        selected_count = 0
        for i, annotation in zip(indices, annotations, strict=True):
            annotation = plain(annotation)
            audit.append({**annotation, "step": step, "arm": arm,
                          "sample_id": records[i]["sample_id"],
                          "rollout_index": records[i]["rollout_index"]})
            selection = annotation.get("selection")
            if selection is None:
                continue
            before = int(selection.get("supervision_before_offset", selection["before_offset"]))
            after = int(selection.get("supervision_after_offset", selection["after_offset"]))
            scope = str(selection.get("supervision_scope", "selected_span"))
            if not 0 <= before < after <= len(records[i]["response_ids"]):
                raise ValueError("selector returned an invalid supervision interval")
            original_mask = output.batch["response_mask"][i, before:after]
            if scope == "selected_span" and not bool(original_mask.bool().all()):
                raise ValueError("selected span includes deterministic protocol tokens instead of sampled reasoning")
            if not bool(original_mask.bool().any()):
                raise ValueError("supervision interval has no trainable response tokens")
            ids = torch.tensor(annotation["teacher_topk_ids"], dtype=torch.int32)
            lps = torch.tensor(annotation["teacher_topk_logprobs"], dtype=torch.float32)
            if ids.shape != (after - before, int(cfg.topk)) or lps.shape != ids.shape:
                raise ValueError("teacher span target shape mismatch")
            if not torch.isfinite(lps).all() or (lps > 1e-4).any():
                raise ValueError("invalid teacher log probabilities")
            if (lps.exp().sum(-1) > 1.001).any():
                raise ValueError("teacher top-k probability mass exceeds one")
            # vLLM/verl arrays store prediction rows, not target-token rows.
            teacher_ids[i, pwidth + before - 1:pwidth + after - 1] = ids
            teacher_logprobs[i, pwidth + before - 1:pwidth + after - 1] = lps
            # ROSD-style suffixes may cross deterministic protocol tokens from
            # the staged rollout. Preserve the native response mask so those
            # positions never acquire a loss merely because the interval is
            # wider than the diagnostic reasoning span.
            mask[i, before:after] = original_mask
            selected_count += 1
        append_rows(root / "span_annotations.jsonl", audit)
        output.batch["teacher_ids"] = teacher_ids
        output.batch["teacher_logprobs"] = teacher_logprobs
        output.batch["response_mask"] = mask
        output.batch["error_span_active_sequences"] = torch.full((len(output),), selected_count, dtype=torch.int64)
        metrics = {
            "error_span/selected_trajectories": selected_count,
            "error_span/selected_tokens": int(mask.sum()),
            "error_span/supervision_scope_error_suffix": float(
                str(cfg.get("supervision_scope", "selected_span")) == "error_suffix"
            ),
            "error_span/assigned_trajectories": len(records),
            "error_span/method_rosd": float(is_rosd),
            "error_span/annotation_replayed": float(replay_annotations),
            "error_span/groups_without_success": (
                sum(record["selected_rollout_index"] is None for record in reference_records)
                if is_rosd
                else 0
            ),
            "error_span/reflection_quote_fallbacks": sum(
                bool((row.get("selection") or {}).get("fallback_to_full_response", False))
                for row in audit
            ),
            "error_span/reflection_format_failures": sum(
                row.get("skip_reason") == "rosd_reflection_format_failure" for row in audit
            ),
            "error_span/reflection_retries": sum(
                max(
                    0,
                    int(
                        (row.get("request") or {}).get("reflection", {}).get("attempt_count", 1)
                    )
                    - 1,
                )
                for row in audit
            ),
            "error_span/reference_groups": sum(r["reference_reasoning"] is not None for r in reference_records),
            "error_span/question_groups": len(groups),
            "error_span/raw_accuracy": sum(r["hit"] for r in records) / len(records),
            "error_span/raw_mean_iou": sum(r["iou"] for r in records) / len(records),
            "error_span/raw_invalid_rate": sum(not r["parse_valid"] for r in records) / len(records),
            "error_span/annotation_seconds": time.perf_counter() - started,
        }
        # Existing RayPPOTrainer already consumes this metrics hook.
        output.meta_info["routed_grounding_metrics"] = metrics
        append_rows(root / "span_steps.jsonl", [dict(step=step, arm=arm, **metrics)])
        return output
