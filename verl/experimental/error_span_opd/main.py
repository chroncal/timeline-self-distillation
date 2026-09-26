"""Small recipe extension of verl's legacy TaskRunner and RayPPOTrainer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.experimental.error_span_opd.evaluation import summarize_validation


def _finite_metric_values(value):
    if isinstance(value, dict):
        return all(_finite_metric_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_metric_values(item) for item in value)
    if hasattr(value, "tolist"):
        return _finite_metric_values(value.tolist())
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return True


def launch_receipt(config):
    repo = Path(__file__).resolve().parents[3]
    root = Path(config.trainer.default_local_dir).resolve()
    if root.exists():
        raise FileExistsError(f"new runs require an unused output directory: {root}")
    sources = [Path(p).resolve() for p in [*config.data.train_files, *config.data.val_files]]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    teacher_evidence_sources = {}
    annotation_paths = config.error_span_opd.get("finecops_positive_annotation_paths", {})
    for split, configured_path in annotation_paths.items():
        source = Path(str(configured_path)).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        teacher_evidence_sources[str(split)] = source
    root.mkdir(parents=True)
    snapshot = root / "code_snapshot"
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((repo / "configs/error_span_opd").glob("*.yaml"))
    if config.error_span_opd.get("protocol_path"):
        files.append(repo / str(config.error_span_opd.protocol_path))
    files.append(repo / "scripts/run_error_span_single.py")
    files += [repo / "verl/utils/metric/utils.py", repo / "verl/trainer/ppo/ray_trainer.py", repo / "verl/trainer/distillation/losses.py", repo / "verl/trainer/distillation/fsdp/losses.py",
              repo / "verl/experimental/routed_grounding/agent_loop.py", repo / "reasoning_checkpoints/extractor.py",
              repo / "checkpoint_repair_upper_bound/run_contrastive_bbox.py",
              repo / "checkpoint_repair_upper_bound/error_span_opd_protocol_20260909.json",
              repo / "checkpoint_repair_upper_bound/error_span_opd_overlay_target_protocol_20260910.json"]
    hashes = {}
    for file in files:
        dest = snapshot / file.relative_to(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    OmegaConf.save(config, root / "resolved_config.yaml", resolve=True)
    models = {}
    for role, path in (("student", config.actor_rollout_ref.model.path),
                       ("teacher", config.error_span_opd.teacher_model)):
        model = Path(path)
        models[role] = {"path": str(model), "files": {}}
        for file in model.iterdir():
            if file.suffix in {".json", ".jinja", ".safetensors"}:
                stat = file.stat()
                entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                if file.suffix != ".safetensors":
                    entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                models[role]["files"][file.name] = entry
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   run_kind=config.error_span_opd.run_kind, command=[sys.executable, *sys.argv],
                   git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                   git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), code_sha256=hashes,
                   data_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   teacher_evidence_sha256={str(split): {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                            for split, path in teacher_evidence_sources.items()},
                   models=models)
    (root / "launch_receipt.json").write_text(json.dumps(payload, indent=2) + "\n")


class ErrorSpanTrainer(RayPPOTrainer):
    def _update_actor(self, batch):
        active = int((batch.batch["response_mask"].sum(-1) > 0).sum())
        if active == 0:
            # A reported step must correspond to an actual optimizer update.
            # Stop and diagnose rather than count empty updates or resample.
            raise RuntimeError("entire rollout batch has no eligible error-span signal; "
                               "stopped before optimizer update, without resampling")
        # Sample trajectories at rollout temperature; distill raw distributions at T=1.
        batch.meta_info["distillation_temperature"] = float(self.config.error_span_opd.distillation_temperature)
        result = super()._update_actor(batch)
        if not _finite_metric_values(result.meta_info["metrics"]):
            raise RuntimeError("non-finite optimizer/update metrics; stopped before saving this checkpoint")
        result.meta_info["metrics"]["error_span/skipped_optimizer_step"] = 0
        result.meta_info["metrics"]["error_span/active_sequences"] = active
        self._completed_optimizer_updates = getattr(self, "_completed_optimizer_updates", 0) + 1
        if self._completed_optimizer_updates != self.global_steps:
            raise RuntimeError("optimizer updates and reported global step diverged")
        return result

    def _save_checkpoint(self):
        super()._save_checkpoint()
        actor = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "actor"
        files = [file for file in actor.rglob("*") if file.is_file()]
        if not files:
            raise RuntimeError("checkpoint save returned without actor files")
        started = time.perf_counter()
        # The upstream fit loop synchronizes these reloaded weights to rollout
        # replicas immediately afterwards, before validation. Do not reload the
        # dataloader or change the sequence of training questions.
        self.actor_rollout_wg.load_checkpoint(str(actor), del_local_after_load=False)
        receipt = dict(step=self.global_steps, checkpoint=str(actor),
                       optimizer_updates=self._completed_optimizer_updates,
                       restored_from_saved_checkpoint=True,
                       reload_seconds=time.perf_counter() - started,
                       files={str(file.relative_to(actor)): file.stat().st_size for file in files})
        (actor.parent / "checkpoint_reload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        self._last_reloaded_checkpoint_step = self.global_steps

    def _validate(self):
        if self.global_steps and getattr(self, "_last_reloaded_checkpoint_step", None) != self.global_steps:
            raise RuntimeError("validation requires a saved and reloaded checkpoint at this step")
        metrics = super()._validate()
        root = Path(self.config.trainer.default_local_dir)
        raw = root / "monitor_draws.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row["step"] == self.global_steps]
        import pyarrow.parquet as pq
        expected_ids = {"ref_adv": [], "finecops_val": []}
        expected_examples = {}
        source_hashes = json.loads((root / "launch_receipt.json").read_text())["data_sha256"]
        for source in self.config.data.val_files:
            source_path = Path(source).resolve()
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hashes[str(source_path)]:
                raise ValueError("validation manifest changed after launch")
            for item in pq.read_table(source, columns=["extra_info"]).to_pylist():
                extra = item["extra_info"]
                sample = str(extra["sample_id"])
                if sample.startswith("ref-adv-s-"):
                    name = "ref_adv"
                elif sample.startswith("finecops-ref-val-"):
                    name = "finecops_val"
                else:
                    raise ValueError(f"unexpected validation source: {sample}")
                expected_ids[name].append(sample)
                expected_examples[sample] = extra
        for row in rows:
            extra = expected_examples.get(row["sample_id"])
            if extra is None or any(row[key] != extra[key] for key in ("image_id", "expression", "image_path")):
                raise ValueError("validation image/question identity differs from manifest")
            if row["ground_truth_bbox"] != list(extra["target_bbox_normalized"]):
                raise ValueError("validation GT differs from manifest")
        if len(expected_ids["ref_adv"]) != int(self.config.error_span_opd.monitor_samples):
            raise ValueError("Ref-Adv manifest count differs from configuration")
        if len(expected_ids["finecops_val"]) != int(self.config.error_span_opd.finecops_val_samples):
            raise ValueError("FineCops validation manifest count differs from configuration")
        for name, summary in summarize_validation(rows, expected_ids).items():
            for key in ("accuracy", "mean_iou", "invalid_rate"):
                metrics[f"{name}/{key}"] = summary[key]
            receipt = dict(step=self.global_steps, benchmark=name, **summary,
                           checkpoint=str(root / f"global_step_{self.global_steps}" / "actor")
                           if self.global_steps else self.config.actor_rollout_ref.model.path,
                           restored_from_saved_checkpoint=bool(self.global_steps),
                           optimizer_updates=getattr(self, "_completed_optimizer_updates", 0))
            with (root / f"{name}_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(receipt, allow_nan=False) + "\n")
            print(json.dumps({"checkpoint_evaluation": receipt}), flush=True)
        return metrics


@ray.remote
class ErrorSpanTaskRunner(BaseTaskRunner):
    def run(self, config):
        import verl.trainer.main_ppo_v0 as upstream
        # This pinned verl recipe API accepts a TaskRunner but not a trainer
        # class. Change only the trainer factory in this isolated Ray process;
        # invoke the upstream setup/worker/fit implementation unchanged.
        original = upstream.RayPPOTrainer
        upstream.RayPPOTrainer = ErrorSpanTrainer
        try:
            return upstream.TaskRunner.__ray_metadata__.modified_class.run(self, config)
        finally:
            upstream.RayPPOTrainer = original


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    if config.actor_rollout_ref.actor.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError("this recipe uses one complete rollout batch per optimizer update")
    if config.actor_rollout_ref.actor.ppo_epochs != 1:
        raise ValueError("this recipe requires one update pass per current-policy rollout batch")
    from verl.utils.config import validate_config
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    launch_receipt(config)
    run_ppo(config, task_runner_class=ErrorSpanTaskRunner)


if __name__ == "__main__":
    main()
