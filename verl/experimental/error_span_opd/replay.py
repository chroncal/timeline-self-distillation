"""Replay recorded Student agent-loop requests for the technical smoke run.

The replay client is deliberately small: it only replaces the Student server
for one agent-loop invocation.  Annotation, Teacher requests, optimisation,
and evaluation continue to use their normal clients.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.error_span_opd.worker import ErrorSpanAgentLoop
from verl.workers.rollout.replica import TokenOutput


class ReplayError(RuntimeError):
    """The recorded trajectory cannot safely answer a native request."""


def _plain(value: Any) -> Any:
    """Match the JSON representation used by native phase receipts."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ids(value: Any, name: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ReplayError(f"{name} must be a flat token-id list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ReplayError(f"{name} contains a non-integer token id") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReplayError(f"replay source is missing {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReplayError(f"invalid JSON in {path}:{line_number}") from error
        if not isinstance(row, Mapping):
            raise ReplayError(f"replay row {path}:{line_number} is not an object")
        rows.append(dict(row))
    return rows


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_id(kwargs: Mapping[str, Any]) -> str | None:
    value = kwargs.get("sample_id")
    if value is None:
        extra = kwargs.get("extra_info")
        if isinstance(extra, Mapping):
            value = extra.get("sample_id")
        else:
            value = getattr(extra, "sample_id", None)
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    return None if value is None else str(value)


class _Corpus:
    def __init__(
        self,
        source: str | Path,
        *,
        expected_run_kind: str = "technical_smoke",
        expected_training_rows: int = 128,
        expected_validation_rows: int = 4,
        expected_training_steps: Sequence[int] = (1,),
        allow_update_evidence: bool = False,
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        if not self.source.is_dir():
            raise ReplayError(f"replay_source must be a directory: {self.source}")
        launch_path = self.source / "launch_receipt.json"
        try:
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ReplayError(f"replay source has no valid launch_receipt.json: {self.source}") from error
        if launch.get("run_kind") != expected_run_kind:
            raise ReplayError(f"replay is restricted to a {expected_run_kind} source run")
        self.launch = launch
        self.expected_run_kind = expected_run_kind
        # Ordinary replay accepts only immutable pre-update smoke evidence.
        # Interrupted-run recovery opts in explicitly and performs its own
        # update-ledger and policy-version checks before consuming the corpus.
        if not allow_update_evidence:
            for name in ("span_steps.jsonl", "optimizer_steps.jsonl", "optimizer_updates.jsonl"):
                path = self.source / name
                if path.is_file() and path.read_text(encoding="utf-8").strip():
                    raise ReplayError(f"replay source contains optimizer/update evidence: {path}")
            for path in self.source.iterdir():
                match = re.fullmatch(r"global_step_(\d+)", path.name)
                if match and int(match.group(1)) > 0:
                    raise ReplayError(f"replay source contains checkpoint/update evidence: {path}")

        train_path = self.source / "training_draws.jsonl"
        val_path = self.source / "monitor_draws.jsonl"
        train = _read_jsonl(train_path)
        val = _read_jsonl(val_path)
        if len(train) != expected_training_rows or len(val) != expected_validation_rows:
            raise ReplayError(
                f"replay requires {expected_training_rows} training and {expected_validation_rows} validation rows; "
                f"got {len(train)} and {len(val)}"
            )
        self.source_hash = _json_hash(
            {name: _file_hash(self.source / name) for name in ("launch_receipt.json", "training_draws.jsonl", "monitor_draws.jsonl")}
        )
        allowed_training_steps = {int(step) for step in expected_training_steps}
        if not allowed_training_steps:
            raise ReplayError("replay requires at least one expected training step")
        self.records: dict[tuple[str, bool, int], dict[str, Any]] = {}
        for row, expected_validate in [*[(r, False) for r in train], *[(r, True) for r in val]]:
            row_step = int(row.get("step", -1))
            expected_steps = {0} if expected_validate else allowed_training_steps
            if bool(row.get("validate")) != expected_validate or row_step not in expected_steps:
                raise ReplayError(
                    "replay rows fall outside the required validation step 0 / configured training steps"
                )
            sid = row.get("sample_id")
            phases = row.get("student_phase_receipts")
            if sid is None or not isinstance(phases, list) or not phases:
                raise ReplayError("replay row lacks sample_id or student phase receipts")
            seed = phases[0].get("sampling", {}).get("seed")
            if seed is None:
                raise ReplayError(f"replay row {sid!r} has no recorded sampling seed")
            key = (str(sid), expected_validate, int(seed))
            if key in self.records:
                raise ReplayError(f"duplicate replay key {key!r}")
            prompt = _ids(row.get("prompt_token_ids"), "prompt_token_ids")
            response = _ids(row.get("response_ids"), "response_ids")
            normalized: list[dict[str, Any]] = []
            for phase in phases:
                if not isinstance(phase, Mapping) or phase.get("kind") not in {"student_reasoning", "student_bbox"}:
                    raise ReplayError(f"row {sid!r} has an invalid Student phase")
                phase = dict(phase)
                token_ids = _ids(phase.get("token_ids"), "phase.token_ids")
                if phase.get("sampling", {}).get("seed") != seed:
                    raise ReplayError(f"row {sid!r} has inconsistent phase seeds")
                if phase.get("token_count") is not None and int(phase["token_count"]) != len(token_ids):
                    raise ReplayError(f"row {sid!r} has an inconsistent phase token count")
                phase["token_ids"] = token_ids
                normalized.append(phase)
            reasoning = normalized[0]
            if reasoning["kind"] != "student_reasoning" or response[: len(reasoning["token_ids"])] != reasoning["token_ids"]:
                raise ReplayError(f"row {sid!r} does not preserve the reasoning response prefix")
            bbox_prompt: list[int] | None = None
            if len(normalized) > 1:
                bbox = normalized[1]
                if bbox["kind"] != "student_bbox":
                    raise ReplayError(f"row {sid!r} has an unexpected phase order")
                prefix_count = int(bbox.get("prompt_token_count", 0)) - len(prompt) - len(reasoning["token_ids"])
                if prefix_count < 0 or len(response) < len(reasoning["token_ids"]) + prefix_count:
                    raise ReplayError(f"row {sid!r} has an invalid bbox prompt prefix")
                bbox_prefix = response[len(reasoning["token_ids"]): len(reasoning["token_ids"]) + prefix_count]
                if response != reasoning["token_ids"] + bbox_prefix + bbox["token_ids"]:
                    raise ReplayError(f"row {sid!r} response does not preserve the staged bbox composition")
                if bbox.get("coordinate_prefix_token_count") not in (None, prefix_count):
                    raise ReplayError(f"row {sid!r} has an invalid coordinate prefix count")
                bbox_prompt = prompt + reasoning["token_ids"] + bbox_prefix
            record = {
                "sample_id": str(sid),
                "validate": expected_validate,
                "step": row_step,
                "seed": int(seed),
                "prompt": prompt,
                "response": response,
                "phases": normalized,
                "bbox_prompt": bbox_prompt,
                "record_hash": _json_hash(row),
                "expression": row.get("expression"),
                "image_path": row.get("image_path"),
                "image_id": row.get("image_id"),
                "ground_truth_bbox": row.get("ground_truth_bbox"),
                "group_id": row.get("group_id"),
                "rollout_index": row.get("rollout_index"),
            }
            self.records[key] = record


class RecordedClient:
    """Read-only per-trajectory server adapter with exact request checks."""

    def __init__(self, real_client: Any, corpus: _Corpus, sample_id: str | None, validate: bool) -> None:
        self.real_client = real_client
        self.corpus = corpus
        self.sample_id = sample_id
        self.validate = bool(validate)
        self.record: dict[str, Any] | None = None
        self.phase_index = 0
        self.fallback = False
        self.native_validation_bypassed = False
        self.requests: list[dict[str, Any]] = []

    async def generate(self, **request: Any) -> TokenOutput:
        sampling = request.get("sampling_params", {})
        seed = sampling.get("seed") if isinstance(sampling, Mapping) else None
        if self.record is None and not self.fallback:
            if seed is None:
                if not self.validate:
                    raise ReplayError("training replay request has no sampling_params.seed")
                self.fallback = True
            else:
                self.record = self.corpus.records.get((self.sample_id or "", self.validate, int(seed)))
                if self.record is None:
                    if not self.validate:
                        raise ReplayError(f"no recorded training trajectory for sample_id={self.sample_id!r}, seed={seed}")
                    self.fallback = True
        if self.fallback:
            return await self.real_client.generate(**request)
        assert self.record is not None
        if self.phase_index >= len(self.record["phases"]):
            raise ReplayError(f"replay received an unexpected extra Student request for {self.sample_id!r}")
        phase = self.record["phases"][self.phase_index]
        expected_prompt = self.record["prompt"] if self.phase_index == 0 else self.record["bbox_prompt"]
        actual_prompt = _ids(request.get("prompt_ids"), "request.prompt_ids")
        if expected_prompt is None or actual_prompt != expected_prompt:
            raise ReplayError(f"replay prompt mismatch for {self.sample_id!r} phase {phase['kind']}")
        if _plain(sampling) != _plain(phase.get("sampling", {})):
            raise ReplayError(f"replay sampling parameters mismatch for {self.sample_id!r} phase {phase['kind']}")
        self.requests.append({"kind": phase["kind"], "prompt_hash": _json_hash(actual_prompt), "sampling_hash": _json_hash(sampling)})
        self.phase_index += 1
        original_extra = copy.deepcopy(phase.get("server_extra_fields") or {})
        extra = dict(original_extra) if isinstance(original_extra, Mapping) else {"recorded_server_extra_fields": original_extra}
        extra.update(
            {
                "replay_source": str(self.corpus.source),
                "replay_source_sha256": self.corpus.source_hash,
                "replay_record_sha256": self.record["record_hash"],
                "replay_phase_sha256": _json_hash(phase),
                "replay_original_server_extra_fields": original_extra,
            }
        )
        return TokenOutput(
            token_ids=list(phase["token_ids"]),
            log_probs=copy.deepcopy(phase.get("logprobs")),
            stop_reason=phase.get("stop_reason"),
            extra_fields=extra,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "replayed": not self.fallback,
            "replay_source": str(self.corpus.source),
            "replay_source_sha256": self.corpus.source_hash,
            "replay_record_sha256": self.record.get("record_hash") if self.record else None,
            "replayed_phase_count": self.phase_index,
            "validation_real_backend_fallback": bool(self.fallback and self.validate),
            "validation_replay_bypassed_routing": self.native_validation_bypassed,
            "request_hashes": list(self.requests),
        }


@register("error_span_opd_replay")
class ReplayErrorSpanAgentLoop(ErrorSpanAgentLoop):
    """Run the native staged loop against an immutable recorded trajectory."""

    def __init__(self, *args: Any, replay_source: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config.error_span_opd.run_kind != "technical_smoke":
            raise ReplayError("ReplayErrorSpanAgentLoop is restricted to run_kind=technical_smoke")
        trainer = self.config.trainer
        if int(trainer.total_training_steps) != 1 or str(trainer.resume_mode) != "disable":
            raise ReplayError("replay requires trainer.total_training_steps=1 and trainer.resume_mode=disable")
        configured_model = str(self.config.actor_rollout_ref.model.path)
        corpus = _Corpus(replay_source)
        recorded_model = str(json.loads((corpus.source / "launch_receipt.json").read_text(encoding="utf-8"))["models"]["student"]["path"])
        if configured_model != recorded_model:
            raise ReplayError(f"replay Student model mismatch: config={configured_model!r}, source={recorded_model!r}")
        self.replay_corpus = corpus

    async def run(self, sampling_params: dict[str, Any], validate: bool = False, **kwargs: Any) -> Any:
        sid = _sample_id(kwargs)
        seed = sampling_params.get("seed")
        known_validation = False
        if validate and seed is not None:
            try:
                known_validation = (sid or "", True, int(seed)) in self.replay_corpus.records
            except (TypeError, ValueError):
                known_validation = False
        client = RecordedClient(self.server_manager, self.replay_corpus, sid, validate)
        client.native_validation_bypassed = known_validation
        original_client = self.server_manager
        self.server_manager = client
        try:
            # Baseline monitor receipts contain only the two staged Student
            # requests.  Native validation may otherwise issue four probes;
            # known replay rows therefore take the standard early-return path.
            output = await super().run(sampling_params, validate=(validate and not known_validation), **kwargs)
        finally:
            self.server_manager = original_client
        if client.record is not None and not client.fallback:
            if client.phase_index != len(client.record["phases"]):
                raise ReplayError(f"replay ended before all recorded phases for {client.sample_id!r}")
            if _ids(output.prompt_ids, "output.prompt_ids") != client.record["prompt"]:
                raise ReplayError(f"replay output prompt changed for {client.sample_id!r}")
            if _ids(output.response_ids, "output.response_ids") != client.record["response"]:
                raise ReplayError(f"replay output response changed for {client.sample_id!r}")
        provenance = client.provenance()
        output.extra_fields["replay_provenance"] = provenance
        receipt = output.extra_fields.get("routed_grounding_receipt")
        if isinstance(receipt, Mapping):
            receipt = dict(receipt)
            receipt["replay_provenance"] = provenance
            output.extra_fields["routed_grounding_receipt"] = receipt
        return output


__all__ = ["RecordedClient", "ReplayError", "ReplayErrorSpanAgentLoop"]
