"""Pure data and token-prefix contracts for formal MM-GCoT v2 training.

The formal trainer consumes a frozen natural trajectory and a frozen v3p5
bridge.  This module keeps that boundary independent of the model runtime:
records are validated before use, bridge text is parsed by the frozen parser,
and every L/R/E prefix is made from the saved token IDs.  No reference box is
part of any function that builds an inference context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Literal

from mmgcot_diagnostic.protocol import BOX_OPEN, bbox_suffix as unconditioned_bbox_suffix
from mmgcot_diagnostic.protocol import early_offset as protocol_early_offset
from mmgcot_timeline_training import bridge_v3


BridgeStatus = Literal["valid", "format_or_incomplete"]
ArmName = Literal["L", "R", "E"]
ARMS: tuple[ArmName, ...] = ("L", "R", "E")
V3P5_VERSION = bridge_v3.BRIDGE_VERSION
MAX_REASONING_TOKENS = 4096
MAX_BRIDGE_TOKENS = 160


class FormalContextError(ValueError):
    """Base error for malformed or drifted formal-context records."""


class ProvenanceError(FormalContextError):
    """Raised when a frozen record no longer matches its provenance."""


def _as_token_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FormalContextError(f"{field} must be a sequence of token IDs")
    result: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise FormalContextError(f"{field}[{index}] is not a non-negative integer")
        result.append(token)
    return tuple(result)


def _require_text(record: Mapping[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = record.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise FormalContextError(f"record field {field!r} must be a non-empty string")
    return value


def _optional_text(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise FormalContextError(f"record field {field!r} must be a string or null")
    return value


def _optional_int(record: Mapping[str, Any], field: str) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalContextError(f"record field {field!r} must be an integer or null")
    return value


def _hash_json(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FormalContextError("provenance payload is not finite JSON") from error
    return hashlib.sha256(payload).hexdigest()


_HASH_FIELDS = frozenset(
    {
        "record_sha256",
        "source_record_sha256",
        "trajectory_sha256",
        "trajectory_provenance_sha256",
        "provenance_sha256",
    }
)


def _is_reference_field(name: str) -> bool:
    normalized = name.casefold()
    return normalized in {
        "ground_truth_bbox",
        "gt_bbox",
        "ground_truth",
        "gt",
    } or normalized.startswith("ground_truth_")


def _provenance_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical inference-record projection used for hashing.

    Hash fields are excluded so a hash can validate its own record.  Reference
    annotations are excluded deliberately: they are not inference inputs and
    must not become part of the context provenance contract.
    """

    return {
        str(key): value
        for key, value in sorted(record.items(), key=lambda item: str(item[0]))
        if str(key) not in _HASH_FIELDS and not _is_reference_field(str(key))
    }


def trajectory_provenance_hash(record: Mapping[str, Any]) -> str:
    """Hash all frozen record content except hashes and reference annotations."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return _hash_json(_provenance_payload(record))


def token_ids_sha256(ids: Sequence[int]) -> str:
    """Hash token IDs with the same compact JSON contract as bridge v2."""

    return _hash_json(list(_as_token_tuple(ids, field="token_ids")))


@dataclass(frozen=True)
class TrajectoryProvenance:
    """Hashes that identify the frozen trajectory used by a context."""

    sample_id: str
    image_id: str
    trajectory_index: int
    image_sha256: str
    prompt_token_ids_sha256: str
    reasoning_token_ids_sha256: str
    trajectory_sha256: str
    source_record_sha256: str | None = None

    @property
    def hashes(self) -> Mapping[str, str]:
        values = {
            "image_sha256": self.image_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "reasoning_token_ids_sha256": self.reasoning_token_ids_sha256,
            "trajectory_sha256": self.trajectory_sha256,
        }
        if self.source_record_sha256 is not None:
            values["source_record_sha256"] = self.source_record_sha256
        return MappingProxyType(values)


@dataclass(frozen=True)
class BridgeDescription:
    """The typed result of the frozen v3p5 bridge parser."""

    task_answer: str
    target_entity_reference: str
    bridge_parse_status: BridgeStatus | str
    task_answer_status: str
    target_entity_reference_status: str
    bridge_version: str = V3P5_VERSION

    @property
    def usable(self) -> bool:
        return (
            self.bridge_parse_status == "valid"
            and self.target_entity_reference_status == "usable"
            and bool(self.target_entity_reference.strip())
            and self.target_entity_reference.strip().casefold() != "unresolved"
        )

    @property
    def target_description(self) -> str | None:
        return self.target_entity_reference if self.usable else None


def parse_v3p5(text: str, *, completed: bool = True) -> BridgeDescription:
    """Parse one bridge continuation using the unchanged v3p5 parser."""

    if not isinstance(text, str):
        raise TypeError("bridge text must be a string")
    parsed = bridge_v3.parse_bridge(text, completed=completed)
    return BridgeDescription(
        task_answer=parsed["task_answer"],
        target_entity_reference=parsed["target_entity_reference"],
        bridge_parse_status=parsed["bridge_parse_status"],
        task_answer_status=parsed["task_answer_status"],
        target_entity_reference_status=parsed["target_entity_reference_status"],
    )


parse_bridge_v3p5 = parse_v3p5
parse_v3p5_bridge = parse_v3p5


@dataclass(frozen=True)
class FrozenTrajectory:
    """Validated model-generated trajectory data, with no reference box field."""

    sample_id: str
    image_id: str
    image_path: str
    image_sha256: str
    question: str
    trajectory_index: int
    prompt_token_ids: tuple[int, ...]
    rendered_prompt: str
    reasoning_token_ids: tuple[int, ...]
    reasoning_token_ids_sha256: str
    prompt_token_ids_sha256: str
    bridge_version: str
    bridge_parse_status: str
    target_entity_reference: str
    target_entity_reference_status: str
    task_answer: str
    task_answer_status: str
    bridge_text: str | None
    bridge_completed: bool
    bridge_token_ids: tuple[int, ...]
    reasoning_finish: str | None
    reasoning_seed: int | None
    bridge_seed: int | None
    trajectory_sha256: str
    source_record_sha256: str | None = None

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "FrozenTrajectory":
        if isinstance(record, cls):
            return record
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping or FrozenTrajectory")

        sample_id = _require_text(record, "sample_id")
        image_id = _require_text(record, "image_id")
        image_path = _require_text(record, "image_path")
        image_sha256 = _require_text(record, "image_sha256")
        question = _require_text(record, "question")
        trajectory_index = record.get("trajectory_index")
        if isinstance(trajectory_index, bool) or not isinstance(trajectory_index, int):
            raise FormalContextError("trajectory_index must be a non-negative integer")
        if trajectory_index < 0:
            raise FormalContextError("trajectory_index must be non-negative")

        prompt_ids = _as_token_tuple(record.get("prompt_token_ids"), field="prompt_token_ids")
        rendered_prompt = _require_text(record, "rendered_prompt", allow_empty=True)
        reasoning_ids = _as_token_tuple(
            record.get("reasoning_token_ids"), field="reasoning_token_ids"
        )
        if len(reasoning_ids) > MAX_REASONING_TOKENS:
            raise FormalContextError(
                f"reasoning_token_ids exceeds the frozen limit {MAX_REASONING_TOKENS}"
            )

        reasoning_hash = _require_text(record, "reasoning_token_ids_sha256")
        actual_reasoning_hash = token_ids_sha256(reasoning_ids)
        if reasoning_hash != actual_reasoning_hash:
            raise ProvenanceError("reasoning token hash does not match saved token IDs")
        prompt_hash = token_ids_sha256(prompt_ids)
        stored_prompt_hash = record.get("prompt_token_ids_sha256")
        if stored_prompt_hash is not None and stored_prompt_hash != prompt_hash:
            raise ProvenanceError("prompt token hash does not match saved token IDs")

        bridge_version = _require_text(record, "bridge_version")
        if bridge_version != V3P5_VERSION:
            raise FormalContextError(
                f"formal context requires bridge {V3P5_VERSION!r}, got {bridge_version!r}"
            )

        generation = record.get("bridge_generation")
        if generation is not None and not isinstance(generation, Mapping):
            raise FormalContextError("bridge_generation must be a mapping")
        bridge_text = (
            generation.get("text") if isinstance(generation, Mapping) else record.get("bridge_text")
        )
        if bridge_text is not None and not isinstance(bridge_text, str):
            raise FormalContextError("bridge generation text must be a string or null")
        bridge_completed = bool(
            generation.get("completed", False) if isinstance(generation, Mapping) else False
        )
        bridge_ids_value = (
            generation.get("token_ids", ()) if isinstance(generation, Mapping) else ()
        )
        bridge_ids = _as_token_tuple(bridge_ids_value, field="bridge_generation.token_ids")

        bridge_parse_status = str(record.get("bridge_parse_status", "format_or_incomplete"))
        target_reference = str(record.get("target_entity_reference", ""))
        target_status = str(record.get("target_entity_reference_status", "invalid"))
        task_answer = str(record.get("task_answer", ""))
        task_status = str(record.get("task_answer_status", "invalid"))

        supplied_hashes: dict[str, str] = {}
        for field in (
            "source_record_sha256",
            "trajectory_provenance_sha256",
            "trajectory_sha256",
            "record_sha256",
        ):
            candidate = record.get(field)
            if candidate is not None:
                if not isinstance(candidate, str) or not candidate:
                    raise FormalContextError(f"record field {field!r} must be a hash string")
                supplied_hashes[field] = candidate

        canonical_hash = trajectory_provenance_hash(record)
        if any(value != canonical_hash for value in supplied_hashes.values()):
            raise ProvenanceError("frozen trajectory provenance hash mismatch")
        source_hash = supplied_hashes.get("source_record_sha256")
        if source_hash is None:
            source_hash = next(iter(supplied_hashes.values()), None)
        if len(bridge_ids) > MAX_BRIDGE_TOKENS:
            raise FormalContextError(
                f"bridge_generation.token_ids exceeds the frozen limit {MAX_BRIDGE_TOKENS}"
            )

        return cls(
            sample_id=sample_id,
            image_id=image_id,
            image_path=image_path,
            image_sha256=image_sha256,
            question=question,
            trajectory_index=trajectory_index,
            prompt_token_ids=prompt_ids,
            rendered_prompt=rendered_prompt,
            reasoning_token_ids=reasoning_ids,
            reasoning_token_ids_sha256=reasoning_hash,
            prompt_token_ids_sha256=prompt_hash,
            bridge_version=bridge_version,
            bridge_parse_status=bridge_parse_status,
            target_entity_reference=target_reference,
            target_entity_reference_status=target_status,
            task_answer=task_answer,
            task_answer_status=task_status,
            bridge_text=bridge_text,
            bridge_completed=bridge_completed,
            bridge_token_ids=bridge_ids,
            reasoning_finish=_optional_text(record, "reasoning_finish"),
            reasoning_seed=_optional_int(record, "reasoning_seed"),
            bridge_seed=_optional_int(record, "bridge_seed"),
            trajectory_sha256=canonical_hash,
            source_record_sha256=source_hash,
        )

    @property
    def provenance(self) -> TrajectoryProvenance:
        return TrajectoryProvenance(
            sample_id=self.sample_id,
            image_id=self.image_id,
            trajectory_index=self.trajectory_index,
            image_sha256=self.image_sha256,
            prompt_token_ids_sha256=self.prompt_token_ids_sha256,
            reasoning_token_ids_sha256=self.reasoning_token_ids_sha256,
            trajectory_sha256=self.trajectory_sha256,
            source_record_sha256=self.source_record_sha256,
        )

    @property
    def provenance_hashes(self) -> Mapping[str, str]:
        return self.provenance.hashes

    def inference_input(self) -> dict[str, str]:
        """Return the complete model input allowlist for inference."""

        return {"image_path": self.image_path, "question": self.question}


TrajectoryRecord = FrozenTrajectory
FrozenTrajectoryRecord = FrozenTrajectory


def validate_frozen_trajectory(
    record: Mapping[str, Any] | FrozenTrajectory,
    expected: Mapping[str, Any] | None = None,
) -> FrozenTrajectory:
    """Validate a frozen trajectory and optional selection-row identity."""

    trajectory = FrozenTrajectory.from_mapping(record)
    if expected is None:
        return trajectory
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a mapping or null")

    identity_fields = ("sample_id", "image_id", "image_sha256", "image_path", "question")
    for field in identity_fields:
        if field in expected and str(expected[field]) != str(getattr(trajectory, field)):
            raise ProvenanceError(f"frozen record changed on {field}")
    if "trajectory_index" in expected and int(expected["trajectory_index"]) != trajectory.trajectory_index:
        raise ProvenanceError("frozen record changed on trajectory_index")
    if "reasoning_token_ids_sha256" in expected and str(
        expected["reasoning_token_ids_sha256"]
    ) != trajectory.reasoning_token_ids_sha256:
        raise ProvenanceError("frozen record changed on reasoning_token_ids_sha256")
    return trajectory


validate_trajectory = validate_frozen_trajectory
validate_trajectory_record = validate_frozen_trajectory


def parse_trajectory_bridge(
    record: FrozenTrajectory | Mapping[str, Any],
) -> BridgeDescription:
    """Parse and validate the bridge attached to a frozen trajectory."""

    trajectory = FrozenTrajectory.from_mapping(record)
    if trajectory.bridge_text is not None:
        parsed = parse_v3p5(trajectory.bridge_text, completed=trajectory.bridge_completed)
        stored = {
            "task_answer": trajectory.task_answer,
            "target_entity_reference": trajectory.target_entity_reference,
            "bridge_parse_status": trajectory.bridge_parse_status,
            "task_answer_status": trajectory.task_answer_status,
            "target_entity_reference_status": trajectory.target_entity_reference_status,
        }
        for field, value in stored.items():
            if value != parsed.__dict__[field]:
                raise ProvenanceError(f"stored bridge field drifted: {field}")
        return parsed

    # Minimal frozen fixtures may carry only the already parsed v3p5 fields.
    return BridgeDescription(
        task_answer=trajectory.task_answer,
        target_entity_reference=trajectory.target_entity_reference,
        bridge_parse_status=trajectory.bridge_parse_status,
        task_answer_status=trajectory.task_answer_status,
        target_entity_reference_status=trajectory.target_entity_reference_status,
    )


parse_bridge_record = parse_trajectory_bridge


def bbox_suffix_v3p5(question: str, target_entity_reference: str | None) -> str:
    """Return the frozen v3p5 bbox suffix, with the protocol fallback."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if (
        target_entity_reference is None
        or not isinstance(target_entity_reference, str)
        or not target_entity_reference.strip()
        or target_entity_reference.strip().casefold() == "unresolved"
    ):
        return unconditioned_bbox_suffix(question, None)
    return bridge_v3.bbox_suffix_v3(question, target_entity_reference)


bbox_suffix = bbox_suffix_v3p5
build_bbox_suffix = bbox_suffix_v3p5


def suffix_for_trajectory(record: FrozenTrajectory | Mapping[str, Any]) -> tuple[str, BridgeDescription]:
    trajectory = FrozenTrajectory.from_mapping(record)
    bridge = parse_trajectory_bridge(trajectory)
    return bbox_suffix_v3p5(trajectory.question, bridge.target_description), bridge


def early_offset(length: int, sentence_boundaries: Iterable[int]) -> int:
    """Use the diagnostic's fixed quarter-length sentence-boundary rule."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative integer")
    boundaries = []
    for boundary in sentence_boundaries:
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise ValueError("sentence boundaries must be integers")
        if boundary < 0 or boundary > length:
            raise ValueError("sentence boundary is outside the reasoning sequence")
        boundaries.append(boundary)
    return protocol_early_offset(length, boundaries)


def compute_early_offset(
    reasoning_token_ids: Sequence[int] | int,
    sentence_boundaries: Iterable[int] = (),
) -> int:
    length = (
        reasoning_token_ids
        if isinstance(reasoning_token_ids, int) and not isinstance(reasoning_token_ids, bool)
        else len(_as_token_tuple(reasoning_token_ids, field="reasoning_token_ids"))
    )
    return early_offset(length, sentence_boundaries)


@dataclass(frozen=True)
class NativeOpeningSplit:
    """Native tokenization of a suffix split before its final opening token."""

    suffix_text: str
    suffix_token_ids: tuple[int, ...]
    body_token_ids: tuple[int, ...]
    opening_token_ids: tuple[int, ...]
    opening_token_id: int

    @property
    def prefix_token_ids(self) -> tuple[int, ...]:
        return self.body_token_ids

    @property
    def opening_id(self) -> int:
        return self.opening_token_id

    def __iter__(self):
        # Convenient unpacking for callers that only need cache-body IDs and
        # the single native token that supplies the first bbox prediction.
        yield self.body_token_ids
        yield self.opening_token_id


def _encode_exact(tokenizer: Any, text: str, *, label: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    ids = _as_token_tuple(encoded, field=f"{label}_token_ids")
    if not ids:
        raise AssertionError(f"tokenizer returned no IDs for {label}")
    try:
        decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
    except TypeError:
        decoded = tokenizer.decode(list(ids))
    if decoded != text:
        raise AssertionError(f"{label} does not round-trip through the tokenizer")
    return ids


def split_native_opening(tokenizer: Any, suffix_text: str) -> NativeOpeningSplit:
    """Split a suffix at the final native token of ``BOX_OPEN``.

    The complete suffix is encoded once.  The opening is never re-tokenized as
    a replacement prefix; only its final token is fed after the cached body,
    matching the formal trainer's causal step.  The assertion catches a
    tokenizer/model contract drift before any GPU work begins.
    """

    if not isinstance(suffix_text, str) or not suffix_text.endswith(BOX_OPEN):
        raise AssertionError("suffix must end with the frozen bbox opening")
    suffix_ids = _encode_exact(tokenizer, suffix_text, label="suffix")
    opening_ids = _encode_exact(tokenizer, BOX_OPEN, label="opening")
    if len(suffix_ids) < 2:
        raise AssertionError("native suffix must contain a body before its opening token")
    if suffix_ids[-1] != opening_ids[-1]:
        raise AssertionError("native suffix opening final token differs from BOX_OPEN")
    return NativeOpeningSplit(
        suffix_text=suffix_text,
        suffix_token_ids=suffix_ids,
        body_token_ids=suffix_ids[:-1],
        opening_token_ids=opening_ids,
        opening_token_id=suffix_ids[-1],
    )


split_opening_last_token = split_native_opening
native_opening_split = split_native_opening


@dataclass(frozen=True)
class ArmTokenPrefix:
    """The exact token sequences used to construct one L/R/E cache state."""

    arm: ArmName
    reasoning_prefix_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]
    context_token_ids: tuple[int, ...]
    opening_prefixed_token_ids: tuple[int, ...]

    @property
    def prefix_token_ids(self) -> tuple[int, ...]:
        return self.continuation_token_ids

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.context_token_ids


def build_lre_token_prefixes(
    prompt_token_ids: Sequence[int],
    reasoning_token_ids: Sequence[int],
    suffix_body_token_ids: Sequence[int],
    opening_token_id: int,
    early_token_offset: int,
) -> Mapping[str, ArmTokenPrefix]:
    """Build deterministic L/R/E prefixes from raw saved token IDs."""

    prompt = _as_token_tuple(prompt_token_ids, field="prompt_token_ids")
    reasoning = _as_token_tuple(reasoning_token_ids, field="reasoning_token_ids")
    suffix_body = _as_token_tuple(suffix_body_token_ids, field="suffix_body_token_ids")
    if isinstance(opening_token_id, bool) or not isinstance(opening_token_id, int):
        raise ValueError("opening_token_id must be an integer")
    if (
        isinstance(early_token_offset, bool)
        or not isinstance(early_token_offset, int)
        or not 0 <= early_token_offset <= len(reasoning)
    ):
        raise ValueError("early_token_offset must be within reasoning_token_ids")

    histories: dict[ArmName, tuple[int, ...]] = {
        "L": reasoning,
        "R": (),
        "E": reasoning[:early_token_offset],
    }
    result: dict[str, ArmTokenPrefix] = {}
    for arm in ARMS:
        history = histories[arm]
        continuation = history + suffix_body
        context = prompt + continuation
        result[arm] = ArmTokenPrefix(
            arm=arm,
            reasoning_prefix_ids=history,
            continuation_token_ids=continuation,
            context_token_ids=context,
            opening_prefixed_token_ids=context + (opening_token_id,),
        )
    return MappingProxyType(result)


build_lre_prefixes = build_lre_token_prefixes
build_arm_prefixes = build_lre_token_prefixes


@dataclass(frozen=True)
class FormalContext:
    """Complete CPU-built context contract for the formal L/R/E trainer."""

    trajectory: FrozenTrajectory
    bridge: BridgeDescription
    suffix_text: str
    suffix_token_ids: tuple[int, ...]
    suffix_body_token_ids: tuple[int, ...]
    opening_token_ids: tuple[int, ...]
    opening_token_id: int
    early_token_offset: int
    arm_prefixes: Mapping[str, ArmTokenPrefix]
    fallback_used: bool

    @property
    def early_offset(self) -> int:
        return self.early_token_offset

    @property
    def target_description(self) -> str | None:
        return self.bridge.target_description

    @property
    def has_description(self) -> bool:
        return self.target_description is not None

    @property
    def target_entity_reference(self) -> str | None:
        return self.target_description

    @property
    def prefixes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(
            {arm: self.arm_prefixes[arm].reasoning_prefix_ids for arm in ARMS}
        )

    @property
    def token_prefixes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(
            {arm: self.arm_prefixes[arm].continuation_token_ids for arm in ARMS}
        )

    @property
    def context_token_prefixes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(
            {arm: self.arm_prefixes[arm].context_token_ids for arm in ARMS}
        )

    @property
    def L(self) -> ArmTokenPrefix:
        return self.arm_prefixes["L"]

    @property
    def R(self) -> ArmTokenPrefix:
        return self.arm_prefixes["R"]

    @property
    def E(self) -> ArmTokenPrefix:
        return self.arm_prefixes["E"]

    @property
    def provenance(self) -> TrajectoryProvenance:
        return self.trajectory.provenance


FormalContextRecord = FormalContext


def build_formal_context(
    record: Mapping[str, Any] | FrozenTrajectory,
    tokenizer: Any,
    sentence_boundaries: Iterable[int] = (),
    *,
    boundaries: Iterable[int] | None = None,
    expected: Mapping[str, Any] | None = None,
) -> FormalContext:
    """Validate a trajectory and build the fixed v3p5 L/R/E context."""

    if boundaries is not None:
        if tuple(sentence_boundaries):
            raise ValueError("provide sentence_boundaries or boundaries, not both")
        sentence_boundaries = boundaries
    trajectory = validate_frozen_trajectory(record, expected=expected)
    bridge = parse_trajectory_bridge(trajectory)
    suffix_text = bbox_suffix_v3p5(trajectory.question, bridge.target_description)
    split = split_native_opening(tokenizer, suffix_text)
    early = compute_early_offset(trajectory.reasoning_token_ids, sentence_boundaries)
    prefixes = build_lre_token_prefixes(
        trajectory.prompt_token_ids,
        trajectory.reasoning_token_ids,
        split.body_token_ids,
        split.opening_token_id,
        early,
    )
    return FormalContext(
        trajectory=trajectory,
        bridge=bridge,
        suffix_text=suffix_text,
        suffix_token_ids=split.suffix_token_ids,
        suffix_body_token_ids=split.body_token_ids,
        opening_token_ids=split.opening_token_ids,
        opening_token_id=split.opening_token_id,
        early_token_offset=early,
        arm_prefixes=prefixes,
        fallback_used=not bridge.usable,
    )


build_context = build_formal_context
build_trajectory_context = build_formal_context


def inference_input(record: Mapping[str, Any] | FrozenTrajectory) -> dict[str, str]:
    """Return only image/question fields allowed to enter model inference."""

    return FrozenTrajectory.from_mapping(record).inference_input()


model_input = inference_input


__all__ = [
    "ARMS",
    "ArmName",
    "ArmTokenPrefix",
    "BridgeDescription",
    "BridgeStatus",
    "FormalContext",
    "FormalContextError",
    "FormalContextRecord",
    "FrozenTrajectory",
    "FrozenTrajectoryRecord",
    "MAX_REASONING_TOKENS",
    "MAX_BRIDGE_TOKENS",
    "NativeOpeningSplit",
    "ProvenanceError",
    "TrajectoryProvenance",
    "TrajectoryRecord",
    "V3P5_VERSION",
    "bbox_suffix",
    "bbox_suffix_v3p5",
    "build_arm_prefixes",
    "build_bbox_suffix",
    "build_context",
    "build_formal_context",
    "build_lre_prefixes",
    "build_lre_token_prefixes",
    "build_trajectory_context",
    "compute_early_offset",
    "early_offset",
    "inference_input",
    "model_input",
    "native_opening_split",
    "parse_bridge_record",
    "parse_bridge_v3p5",
    "parse_trajectory_bridge",
    "parse_v3p5",
    "parse_v3p5_bridge",
    "split_native_opening",
    "split_opening_last_token",
    "suffix_for_trajectory",
    "token_ids_sha256",
    "trajectory_provenance_hash",
    "validate_frozen_trajectory",
    "validate_trajectory",
    "validate_trajectory_record",
]
