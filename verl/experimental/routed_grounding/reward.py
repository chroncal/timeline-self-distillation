# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only reward for the routed-grounding validation path.

The legacy RayPPO reward manager still asks for a scalar score during
validation, even when task rewards and policy-gradient updates are disabled
for the distillation experiment.  ``refcocog_umd`` is not one of verl's
default reward sources, so this small adapter keeps validation on the same
geometry contract as the routed agent loop instead of falling through to the
default ``NotImplementedError``.

The routing receipt is intentionally the source of truth.  It contains every
candidate instance and the target annotation, whereas ``ground_truth`` only
contains the target box prepared for the model prompt.  No image, torch,
numpy, or GPU dependency is needed here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load_router() -> Any:
    """Load the dependency-free router in both package and file-loader modes.

    ``verl.trainer.ppo.reward`` loads a configured custom reward by file path,
    which means this module has no package context on some legacy workers.
    Loading the sibling router directly preserves the CPU-only contract in
    that mode.  Normal package imports reuse the already-imported router.
    """

    try:
        from . import router as router_module

        return router_module
    except (ImportError, ModuleNotFoundError):
        router_path = Path(__file__).resolve().with_name("router.py")
        module_name = "_routed_grounding_reward_router"
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            return loaded
        spec = importlib.util.spec_from_file_location(module_name, router_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load CPU router from {router_path}") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


_ROUTER = _load_router()
Instance = _ROUTER.Instance
coerce_instances = _ROUTER.coerce_instances
match_prediction = _ROUTER.match_prediction
parse_response = _ROUTER.parse_response
xywh_to_xyxy = _ROUTER.xywh_to_xyxy
xyxy_iou = _ROUTER.xyxy_iou


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, Mapping) else None
    return None


def _same_id(left: Any, right: Any) -> bool:
    """Compare JSON scalar IDs without treating ``17`` and ``17.5`` alike."""

    if left == right:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError, OverflowError):
        return str(left).strip() == str(right).strip()
    return math.isfinite(left_number) and math.isfinite(right_number) and left_number == right_number


def _instance_id(instance: Mapping[str, Any]) -> Any:
    return instance.get("ann_id", instance.get("id"))


def _instance_values(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Normalize list and id-keyed mapping forms used by routing receipts."""

    if isinstance(value, Mapping):
        if "bbox" in value or "bbox_xywh" in value:
            return (value,)
        values = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                candidate = dict(item)
                candidate.setdefault("ann_id", key)
                values.append(candidate)
        return tuple(values)
    if isinstance(value, str | bytes | bytearray) or value is None:
        return ()
    try:
        return tuple(item for item in value if isinstance(item, Mapping))
    except TypeError:
        return ()


def _record_from_extra(extra_info: Any) -> Mapping[str, Any] | None:
    """Read the complete routing record from ``extra_info``."""

    extra = _as_mapping(extra_info)
    if extra is None:
        return None
    record = _as_mapping(extra.get("routing_record_json"))
    if record is None:
        return None
    # Accept a nested receipt without allowing any other extra_info field to
    # replace the manifest's instances or target.
    nested = record.get("routing_record")
    return _as_mapping(nested) if isinstance(nested, Mapping) else record


def _finite_dimension(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _target_instance(instances: Sequence[Mapping[str, Any]], target_id: Any) -> Mapping[str, Any] | None:
    if target_id is None:
        return None
    return next(
        (instance for instance in instances if _same_id(_instance_id(instance), target_id)),
        None,
    )


def _target_xyxy(
    record: Mapping[str, Any], target: Mapping[str, Any] | None
) -> tuple[tuple[float, float, float, float] | None, float | None, float | None]:
    """Return target ``xyxy`` in source coordinates and image dimensions."""

    width = _finite_dimension(record.get("image_width", record.get("width")))
    height = _finite_dimension(record.get("image_height", record.get("height")))
    bbox: Any = None
    if target is not None:
        bbox = target.get("bbox_xywh", target.get("bbox"))
        width = _finite_dimension(target.get("image_width", target.get("width"))) or width
        height = _finite_dimension(target.get("image_height", target.get("height"))) or height
    if bbox is None:
        bbox = record.get("target_bbox", record.get("target_bbox_xywh"))
    if bbox is None:
        return None, width, height
    try:
        return xywh_to_xyxy(bbox), width, height
    except (TypeError, ValueError, OverflowError):
        return None, width, height


def _prediction_to_source(
    bbox: Sequence[float], width: float | None, height: float | None
) -> tuple[float, float, float, float]:
    if width is None or height is None:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0


def _target_iou(bbox: Sequence[float], record: Mapping[str, Any], target: Mapping[str, Any] | None) -> float:
    target_box, width, height = _target_xyxy(record, target)
    if target_box is None:
        return 0.0
    try:
        score = xyxy_iou(_prediction_to_source(bbox, width, height), target_box)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return float(score) if math.isfinite(float(score)) else 0.0


def _zero_result(*, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "score": 0.0,
        "acc@.5": 0.0,
        "acc@.75": 0.0,
        "acc@.9": 0.0,
        # Decimal aliases make the metrics convenient for generic loggers
        # while retaining the exact protocol spellings above.
        "acc@0.5": 0.0,
        "acc@0.75": 0.0,
        "acc@0.9": 0.0,
        "bbox_valid": False,
        "wrong_instance": False,
        "target_match": False,
        "target_match_iou": 0.0,
        "match_status": "invalid",
        "match_iou": 0.0,
        "error": error or "",
        "match_error": "",
    }
    return result


def _restore_prompt_owned_think(response: Any) -> Any:
    """Restore Qwen's opening tag when it belongs to the chat prompt tokens."""

    if isinstance(response, str) and not response.startswith("<think>") and "</think>" in response:
        return "<think>" + response
    return response


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """Return target IoU and auditable router metrics for one response.

    ``data_source`` and ``ground_truth`` remain in the public legacy reward
    signature.  Geometry is deliberately taken from the complete
    ``routing_record_json`` receipt so a distractor annotation can never be
    mistaken for the target.  Malformed responses or records fail closed with
    zero IoU and a serializable result rather than raising from validation.
    """

    del data_source, ground_truth, kwargs
    record = _record_from_extra(extra_info)
    if record is None:
        return _zero_result(error="extra_info.routing_record_json is missing or invalid")

    target_id = record.get("target_ann_id", record.get("target_id"))
    instances = _instance_values(record.get("instances"))
    target = _target_instance(instances, target_id)
    try:
        # ``routing_record_json`` preserves COCO metadata such as category
        # and category_id.  Reuse the router's coercion boundary so that
        # those non-geometric fields are ignored exactly as they are in the
        # agent loop.
        router_instances = coerce_instances(instances)
    except (TypeError, ValueError, OverflowError) as error:
        return _zero_result(error=f"invalid routing instances: {error}")

    parsed = parse_response(_restore_prompt_owned_think(solution_str))
    if not parsed.format_valid or parsed.bbox is None:
        result = _zero_result(error=parsed.error or "invalid response")
        result["bbox_valid"] = False
        return result

    score = _target_iou(parsed.bbox, record, target)
    match = match_prediction(parsed.bbox, router_instances)
    target_match = match.status == "matched" and _same_id(match.ann_id, target_id)
    wrong_instance = match.status == "matched" and not target_match
    target_match_iou = float(match.iou) if target_match else 0.0
    result = {
        "score": score,
        "acc@.5": float(score >= 0.5),
        "acc@.75": float(score >= 0.75),
        "acc@.9": float(score >= 0.9),
        "acc@0.5": float(score >= 0.5),
        "acc@0.75": float(score >= 0.75),
        "acc@0.9": float(score >= 0.9),
        "bbox_valid": True,
        "wrong_instance": wrong_instance,
        "target_match": target_match,
        "target_match_iou": target_match_iou,
        "match_status": match.status,
        "match_iou": float(match.iou),
        "error": "",
        "match_error": match.error or "",
    }
    return result


__all__ = ["compute_score"]
