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

"""CPU-only contracts for routed visual-grounding Teacher targets.

This module intentionally contains no model, tokenizer, or accelerator code.  It
is the small text/geometry boundary shared by data preparation and verification:

* a ground-truth crop is made around the box centre with a 1.5x extent;
* coordinates emitted for that crop can be mapped back to the source's
  normalized ``[0, 1000]`` coordinate system;
* localization and referent repair targets have explicit textual contracts;
* target acceptance is decided by geometry rather than by generation success.

The only accepted answer object is ``{"bbox": [x1, y1, x2, y2]}``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

COORDINATE_SCALE = 1000.0
DEFAULT_CROP_SCALE = 1.5
IOU_THRESHOLD = 0.5
LOCALIZATION_GAIN = 0.05
BBOX_FIELD = "bbox"
LOCALIZATION = "localization"
REFERENT = "referent"
TeacherMode: TypeAlias = Literal["localization", "referent"]
BBox: TypeAlias = tuple[float, float, float, float]

DEFAULT_REFERENT_CORRECTION = "The referent is the described object."


def _mode(value: str) -> TeacherMode:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {LOCALIZATION, "localize", "localisation"}:
        return LOCALIZATION
    if normalized in {REFERENT, "reference", "referential"}:
        return REFERENT
    raise ValueError(f"unknown Teacher target mode: {value!r}")


def _numbers(value: Sequence[Any], *, name: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes):
        raise ValueError(f"{name} must be a numeric sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    result: list[float] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{name} coordinates must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} coordinates must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} coordinates must be finite")
        result.append(number)
    return tuple(result)


def _bbox(
    value: Sequence[Any],
    *,
    name: str = "bbox",
    scale: float = COORDINATE_SCALE,
    positive: bool = True,
) -> BBox:
    values = _numbers(value, name=name)
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four coordinates")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("coordinate scale must be finite and positive")
    x1, y1, x2, y2 = values
    if any(number < 0.0 or number > float(scale) for number in values):
        raise ValueError(f"{name} coordinates must lie in [0, {scale}]")
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{name} coordinates must be ordered xyxy")
    if positive and (x2 <= x1 or y2 <= y1):
        raise ValueError(f"{name} must have positive width and height")
    return x1, y1, x2, y2


def _point(value: Sequence[Any], *, name: str, scale: float) -> tuple[float, float]:
    values = _numbers(value, name=name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    if any(number < 0.0 or number > scale for number in values):
        raise ValueError(f"{name} coordinates must lie in [0, {scale}]")
    return values[0], values[1]


def bbox_iou(first: Sequence[Any], second: Sequence[Any]) -> float:
    """Return IoU for two positive, normalized ``xyxy`` boxes."""

    a = _bbox(first, name="first bbox")
    b = _bbox(second, name="second bbox")
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _image_size(image: Any, image_size: Sequence[Any] | None) -> tuple[int, int]:
    """Extract ``(width, height)`` from a PIL-like/array-like image."""

    candidate: Any = image_size
    if candidate is None:
        candidate = getattr(image, "size", None)
        # numpy's ``size`` is an integer number of elements, not dimensions.
        if not isinstance(candidate, Sequence) or isinstance(candidate, str | bytes):
            candidate = None
    if candidate is None:
        shape = getattr(image, "shape", None)
        if shape is not None:
            shape = tuple(shape)
            if len(shape) < 2:
                shape = None
            elif len(shape) == 2:
                # Array convention is (height, width).
                candidate = (shape[1], shape[0])
            elif shape[-1] in (1, 3, 4):
                candidate = (shape[1], shape[0])
            else:
                # Also accommodate channel-first arrays.
                candidate = (shape[-1], shape[-2])
    if candidate is None:
        try:
            height = len(image)
            width = len(image[0])
            candidate = (width, height)
        except (IndexError, TypeError, AttributeError) as exc:
            raise ValueError("image_size is required for an image without dimensions") from exc
    try:
        values = tuple(candidate)
    except TypeError as exc:
        raise ValueError("image_size must be (width, height)") from exc
    if len(values) != 2:
        raise ValueError("image_size must contain width and height")
    width, height = values
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("image dimensions must be positive integers")
    try:
        width_int, height_int = int(width), int(height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image dimensions must be positive integers") from exc
    if width_int <= 0 or height_int <= 0 or width_int != float(width) or height_int != float(height):
        raise ValueError("image dimensions must be positive integers")
    return width_int, height_int


@dataclass(frozen=True)
class CropTransform:
    """Affine mapping from normalized crop coordinates to source coordinates.

    ``crop_box_px`` uses half-open pixel coordinates ``(left, top, right,
    bottom)``.  ``map_bbox`` and ``map_point`` consume coordinates normalized
    to ``[0, 1000]`` relative to the crop and return normalized source
    coordinates in the same range.
    """

    image_size: tuple[int, int]
    crop_box_px: tuple[int, int, int, int]
    coordinate_scale: float = COORDINATE_SCALE

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        if len(self.crop_box_px) != 4:
            raise ValueError("crop_box_px must contain four coordinates")
        left, top, right, bottom = self.crop_box_px
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("crop_box_px must be a non-empty box inside image_size")
        if self.coordinate_scale <= 0 or not math.isfinite(self.coordinate_scale):
            raise ValueError("coordinate_scale must be finite and positive")

    @property
    def image_width(self) -> int:
        return self.image_size[0]

    @property
    def image_height(self) -> int:
        return self.image_size[1]

    @property
    def crop_width(self) -> int:
        return self.crop_box_px[2] - self.crop_box_px[0]

    @property
    def crop_height(self) -> int:
        return self.crop_box_px[3] - self.crop_box_px[1]

    @property
    def crop_bbox(self) -> BBox:
        """The crop extent in the source's normalized coordinate system."""

        left, top, right, bottom = self.crop_box_px
        return (
            left / self.image_width * self.coordinate_scale,
            top / self.image_height * self.coordinate_scale,
            right / self.image_width * self.coordinate_scale,
            bottom / self.image_height * self.coordinate_scale,
        )

    @property
    def affine(self) -> tuple[float, float, float, float]:
        """Return ``(x_scale, y_scale, x_offset, y_offset)`` for crop norm."""

        left, top, _, _ = self.crop_box_px
        return (
            self.crop_width / self.image_width,
            self.crop_height / self.image_height,
            left / self.image_width * self.coordinate_scale,
            top / self.image_height * self.coordinate_scale,
        )

    def map_point(self, point: Sequence[Any]) -> tuple[float, float]:
        """Map one normalized crop point to normalized source coordinates."""

        x, y = _point(point, name="crop point", scale=self.coordinate_scale)
        x_scale, y_scale, x_offset, y_offset = self.affine
        return x * x_scale + x_offset, y * y_scale + y_offset

    crop_to_original = map_point
    crop_point_to_original = map_point

    def map_pixel_point(self, point: Sequence[Any]) -> tuple[float, float]:
        """Map a crop-pixel point to normalized source coordinates."""

        x, y = _numbers(point, name="crop pixel point")
        if len((x, y)) != 2 or x < 0.0 or y < 0.0 or x > self.crop_width or y > self.crop_height:
            raise ValueError("crop pixel point must lie inside the crop")
        left, top, _, _ = self.crop_box_px
        return (
            (left + x) / self.image_width * self.coordinate_scale,
            (top + y) / self.image_height * self.coordinate_scale,
        )

    def map_bbox(self, bbox: Sequence[Any]) -> BBox:
        """Map a normalized crop ``xyxy`` bbox to normalized source coords."""

        x1, y1, x2, y2 = _bbox(bbox, name="crop bbox", scale=self.coordinate_scale)
        left, top = self.map_point((x1, y1))
        right, bottom = self.map_point((x2, y2))
        return left, top, right, bottom

    crop_bbox_to_original = map_bbox
    map_crop_bbox = map_bbox

    def map_pixel_bbox(self, bbox: Sequence[Any]) -> BBox:
        """Map a crop-pixel ``xyxy`` bbox to normalized source coordinates."""

        values = _numbers(bbox, name="crop pixel bbox")
        if len(values) != 4:
            raise ValueError("crop pixel bbox must contain exactly four coordinates")
        x1, y1, x2, y2 = values
        if not (0.0 <= x1 < x2 <= self.crop_width and 0.0 <= y1 < y2 <= self.crop_height):
            raise ValueError("crop pixel bbox must be ordered and inside the crop")
        left, top = self.map_pixel_point((x1, y1))
        right, bottom = self.map_pixel_point((x2, y2))
        return left, top, right, bottom

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": [self.image_width, self.image_height],
            "crop_box_px": list(self.crop_box_px),
            "crop_size": [self.crop_width, self.crop_height],
            "crop_bbox": list(self.crop_bbox),
            "affine": list(self.affine),
            "coordinate_scale": self.coordinate_scale,
        }

    as_dict = to_dict

    def __call__(self, bbox: Sequence[Any]) -> BBox:
        return self.map_bbox(bbox)

    def __iter__(self) -> Iterator[int]:
        return iter(self.crop_box_px)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self.crop_box_px[key]
        return self.to_dict()[key]


@dataclass(frozen=True)
class CropResult:
    """The cropped image and its source-coordinate transform."""

    crop: Any
    mapping: CropTransform

    @property
    def crop_box_px(self) -> tuple[int, int, int, int]:
        return self.mapping.crop_box_px

    @property
    def crop_bbox(self) -> BBox:
        return self.mapping.crop_bbox

    def __iter__(self) -> Iterator[Any]:
        # This permits the convenient ``crop, mapping = crop_gt_bbox(...)``.
        yield self.crop
        yield self.mapping

    def to_dict(self) -> dict[str, Any]:
        return {"crop": self.crop, "mapping": self.mapping.to_dict()}


def compute_gt_crop_transform(
    gt_bbox: Sequence[Any],
    image_size: Sequence[Any],
    *,
    expansion: float = DEFAULT_CROP_SCALE,
    coordinate_scale: float = COORDINATE_SCALE,
) -> CropTransform:
    """Compute a centered 1.5x GT crop and clip it to image boundaries.

    ``gt_bbox`` is in normalized source coordinates.  Pixel crop boundaries
    are rounded outward (floor for the left/top and ceil for right/bottom),
    which ensures the requested GT region is never lost to rounding.
    """

    width, height = _image_size(None, image_size)
    if expansion <= 0.0 or not math.isfinite(float(expansion)):
        raise ValueError("expansion must be finite and positive")
    x1, y1, x2, y2 = _bbox(gt_bbox, name="gt bbox", scale=coordinate_scale)
    px1, py1 = x1 / coordinate_scale * width, y1 / coordinate_scale * height
    px2, py2 = x2 / coordinate_scale * width, y2 / coordinate_scale * height
    center_x, center_y = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    crop_width, crop_height = (px2 - px1) * expansion, (py2 - py1) * expansion
    raw_left, raw_top = center_x - crop_width / 2.0, center_y - crop_height / 2.0
    raw_right, raw_bottom = center_x + crop_width / 2.0, center_y + crop_height / 2.0
    left = max(0, math.floor(raw_left))
    top = max(0, math.floor(raw_top))
    right = min(width, math.ceil(raw_right))
    bottom = min(height, math.ceil(raw_bottom))
    # A valid positive GT box normally makes this unnecessary, but retain a
    # one-pixel crop for extremely small images/boxes after integer rounding.
    if right <= left:
        right = min(width, left + 1)
        left = max(0, right - 1)
    if bottom <= top:
        bottom = min(height, top + 1)
        top = max(0, bottom - 1)
    return CropTransform(
        image_size=(width, height),
        crop_box_px=(left, top, right, bottom),
        coordinate_scale=float(coordinate_scale),
    )


def _slice_image(image: Any, crop_box: tuple[int, int, int, int]) -> Any:
    left, top, right, bottom = crop_box
    crop_method = getattr(image, "crop", None)
    if callable(crop_method):
        return crop_method(crop_box)
    try:
        rows = image[top:bottom]
        result = rows[:, left:right]  # numpy/torch-like arrays
    except (TypeError, IndexError, AttributeError):
        try:
            result = [row[left:right] for row in image[top:bottom]]
        except (TypeError, IndexError, AttributeError) as exc:
            raise TypeError("image must provide crop() or 2-D slicing") from exc
    copy_method = getattr(result, "copy", None)
    return copy_method() if callable(copy_method) else result


def crop_gt_bbox(
    image: Any,
    gt_bbox: Sequence[Any],
    *,
    image_size: Sequence[Any] | None = None,
    expansion: float = DEFAULT_CROP_SCALE,
    coordinate_scale: float = COORDINATE_SCALE,
) -> CropResult:
    """Crop ``image`` around ``gt_bbox`` and return the coordinate mapping."""

    size = _image_size(image, image_size)
    mapping = compute_gt_crop_transform(
        gt_bbox,
        size,
        expansion=expansion,
        coordinate_scale=coordinate_scale,
    )
    return CropResult(_slice_image(image, mapping.crop_box_px), mapping)


# Descriptive aliases used by callers that prefer a verb-first name.
build_gt_crop = crop_gt_bbox
make_gt_crop = crop_gt_bbox
gt_bbox_crop = crop_gt_bbox
crop_image_around_gt = crop_gt_bbox
crop_box_from_gt_bbox = compute_gt_crop_transform
compute_crop_transform = compute_gt_crop_transform


def map_crop_bbox_to_original(crop_bbox: Sequence[Any], mapping: CropTransform | CropResult) -> BBox:
    """Map a normalized crop bbox back to the source's ``[0,1000]`` space."""

    transform = mapping.mapping if isinstance(mapping, CropResult) else mapping
    if not isinstance(transform, CropTransform):
        raise TypeError("mapping must be a CropTransform or CropResult")
    return transform.map_bbox(crop_bbox)


def map_crop_point_to_original(crop_point: Sequence[Any], mapping: CropTransform | CropResult) -> tuple[float, float]:
    transform = mapping.mapping if isinstance(mapping, CropResult) else mapping
    if not isinstance(transform, CropTransform):
        raise TypeError("mapping must be a CropTransform or CropResult")
    return transform.map_point(crop_point)


def gt_overlay(image: Any, gt_bbox: Sequence[Any], *, color: Any = (255, 0, 0), width: int = 3) -> Any:
    """Return a smoke-test-only image with the GT rectangle drawn on it.

    The overlay is deliberately not used by target construction.  PIL is
    imported lazily so the core module remains usable in CPU text-only jobs.
    """

    if width < 1:
        raise ValueError("overlay width must be positive")
    image_width, image_height = _image_size(image, None)
    x1, y1, x2, y2 = _bbox(gt_bbox, name="gt bbox")
    pixels = (
        x1 / COORDINATE_SCALE * image_width,
        y1 / COORDINATE_SCALE * image_height,
        x2 / COORDINATE_SCALE * image_width,
        y2 / COORDINATE_SCALE * image_height,
    )
    copy_method = getattr(image, "copy", None)
    output = copy_method() if callable(copy_method) else image
    try:
        from PIL import ImageDraw

        ImageDraw.Draw(output).rectangle(pixels, outline=color, width=width)
        return output
    except ImportError as exc:
        raise RuntimeError("gt_overlay requires Pillow and is only a smoke helper") from exc


make_gt_overlay = gt_overlay


def _reasoning_text(reasoning: Any, *, name: str = "reasoning") -> str:
    if isinstance(reasoning, Mapping):
        for key in ("reasoning", "thinking", "prefix"):
            if key in reasoning:
                reasoning = reasoning[key]
                break
    if isinstance(reasoning, Sequence) and not isinstance(reasoning, str | bytes):
        pieces = [str(piece).strip() for piece in reasoning]
        if any(not piece for piece in pieces):
            raise ValueError(f"{name} contains an empty sentence")
        text = "\n".join(pieces)
    else:
        text = str(reasoning).strip()
    if text.startswith("<think>"):
        if text.count("<think>") != 1:
            raise ValueError(f"{name} must contain at most one <think> marker")
        text = text[len("<think>") :]
    if "</think>" in text:
        if text.count("</think>") != 1:
            raise ValueError(f"{name} must contain at most one </think> marker")
        text = text.split("</think>", 1)[0]
    text = text.strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _render_bbox_json(bbox: Sequence[Any], *, answer_key: str = BBOX_FIELD) -> str:
    if answer_key != BBOX_FIELD:
        raise ValueError(f"Teacher targets must use the canonical bbox key, got {answer_key!r}")
    values = _bbox(bbox, name="target bbox")
    rendered: list[int | float] = [int(value) if value.is_integer() else value for value in values]
    return json.dumps({answer_key: rendered}, ensure_ascii=False, separators=(",", ":"))


def render_bbox_json(bbox: Sequence[Any], *, answer_key: str = BBOX_FIELD) -> str:
    """Render one canonical bbox JSON object without surrounding text."""

    return _render_bbox_json(bbox, answer_key=answer_key)


def build_localization_teacher_prompt(
    reasoning: Any,
    *,
    query: str | None = None,
) -> str:
    """Build the localization prompt that carries complete reasoning."""

    complete = _reasoning_text(reasoning)
    query_text = "" if query is None else f"\nReferring expression:\n{str(query).strip()}\n"
    return (
        "Localization repair. Use the complete reasoning below as context. "
        "You receive the full original image first and a high-resolution GT-centered crop second. "
        "Use the crop only as privileged evidence; return coordinates in the first full image. "
        "Return exactly one bbox JSON object and no explanation."
        f"{query_text}\nComplete reasoning:\n{complete}\n\n"
        'The answer must be only {"bbox": [x1, y1, x2, y2]} with normalized coordinates in 0..1000.'
    )


def build_referent_teacher_prompt(
    original_response: Any,
    *,
    query: str | None = None,
) -> str:
    """Build the referent prompt for one short prefix correction sentence."""

    prefix = _extract_think_prefix(original_response)
    query_text = "" if query is None else f"\nReferring expression:\n{str(query).strip()}\n"
    return (
        "Referent repair. Continue from the original reasoning prefix with exactly one short "
        "correction sentence, close the reasoning, and emit exactly one bbox JSON object. "
        "You receive the full original image first and a high-resolution GT-centered crop second; "
        "all bbox coordinates must refer to the first full image."
        f"{query_text}\nOriginal reasoning prefix:\n{prefix}\n\n"
        'Use the answer shape {"bbox": [x1, y1, x2, y2]} with normalized coordinates in 0..1000.'
    )


build_localization_prompt = build_localization_teacher_prompt
build_referent_prompt = build_referent_teacher_prompt


def build_localization_teacher_target(
    reasoning: Any,
    target_bbox: Sequence[Any],
    *,
    answer_key: str = BBOX_FIELD,
) -> str:
    """Preserve complete reasoning and append a bbox-only answer."""

    complete = _reasoning_text(reasoning)
    answer = _render_bbox_json(target_bbox, answer_key=answer_key)
    return f"<think>\n{complete}\n</think><answer>{answer}</answer>"


def _extract_think_prefix(original_response: Any) -> str:
    text = str(original_response).strip()
    if not text:
        raise ValueError("original response/prefix must be non-empty")
    if text.count("<think>") > 1 or text.count("</think>") > 1:
        raise ValueError("original response must contain at most one thinking block")
    if "<think>" in text:
        if not text.startswith("<think>"):
            raise ValueError("original response must start with <think>")
        prefix = text.split("</think>", 1)[0] if "</think>" in text else text
    else:
        prefix = f"<think>\n{text}"
    if not prefix.startswith("<think>"):
        raise ValueError("original response must provide a <think> prefix")
    if not prefix[len("<think>") :].strip():
        raise ValueError("original reasoning prefix must be non-empty")
    return prefix.rstrip()


def _validate_correction_sentence(sentence: str) -> str:
    text = str(sentence).strip()
    if not text or any(marker in text for marker in ("<think>", "</think>", "<answer>", "</answer>")):
        raise ValueError("correction sentence must be plain text")
    if len(re.findall(r"[.!?]", text)) != 1 or not re.search(r"[.!?]$", text):
        raise ValueError("referent correction must contain exactly one short sentence")
    if len(text.split()) > 24:
        raise ValueError("referent correction sentence is too long")
    return text


def build_referent_teacher_target(
    original_response: Any,
    target_bbox: Sequence[Any],
    *,
    correction_sentence: str = DEFAULT_REFERENT_CORRECTION,
    answer_key: str = BBOX_FIELD,
) -> str:
    """Continue the original prefix with one correction sentence and bbox."""

    prefix = _extract_think_prefix(original_response)
    correction = _validate_correction_sentence(correction_sentence)
    answer = _render_bbox_json(target_bbox, answer_key=answer_key)
    return f"{prefix}\n{correction}</think><answer>{answer}</answer>"


build_localization_target = build_localization_teacher_target
make_localization_teacher_target = build_localization_teacher_target
build_referent_target = build_referent_teacher_target
make_referent_teacher_target = build_referent_teacher_target


@dataclass(frozen=True)
class TeacherContract:
    """A prompt/target pair and its validated target bbox."""

    mode: TeacherMode
    prompt: str
    target: str
    target_bbox: BBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "prompt": self.prompt,
            "target": self.target,
            "target_bbox": list(self.target_bbox),
        }


def build_teacher_contract(
    mode: str,
    *,
    target_bbox: Sequence[Any],
    reasoning: Any | None = None,
    original_response: Any | None = None,
    query: str | None = None,
    correction_sentence: str = DEFAULT_REFERENT_CORRECTION,
    answer_key: str = BBOX_FIELD,
) -> TeacherContract:
    """Build either explicit Teacher prompt/target contract."""

    selected = _mode(mode)
    target = _bbox(target_bbox, name="target bbox")
    if selected == LOCALIZATION:
        if reasoning is None:
            raise ValueError("localization contract requires complete reasoning")
        prompt = build_localization_teacher_prompt(reasoning, query=query)
        rendered_target = build_localization_teacher_target(reasoning, target, answer_key=answer_key)
    else:
        if original_response is None:
            raise ValueError("referent contract requires an original response/prefix")
        prompt = build_referent_teacher_prompt(original_response, query=query)
        rendered_target = build_referent_teacher_target(
            original_response,
            target,
            correction_sentence=correction_sentence,
            answer_key=answer_key,
        )
    return TeacherContract(selected, prompt, rendered_target, target)


build_localization_contract = lambda reasoning, target_bbox, **kwargs: build_teacher_contract(
    LOCALIZATION, reasoning=reasoning, target_bbox=target_bbox, **kwargs
)
build_referent_contract = lambda original_response, target_bbox, **kwargs: build_teacher_contract(
    REFERENT, original_response=original_response, target_bbox=target_bbox, **kwargs
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


@dataclass(frozen=True)
class ParsedTeacherTarget:
    """Validated text target returned by :func:`parse_teacher_target`."""

    mode: TeacherMode | None
    reasoning: str
    bbox: BBox
    answer_key: str
    correction_sentence: str | None = None

    @property
    def target_bbox(self) -> BBox:
        return self.bbox

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reasoning": self.reasoning,
            "bbox": list(self.bbox),
            "answer_key": self.answer_key,
            "correction_sentence": self.correction_sentence,
        }

    as_dict = to_dict


_TARGET_RE = re.compile(
    r"\A\s*<think>(?P<reasoning>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)


def parse_teacher_target(
    response: str,
    *,
    mode: str | None = None,
    original_response: Any | None = None,
) -> ParsedTeacherTarget | None:
    """Strictly parse a two-tag Teacher target, returning ``None`` on failure."""

    if not isinstance(response, str):
        return None
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return None
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return None
    match = _TARGET_RE.fullmatch(response)
    if match is None:
        return None
    reasoning = match.group("reasoning").strip()
    if not reasoning:
        return None
    try:
        value = json.loads(
            match.group("answer").strip(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or len(value) != 1:
        return None
    keys = set(value)
    if keys != {BBOX_FIELD}:
        return None
    answer_key = next(iter(keys))
    try:
        parsed_bbox = _bbox(value[answer_key], name="target bbox")
    except (TypeError, ValueError):
        return None
    selected: TeacherMode | None = None
    if mode is not None:
        try:
            selected = _mode(mode)
        except ValueError:
            return None
    correction: str | None = None
    if selected == REFERENT:
        if original_response is not None:
            try:
                prefix = _extract_think_prefix(original_response)
            except ValueError:
                return None
            if not reasoning.startswith(prefix[len("<think>") :].strip()):
                return None
            correction = reasoning[len(prefix[len("<think>") :].strip()) :].strip()
        else:
            # Without the source prefix, require a final one-sentence suffix;
            # this still catches accidental multi-sentence referent targets.
            sentence_match = re.search(r"([^.!?]*[.!?])\s*\Z", reasoning)
            if sentence_match is not None:
                correction = sentence_match.group(1).strip()
        if correction is not None:
            try:
                correction = _validate_correction_sentence(correction)
            except ValueError:
                return None
    return ParsedTeacherTarget(selected, reasoning, parsed_bbox, answer_key, correction)


parse_teacher_target_response = parse_teacher_target


class VerificationResult(dict[str, Any]):
    """JSON-serializable verifier result with attribute convenience access."""

    def __init__(
        self,
        *,
        accepted: bool,
        reason: str,
        target_bbox: BBox,
        teacher_bbox: BBox | None,
        iou: float | None,
        matched_ann_id: Any | None = None,
    ) -> None:
        super().__init__(
            accepted=bool(accepted),
            reason=str(reason),
            target_bbox=list(target_bbox),
            teacher_bbox=None if teacher_bbox is None else list(teacher_bbox),
            iou=None if iou is None else float(iou),
            matched_ann_id=matched_ann_id,
        )

    @property
    def accepted(self) -> bool:
        return bool(self["accepted"])

    @property
    def reason(self) -> str:
        return str(self["reason"])

    @property
    def target_bbox(self) -> list[float]:
        return self["target_bbox"]

    @property
    def iou(self) -> float | None:
        return self["iou"]

    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    as_dict = to_dict


def _parse_verifier_args(
    args: tuple[Any, ...],
    *,
    mode: str | None,
    teacher_output: str | None,
    target_bbox: Sequence[Any] | None,
    student_bbox: Sequence[Any] | None,
) -> tuple[TeacherMode, str, Sequence[Any], Sequence[Any] | None]:
    """Accept both ``(output, target, mode=...)`` and ``(mode, output, target)``."""

    values = list(args)
    if values:
        if isinstance(values[0], str) and values[0].strip().lower().replace("-", "_") in {
            LOCALIZATION,
            REFERENT,
            "localize",
            "reference",
        }:
            if mode is not None:
                raise TypeError("mode supplied both positionally and by keyword")
            mode, values = values[0], values[1:]
        if teacher_output is None and values:
            teacher_output, values = values[0], values[1:]
        if target_bbox is None and values:
            target_bbox, values = values[0], values[1:]
        if student_bbox is None and values:
            student_bbox, values = values[0], values[1:]
        if values:
            raise TypeError("too many positional arguments")
    if mode is None:
        mode = REFERENT
    if teacher_output is None or target_bbox is None:
        raise TypeError("teacher_output and target_bbox are required")
    return _mode(mode), teacher_output, target_bbox, student_bbox


def verify_teacher_target(
    *args: Any,
    mode: str | None = None,
    teacher_output: str | None = None,
    target_bbox: Sequence[Any] | None = None,
    student_bbox: Sequence[Any] | None = None,
    instances: Sequence[Any] | None = None,
    target_ann_id: Any | None = None,
) -> VerificationResult:
    """Verify a Teacher target using mode-specific IoU gates.

    Accepted call forms are ``verify_teacher_target(output, target, mode=...)``
    and ``verify_teacher_target(mode, output, target, student_bbox)``.  A
    referent target must geometrically match ``target_ann_id`` and needs IoU
    >= 0.5. A localization target must match ``target_ann_id`` and improve IoU
    by at least 0.05 over the Student box.
    """

    try:
        selected, output, target, student = _parse_verifier_args(
            args,
            mode=mode,
            teacher_output=teacher_output,
            target_bbox=target_bbox,
            student_bbox=student_bbox,
        )
        target_box = _bbox(target, name="target bbox")
    except (TypeError, ValueError) as exc:
        # The caller did not provide a usable target; retain the required
        # JSON shape even though no meaningful IoU exists.
        fallback = (0.0, 0.0, 1.0, 1.0)
        return VerificationResult(
            accepted=False,
            reason=f"invalid_input: {exc}",
            target_bbox=fallback,
            teacher_bbox=None,
            iou=None,
        )

    parsed = parse_teacher_target(output, mode=selected)
    if parsed is None:
        return VerificationResult(
            accepted=False,
            reason="malformed_target",
            target_bbox=target_box,
            teacher_bbox=None,
            iou=None,
        )
    score = bbox_iou(parsed.bbox, target_box)
    if instances is None or target_ann_id is None:
        return VerificationResult(
            accepted=False,
            reason="instances_and_target_ann_id_required",
            target_bbox=target_box,
            teacher_bbox=parsed.bbox,
            iou=score,
        )
    from verl.experimental.routed_grounding.router import match_prediction

    match = match_prediction(parsed.bbox, instances)
    if match.status != "matched" or match.ann_id != target_ann_id:
        return VerificationResult(
            accepted=False,
            reason="teacher_bbox_does_not_match_target_ann_id",
            target_bbox=target_box,
            teacher_bbox=parsed.bbox,
            iou=score,
            matched_ann_id=match.ann_id,
        )
    if selected == REFERENT:
        if score < IOU_THRESHOLD:
            return VerificationResult(
                accepted=False,
                reason="target_iou_below_0.5",
                target_bbox=target_box,
                teacher_bbox=parsed.bbox,
                iou=score,
                matched_ann_id=match.ann_id,
            )
        return VerificationResult(
            accepted=True,
            reason="accepted_target_match_and_iou",
            target_bbox=target_box,
            teacher_bbox=parsed.bbox,
            iou=score,
            matched_ann_id=match.ann_id,
        )
    if student is None:
        return VerificationResult(
            accepted=False,
            reason="student_bbox_required",
            target_bbox=target_box,
            teacher_bbox=parsed.bbox,
            iou=score,
            matched_ann_id=match.ann_id,
        )
    try:
        student_score = bbox_iou(student, target_box)
    except (TypeError, ValueError) as exc:
        return VerificationResult(
            accepted=False,
            reason=f"invalid_student_bbox: {exc}",
            target_bbox=target_box,
            teacher_bbox=parsed.bbox,
            iou=score,
            matched_ann_id=match.ann_id,
        )
    if score - student_score < LOCALIZATION_GAIN:
        return VerificationResult(
            accepted=False,
            reason="insufficient_iou_improvement",
            target_bbox=target_box,
            teacher_bbox=parsed.bbox,
            iou=score,
            matched_ann_id=match.ann_id,
        )
    return VerificationResult(
        accepted=True,
        reason="accepted_target_match_and_iou_gain",
        target_bbox=target_box,
        teacher_bbox=parsed.bbox,
        iou=score,
        matched_ann_id=match.ann_id,
    )


verify_target = verify_teacher_target
verify_teacher = verify_teacher_target


__all__ = [
    "BBOX_FIELD",
    "BBox",
    "COORDINATE_SCALE",
    "CropResult",
    "CropTransform",
    "DEFAULT_CROP_SCALE",
    "DEFAULT_REFERENT_CORRECTION",
    "IOU_THRESHOLD",
    "LOCALIZATION",
    "LOCALIZATION_GAIN",
    "ParsedTeacherTarget",
    "REFERENT",
    "TeacherContract",
    "TeacherMode",
    "VerificationResult",
    "bbox_iou",
    "build_gt_crop",
    "build_localization_contract",
    "build_localization_prompt",
    "build_localization_target",
    "build_localization_teacher_prompt",
    "build_localization_teacher_target",
    "build_referent_contract",
    "build_referent_prompt",
    "build_referent_target",
    "build_referent_teacher_prompt",
    "build_referent_teacher_target",
    "build_teacher_contract",
    "compute_crop_transform",
    "compute_gt_crop_transform",
    "crop_box_from_gt_bbox",
    "crop_gt_bbox",
    "crop_image_around_gt",
    "gt_bbox_crop",
    "gt_overlay",
    "make_gt_crop",
    "make_gt_overlay",
    "make_localization_teacher_target",
    "make_referent_teacher_target",
    "map_crop_bbox_to_original",
    "map_crop_point_to_original",
    "parse_teacher_target",
    "parse_teacher_target_response",
    "render_bbox_json",
    "verify_teacher",
    "verify_teacher_target",
    "verify_target",
]
