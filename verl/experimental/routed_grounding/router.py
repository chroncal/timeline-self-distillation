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

"""CPU-only parsing, geometry matching, and routing for grounded answers.

The router in this module is deliberately small and dependency free.  It is
used by the experiment's evaluator, where an answer is a strict two-tag
record and COCO annotations are the only image-side source of truth.  All
coordinates emitted by a model are in the usual ``[0, 1000]`` normalized
coordinate system; annotation boxes and masks remain in image pixels.

There are three intentionally separate pieces:

* :func:`parse_response` validates the wire format and retains useful parser
  state, including deterministic prefixes for truncated responses.
* :func:`match_prediction` maps a valid box to an annotation.  A unique mask
  hit at the predicted-box centre takes precedence over box IoU; otherwise a
  conservative IoU threshold and margin are applied.
* :func:`route_response` turns the original match and four probe matches into
  one of the experiment's route labels.

No torch, numpy, PIL, shapely, or pycocotools import is required.  Polygon
and both common COCO RLE representations are handled with the Python
standard library so the helpers can run in a CPU-only unit-test environment.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

COORDINATE_MIN = 0.0
COORDINATE_MAX = 1000.0
IOU_THRESHOLD = 0.1
IOU_MARGIN = 0.05
TARGET_IOU_THRESHOLD = 0.5
EXPECTED_PROBE_COUNT = 4

MatchStatus = Literal["matched", "background", "ambiguous", "invalid"]
RouteLabel = Literal[
    "correct",
    "localization",
    "localization_recoverable",
    "referent_unrecoverable",
    "uncertain",
]


def _is_number(value: Any) -> bool:
    """Return whether *value* is a finite, non-boolean real number."""

    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number)


def _number(value: Any, *, label: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_prediction_bbox(value: Any) -> tuple[float, float, float, float]:
    """Validate a normalized ``xyxy`` prediction in the closed 0..1000 range.

    A box must have positive width and height.  Keeping ordering strict here
    makes malformed model output distinguishable from a low-overlap (but
    valid) prediction during routing.
    """

    if isinstance(value, str | bytes | bytearray):
        raise ValueError("bbox must contain exactly four numeric coordinates")
    try:
        coordinates = tuple(value)
    except TypeError as error:
        raise ValueError("bbox must contain exactly four numeric coordinates") from error
    if len(coordinates) != 4:
        raise ValueError("bbox must contain exactly four numeric coordinates")

    result = tuple(_number(coordinate, label="bbox coordinate") for coordinate in coordinates)
    if any(coordinate < COORDINATE_MIN or coordinate > COORDINATE_MAX for coordinate in result):
        raise ValueError("bbox coordinates must be finite values in 0..1000")
    x1, y1, x2, y2 = result
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox must have positive width and height and ordered corners")
    return result


# Common spelling used by geometry callers.
validate_bbox = validate_prediction_bbox


def xywh_to_xyxy(value: Sequence[Any]) -> tuple[float, float, float, float]:
    """Convert an image-pixel COCO ``xywh`` box to ``xyxy``."""

    if isinstance(value, str | bytes | bytearray):
        raise ValueError("bbox_xywh must contain exactly four numeric values")
    try:
        coordinates = tuple(value)
    except TypeError as error:
        raise ValueError("bbox_xywh must contain exactly four numeric values") from error
    if len(coordinates) != 4:
        raise ValueError("bbox_xywh must contain exactly four numeric values")
    x, y, width, height = (_number(coordinate, label="bbox_xywh value") for coordinate in coordinates)
    if width <= 0 or height <= 0:
        raise ValueError("bbox_xywh width and height must be positive")
    if x < 0 or y < 0:
        raise ValueError("bbox_xywh x and y must be non-negative")
    return x, y, x + width, y + height


bbox_xywh_to_xyxy = xywh_to_xyxy


def xyxy_iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Compute inclusive-of-neither-edge IoU for two ``xyxy`` boxes."""

    try:
        a = tuple(float(value) for value in left)
        b = tuple(float(value) for value in right)
    except (TypeError, ValueError) as error:
        raise ValueError("IoU boxes must contain four numeric values") from error
    if len(a) != 4 or len(b) != 4:
        raise ValueError("IoU boxes must contain four numeric values")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("IoU boxes must contain finite values")
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if a_area <= 0 or b_area <= 0:
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a_area + b_area - intersection
    return intersection / union if union > 0 else 0.0


iou_xyxy = xyxy_iou
bbox_iou = xyxy_iou


def _coerce_dimension(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    number = _number(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


@dataclass(frozen=True, init=False)
class Instance:
    """A COCO-style annotation used by the CPU matcher.

    ``bbox_xywh`` is in source-image pixels.  ``segmentation`` accepts a COCO
    polygon (flat or list-of-polygons), an uncompressed RLE mapping, or a
    compressed COCO RLE mapping.  ``image_width`` and ``image_height`` are
    used to map normalized model coordinates to source pixels.  The aliases
    ``bbox``, ``width``, and ``height`` are accepted to make loading simple
    annotation dictionaries less error prone.
    """

    ann_id: Any
    bbox_xywh: tuple[float, float, float, float]
    segmentation: Any
    image_width: float | None
    image_height: float | None

    def __init__(
        self,
        ann_id: Any,
        bbox_xywh: Sequence[Any] | None = None,
        segmentation: Any = None,
        image_width: Any = None,
        image_height: Any = None,
        *,
        bbox: Sequence[Any] | None = None,
        width: Any = None,
        height: Any = None,
    ) -> None:
        if bbox_xywh is None:
            bbox_xywh = bbox
        elif bbox is not None and tuple(bbox_xywh) != tuple(bbox):
            raise ValueError("bbox and bbox_xywh disagree")
        if bbox_xywh is None:
            raise ValueError("Instance requires bbox_xywh (or bbox)")
        canonical_bbox = tuple(_number(value, label="bbox_xywh value") for value in bbox_xywh)
        # Validate while retaining xywh rather than silently clipping it.
        xywh_to_xyxy(canonical_bbox)

        if image_width is None:
            image_width = width
        elif width is not None and float(image_width) != float(width):
            raise ValueError("width and image_width disagree")
        if image_height is None:
            image_height = height
        elif height is not None and float(image_height) != float(height):
            raise ValueError("height and image_height disagree")

        object.__setattr__(self, "ann_id", ann_id)
        object.__setattr__(self, "bbox_xywh", canonical_bbox)
        object.__setattr__(self, "segmentation", segmentation)
        object.__setattr__(self, "image_width", _coerce_dimension(image_width, label="image_width"))
        object.__setattr__(self, "image_height", _coerce_dimension(image_height, label="image_height"))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The source-image box as ``xywh`` (the COCO field spelling)."""

        return self.bbox_xywh

    @property
    def width(self) -> float | None:
        return self.image_width

    @property
    def height(self) -> float | None:
        return self.image_height

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return xywh_to_xyxy(self.bbox_xywh)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ann_id": self.ann_id,
            "bbox_xywh": list(self.bbox_xywh),
            "segmentation": _jsonable(self.segmentation),
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    as_dict = to_dict


def _coerce_instance(value: Instance | Mapping[str, Any]) -> Instance:
    if isinstance(value, Instance):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("instances must contain Instance objects or mappings")
    if "ann_id" in value:
        ann_id = value["ann_id"]
    elif "id" in value:
        ann_id = value["id"]
    else:
        raise ValueError("instance mapping is missing ann_id")
    bbox = value.get("bbox_xywh", value.get("bbox"))
    segmentation = value.get("segmentation")
    image_width = value.get("image_width", value.get("width"))
    image_height = value.get("image_height", value.get("height"))
    return Instance(
        ann_id=ann_id,
        bbox_xywh=bbox,
        segmentation=segmentation,
        image_width=image_width,
        image_height=image_height,
    )


def coerce_instances(values: Iterable[Instance | Mapping[str, Any]]) -> tuple[Instance, ...]:
    return tuple(_coerce_instance(value) for value in values)


def _normalized_to_image_bbox(bbox: Sequence[float], instance: Instance) -> tuple[float, float, float, float]:
    width, height = instance.image_width, instance.image_height
    if width is None or height is None:
        # This fallback is useful for synthetic tests and annotations whose
        # geometry is already in the normalized coordinate system.
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    scale_x, scale_y = width / COORDINATE_MAX, height / COORDINATE_MAX
    x1, y1, x2, y2 = bbox
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


def _normalized_to_image_point(point: tuple[float, float], instance: Instance) -> tuple[float, float]:
    width, height = instance.image_width, instance.image_height
    if width is None or height is None:
        return point
    return point[0] * width / COORDINATE_MAX, point[1] * height / COORDINATE_MAX


def _point_on_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9


def _point_in_polygon(point: tuple[float, float], coordinates: Sequence[Any]) -> bool:
    if isinstance(coordinates, str | bytes | bytearray):
        return False
    try:
        values = tuple(float(value) for value in coordinates)
    except (TypeError, ValueError):
        return False
    if len(values) < 6 or len(values) % 2:
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    vertices = tuple((values[index], values[index + 1]) for index in range(0, len(values), 2))
    inside = False
    for index, vertex in enumerate(vertices):
        previous = vertices[index - 1]
        if _point_on_segment(point, previous, vertex):
            return True
        x1, y1 = previous
        x2, y2 = vertex
        px, py = point
        if (y1 > py) != (y2 > py):
            intersection_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < intersection_x:
                inside = not inside
    return inside


def _polygon_list(segmentation: Any) -> tuple[Sequence[Any], ...]:
    """Normalize COCO polygon variants into a tuple of flat polygons."""

    if not isinstance(segmentation, list | tuple) or not segmentation:
        return ()
    if all(not isinstance(value, list | tuple) for value in segmentation):
        return (segmentation,)
    polygons: list[Sequence[Any]] = []
    for polygon in segmentation:
        if isinstance(polygon, list | tuple):
            polygons.append(polygon)
    return tuple(polygons)


def _decode_compressed_rle_counts(value: str | bytes) -> tuple[int, ...]:
    """Decode the compact ASCII count format emitted by COCO mask RLE."""

    text = value.decode("ascii") if isinstance(value, bytes) else value
    counts: list[int] = []
    index = 0
    while index < len(text):
        number = 0
        shift = 0
        while True:
            if index >= len(text):
                raise ValueError("truncated compressed COCO RLE counts")
            character = ord(text[index]) - 48
            index += 1
            number |= (character & 0x1F) << shift
            if character & 0x20:
                shift += 5
            else:
                if character & 0x10:
                    number |= -1 << (shift + 5)
                break
        # COCO's compact format delta-codes count m against count m-2 once
        # m > 2 (not against the immediately preceding run).
        if len(counts) > 2:
            number += counts[-2]
        counts.append(number)
    return tuple(counts)


def _rle_contains(point: tuple[float, float], segmentation: Mapping[str, Any]) -> bool:
    counts_value = segmentation.get("counts")
    size = segmentation.get("size")
    if not isinstance(size, list | tuple) or len(size) != 2:
        return False
    try:
        height, width = int(size[0]), int(size[1])
    except (TypeError, ValueError, OverflowError):
        return False
    if height <= 0 or width <= 0:
        return False
    x, y = point
    pixel_x, pixel_y = math.floor(x), math.floor(y)
    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
        return False
    if isinstance(counts_value, str | bytes):
        try:
            counts = _decode_compressed_rle_counts(counts_value)
        except (UnicodeError, ValueError):
            return False
    elif isinstance(counts_value, list | tuple):
        try:
            counts = tuple(int(count) for count in counts_value)
        except (TypeError, ValueError, OverflowError):
            return False
    else:
        return False
    if any(count < 0 for count in counts):
        return False

    # COCO RLE is column-major: y changes fastest.  The resulting flat index
    # is therefore y + x * height.
    flat_index = pixel_y + pixel_x * height
    offset = 0
    for run_index, run_length in enumerate(counts):
        if flat_index < offset + run_length:
            # COCO runs alternate empty and filled pixels and start with an
            # empty run.  Thus odd-numbered runs are the mask foreground.
            return run_index % 2 == 1
        offset += run_length
        if offset > height * width:
            return False
    return False


def point_in_segmentation(point: tuple[float, float], segmentation: Any) -> bool:
    """Test a source-image point against a COCO polygon or RLE mask."""

    if isinstance(segmentation, Mapping):
        return _rle_contains(point, segmentation)
    return any(_point_in_polygon(point, polygon) for polygon in _polygon_list(segmentation))


point_in_mask = point_in_segmentation


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ResponsePrefixes:
    """Deterministic prefixes used by truncated reasoning/bbox probes."""

    raw_response: str
    bbox_prefix: str | None
    reasoning_prefix: str | None
    bbox_prefix_valid: bool
    reasoning_prefix_valid: bool
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.bbox_prefix_valid and self.reasoning_prefix_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "bbox_prefix": self.bbox_prefix,
            "reasoning_prefix": self.reasoning_prefix,
            "bbox_prefix_valid": self.bbox_prefix_valid,
            "reasoning_prefix_valid": self.reasoning_prefix_valid,
            "valid": self.valid,
            "error": self.error,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        # Tuple unpacking is useful for tiny callers while named fields remain
        # the canonical interface.
        yield self.bbox_prefix
        yield self.reasoning_prefix


_BBOX_OPEN_RE = re.compile(r'<answer>\{"bbox":\[\s*', re.DOTALL)
_COORDINATE_TOKEN_RE = re.compile(r"(?:[-+]?\d+(?:\.\d*)?|[-+]?\.\d+)(?:[eE][-+]?\d+)?")


def get_response_prefixes(response: str) -> ResponsePrefixes:
    """Return the prefix before the first bbox coordinate and before ``</think>``.

    Prefix extraction is intentionally independent of full parsing.  The bbox
    prefix is found after the exact ``<answer>{"bbox":[`` opener (allowing
    whitespace immediately before the first value); the reasoning prefix ends
    immediately before the first closing ``</think>``.  A failed extraction is
    retained as state instead of being guessed, because the route must then be
    considered uncertain.
    """

    if not isinstance(response, str):
        return ResponsePrefixes(
            raw_response=str(response),
            bbox_prefix=None,
            reasoning_prefix=None,
            bbox_prefix_valid=False,
            reasoning_prefix_valid=False,
            error="response must be text",
        )

    reasoning_end = response.find("</think>")
    reasoning_prefix = response[:reasoning_end] if reasoning_end >= 0 else None

    bbox_prefix: str | None = None
    answer_match = _BBOX_OPEN_RE.search(response)
    if answer_match:
        # The opener regex consumes whitespace before the first coordinate. A
        # non-empty token is required so ``[ ]`` is recorded as a failure.
        coordinate_match = _COORDINATE_TOKEN_RE.match(response, answer_match.end())
        if coordinate_match:
            bbox_prefix = response[: coordinate_match.start()]

    errors: list[str] = []
    if reasoning_prefix is None:
        errors.append("missing </think> prefix boundary")
    if bbox_prefix is None:
        errors.append("missing bbox first-coordinate prefix boundary")
    return ResponsePrefixes(
        raw_response=response,
        bbox_prefix=bbox_prefix,
        reasoning_prefix=reasoning_prefix,
        bbox_prefix_valid=bbox_prefix is not None,
        reasoning_prefix_valid=reasoning_prefix is not None,
        error="; ".join(errors) or None,
    )


response_prefixes = get_response_prefixes
truncate_response = get_response_prefixes
truncate_response_prefixes = get_response_prefixes


@dataclass(frozen=True)
class ParseResult:
    """A JSON-serializable strict-response parse receipt."""

    raw_response: str
    format_valid: bool
    reasoning: str | None
    bbox: tuple[float, float, float, float] | None
    error: str | None
    bbox_prefix: str | None
    reasoning_prefix: str | None
    bbox_prefix_valid: bool
    reasoning_prefix_valid: bool
    response_prefixes: ResponsePrefixes = field(repr=False)

    @property
    def valid(self) -> bool:
        return self.format_valid

    @property
    def parse_valid(self) -> bool:
        return self.format_valid

    @property
    def bbox_value(self) -> tuple[float, float, float, float] | None:
        return self.bbox

    @property
    def raw(self) -> str:
        return self.raw_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "raw": self.raw_response,
            "format_valid": self.format_valid,
            "valid": self.format_valid,
            "parse_valid": self.format_valid,
            "reasoning": self.reasoning,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "bbox_value": list(self.bbox) if self.bbox is not None else None,
            "error": self.error,
            "bbox_prefix": self.bbox_prefix,
            "reasoning_prefix": self.reasoning_prefix,
            "bbox_prefix_valid": self.bbox_prefix_valid,
            "reasoning_prefix_valid": self.reasoning_prefix_valid,
            "prefixes_valid": self.bbox_prefix_valid and self.reasoning_prefix_valid,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


_RESPONSE_RE = re.compile(r"\A<think>(?P<reasoning>.*?)</think><answer>(?P<payload>.*)</answer>\Z", re.DOTALL)


def _invalid_parse(response: Any, prefixes: ResponsePrefixes, error: str) -> ParseResult:
    text = response if isinstance(response, str) else str(response)
    return ParseResult(
        raw_response=text,
        format_valid=False,
        reasoning=None,
        bbox=None,
        error=error,
        bbox_prefix=prefixes.bbox_prefix,
        reasoning_prefix=prefixes.reasoning_prefix,
        bbox_prefix_valid=prefixes.bbox_prefix_valid,
        reasoning_prefix_valid=prefixes.reasoning_prefix_valid,
        response_prefixes=prefixes,
    )


def parse_response(response: str) -> ParseResult:
    """Strictly parse ``<think>...</think><answer>{"bbox":[...]}</answer>``.

    The outer tags and ordering are exact, the JSON object has exactly one
    key (``bbox``), and all four coordinates are finite normalized values with
    positive area.  No prefix/suffix or alternate answer schema is accepted.
    """

    prefixes = get_response_prefixes(response)
    if not isinstance(response, str):
        return _invalid_parse(response, prefixes, "response must be text")
    match = _RESPONSE_RE.fullmatch(response)
    if match is None:
        return _invalid_parse(response, prefixes, "response must exactly match the required think/answer tags")

    payload = match.group("payload")
    if not payload or payload != payload.strip():
        return _invalid_parse(response, prefixes, "answer payload must not have surrounding whitespace")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        return _invalid_parse(response, prefixes, f"answer payload is not valid JSON: {error.msg}")
    if not isinstance(decoded, dict) or set(decoded) != {"bbox"}:
        return _invalid_parse(response, prefixes, 'answer JSON must contain exactly the "bbox" key')
    try:
        bbox = validate_prediction_bbox(decoded["bbox"])
    except (TypeError, ValueError) as error:
        return _invalid_parse(response, prefixes, str(error))
    if not prefixes.valid:
        return _invalid_parse(response, prefixes, prefixes.error or "response prefix truncation failed")
    return ParseResult(
        raw_response=response,
        format_valid=True,
        reasoning=match.group("reasoning"),
        bbox=bbox,
        error=None,
        bbox_prefix=prefixes.bbox_prefix,
        reasoning_prefix=prefixes.reasoning_prefix,
        bbox_prefix_valid=prefixes.bbox_prefix_valid,
        reasoning_prefix_valid=prefixes.reasoning_prefix_valid,
        response_prefixes=prefixes,
    )


parse_grounding_response = parse_response
parse_grounding = parse_response
parse = parse_response


@dataclass(frozen=True)
class MatchResult:
    """Receipt for mapping one prediction to an annotation or background."""

    status: MatchStatus
    ann_id: Any | None
    iou: float
    second_iou: float
    margin: float
    method: str
    center: tuple[float, float] | None
    candidate_ious: tuple[tuple[Any, float], ...]
    error: str | None = None

    @property
    def matched(self) -> bool:
        return self.status == "matched"

    @property
    def label(self) -> MatchStatus:
        return self.status

    @property
    def match_status(self) -> MatchStatus:
        return self.status

    @property
    def target(self) -> Any | None:
        return self.ann_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "label": self.status,
            "ann_id": self.ann_id,
            "iou": self.iou,
            "second_iou": self.second_iou,
            "margin": self.margin,
            "method": self.method,
            "center": list(self.center) if self.center is not None else None,
            "candidate_ious": [[ann_id, value] for ann_id, value in self.candidate_ious],
            "error": self.error,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _invalid_match(error: str) -> MatchResult:
    return MatchResult(
        status="invalid",
        ann_id=None,
        iou=0.0,
        second_iou=0.0,
        margin=0.0,
        method="invalid",
        center=None,
        candidate_ious=(),
        error=error,
    )


def match_prediction(
    prediction_bbox: Sequence[Any],
    instances: Iterable[Instance | Mapping[str, Any]],
    *,
    iou_threshold: float = IOU_THRESHOLD,
    margin_threshold: float = IOU_MARGIN,
) -> MatchResult:
    """Match a normalized prediction to instances using mask-then-IoU logic."""

    try:
        prediction = validate_prediction_bbox(prediction_bbox)
        if not _is_number(iou_threshold) or not _is_number(margin_threshold):
            raise ValueError("matching thresholds must be finite numbers")
        iou_threshold = float(iou_threshold)
        margin_threshold = float(margin_threshold)
        if iou_threshold < 0 or margin_threshold < 0:
            raise ValueError("matching thresholds must be non-negative")
        normalized_instances = coerce_instances(instances)
    except (TypeError, ValueError) as error:
        return _invalid_match(str(error))

    x1, y1, x2, y2 = prediction
    normalized_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    center_mask_hits: list[int] = []
    iou_records: list[tuple[Any, float]] = []
    for index, instance in enumerate(normalized_instances):
        image_center = _normalized_to_image_point(normalized_center, instance)
        if instance.segmentation is not None and point_in_segmentation(image_center, instance.segmentation):
            center_mask_hits.append(index)
        prediction_image_bbox = _normalized_to_image_bbox(prediction, instance)
        iou_records.append((instance.ann_id, xyxy_iou(prediction_image_bbox, instance.bbox_xyxy)))

    ordered = sorted(enumerate(iou_records), key=lambda item: (-item[1][1], item[0]))
    sorted_scores = [record[1][1] for record in ordered]
    top_iou = sorted_scores[0] if sorted_scores else 0.0
    second_iou = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = top_iou - second_iou
    candidate_ious = tuple(record for _, record in ordered)

    if len(center_mask_hits) == 1:
        selected_index = center_mask_hits[0]
        selected_ann_id = normalized_instances[selected_index].ann_id
        selected_iou = iou_records[selected_index][1]
        other_scores = [score for index, (_, score) in enumerate(iou_records) if index != selected_index]
        selected_second = max(other_scores, default=0.0)
        return MatchResult(
            status="matched",
            ann_id=selected_ann_id,
            iou=selected_iou,
            second_iou=selected_second,
            margin=selected_iou - selected_second,
            method="center_mask",
            center=normalized_center,
            candidate_ious=candidate_ious,
        )

    if not normalized_instances or top_iou < iou_threshold:
        return MatchResult(
            status="background",
            ann_id=None,
            iou=top_iou,
            second_iou=second_iou,
            margin=margin,
            method="bbox_iou",
            center=normalized_center,
            candidate_ious=candidate_ious,
            error="no candidate reaches the IoU threshold",
        )
    if margin < margin_threshold:
        return MatchResult(
            status="ambiguous",
            ann_id=None,
            iou=top_iou,
            second_iou=second_iou,
            margin=margin,
            method="bbox_iou",
            center=normalized_center,
            candidate_ious=candidate_ious,
            error="top candidates do not clear the IoU margin",
        )

    selected_ann_id = ordered[0][1][0]
    return MatchResult(
        status="matched",
        ann_id=selected_ann_id,
        iou=top_iou,
        second_iou=second_iou,
        margin=margin,
        method="bbox_iou_fallback" if len(center_mask_hits) > 1 else "bbox_iou",
        center=normalized_center,
        candidate_ious=candidate_ious,
    )


match_bbox = match_prediction
match_box = match_prediction


def _target_equal(left: Any, right: Any) -> bool:
    return left == right


@dataclass(frozen=True)
class RouteResult:
    """A route decision plus all receipts needed to audit it."""

    route: RouteLabel
    parsed: ParseResult
    original_match: MatchResult | None
    probe_matches: tuple[MatchResult, ...]
    target_ann_id: Any | None
    error: str | None = None

    @property
    def label(self) -> RouteLabel:
        return self.route

    @property
    def category(self) -> RouteLabel:
        return self.route

    @property
    def valid(self) -> bool:
        return self.parsed.format_valid and self.parsed.response_prefixes.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "label": self.route,
            "category": self.route,
            "parsed": self.parsed.to_dict(),
            "original_match": self.original_match.to_dict() if self.original_match is not None else None,
            "probe_matches": [match.to_dict() for match in self.probe_matches],
            "target_ann_id": self.target_ann_id,
            "error": self.error,
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _coerce_probe_values(probes: Any) -> tuple[Any, ...]:
    if probes is None:
        return ()
    if isinstance(probes, Mapping):
        return tuple(probes.values())
    if isinstance(probes, str | bytes | bytearray):
        return (probes,)
    try:
        return tuple(probes)
    except TypeError:
        return (probes,)


def _empty_invalid_parse(error: str) -> ParseResult:
    prefixes = ResponsePrefixes("", None, None, False, False, error)
    return _invalid_parse("", prefixes, error)


def _as_parsed(value: Any) -> ParseResult:
    if isinstance(value, ParseResult):
        return value
    if isinstance(value, str):
        return parse_response(value)
    # A mapping receipt can be useful when routing serialized probe data.
    if isinstance(value, Mapping):
        if "raw_response" in value and isinstance(value["raw_response"], str):
            return parse_response(value["raw_response"])
        if "bbox" in value:
            try:
                bbox = validate_prediction_bbox(value["bbox"])
            except (TypeError, ValueError) as error:
                return _empty_invalid_parse(str(error))
            raw = str(value.get("raw_response", ""))
            prefixes = get_response_prefixes(raw)
            return ParseResult(
                raw,
                True,
                value.get("reasoning"),
                bbox,
                None,
                prefixes.bbox_prefix,
                prefixes.reasoning_prefix,
                prefixes.bbox_prefix_valid,
                prefixes.reasoning_prefix_valid,
                prefixes,
            )
    return _empty_invalid_parse("response must be text, ParseResult, or a parse mapping")


def _as_match(value: Any, instances: tuple[Instance, ...]) -> MatchResult:
    if isinstance(value, MatchResult):
        return value
    if isinstance(value, ParseResult):
        if not value.format_valid or value.bbox is None:
            return _invalid_match(value.error or "invalid parsed response")
        return match_prediction(value.bbox, instances)
    if isinstance(value, str):
        parsed = parse_response(value)
        return _as_match(parsed, instances)
    if isinstance(value, Mapping):
        if "status" in value and "iou" in value:
            try:
                status = str(value["status"])
                if status not in {"matched", "background", "ambiguous", "invalid"}:
                    raise ValueError("unknown match status")
                candidate_ious = tuple((item[0], float(item[1])) for item in value.get("candidate_ious", ()))
                center = value.get("center")
                return MatchResult(
                    status=status,  # type: ignore[arg-type]
                    ann_id=value.get("ann_id"),
                    iou=float(value.get("iou", 0.0)),
                    second_iou=float(value.get("second_iou", 0.0)),
                    margin=float(value.get("margin", 0.0)),
                    method=str(value.get("method", "serialized")),
                    center=tuple(center) if center is not None else None,
                    candidate_ious=candidate_ious,
                    error=value.get("error"),
                )
            except (TypeError, ValueError, KeyError) as error:
                return _invalid_match(str(error))
        parsed = _as_parsed(value)
        return _as_match(parsed, instances)
    try:
        return match_prediction(value, instances)
    except (TypeError, ValueError) as error:
        return _invalid_match(str(error))


def route_response(
    response: Any,
    instances: Iterable[Instance | Mapping[str, Any]],
    target_ann_id: Any = None,
    probes: Iterable[Any] | None = None,
    *,
    target_id: Any = None,
    probe_responses: Iterable[Any] | None = None,
    probe_matches: Iterable[Any] | None = None,
) -> RouteResult:
    """Route an original response, optionally using four probe responses.

    A target original match is routed directly by its IoU.  Every other
    decisive state (including original background and original ambiguity) is
    checked with exactly four probes.  Three or more target probe matches are
    ``localization_recoverable``.  With zero target matches, three or more
    matches to the same wrong annotation are ``referent_unrecoverable``; three
    or more background probes are also ``referent_unrecoverable``.  Ambiguous probes are not background and
    do not contribute to either unrecoverable condition; all remaining cases
    are ``uncertain``.
    """

    normalized_instances = coerce_instances(instances)
    if target_ann_id is None:
        target_ann_id = target_id
    if probes is None:
        probes = probe_responses if probe_responses is not None else probe_matches
    parsed = _as_parsed(response)
    probe_values = _coerce_probe_values(probes)

    if not parsed.format_valid or parsed.bbox is None:
        return RouteResult(
            route="uncertain",
            parsed=parsed,
            original_match=None,
            probe_matches=(),
            target_ann_id=target_ann_id,
            error=parsed.error or "invalid response",
        )
    if not parsed.response_prefixes.valid:
        return RouteResult(
            route="uncertain",
            parsed=parsed,
            original_match=None,
            probe_matches=(),
            target_ann_id=target_ann_id,
            error=parsed.response_prefixes.error or "response prefix truncation failed",
        )
    if target_ann_id is None:
        return RouteResult(
            route="uncertain",
            parsed=parsed,
            original_match=None,
            probe_matches=(),
            target_ann_id=None,
            error="target_ann_id is required",
        )

    original_match = match_prediction(parsed.bbox, normalized_instances)
    if original_match.status == "matched" and _target_equal(original_match.ann_id, target_ann_id):
        route: RouteLabel = "correct" if original_match.iou >= TARGET_IOU_THRESHOLD else "localization"
        return RouteResult(route, parsed, original_match, (), target_ann_id)

    matches = tuple(_as_match(value, normalized_instances) for value in probe_values)
    if len(matches) != EXPECTED_PROBE_COUNT:
        return RouteResult(
            route="uncertain",
            parsed=parsed,
            original_match=original_match,
            probe_matches=matches,
            target_ann_id=target_ann_id,
            error=f"expected exactly {EXPECTED_PROBE_COUNT} probes",
        )
    target_count = sum(match.status == "matched" and _target_equal(match.ann_id, target_ann_id) for match in matches)
    if target_count >= 3:
        return RouteResult("localization_recoverable", parsed, original_match, matches, target_ann_id)

    if target_count == 0:
        background_count = sum(match.status == "background" for match in matches)
        if background_count >= 3:
            return RouteResult("referent_unrecoverable", parsed, original_match, matches, target_ann_id)

        wrong_ids = [match.ann_id for match in matches if match.status == "matched"]
        if wrong_ids:
            # A repeated wrong annotation is evidence of a stable referent,
            # while an isolated wrong match is not enough to call the probe
            # set unrecoverable.  ``repr`` is avoided here: equality is the
            # same semantic relation used for target IDs and works for JSON
            # scalar IDs as well as strings.
            for candidate in wrong_ids:
                if sum(_target_equal(candidate, wrong_id) for wrong_id in wrong_ids) >= 3:
                    return RouteResult("referent_unrecoverable", parsed, original_match, matches, target_ann_id)

    statuses = ", ".join(match.status for match in matches)
    return RouteResult(
        "uncertain",
        parsed,
        original_match,
        matches,
        target_ann_id,
        f"four probes do not meet a route threshold (target={target_count}, statuses={statuses})",
    )


route_grounding = route_response
route_prediction = route_response
route = route_response


class GroundingRouter:
    """Convenience wrapper for repeated routing against one annotation set."""

    def __init__(
        self,
        instances: Iterable[Instance | Mapping[str, Any]],
        target_ann_id: Any = None,
        *,
        target_id: Any = None,
    ):
        self.instances = coerce_instances(instances)
        self.target_ann_id = target_ann_id if target_ann_id is not None else target_id

    def match(self, prediction_bbox: Sequence[Any]) -> MatchResult:
        return match_prediction(prediction_bbox, self.instances)

    def route(self, response: Any, probes: Iterable[Any] | None = None) -> RouteResult:
        return route_response(response, self.instances, self.target_ann_id, probes)

    def parse(self, response: str) -> ParseResult:
        return parse_response(response)


__all__ = [
    "COORDINATE_MIN",
    "COORDINATE_MAX",
    "IOU_THRESHOLD",
    "IOU_MARGIN",
    "TARGET_IOU_THRESHOLD",
    "EXPECTED_PROBE_COUNT",
    "Instance",
    "ResponsePrefixes",
    "ParseResult",
    "MatchResult",
    "RouteResult",
    "GroundingRouter",
    "validate_prediction_bbox",
    "validate_bbox",
    "xywh_to_xyxy",
    "bbox_xywh_to_xyxy",
    "xyxy_iou",
    "iou_xyxy",
    "bbox_iou",
    "point_in_segmentation",
    "point_in_mask",
    "coerce_instances",
    "get_response_prefixes",
    "response_prefixes",
    "truncate_response",
    "truncate_response_prefixes",
    "parse_response",
    "parse_grounding_response",
    "parse_grounding",
    "parse",
    "match_prediction",
    "match_bbox",
    "match_box",
    "route_response",
    "route_grounding",
    "route_prediction",
    "route",
]
