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

"""CPU-only grounding response parser and router."""

from .router import (
    COORDINATE_MAX,
    COORDINATE_MIN,
    EXPECTED_PROBE_COUNT,
    IOU_MARGIN,
    IOU_THRESHOLD,
    TARGET_IOU_THRESHOLD,
    GroundingRouter,
    Instance,
    MatchResult,
    ParseResult,
    ResponsePrefixes,
    RouteResult,
    bbox_iou,
    bbox_xywh_to_xyxy,
    coerce_instances,
    get_response_prefixes,
    iou_xyxy,
    match_bbox,
    match_box,
    match_prediction,
    parse,
    parse_grounding,
    parse_grounding_response,
    parse_response,
    point_in_mask,
    point_in_segmentation,
    response_prefixes,
    route,
    route_grounding,
    route_prediction,
    route_response,
    truncate_response,
    truncate_response_prefixes,
    validate_bbox,
    validate_prediction_bbox,
    xywh_to_xyxy,
    xyxy_iou,
)

__all__ = [
    "COORDINATE_MAX",
    "COORDINATE_MIN",
    "EXPECTED_PROBE_COUNT",
    "IOU_MARGIN",
    "IOU_THRESHOLD",
    "TARGET_IOU_THRESHOLD",
    "GroundingRouter",
    "Instance",
    "MatchResult",
    "ParseResult",
    "ResponsePrefixes",
    "RouteResult",
    "bbox_iou",
    "bbox_xywh_to_xyxy",
    "coerce_instances",
    "get_response_prefixes",
    "iou_xyxy",
    "match_bbox",
    "match_box",
    "match_prediction",
    "parse",
    "parse_grounding",
    "parse_grounding_response",
    "parse_response",
    "point_in_mask",
    "point_in_segmentation",
    "response_prefixes",
    "route",
    "route_grounding",
    "route_prediction",
    "route_response",
    "truncate_response",
    "truncate_response_prefixes",
    "validate_bbox",
    "validate_prediction_bbox",
    "xywh_to_xyxy",
    "xyxy_iou",
]
