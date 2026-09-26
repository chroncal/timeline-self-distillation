"""Frozen semantic contract for the model-only MM-GCoT target bridge v2."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


BRIDGE_VERSION = "mmgcot_target_bridge_v2"
TASK_VALUE = r'[^"\\\r\n<>]{1,160}'
ENTITY_VALUE = r'[^"\\\r\n<>]{1,240}'
BRIDGE_REGEX = (
    rf'({TASK_VALUE})"</task_answer>\n'
    rf'<target_entity>"({ENTITY_VALUE})"</target_entity>'
)
BRIDGE_PROMPT = (
    "\nConvert the completed reasoning into two distinct fields. Copy the student's "
    "own final conclusion; do not independently solve the image, repair a mistaken "
    "choice, or invent a new target.\n"
    "- task_answer: the final answer to the original question, such as the concluded "
    "attribute, material, shape, action, or object category.\n"
    "- target_entity: one concise noun phrase referring to the particular visible "
    "image instance whose bounding box should be returned. For an attribute question, "
    "name the object that owns the attribute, not the attribute value. For an object "
    "identification question, refer to the selected visible instance; the concluded "
    "category may be its head noun. Preserve only identifying appearance or relations "
    "already used in the completed reasoning.\n"
    "The two fields have different roles even when they share a word. Do not include "
    "coordinates, box edges, confidence, or explanation. Use UNRESOLVED for either "
    "field when the completed reasoning did not determine it.\n"
    "</think>\n<task_answer>\""
)


def bbox_suffix_v2(question: str, target_entity_reference: str) -> str:
    """Condition bbox generation on the entity field, never on task_answer."""
    from mmgcot_diagnostic.protocol import BOX_OPEN

    return (
        f"\nThe image entity to locate is: {target_entity_reference}. "
        "Locate exactly that one visible image instance. The original question is "
        "context only; do not locate an answer value merely because it answers the question. "
        f"Original question: {question}\n"
        "Return only its bounding box in the original image as xmin,ymin,xmax,ymax "
        "on a 0 to 1000 normalized scale.\n" + BOX_OPEN
    )

STUDENT_SELECTION_LABELS = (
    "none",
    "selected_different_entity",
    "selection_not_unique",
    "selection_missing",
    "cannot_determine",
)
EXTRACTOR_CONTENT_ERROR_LABELS = (
    "none",
    "answer_value_not_entity",
    "reference_mismatches_student_selection",
    "reference_not_supported_by_reasoning",
    "reference_not_unique",
    "unresolved_consistent",
    "cannot_determine",
)
TARGET_REFERENCE_REVIEW_LABELS = (
    "same_target",
    "different_target",
    "description_not_unique",
    "cannot_determine",
)


def parse_bridge(text: str, *, completed: bool) -> dict[str, str]:
    """Parse a constrained bridge continuation without repairing its content."""
    match = re.fullmatch(BRIDGE_REGEX, text) if completed else None
    if match is None:
        return {
            "task_answer": "",
            "target_entity_reference": "",
            "bridge_parse_status": "format_or_incomplete",
            "task_answer_status": "invalid",
            "target_entity_reference_status": "invalid",
        }
    task_answer, target = (value.strip() for value in match.groups())
    task_status = "unresolved" if task_answer.casefold() == "unresolved" else "usable"
    target_status = "unresolved" if target.casefold() == "unresolved" else "usable"
    return {
        "task_answer": task_answer,
        "target_entity_reference": target,
        "bridge_parse_status": "valid",
        "task_answer_status": task_status,
        "target_entity_reference_status": target_status,
    }


def token_ids_sha256(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def bridge_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """Predeclared semantic interface gate, independent of GT correctness rate.

    The student may choose the wrong entity.  The bridge passes only by faithfully
    serializing that choice; ``same_gt_entity`` is deliberately not a gate input.
    """
    total = int(summary["cases"])
    valid = int(summary["valid_parse"])
    content = summary["extractor_content_error_counts"]
    faithful = int(content.get("none", 0)) + int(content.get("unresolved_consistent", 0))
    answer_only = int(content.get("answer_value_not_entity", 0))
    reasons: list[str] = []
    if valid / total < 0.95:
        reasons.append(f"valid parse rate {valid / total:.3f} < 0.95")
    if faithful / total < 0.90:
        reasons.append(f"faithful serialization rate {faithful / total:.3f} < 0.90")
    if answer_only / total > 0.05:
        reasons.append(f"answer-only target rate {answer_only / total:.3f} > 0.05")
    return not reasons, reasons
