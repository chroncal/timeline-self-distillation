"""Versioned instance-description prompts and non-oracular bridge diagnostics.

Format completion is not evidence of visual instance binding. Echo detection
is a warning, not a sample filter or a substitute for semantic evaluation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

LEGACY_ENTITY_OPEN = (
    "\nIdentify only the final target object in words, preserving its appearance and relations. "
    'Do not give coordinates.\n</think>\n<target>"'
)
INSTANCE_ENTITY_OPEN = (
    "\nDescribe the particular image instance selected by the completed reasoning above. "
    "Extract the final choice, not merely the user's request and not an earlier rejected candidate. "
    "Preserve its identifying appearance, relative position, and distinguishing relationships "
    "when the reasoning explicitly provides them, so that another reader can distinguish this "
    "instance from the alternatives without redoing the reasoning. "
    "Use only information stated in that reasoning; do not invent or fill in missing details. "
    "Do not include coordinate values, estimated box edges, a bounding box, or an explanation. "
    "Return one concise identifying noun phrase. If the reasoning did not resolve a target "
    'instance, output UNRESOLVED.\n</think>\n<target>"'
)
# Keep the exact original grammar and budget for the matched bridge comparison.
ENTITY_REGEX = r'[^"\\\r\n<>0-9]{1,240}"'
BRIDGE_VERSIONS = ("instance_v2", "legacy_v1")


def bridge_prompt(version: str) -> str:
    if version == "instance_v2":
        return INSTANCE_ENTITY_OPEN
    if version == "legacy_v1":
        return LEGACY_ENTITY_OPEN
    raise ValueError(f"unknown entity bridge version: {version}")


def _normalize_request(text: str) -> str:
    normalized = " ".join(text.casefold().strip().rstrip(".?!").split())
    return re.sub(r"^(?:a|an|the)\s+", "", normalized)


def inspect_bridge_result(result: Mapping[str, Any], expression: str, version: str) -> dict[str, Any]:
    """Preserve model text; expose echo/unresolved states without guessing identity.

    No GT, image annotation, sample-specific rewrite, or fallback is available
    here. A non-echo phrase remains *unverified*: extra words can be hallucinated.
    """
    bridge_prompt(version)  # Validate version even when generation failed.
    text = result.get("text", "")
    issues = []
    if not result.get("completed", False):
        issues.append("generation_incomplete")
    if not isinstance(text, str) or re.fullmatch(ENTITY_REGEX, text) is None:
        issues.append("invalid_entity_grammar")
    entity = text[:-1].strip() if isinstance(text, str) and not issues else ""
    if not entity:
        issues.append("empty_entity")
    format_valid = not issues
    unresolved = format_valid and entity.casefold() == "unresolved"
    echo = format_valid and _normalize_request(entity) == _normalize_request(expression)
    if unresolved:
        issues.append("model_reports_unresolved")
    if echo:
        issues.append("request_echo_no_new_instance_information")
    semantic_status = (
        "invalid"
        if not format_valid
        else "unresolved"
        if unresolved
        else "request_echo"
        if echo
        else "needs_semantic_review"
    )
    return {
        "version": version,
        "entity": entity,
        "format_valid": format_valid,
        "usable_for_probe": format_valid and not unresolved,
        "echoes_request": echo,
        "semantic_status": semantic_status,
        "instance_binding_verified": False,
        "issues": issues,
    }
