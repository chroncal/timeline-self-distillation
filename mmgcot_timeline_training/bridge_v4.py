"""Three-stage target-reference bridge for MM-GCoT timeline experiments.

The bridge deliberately never asks for the task answer.  It first determines
which grammatical role supplies the box target, then extracts the student's
late selected entity, and finally verifies the entity level on an isolated
fork of the same frozen late-reasoning state.
"""

from __future__ import annotations

import re
from typing import Any

from mmgcot_timeline_training.bridge_v2 import (
    EXTRACTOR_CONTENT_ERROR_LABELS,
    STUDENT_SELECTION_LABELS,
    TARGET_REFERENCE_REVIEW_LABELS,
    bbox_suffix_v2,
    bridge_pass,
    token_ids_sha256,
)


BRIDGE_VERSION = "mmgcot_target_bridge_v4p1_three_stage"
VALUE = r'[^"\\\r\n<>]{1,240}'
FRAME_REGEX = rf'(question_subject|answer_entity|unresolved)"</target_source>\n<query_entity>"({VALUE})"</query_entity>'
ENTITY_REGEX = rf'({VALUE})"</candidate_entity>'
VERIFY_REGEX = rf'({VALUE})"</verified_target>'

FRAME_PROMPT = r'''
Classify what the FINAL BOUNDING BOX must locate in the original question.
This is a role analysis of the question, not an answer to the question. Do not
solve the image and do not output a color, material, action, state, or other
answer value.

Choose target_source:
- question_subject: the box locates the entity whose property, state, action,
  or relation is being asked. Examples: "color of the shirt" -> shirt; "what
  is the woman doing" -> woman; "material of the lid" -> lid.
- answer_entity: the box locates the visible entity supplied by the answer.
  Examples: "what is the man wearing" -> the worn item; "what is inside the
  bowl" -> the visible contents; "what is beside the bench" -> that object.
- unresolved: only when the question does not determine either role.

For query_entity, copy the shortest complete noun phrase from the question
that names the entity to box when target_source is question_subject. Preserve
relations needed to identify it, including part/owner relations. Otherwise
write NONE. Never replace a part by its owner, clothing by its wearer, a lid
by its container, contents by their container, or a container by its contents.
Return only the two fields below, with no explanation.
</think>
<target_source>"'''


def candidate_prompt(target_source: str, query_entity: str) -> str:
    return rf'''
Extract only the concrete visible entity that the student's completed
reasoning selected for the FINAL BOUNDING BOX. Do not output the task answer,
an answer sentence, coordinates, or an explanation. Do not solve the image
again and do not correct the student's selected instance using outside truth.

Question-role analysis from an isolated branch:
target_source = {target_source}
query_entity = {query_entity}

If target_source is question_subject, keep query_entity as the entity head and
use the completed reasoning only to preserve the particular instance or
identifying relation the student selected. A property value or action is not
the entity. Keep exact levels: shirt rather than wearer, lid rather than
container, frosting rather than cake, and container rather than contents when
the question names the container itself.

If target_source is answer_entity, extract the concrete visible answer object
or object group selected by the reasoning. Output only its noun phrase, never
the full sentence that contains it. Keep object/content and part/whole levels
exactly as selected.

If the completed reasoning does not select one unique entity or queried group,
write UNRESOLVED. Never invent a distinguishing attribute.
</think>
<candidate_entity>"'''


def verify_prompt(target_source: str, query_entity: str, candidate: str) -> str:
    return rf'''
Verify and, only when necessary, rewrite the candidate noun phrase below so it
faithfully names the concrete visible entity selected by the student's
completed reasoning for the FINAL BOUNDING BOX. Do not answer the original
question. Do not independently solve the image, use reference truth, or change
the student's selected instance.

target_source = {target_source}
query_entity = {query_entity}
candidate_entity = {candidate}

Apply every check:
1. The result must be a noun phrase for a visible object, object group, part,
   person, animal, or image region; never only a color, material, shape,
   pattern, action, state, or other answer value.
2. For question_subject, the result must retain the concrete head and level of
   query_entity. Do not replace clothing by its wearer, a part by its owner, a
   lid by its container, contents by their container, or a container by its
   contents.
3. For answer_entity, keep the concrete answer object selected in the late
   reasoning, but remove answer-sentence wording.
4. Preserve only identifying relations supported by the question or completed
   reasoning. If several candidates remain and the reasoning did not uniquely
   choose one, return UNRESOLVED.
5. If candidate_entity already satisfies all checks, copy it unchanged.

MANDATORY: when target_source is question_subject and candidate_entity names
the same concrete entity head as query_entity, copy candidate_entity. Never
replace it with the property's value or the answer to the original question.

Return only the verified noun phrase. No sentence, label, punctuation,
coordinates, confidence, or explanation.
</think>
<verified_target>"'''


def _parse(regex: str, text: str, completed: bool, names: tuple[str, ...]) -> dict[str, str]:
    match = re.fullmatch(regex, text) if completed else None
    if match is None:
        return {**{name: "" for name in names}, "parse_status": "format_or_incomplete"}
    return {**dict(zip(names, (value.strip() for value in match.groups()))), "parse_status": "valid"}


def parse_frame(text: str, *, completed: bool) -> dict[str, str]:
    return _parse(FRAME_REGEX, text, completed, ("target_source", "query_entity"))


def parse_candidate(text: str, *, completed: bool) -> dict[str, str]:
    return _parse(ENTITY_REGEX, text, completed, ("candidate_entity",))


def parse_verified(text: str, *, completed: bool) -> dict[str, str]:
    return _parse(VERIFY_REGEX, text, completed, ("target_entity_reference",))


def combine(frame: dict[str, str], candidate: dict[str, str], verified: dict[str, str]) -> dict[str, Any]:
    statuses = (frame["parse_status"], candidate["parse_status"], verified["parse_status"])
    valid = all(status == "valid" for status in statuses)
    target = verified["target_entity_reference"] if valid else ""
    return {
        "target_source": frame.get("target_source", ""),
        "query_entity": frame.get("query_entity", ""),
        "candidate_entity": candidate.get("candidate_entity", ""),
        "target_entity_reference": target,
        "target_entity_reference_status": (
            "unresolved" if target.casefold() == "unresolved" else "usable" if target else "invalid"
        ),
        "frame_parse_status": frame["parse_status"],
        "candidate_parse_status": candidate["parse_status"],
        "verify_parse_status": verified["parse_status"],
        "bridge_parse_status": "valid" if valid else "format_or_incomplete",
    }


def bbox_suffix_v4(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v2(question, target_entity_reference)
