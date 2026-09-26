"""Concise prompt-only refinement of the stronger v3p5 dual-field bridge.

Unlike v6, this adds no case-specific demonstration and does not alter the
generation structure.  It puts role and group checks immediately before output.
"""

from __future__ import annotations

from mmgcot_timeline_training.bridge_v2 import (
    BRIDGE_REGEX,
    EXTRACTOR_CONTENT_ERROR_LABELS,
    STUDENT_SELECTION_LABELS,
    TARGET_REFERENCE_REVIEW_LABELS,
    bbox_suffix_v2,
    bridge_pass,
    parse_bridge,
    token_ids_sha256,
)
from mmgcot_timeline_training.bridge_v3 import BRIDGE_PROMPT as V3_PROMPT


BRIDGE_VERSION = "mmgcot_target_bridge_v10_concise_dual_field"
_SUFFIX = '</think>\n<task_answer>"'
assert V3_PROMPT.endswith(_SUFFIX)
BRIDGE_PROMPT = V3_PROMPT[:-len(_SUFFIX)] + r'''

FINAL TARGET CHECK, applied once after finding the student's answer:
1. Decide the localization role from the question. For a property OF X or an
   action BY X, target X at its exact part/object level. For what a person
   wears/carries or what is on/in X, target the selected visible items, not
   the wearer or container. Never substitute a nearby related noun.
2. Name the student's final selected instance or complete explicitly selected
   group as ONE referring noun phrase. If your draft is a full answer sentence,
   keep its selected entity/group and remove the answer-sentence wording.
3. Preserve supported relations that distinguish same-kind instances. If the
   student's final reasoning remains undecided between different instances or
   object levels, output UNRESOLVED rather than choosing one.
4. The target must be a visible entity, part, or selected group. It must NEVER
   be only task_answer when task_answer is a color, material, shape, pattern,
   action, state, or other property value.
''' + _SUFFIX


def bbox_suffix_v10(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v2(question, target_entity_reference)
