"""A short prompted self-audit before serializing the selected bbox entity.

The frozen natural reasoning is never changed.  This branch permits at most
192 extra thinking tokens to distinguish the queried role from its answer.
"""

from __future__ import annotations

from mmgcot_timeline_training.bridge_v5 import (
    BRIDGE_REGEX,
    EXTRACTOR_CONTENT_ERROR_LABELS,
    STUDENT_SELECTION_LABELS,
    TARGET_REFERENCE_REVIEW_LABELS,
    bbox_suffix_v5,
    bridge_pass,
    parse_bridge,
    token_ids_sha256,
)


BRIDGE_VERSION = "mmgcot_target_bridge_v9_thinking_audit"
BRIDGE_PROMPT = "dynamic: analysis_prompt(question); then <=192 thinking tokens"
FINAL_SUFFIX = '</think>\n<target_entity>"'


def analysis_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("empty original question")
    if any(marker in question for marker in ("</think>", "<|im_start|>", "<|im_end|>")):
        raise ValueError("chat-control token in original question")
    return f'''
Before giving the bounding-box target, audit the entity you selected in your
reasoning. This is NOT another image question and must not change your choice.
The original question was: {question}

In this private audit, explicitly determine:
1. The queried BOX ROLE: property owner/actor, answer object, worn or carried
   item, contents on/in an object, or other. A material, color, pattern,
   action, or full answer sentence is never itself the box entity.
2. The exact object, object part, or full explicitly selected group in your
   preceding reasoning. Keep the subject/owner relation and every supported
   identifying relation. A fork on a table does not replace the table whose
   material was asked. A bagel does not replace selected cheese and ham on
   a sandwich. A wearer does not replace the selected garment.
3. Whether you actually settled on one entity or queried group. If you ended
   undecided between levels such as a bag and carrots inside it, the target
   must be UNRESOLVED. Do not guess or invent a distinguishing feature.

Think through those three checks briefly, then end thinking. After thinking,
emit only a concrete referring noun phrase for YOUR selected box entity or
UNRESOLVED. Never output the task-answer value as an entity.'''


def bbox_suffix_v9(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v5(question, target_entity_reference)
