"""Prompt-only refinement of the dual-field v3p5 target bridge.

The v3p5 parser, grammar and semantic gate are unchanged.  Only the prompt
adds final, concrete checks at the point of serializing the target field.
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


BRIDGE_VERSION = "mmgcot_target_bridge_v6p1_prompt_only"

FINAL_CHECK = r'''

FINAL SERIALIZATION CHECK — apply AFTER deciding task_answer and IMMEDIATELY
BEFORE writing target_entity. The target field is a REFERRING NOUN PHRASE, not
the task answer. Keep the exact entity level and the relations that identify
the student's particular selected instance. In each line below, the answer
value and target referent have different jobs:

- Material of the backpack next to the man in a dark blue shirt: if the
  student's answer is canvas, target_entity is "the backpack next to the man
  wearing a dark blue shirt", NEVER "canvas".
- Material of the table with a fork and butterknife: if the answer is wood,
  target_entity is "the table with a fork and butterknife", NEVER "wood".
- Color of the shirt worn by the baby that the woman bends over: target_entity
  is "the shirt worn by the baby", NEVER "the baby" or a color. The shirt is
  the garment being localized; the baby is only the owner relation.
- Material of the top of the three-section container beside the fork: if the
  student selected the lid/top, target_entity is "the lid of the three-section
  container beside the fork", NEVER "the container" or the material.
- What the passenger on the white and green bus is wearing: if the student
  selected both the white top and cloth at the waist, target_entity is "the
  white top and waist cloth worn by the bus passenger". NEVER copy a full
  answer sentence with a verb such as "is wearing".
- What is on or inside a sandwich: if the student selected cheese and ham,
  target_entity is "the cheese and ham inside the sandwich", NEVER the
  sandwich itself or a sentence describing the sandwich.
- The open thing by the cat: if the student ends with "carrots or a bag of
  carrots" and has not resolved bag versus contents, target_entity is
  UNRESOLVED. Do not guess one of the two levels.
- When two beds or cupcakes are discussed, "the bedspread" or "the cupcake"
  alone is not unique. Preserve the student's supported relation, such as
  "the bedspread on the bed with the flowers in front" or "the brown
  chocolate cupcake on the white plate". If the student never selected one,
  write UNRESOLVED.

The examples are only format and level checks. They cannot replace the actual
student choice: if the completed reasoning selected a different entity, name
that entity even when it does not match the reference target. Do not use the
image or an answer key to correct the student's selected object. If you cannot
name one unique selected entity or queried group, write UNRESOLVED.

Before emitting the final field, silently ask: "Is this a concrete noun phrase
for the same selected object, part, or group, with enough supported relations
to identify it?" If no, rewrite the target field now. In an attribute or
action question, target_entity MUST NOT equal task_answer.
'''

_SUFFIX = '</think>\n<task_answer>"'
assert V3_PROMPT.endswith(_SUFFIX)
BRIDGE_PROMPT = V3_PROMPT[:-len(_SUFFIX)] + FINAL_CHECK + _SUFFIX


def bbox_suffix_v6(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v2(question, target_entity_reference)
