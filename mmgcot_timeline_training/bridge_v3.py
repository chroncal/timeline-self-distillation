"""Prompt-only target-bridge v3 contract.

The parser, grammar, review labels, and semantic gate are intentionally
identical to v2.  The only experimental change is the extraction prompt.
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


BRIDGE_VERSION = "mmgcot_target_bridge_v3p5_prompt_only"

# This prompt starts a grammar-constrained continuation immediately after the
# opening quote of task_answer, exactly as v2 did.
BRIDGE_PROMPT = r'''
Read the original question and the student's completed reasoning. Serialize the
student's own conclusion into two fields. Do not solve the image again, use a
reference answer, correct the student's choice, or invent visual details.

FIELD 1 — task_answer:
Write the student's answer to the question. This may be a color, material,
shape, pattern, action, state, object category, or phrase.

FIELD 2 — target_entity:
Write a concrete noun phrase for the visible image region whose bounding box
the student intended to locate. Reuse the target noun and any identifying
relations from the original question, restricted to the instance selected by
the student's reasoning.

HARD RULE: target_entity must denote a visible object, an explicitly queried
group of visible objects, object part, person, animal, or other image region.
A color, material, shape, pattern,
action, state, or answer-value expression is NEVER a target entity by itself.
Do not copy task_answer into target_entity when task_answer is only such a
value. Instead name the object that has that value.

Examples of the required distinction:
- Question: What is the color of the fence that the bird is perched on?
  task_answer: blue
  target_entity: the fence that the bird is perched on
  Forbidden target_entity: blue
- Question: What is the material of the seat on the dark green bench?
  task_answer: wood
  target_entity: the seat on the dark green bench
  Forbidden target_entity: wood
- Question: What is the color of the freight train cars?
  task_answer: grey
  target_entity: the freight train cars
  Forbidden target_entity: grey
- Question: What is the shape of the cutting board under the pizza?
  task_answer: rectangular
  target_entity: the cutting board under the pizza
  Forbidden target_entity: rectangular
- Question: What is the woman on the motorcycle doing?
  task_answer: posing for a photo
  target_entity: the woman on the motorcycle
  Forbidden target_entity: posing for a photo

Before writing target_entity, silently check: "Could this phrase by itself
point to a visible object, queried object group, or image region?" If no, replace it with the
concrete owner/actor/object phrase supported by the question and reasoning.
Never output punctuation, a dictionary-like field such as {color: blue},
coordinates, confidence, or explanation. Include enough relations from the
question to distinguish the selected instance. If the student's reasoning did
not select any entity, output UNRESOLVED; do not guess.

FINAL MANDATORY CHECK IMMEDIATELY BEFORE OUTPUT:
If the question asks for a color, material, shape, pattern, action, or state,
target_entity MUST contain a concrete entity head noun from the question (for
example fence, cars, seat, cutting board, woman). In these questions,
target_entity MUST NOT equal task_answer and MUST NOT be only the answer value.
If it does, rewrite target_entity as the answer-owning object or actor now.

DIRECT-SUBJECT RULE FOR PROPERTY QUESTIONS:
For "What is the [property] of X?", target_entity is X itself: the exact object
or part whose property is requested. Never move upward to X's owner or
container. Therefore "color of the glasses worn by the man" targets the
glasses, not the man; "color of the frosting on the cake" targets the frosting,
not the cake. Keep the complete relation needed to identify X.

UNIQUENESS RULE:
If several matching instances are visible, include every distinguishing
position, appearance, or relation that the student's reasoning actually used.
If the reasoning did not uniquely choose one instance or queried group, output
UNRESOLVED rather than a generic noun or an invented distinction.

CONTENT/OBJECT RULE:
For questions such as "What is inside/on X?", target_entity must be a noun
phrase for the visible content object or objects selected by the reasoning,
not a full answer sentence of the form "X is filled with ...". If no one
visible content target was selected, output UNRESOLVED.

FINAL EXAMPLE: For "What is the color of the glasses worn by the man?",
target_entity is "the glasses worn by the man", NEVER "the man".
For "What material is the book on the laptop made of?", target_entity is
"the book on the laptop", NEVER "paper" or another material value.
</think>
<task_answer>"'''


def bbox_suffix_v3(question: str, target_entity_reference: str) -> str:
    """Alias preserving the unchanged v2 bbox-conditioning contract."""
    return bbox_suffix_v2(question, target_entity_reference)
