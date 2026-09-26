"""Single-field target-only bridge; no task-answer field is generated."""

from __future__ import annotations

import re

from mmgcot_timeline_training.bridge_v2 import (
    ENTITY_VALUE,
    EXTRACTOR_CONTENT_ERROR_LABELS,
    STUDENT_SELECTION_LABELS,
    TARGET_REFERENCE_REVIEW_LABELS,
    bbox_suffix_v2,
    bridge_pass,
    token_ids_sha256,
)


BRIDGE_VERSION = "mmgcot_target_bridge_v5_target_only"
BRIDGE_REGEX = rf'({ENTITY_VALUE})"</target_entity>'
BRIDGE_PROMPT = r'''
Extract the image entity that the student's COMPLETED reasoning intended the
FINAL BOUNDING BOX to locate. Output ONE noun phrase only. This is not a task
answer: NEVER copy the final answer sentence, color, material, pattern, action,
or state into the target field. Do not give coordinates, confidence, or an
explanation. Do not solve the image again or use any reference truth to repair
the student's choice.

First distinguish the role asked by the ORIGINAL QUESTION:
- For a property, state, or action OF X, the box target is the concrete X that
  owns the property or performs the action. Retain all the identifying words
  and relations in the question, then use the student's reasoning to determine
  which instance it finally chose. A shirt's color targets the shirt, not the
  baby wearing it or the color. A lid's material targets the lid, not the box.
- For "What is X wearing/carrying?", the target is the visible worn/carried
  item or explicitly selected group of items, not X. For "What is on/in/inside
  X?", the target is the visible content object or selected group, not X.
- For "What is the thing near X?", the target is the thing selected by the
  student's reasoning, not X.

Keep the exact object level: garment versus wearer, lid versus container,
frosting versus cupcake, bag versus carrots inside the bag, and contents versus
their container. A grammatically valid full answer sentence is still INVALID
as a target reference. Name just the selected object or group in a noun phrase.
Never leave out an identifying relation when several similar objects appear.
Use only relations in the question or the student's reasoning; do not invent
a visual feature. If the reasoning finally chooses a wrong object, preserve
THAT choice instead of silently correcting it to match the question. If it
alternates between two entities without a final choice, write UNRESOLVED.

Examples of the required output (the examples teach wording, not image truth):
- Question: What is the color of the shirt worn by the baby that the woman
  bending over has? Student ultimately selects the baby's shirt.
  Output: the shirt worn by the baby that the woman is bending over
  Wrong: the baby; black; shirt
- Question: What is the material of the top of the three-section container
  beside the little fork? Student selects its lid.
  Output: the lid of the three-section container beside the little fork
  Wrong: the container; plastic
- Question: What is on the sandwich on the table? Student identifies cheese
  and ham inside the bagel sandwich.
  Output: the cheese and ham inside the bagel sandwich
  Wrong: the bagel; the sandwich is a bagel filled with cheese and ham
- Question: What is the passenger on the white and green bus wearing?
  Student selects a white top and cloth at the waist.
  Output: the white top and waist cloth worn by the bus passenger
  Wrong: the passenger; the passenger is wearing a white top and cloth
- Question: What is the open thing beside the cat? Student keeps alternating
  between a bag and the carrots inside it without deciding.
  Output: UNRESOLVED
  Wrong: carrots; bag
- Question: What is the color of the brown chocolate cupcake on the white
  plate holding a cupcake with brown and blue frosting? Student selects that
  brown cupcake.
  Output: the brown chocolate cupcake on the white plate
  Wrong: cupcake; brown

Immediately before output, check that your text is one concrete entity noun
phrase with enough supported relations to distinguish it. If you cannot form
that phrase from the completed reasoning, output UNRESOLVED.
</think>
<target_entity>"'''


def parse_bridge(text: str, *, completed: bool) -> dict[str, str]:
    match = re.fullmatch(BRIDGE_REGEX, text) if completed else None
    if match is None:
        return {
            "target_entity_reference": "",
            "bridge_parse_status": "format_or_incomplete",
            "target_entity_reference_status": "invalid",
        }
    target = match.group(1).strip()
    return {
        "target_entity_reference": target,
        "bridge_parse_status": "valid",
        "target_entity_reference_status": (
            "unresolved" if target.casefold() == "unresolved" else "usable"
        ),
    }


def bbox_suffix_v5(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v2(question, target_entity_reference)
