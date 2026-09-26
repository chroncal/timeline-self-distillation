"""Target-only bridge in a separate user turn after frozen natural reasoning.

The original image/question and sampled reasoning tokens are replayed verbatim.
The added chat turn requests only a reference to the student's selected bbox
entity; it does not supply an answer, ground truth, or new visual evidence.
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


BRIDGE_VERSION = "mmgcot_target_bridge_v7_new_user_turn"

# Matches Qwen3.5's non-thinking assistant generation prompt.  The sampled
# reasoning ends before </think>; closing the old assistant turn does not alter
# any of its frozen token IDs.  Keep this exact suffix in the protocol record.
BRIDGE_PROMPT = r'''</think>
<|im_end|>
<|im_start|>user
Identify the visible entity or entity group that YOUR preceding reasoning
finally selected for the bounding box requested by the original image
question. Return a referring NOUN PHRASE for that selected image region.

This is an entity-extraction task, NOT a request to answer the original
question. Do not write the answer value, a color, material, shape, pattern,
action, state, or a full answer sentence in the target field. Do not localize
the image again. Do not use reference truth to correct a mistaken student
selection: describe the entity actually selected by your preceding reasoning.

Keep the exact target level and supported identifying relations:
- Color of the baby's shirt: name the shirt worn by the baby, not the baby or
  its color. Material of the container lid: name the lid, not the container or
  the material. Shape of the board beneath the pizza: name the board, not
  "rectangular" or the pizza.
- What someone wears or carries: name the selected garment/item or explicitly
  selected group, not the person or an answer sentence.
- What is in/on a sandwich: name the selected contents/group, not the sandwich.
- A phone on a laptop: name the phone, not the laptop. A chair behind a plate:
  name the chair, not the food on the plate.
- If multiple similar objects occur, keep the relation/appearance that the
  preceding reasoning used to identify ONE instance. If it did not settle on
  one instance, or alternated between bag versus carrots inside it without
  resolving that distinction, output UNRESOLVED. Never invent a relation.

Output only the noun phrase, or UNRESOLVED. No explanation or coordinates.
<|im_end|>
<|im_start|>assistant
<think>

</think>

<target_entity>"'''


def bbox_suffix_v7(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v5(question, target_entity_reference)
