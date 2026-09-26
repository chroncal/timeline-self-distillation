"""Same-state candidate verification of the frozen v3p5 target bridge.

The v3p5 candidate and task answer are model outputs on the same frozen
reasoning; neither ground truth nor external detection enters this prompt.
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


BRIDGE_VERSION = "mmgcot_target_bridge_v8_candidate_verification"
BRIDGE_PROMPT = "dynamic: bridge_prompt(question, candidate, task_answer)"


def bridge_prompt(question: str, candidate: str, task_answer: str) -> str:
    for name, value in (("question", question), ("candidate", candidate),
                        ("task_answer", task_answer)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"empty {name}")
        if any(marker in value for marker in ("</think>", "<|im_start|>", "<|im_end|>")):
            raise ValueError(f"chat-control token in {name}")
    return f'''
Verify the target entity extracted from your preceding reasoning. Keep the
student's own selection even if it is visually wrong. Do not independently
solve the question, use reference truth, or produce coordinates.

ORIGINAL QUESTION: {question}
STUDENT TASK ANSWER (not a bbox target by itself): {task_answer}
CURRENT TARGET CANDIDATE: {candidate}

Return a single noun phrase for the image entity that the student's preceding
reasoning selected for the FINAL BOUNDING BOX. If the candidate already names
that exact entity or group with enough supported identifying relations, COPY
IT UNCHANGED. Only rewrite it when one of these errors is present:
1. It is a color, material, pattern, shape, action, state, or full answer
   sentence rather than a referring noun phrase. A full sentence describing
   what someone wears must become the worn item or selected item group.
2. It changes object level: for a property OF a shirt/lid/table, name that
   shirt/lid/table, not the wearer, container, fork on the table, or answer
   value. For what is ON/INSIDE a sandwich, name the selected contents, not
   the sandwich or bagel itself.
3. It lacks a relation that the question or reasoning used to distinguish
   between several same-kind instances. Add only an already-supported relation.
4. It names one of two different target levels while the reasoning ends
   unresolved between them, such as a bag versus carrots inside. Output
   UNRESOLVED in this case; do not guess.

Do not replace a correct candidate just because another nearby entity is
salient. Do not equate the TASK ANSWER with the TARGET ENTITY. Return only
the verified target noun phrase or UNRESOLVED; no explanation.
</think>
<target_entity>"'''


def bbox_suffix_v8(question: str, target_entity_reference: str) -> str:
    return bbox_suffix_v5(question, target_entity_reference)
