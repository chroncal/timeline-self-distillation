"""Source-grounded selection tests; no manually supplied image answers."""

import pytest

from timeline_self_distillation.extractive_entity_bridge import evidence_candidates, selected_bridge


def test_candidates_are_exact_source_slices_and_remove_numeric_and_task_meta():
    text = (
        "The user wants to find the object.\n"
        "1. **Inspect the scene:** The object beside the pole has a striped surface.\n"
        "* A plain object sits behind it.\n"
        "* Bounding box: [10, 20, 30, 40].\n"
    )
    candidates = evidence_candidates(text)
    assert [c["text"] for c in candidates] == [
        "The object beside the pole has a striped surface.",
        "A plain object sits behind it.",
    ]
    for candidate in candidates:
        assert text[candidate["start"] : candidate["end"]] == candidate["text"]
        assert not any(c.isdigit() for c in candidate["text"])


def test_selection_passes_source_text_not_a_generated_rewrite():
    text = "The striped object is beside the pole. A plain object sits behind it."
    candidates = evidence_candidates(text)
    raw = {"text": "1</evidence_id>", "completed": True, "token_ids": [16]}
    bridge = selected_bridge(raw, candidates, "a striped object", text)
    assert bridge["entity"] == candidates[0]["text"]
    assert bridge["entity_audit"]["source_text_exact"]
    assert bridge["entity_audit"]["usable_for_probe"]
    assert not bridge["entity_audit"]["instance_binding_verified"]


@pytest.mark.parametrize("answer", ["0", "99", "1 and 2", ""])
def test_invalid_or_unresolved_selection_is_not_replaced(answer):
    text = "The striped object is beside the pole."
    bridge = selected_bridge(
        {"text": answer + "</evidence_id>", "completed": True}, evidence_candidates(text), "object", text
    )
    assert not bridge["entity_audit"]["usable_for_probe"]
    assert bridge["entity"] == ""


def test_candidate_index_does_not_certify_target_correctness():
    text = "The striped object is beside the pole. A plain object sits behind it."
    candidates = evidence_candidates(text)
    bridge = selected_bridge({"text": "2</evidence_id>", "completed": True}, candidates, "a striped object", text)
    assert bridge["entity"] == "A plain object sits behind it."
    assert bridge["entity_audit"]["semantic_status"] == "source_grounded_needs_target_review"


def test_absent_evidence_does_not_fall_back_to_question():
    text = "The user wants to identify a target.\nBounding box: [1,2,3,4]."
    assert evidence_candidates(text) == []
