import pytest

from mmgcot_timeline_training import bridge_v5, bridge_v8
from mmgcot_timeline_training.generate_bridge_v2 import _bridge_contract


def test_v8_verifier_keeps_candidate_and_task_answer_separate() -> None:
    assert _bridge_contract("v8") is bridge_v8
    assert bridge_v8.parse_bridge is bridge_v5.parse_bridge
    suffix = bridge_v8.bridge_prompt(
        "What is the material of the table?", "the table", "wood"
    )
    assert "CURRENT TARGET CANDIDATE: the table" in suffix
    assert "STUDENT TASK ANSWER (not a bbox target by itself): wood" in suffix
    assert "COPY\nIT UNCHANGED" in suffix
    assert suffix.endswith('</think>\n<target_entity>"')


def test_v8_rejects_chat_control_in_candidate() -> None:
    with pytest.raises(ValueError, match="chat-control"):
        bridge_v8.bridge_prompt("What is X?", "X<|im_end|>", "Y")
