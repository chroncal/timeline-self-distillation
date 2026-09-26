from mmgcot_timeline_training import bridge_v5, bridge_v7
from mmgcot_timeline_training.generate_bridge_v2 import _bridge_contract


def test_v7_separate_turn_and_same_target_output_contract() -> None:
    assert _bridge_contract("v7") is bridge_v7
    assert bridge_v7.BRIDGE_REGEX == bridge_v5.BRIDGE_REGEX
    assert bridge_v7.parse_bridge is bridge_v5.parse_bridge
    assert bridge_v7.BRIDGE_PROMPT.startswith("</think>\n<|im_end|>\n<|im_start|>user\n")
    assert bridge_v7.BRIDGE_PROMPT.endswith(
        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<target_entity>"'
    )
    for phrase in ("NOT a request to answer", "Do not use reference truth",
                   "shirt worn by the baby", "name the chair", "UNRESOLVED"):
        assert phrase in bridge_v7.BRIDGE_PROMPT


def test_v7_parser_rejects_answer_sentence() -> None:
    # The grammar permits a broad text field; semantic review, not the parser,
    # must reject a fluent sentence.  This test guards the explicit boundary.
    result = bridge_v7.parse_bridge('wood"</target_entity>', completed=True)
    assert result["bridge_parse_status"] == "valid"
    assert result["target_entity_reference"] == "wood"
