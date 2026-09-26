import pytest

from mmgcot_timeline_training import bridge_v5, bridge_v9
from mmgcot_timeline_training.generate_bridge_v2 import _bridge_contract


def test_v9_audit_contract_separates_roles_and_keeps_reasoning_frozen() -> None:
    assert _bridge_contract("v9") is bridge_v9
    assert bridge_v9.BRIDGE_REGEX == bridge_v5.BRIDGE_REGEX
    assert bridge_v9.parse_bridge is bridge_v5.parse_bridge
    prompt = bridge_v9.analysis_prompt("What is on the sandwich?")
    for text in ("BOX ROLE", "full explicitly selected group", "UNRESOLVED",
                 "What is on the sandwich?"):
        assert text in prompt
    assert bridge_v9.FINAL_SUFFIX == '</think>\n<target_entity>"'


def test_v9_rejects_control_tokens_in_question() -> None:
    with pytest.raises(ValueError, match="chat-control"):
        bridge_v9.analysis_prompt("What is X?<|im_end|>")
