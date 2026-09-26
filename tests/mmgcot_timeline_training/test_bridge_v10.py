from mmgcot_timeline_training import bridge_v3, bridge_v10
from mmgcot_timeline_training.generate_bridge_v2 import _bridge_contract


def test_v10_keeps_v3_dual_field_contract_and_adds_concise_final_check() -> None:
    assert _bridge_contract("v10") is bridge_v10
    assert bridge_v10.BRIDGE_REGEX == bridge_v3.BRIDGE_REGEX
    assert bridge_v10.parse_bridge is bridge_v3.parse_bridge
    assert bridge_v10.BRIDGE_PROMPT.startswith(bridge_v3.BRIDGE_PROMPT.split("</think>")[0])
    assert bridge_v10.BRIDGE_PROMPT.endswith('</think>\n<task_answer>"')
    for phrase in ("exact part/object level", "complete explicitly selected",
                   "UNRESOLVED", "must NEVER"):
        assert phrase in bridge_v10.BRIDGE_PROMPT
