from mmgcot_timeline_training import bridge_v3, bridge_v6


def test_v6_is_prompt_only_refinement() -> None:
    assert bridge_v6.BRIDGE_REGEX == bridge_v3.BRIDGE_REGEX
    assert bridge_v6.BRIDGE_PROMPT.startswith(bridge_v3.BRIDGE_PROMPT.split("</think>")[0])
    assert bridge_v6.BRIDGE_PROMPT.endswith('</think>\n<task_answer>"')
    for phrase in ("NEVER \"canvas\"", "NEVER \"wood\"", "the shirt worn by the baby",
                   "the lid of the three-section", "UNRESOLVED"):
        assert phrase in bridge_v6.BRIDGE_PROMPT
