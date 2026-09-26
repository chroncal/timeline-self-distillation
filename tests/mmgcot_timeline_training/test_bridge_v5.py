from mmgcot_timeline_training import bridge_v5


def test_target_only_bridge_has_no_answer_field() -> None:
    assert "<task_answer>" not in bridge_v5.BRIDGE_PROMPT
    assert "<target_entity>" in bridge_v5.BRIDGE_PROMPT
    parsed = bridge_v5.parse_bridge(
        'the shirt worn by the baby"</target_entity>', completed=True
    )
    assert parsed["target_entity_reference"] == "the shirt worn by the baby"
    assert "task_answer" not in parsed


def test_incomplete_target_is_not_repaired() -> None:
    assert bridge_v5.parse_bridge('the shirt', completed=False)["bridge_parse_status"] != "valid"
