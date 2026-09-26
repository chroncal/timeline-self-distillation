from mmgcot_timeline_training import bridge_v4


def test_prompts_never_request_task_answer_or_gt() -> None:
    joined = "\n".join((
        bridge_v4.FRAME_PROMPT,
        bridge_v4.candidate_prompt("question_subject", "the shirt worn by the baby"),
        bridge_v4.verify_prompt("question_subject", "the shirt worn by the baby", "the baby"),
    )).lower()
    assert "<task_answer>" not in joined
    assert "reference truth" in joined
    assert "shirt rather than wearer" in joined
    assert "lid rather than" in joined
    assert "container rather than contents" in joined
    assert "copy candidate_entity" in joined


def test_parse_and_combine_valid_three_stages() -> None:
    frame = bridge_v4.parse_frame(
        'question_subject"</target_source>\n<query_entity>"the black shirt worn by the baby"</query_entity>',
        completed=True,
    )
    candidate = bridge_v4.parse_candidate('the black shirt"</candidate_entity>', completed=True)
    verified = bridge_v4.parse_verified('the black shirt worn by the baby"</verified_target>', completed=True)
    result = bridge_v4.combine(frame, candidate, verified)
    assert result["bridge_parse_status"] == "valid"
    assert result["target_source"] == "question_subject"
    assert result["target_entity_reference"] == "the black shirt worn by the baby"


def test_any_incomplete_stage_invalidates_bridge() -> None:
    frame = bridge_v4.parse_frame("", completed=False)
    candidate = bridge_v4.parse_candidate('the lid"</candidate_entity>', completed=True)
    verified = bridge_v4.parse_verified('the lid"</verified_target>', completed=True)
    result = bridge_v4.combine(frame, candidate, verified)
    assert result["bridge_parse_status"] == "format_or_incomplete"
    assert result["target_entity_reference"] == ""
    assert result["target_entity_reference_status"] == "invalid"
