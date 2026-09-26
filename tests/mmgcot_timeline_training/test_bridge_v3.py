from __future__ import annotations

from mmgcot_timeline_training import bridge_v2, bridge_v3
from mmgcot_timeline_training.generate_bridge_v2 import _bridge_seed, parse_args
from mmgcot_diagnostic.protocol import stable_seed


def test_v3_is_prompt_only_change() -> None:
    assert bridge_v3.BRIDGE_PROMPT != bridge_v2.BRIDGE_PROMPT
    assert bridge_v3.BRIDGE_REGEX == bridge_v2.BRIDGE_REGEX
    assert bridge_v3.parse_bridge is bridge_v2.parse_bridge
    assert bridge_v3.bridge_pass is bridge_v2.bridge_pass


def test_v3_prompt_explicitly_forbids_answer_values_as_entities() -> None:
    prompt = bridge_v3.BRIDGE_PROMPT
    assert "HARD RULE" in prompt
    assert "Forbidden target_entity: blue" in prompt
    assert "Could this phrase by itself" in prompt
    assert "Do not solve the image again" in prompt
    assert "MUST NOT equal task_answer" in prompt
    assert "target_entity: the freight train cars" in prompt
    assert "glasses, not the man" in prompt
    assert "output\nUNRESOLVED rather than a generic noun" in prompt


def test_v3_requires_frozen_reasoning() -> None:
    try:
        parse_args([
            "--selection", "selection.jsonl", "--output-dir", "out",
            "--bridge-version", "v3", "--device", "0",
            "--shard-index", "0", "--num-shards", "1",
        ])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("v3 accepted generation without frozen reasoning")


def test_bridge_seed_is_prompt_version_invariant() -> None:
    assert _bridge_seed("sample") == stable_seed("sample", 0, "bridge_v2")
