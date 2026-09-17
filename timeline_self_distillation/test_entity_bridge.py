"""Bridge contract tests; semantic instance fidelity still needs real-model review."""

from copy import deepcopy

import pytest

from timeline_self_distillation.entity_bridge import (
    ENTITY_REGEX,
    LEGACY_ENTITY_OPEN,
    bridge_prompt,
    inspect_bridge_result,
)


def result(text, completed=True):
    return {"text": text, "completed": completed, "token_ids": [1, 2]}


@pytest.mark.parametrize(
    ("expression", "entity"),
    [
        ("the largest fishing ship", "the largest fishing ship"),
        ("the back of a yellow and red bus", "back of a yellow and red bus"),
        ("a white horse", "white horse"),
        ("pizza with greens", "pizza with greens"),
    ],
)
def test_nonempty_request_echo_is_not_certified_as_instance_bridge(expression, entity):
    audit = inspect_bridge_result(result(entity + '"'), expression, "instance_v2")
    assert audit["entity"] == entity
    assert audit["format_valid"]
    assert audit["echoes_request"]
    assert audit["semantic_status"] == "request_echo"
    assert not audit["instance_binding_verified"]


def test_richer_description_preserved_without_claiming_semantic_correctness():
    entity = "the red vessel in the foreground, ahead of the blue vessel"
    raw = result(entity + '"')
    snapshot = deepcopy(raw)
    audit = inspect_bridge_result(raw, "the largest ship", "instance_v2")
    assert raw == snapshot
    assert audit["entity"] == entity
    assert audit["usable_for_probe"]
    assert not audit["echoes_request"]
    assert audit["semantic_status"] == "needs_semantic_review"
    assert not audit["instance_binding_verified"]


@pytest.mark.parametrize(
    "raw",
    [result('a vessel"', False), result("a vessel"), result('"'), result('  "'), result('bus [1,2,3,4]"')],
)
def test_incomplete_or_invalid_output_never_falls_back_to_request(raw):
    audit = inspect_bridge_result(raw, "the original request", "instance_v2")
    assert not audit["format_valid"]
    assert not audit["usable_for_probe"]
    assert audit["entity"] == ""
    assert audit["issues"]


def test_explicit_unresolved_is_recorded_not_substituted():
    audit = inspect_bridge_result(result('UNRESOLVED"'), "a ship", "instance_v2")
    assert audit["format_valid"]
    assert not audit["usable_for_probe"]
    assert audit["semantic_status"] == "unresolved"
    assert audit["entity"] == "UNRESOLVED"


def test_versions_keep_legacy_prompt_and_do_not_include_sample_specific_answers():
    assert bridge_prompt("legacy_v1") == LEGACY_ENTITY_OPEN
    repaired = bridge_prompt("instance_v2")
    assert "completed reasoning" in repaired
    assert "final choice" in repaired
    assert "UNRESOLVED" in repaired
    assert repaired.endswith('</think>\n<target>"')
    assert all(word not in repaired.lower() for word in ("pizza", "horse", "fishing", "bus"))
    assert "0-9" in ENTITY_REGEX
    with pytest.raises(ValueError):
        bridge_prompt("unknown")


def test_actual_generation_seam_uses_full_cache_and_reports_echo(monkeypatch):
    from timeline_self_distillation import run_teacher_pilot as pilot

    late_cache = object()
    forked = object()
    calls = []

    def fake_fork(cache):
        assert cache is late_cache
        return forked

    def fake_generate(model, tokenizer, grammar, cache, suffix, seed, **kwargs):
        calls.append((cache, suffix, seed, kwargs))
        return result('white horse"')

    monkeypatch.setattr(pilot, "fork", fake_fork)
    monkeypatch.setattr(pilot, "constrained_generate", fake_generate)
    output = pilot.generate_entity_bridge(None, None, None, late_cache, "a white horse", 123)
    assert output["entity"] == "white horse"
    assert output["entity_audit"]["semantic_status"] == "request_echo"
    assert calls == [(forked, bridge_prompt("instance_v2"), 123, {"greedy": True, "limit": 96})]
