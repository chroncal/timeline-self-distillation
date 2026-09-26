from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmgcot_timeline_training.prepare_bridge_review_delta import (
    ReviewDeltaError,
    normalize_target_entity_reference,
    prepare_bridge_review_delta,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _package_row(sample_id: str, target: str, *, image_id: str = "image-1") -> dict:
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "image_sha256": "image-hash-" + sample_id,
        "question": "What is the color of the cup?",
        "reasoning_token_ids_sha256": "hash-" + sample_id,
        "task_type": "attribute",
        "split": "dev",
        "bridge_parse_status": "valid",
        "target_entity_reference": target,
    }


def _review_rows(case_id: str, label: str) -> dict:
    return {
        "case_id": case_id,
        "student_target_selection_error": label,
        "extractor_content_error": "none",
        "target_reference_review_label": "same_target",
    }


def _make_inputs(tmp_path: Path) -> dict[str, list[Path] | Path]:
    v6 = tmp_path / "v6.jsonl"
    _write_jsonl(
        v6,
        [
            _package_row("s1", " The CUP. "),
            _package_row("s2", "the plate"),
        ],
    )

    v3p5_a = tmp_path / "v3p5_a.jsonl"
    v3p5_b = tmp_path / "v3p5_b.jsonl"
    _write_jsonl(v3p5_a, [_package_row("s1", "the cup")])
    _write_jsonl(v3p5_b, [_package_row("s2", "the bowl")])

    mappings: list[Path] = []
    reviewer_a: list[Path] = []
    reviewer_b: list[Path] = []
    adjudicated: list[Path] = []
    for stem, sample_id, case_id, label in (
        ("a", "s1", "case-a", "none"),
        ("b", "s2", "case-b", "selected_different_entity"),
    ):
        mapping = tmp_path / f"mapping_{stem}.jsonl"
        a = tmp_path / f"reviewer_a_{stem}.jsonl"
        b = tmp_path / f"reviewer_b_{stem}.jsonl"
        final = tmp_path / f"adjudicated_{stem}.jsonl"
        _write_jsonl(mapping, [{"case_id": case_id, "sample_id": sample_id}])
        _write_jsonl(a, [_review_rows(case_id, label)])
        _write_jsonl(b, [_review_rows(case_id, label)])
        _write_jsonl(final, [{**_review_rows(case_id, label), "reason": "frozen"}])
        mappings.append(mapping)
        reviewer_a.append(a)
        reviewer_b.append(b)
        adjudicated.append(final)
    return {
        "v6": v6,
        "v3p5": [v3p5_a, v3p5_b],
        "mappings": mappings,
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
        "adjudicated": adjudicated,
    }


def _run(tmp_path: Path, inputs: dict[str, list[Path] | Path]) -> tuple[dict, list[dict], list[dict]]:
    unchanged = tmp_path / "unchanged.jsonl"
    changed = tmp_path / "changed.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_bridge_review_delta(
        inputs["v6"],
        inputs["v3p5"],
        inputs["mappings"],
        inputs["reviewer_a"],
        inputs["reviewer_b"],
        inputs["adjudicated"],
        unchanged,
        changed,
        manifest_path,
    )
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
    return manifest, read(unchanged), read(changed)


def test_routes_only_textually_equal_references_and_reuses_adjudication(tmp_path: Path) -> None:
    manifest, unchanged, changed = _run(tmp_path, _make_inputs(tmp_path))

    assert manifest["counts"] == {
        "total": 2,
        "unchanged": 1,
        "changed": 1,
        "new_blind_review_required": 1,
        "old_adjudicated_reviews_reused": 1,
    }
    assert [row["sample_id"] for row in unchanged] == ["s1"]
    assert unchanged[0]["review_route"] == "unchanged"
    assert unchanged[0]["review_route_reasons"] == ["valid_nonempty_reference_equal_after_normalization"]
    assert unchanged[0]["reused_review"]["adjudicated"]["case_id"] == "case-a"
    assert unchanged[0]["reused_review"]["adjudicated"]["student_target_selection_error"] == "none"

    assert [row["sample_id"] for row in changed] == ["s2"]
    assert changed[0]["review_status"] == "needs_new_blind_review"
    assert changed[0]["new_blind_review_required"] is True
    assert changed[0]["new_blind_review_label"] is None
    assert changed[0]["review_route_reasons"] == ["target_text_changed"]
    assert changed[0]["target_entity_reference_comparison"] == "different_after_normalization"
    assert "reused_review" not in changed[0]
    assert "v3p5_target_entity_reference" not in changed[0]

    assert manifest["metadata"]["reference_equal_after_normalization"] == 1
    assert manifest["metadata"]["reference_different_after_normalization"] == 1
    assert len(manifest["input_hashes"]) == 1 + 2 + 2 * 4


def test_reference_normalization_does_not_infer_semantic_equivalence() -> None:
    assert normalize_target_entity_reference("  The\tCup.。 ") == "the cup"
    assert normalize_target_entity_reference("the cup") == "the cup"
    assert normalize_target_entity_reference("a cup") != "the cup"
    assert normalize_target_entity_reference("cup on plate") != "cup beside plate"
    assert normalize_target_entity_reference("cups") != "cup"


@pytest.mark.parametrize(
    "field",
    ["image_id", "image_sha256", "question", "reasoning_token_ids_sha256", "task_type", "split"],
)
def test_metadata_mismatch_fails_closed_without_outputs(tmp_path: Path, field: str) -> None:
    inputs = _make_inputs(tmp_path)
    v3p5_first = inputs["v3p5"][0]
    row = json.loads(v3p5_first.read_text().splitlines()[0])
    row[field] = row[field] + "-changed"
    _write_jsonl(v3p5_first, [row])

    outputs = [tmp_path / "unchanged.jsonl", tmp_path / "changed.jsonl", tmp_path / "manifest.json"]
    with pytest.raises(ReviewDeltaError, match="metadata mismatch"):
        prepare_bridge_review_delta(
            inputs["v6"], inputs["v3p5"], inputs["mappings"], inputs["reviewer_a"],
            inputs["reviewer_b"], inputs["adjudicated"], *outputs,
        )
    assert all(not path.exists() for path in outputs)


@pytest.mark.parametrize("source", ["v6", "v3p5"])
def test_equal_text_with_invalid_parse_needs_new_review(tmp_path: Path, source: str) -> None:
    inputs = _make_inputs(tmp_path)
    path = inputs["v6"] if source == "v6" else inputs["v3p5"][0]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["bridge_parse_status"] = "format_or_incomplete"
    _write_jsonl(path, rows)

    manifest, unchanged, changed = _run(tmp_path, inputs)
    assert unchanged == []
    assert {row["sample_id"] for row in changed} == {"s1", "s2"}
    invalid = next(row for row in changed if row["sample_id"] == "s1")
    assert invalid["review_status"] == "needs_new_blind_review"
    assert invalid["review_route_reasons"] == [f"{source}_bridge_parse_not_valid"]
    assert invalid["target_entity_reference_comparison"] == "equal_after_normalization"
    assert "reused_review" not in invalid
    assert manifest["metadata"]["reference_equal_after_normalization"] == 1
    assert manifest["metadata"]["reference_different_after_normalization"] == 1


@pytest.mark.parametrize("target", ["", "  .。  "])
def test_empty_normalized_targets_never_reuse_labels(tmp_path: Path, target: str) -> None:
    inputs = _make_inputs(tmp_path)
    for path in (inputs["v6"], inputs["v3p5"][0]):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["target_entity_reference"] = target
        _write_jsonl(path, rows)

    _, unchanged, changed = _run(tmp_path, inputs)
    assert unchanged == []
    empty = next(row for row in changed if row["sample_id"] == "s1")
    assert empty["review_route_reasons"] == ["v6_empty_target", "v3p5_empty_target"]
    assert empty["new_blind_review_required"] is True
    assert "reused_review" not in empty


def test_invalid_parse_and_text_change_report_both_reasons(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    rows = [json.loads(line) for line in inputs["v6"].read_text().splitlines()]
    rows[0]["bridge_parse_status"] = "invalid"
    rows[0]["target_entity_reference"] = "the bowl"
    _write_jsonl(inputs["v6"], rows)

    _, _, changed = _run(tmp_path, inputs)
    row = next(row for row in changed if row["sample_id"] == "s1")
    assert row["review_route_reasons"] == ["v6_bridge_parse_not_valid", "target_text_changed"]
    assert row["target_entity_reference_comparison"] == "different_after_normalization"


def test_review_coverage_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    _write_jsonl(inputs["reviewer_b"][1], [{"case_id": "wrong", "label": "same"}])
    outputs = [tmp_path / "unchanged.jsonl", tmp_path / "changed.jsonl", tmp_path / "manifest.json"]

    with pytest.raises(ReviewDeltaError, match="case coverage mismatch"):
        prepare_bridge_review_delta(
            inputs["v6"], inputs["v3p5"], inputs["mappings"], inputs["reviewer_a"],
            inputs["reviewer_b"], inputs["adjudicated"], *outputs,
        )
    assert all(not path.exists() for path in outputs)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    unchanged = tmp_path / "unchanged.jsonl"
    unchanged.write_text("sentinel\n", encoding="utf-8")
    changed = tmp_path / "changed.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_bridge_review_delta(
            inputs["v6"], inputs["v3p5"], inputs["mappings"], inputs["reviewer_a"],
            inputs["reviewer_b"], inputs["adjudicated"], unchanged, changed, manifest,
        )
    assert unchanged.read_text(encoding="utf-8") == "sentinel\n"
    assert not changed.exists()
    assert not manifest.exists()
