"""CPU tests for MM-GCoT qualification ordering, review gates, and quotas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmgcot_diagnostic.qualification import (
    DEFAULT_SEED,
    EligibilitySource,
    FinalizeResult,
    QualificationError,
    _load_reviewed_source,
    _ReviewedCandidate,
    _select_final_split,
    _select_raw_reserve,
    finalize,
    read_jsonl,
    stable_order_rows,
)


def _source(name: str, split: str = "formal") -> EligibilitySource:
    return EligibilitySource(
        name=name,
        split=split,
        role="primary" if name.startswith("primary") else "reserve",
        manifest=Path(f"/{name}.jsonl"),
        private_key=Path(f"/{name}_key.jsonl"),
        review=Path(f"/{name}_review.jsonl"),
    )


def _candidate(
    sample_id: str,
    image_id: str,
    task_type: str,
    *,
    split: str = "formal",
    status: str = "eligible",
    rank: str | None = None,
    source_name: str = "primary_formal",
) -> _ReviewedCandidate:
    row = {
        "sample_id": sample_id,
        "image_id": image_id,
        "task_type": task_type,
        "source_id": sample_id,
        "source_row_index": 0,
    }
    return _ReviewedCandidate(
        source=_source(source_name, split),
        row=row,
        case_id=f"case-{sample_id}",
        status=status,
        review_reason="test",
        stable_rank=rank or sample_id,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stable_order_is_seeded_and_independent_of_input_order() -> None:
    rows = [
        {"sample_id": "s-a", "image_id": "a", "source_id": "a", "source_row_index": 0},
        {"sample_id": "s-b", "image_id": "b", "source_id": "b", "source_row_index": 1},
        {"sample_id": "s-c", "image_id": "c", "source_id": "c", "source_row_index": 2},
    ]
    first = stable_order_rows(rows, split="formal", task_type="attribute", seed=DEFAULT_SEED)
    second = stable_order_rows(list(reversed(rows)), split="formal", task_type="attribute", seed=DEFAULT_SEED)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert first != stable_order_rows(rows, split="formal", task_type="attribute", seed=DEFAULT_SEED + 1)


def test_final_selection_excludes_duplicate_images() -> None:
    candidates = [
        _candidate("a-shared", "shared", "attribute", rank="01"),
        _candidate("o-shared", "shared", "object", rank="02"),
        _candidate("o-unique", "unique", "object", rank="03"),
    ]
    selected, counts, fallbacks = _select_final_split(candidates, split="formal", per_task=1, seed=DEFAULT_SEED)
    assert [row.row["sample_id"] for row in selected] == ["a-shared", "o-unique"]
    assert {row.row["image_id"] for row in selected} == {"shared", "unique"}
    assert counts == {"attribute": 1, "object": 1}
    assert fallbacks == []


def test_reserve_selection_excludes_primary_and_formal_reserve_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeImage:
        def __init__(self, image_id: str) -> None:
            self.image_id = image_id
            self.image_path = str(tmp_path / f"{image_id}.jpg")
            self.sha256 = "0" * 64

    monkeypatch.setattr(
        "mmgcot_diagnostic.qualification._data.materialize_image",
        lambda **kwargs: FakeImage(kwargs["image_id"]),
    )
    candidate = lambda image_id, source_id, task: {
        "image_id": image_id,
        "source_id": source_id,
        "task_type": task,
        "image_ref": f"VG_100K/{image_id}.jpg",
        "source_row_index": 0,
        "source_occurrence": 1,
        "source_file": "/raw.json",
        "question": "q",
        "ground_truth_bbox": [0.1, 0.1, 0.2, 0.2],
        "reference_cot": "c",
        "reference_answer": "a",
    }
    rows = {
        "attribute": [candidate("primary", "p", "attribute"), candidate("a-only", "a", "attribute")],
        "object": [candidate("formal", "f", "object"), candidate("o-only", "o", "object")],
    }
    selected, counts, exclusions = _select_raw_reserve(
        split="pilot",
        candidates_by_task=rows,
        target_per_task=1,
        seed=DEFAULT_SEED,
        excluded_image_ids={"primary", "formal"},
        image_dir=tmp_path,
        local_index={},
    )
    assert {row["image_id"] for row in selected} == {"a-only", "o-only"}
    assert counts == {"attribute": 1, "object": 1}
    assert {row["image_id"] for row in exclusions} <= {"primary", "formal"}


def test_only_eligible_rows_can_be_selected() -> None:
    candidates = [
        _candidate("a-invalid", "a-invalid", "attribute", status="invalid", rank="01"),
        _candidate("a-eligible", "a-eligible", "attribute", status="eligible", rank="02"),
        _candidate("o-eligible", "o-eligible", "object", status="eligible", rank="03"),
    ]
    selected, _, _ = _select_final_split(candidates, split="formal", per_task=1, seed=DEFAULT_SEED)
    assert {row.row["sample_id"] for row in selected} == {"a-eligible", "o-eligible"}
    assert candidates[0].decision == "review_status_not_eligible"


def test_short_task_can_be_filled_by_other_task_and_is_recorded() -> None:
    candidates = [
        _candidate("a-only", "a-only", "attribute", rank="01"),
        _candidate("o-1", "o-1", "object", rank="02"),
        _candidate("o-2", "o-2", "object", rank="03"),
        _candidate("o-3", "o-3", "object", rank="04"),
    ]
    selected, counts, fallbacks = _select_final_split(candidates, split="formal", per_task=2, seed=DEFAULT_SEED)
    assert len(selected) == 4
    assert counts == {"attribute": 1, "object": 3}
    assert fallbacks == [
        {
            "sample_id": "o-3",
            "image_id": "o-3",
            "selected_task_type": "object",
            "for_short_task_type": "attribute",
            "reason": "quota_fallback_other_task_type",
        }
    ]


def test_shortfall_is_fail_closed() -> None:
    candidates = [
        _candidate("a-only", "a-only", "attribute"),
        _candidate("o-only", "o-only", "object"),
    ]
    with pytest.raises(QualificationError, match="formal final quota unmet"):
        _select_final_split(candidates, split="formal", per_task=2, seed=DEFAULT_SEED)


def test_private_key_controls_case_to_sample_mapping_and_review_coverage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    key = tmp_path / "private_key.jsonl"
    review = tmp_path / "review.jsonl"
    rows = [
        {"sample_id": "s1", "image_id": "i1", "task_type": "attribute"},
        {"sample_id": "s2", "image_id": "i2", "task_type": "object"},
    ]
    _write_jsonl(manifest, rows)
    _write_jsonl(key, [{"case_id": "c1", "sample_id": "s1"}, {"case_id": "c2", "sample_id": "s2"}])
    _write_jsonl(review, [{"case_id": "c1", "status": "eligible"}, {"case_id": "c2", "status": "invalid"}])
    source = EligibilitySource("primary_formal", "formal", "primary", manifest, key, review)
    loaded = _load_reviewed_source(source)
    assert [(row.case_id, row.row["sample_id"], row.status) for row in loaded] == [
        ("c1", "s1", "eligible"),
        ("c2", "s2", "invalid"),
    ]

    _write_jsonl(review, [{"case_id": "c1", "status": "eligible"}, {"case_id": "c2", "status": "pending"}])
    with pytest.raises(QualificationError, match="unknown eligibility status"):
        _load_reviewed_source(source)

    _write_jsonl(review, [{"case_id": "c1", "status": "eligible"}])
    with pytest.raises(QualificationError, match="review coverage mismatch"):
        _load_reviewed_source(source)


def test_finalize_writes_new_frozen_artifacts_and_formal_image_exclusion(tmp_path: Path) -> None:
    def make_source(name: str, rows: list[dict]) -> tuple[Path, Path, Path]:
        manifest = tmp_path / f"{name}.jsonl"
        key = tmp_path / f"{name}.key.jsonl"
        review = tmp_path / f"{name}.review.jsonl"
        _write_jsonl(manifest, rows)
        _write_jsonl(
            key,
            [{"case_id": f"{name}-case-{index}", "sample_id": row["sample_id"]} for index, row in enumerate(rows)],
        )
        _write_jsonl(
            review,
            [{"case_id": f"{name}-case-{index}", "status": "eligible"} for index, row in enumerate(rows)],
        )
        return manifest, key, review

    primary_pilot = make_source(
        "primary-pilot",
        [
            {
                "sample_id": "pilot-shared",
                "image_id": "formal-shared",
                "task_type": "attribute",
                "source_id": "p1",
                "source_row_index": 0,
            },
            {
                "sample_id": "pilot-object",
                "image_id": "pilot-object",
                "task_type": "object",
                "source_id": "p2",
                "source_row_index": 1,
            },
        ],
    )
    primary_formal = make_source(
        "primary-formal",
        [
            {
                "sample_id": "formal-attribute",
                "image_id": "formal-shared",
                "task_type": "attribute",
                "source_id": "f1",
                "source_row_index": 0,
            }
        ],
    )
    reserve_pilot = make_source(
        "reserve-pilot",
        [
            {
                "sample_id": "reserve-attribute",
                "image_id": "pilot-attribute",
                "task_type": "attribute",
                "source_id": "rp1",
                "source_row_index": 0,
            }
        ],
    )
    reserve_formal = make_source(
        "reserve-formal",
        [
            {
                "sample_id": "reserve-object",
                "image_id": "formal-object",
                "task_type": "object",
                "source_id": "rf1",
                "source_row_index": 0,
            }
        ],
    )
    output = tmp_path / "frozen"
    result = finalize(
        primary_pilot=primary_pilot[0],
        primary_formal=primary_formal[0],
        reserve_pilot=reserve_pilot[0],
        reserve_formal=reserve_formal[0],
        primary_pilot_key=primary_pilot[1],
        primary_formal_key=primary_formal[1],
        reserve_pilot_key=reserve_pilot[1],
        reserve_formal_key=reserve_formal[1],
        primary_pilot_review=primary_pilot[2],
        primary_formal_review=primary_formal[2],
        reserve_pilot_review=reserve_pilot[2],
        reserve_formal_review=reserve_formal[2],
        output_dir=output,
        pilot_per_task=1,
        formal_per_task=1,
    )
    assert isinstance(result, FinalizeResult)
    formal_rows = read_jsonl(result.formal_path)
    pilot_rows = read_jsonl(result.pilot_path)
    assert {row["image_id"] for row in formal_rows} == {"formal-shared", "formal-object"}
    assert {row["image_id"] for row in pilot_rows} == {"pilot-attribute", "pilot-object"}
    assert not ({row["image_id"] for row in formal_rows} & {row["image_id"] for row in pilot_rows})
    mapping = read_jsonl(result.review_mapping_path)
    shared = next(row for row in mapping if row["sample_id"] == "pilot-shared")
    assert shared["case_id"] == "primary-pilot-case-0"
    assert shared["selected"] is False
    assert shared["decision"] == "excluded_formal_image"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["counts"]["selected_total"] == 4
    assert result.hashes_path.is_file()
