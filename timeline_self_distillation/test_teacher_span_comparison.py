"""CPU contracts for the frozen, cumulative span comparison."""

import pytest

from timeline_self_distillation import run_teacher_span_comparison as probe


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def row(text="First.\nSecond.\nThird.\nLast."):
    return {"reasoning_ids": list(map(ord, text)), "reasoning_text": text, "first_span_offset": 7}


def test_original_splitter_and_cumulative_prefixes_are_used():
    anchors = probe.span_anchors(CharacterTokenizer(), row())
    assert [anchors[c]["offset"] for c in probe.CONDITIONS] == [7, 15, 22, 27]
    assert anchors["span3"]["prefix_text"] == "First.\nSecond.\nThird.\n"
    assert anchors["span3"]["span_text"] == "Third.\n"


def test_no_silent_boundary_fallback():
    bad = row()
    bad["first_span_offset"] = 6
    with pytest.raises(ValueError, match="span1 differs"):
        probe.span_anchors(CharacterTokenizer(), bad)
    with pytest.raises(ValueError, match="fewer than three"):
        probe.span_anchors(CharacterTokenizer(), row("Only once."))


def test_capture_replays_full_reasoning_once_and_retains_all_earlier_tokens(monkeypatch):
    calls = []

    def advance(model, cache, ids):
        calls.extend(ids)
        cache.extend(ids)
        return cache, None

    monkeypatch.setattr(probe, "fork", lambda cache: cache.copy())
    monkeypatch.setattr(probe, "_advance", advance)
    original = [-1]
    ids = list(range(10))
    anchors = {c: {"offset": o} for c, o in zip(probe.CONDITIONS, (2, 4, 7, 10), strict=True)}
    saved = probe.capture_prefixes(None, original, ids, anchors)
    assert original == [-1]
    assert calls == ids
    assert saved["span2"] == [-1, 0, 1, 2, 3]
    assert saved["span3"] == [-1, *range(7)]
    assert saved["late"] == [-1, *ids]


def records():
    return [
        {
            "condition": c,
            "sample_id": s,
            "mode": mode,
            "draw": d,
            "seed": probe.micro_seed(probe.SEED, order, d),
            "iou": 0.5,
            "parse_valid": True,
            "bbox": [0, 0, 100, 100],
            "ids": [1, 2],
        }
        for c in probe.CONDITIONS
        for order, s in enumerate(probe.EXPECTED_SAMPLES)
        for mode, n in (("random", probe.DRAWS), ("greedy", 1))
        for d in range(n)
    ]


def test_summary_all_340_and_invalid_counted_zero():
    raw = records()
    raw[0].update(iou=0.0, parse_valid=False)
    result = probe.summarize(raw)
    assert result["total_outputs"] == 340
    assert result["conditions"]["span1"]["random"]["mean_iou"] == pytest.approx(39.5 / 80)
    assert result["conditions"]["span1"]["random"]["invalid_count"] == 1
    assert result["conditions"]["span2"]["random"]["count"] == 80
    assert result["conditions"]["span3"]["greedy"]["count"] == 5
    assert len(result["conditions"]["span3"]["random"]["per_sample"]) == 5


def test_summary_rejects_duplicates_and_nonfinite():
    raw = records()
    raw[0] = raw[1].copy()
    with pytest.raises(ValueError, match="distinct"):
        probe.summarize(raw)
    raw = records()
    raw[0]["iou"] = float("nan")
    with pytest.raises(FloatingPointError):
        probe.summarize(raw)


def test_late_parity_checks_all85_ids_boxes_and_scores():
    raw = records()
    reference = [dict(r, model="base") for r in raw if r["condition"] == "late"]
    assert probe.late_parity(raw, reference)["all_exact"]
    reference[0]["ids"] = [1, 3]
    assert not probe.late_parity(raw, reference)["all_exact"]
    with pytest.raises(ValueError, match="85"):
        probe.late_parity(raw, reference[:-1])
