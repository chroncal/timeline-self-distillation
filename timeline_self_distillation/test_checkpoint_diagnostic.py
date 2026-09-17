"""CPU contract tests for the read-only checkpoint diagnostic helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import timeline_self_distillation.run_checkpoint_diagnostic as diagnostic


class _Tokenizer:
    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in ids)


def test_digit_rows_follow_exact_token_isdigit_and_keep_punctuation_out():
    tokenizer = _Tokenizer({1: "1", 2: ",", 3: "1,", 4: " 2", 5: "20"})

    rows = diagnostic.digit_token_positions(tokenizer, [1, 2, 3, 4, 5])

    assert [(row["position"], row["token_id"]) for row in rows] == [(0, 1), (4, 5)]


def test_canonical_bbox_rounds_clamps_and_uses_box_regex():
    bbox, tail = diagnostic.canonical_bbox_tail([1.49, -2.0, 1000.6, 999.6])

    assert bbox == [1, 0, 1000, 1000]
    assert tail == "1,0,1000,1000]}</answer>"


def test_token_distribution_metrics_matches_known_kl_and_is_finite():
    p = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    q = torch.log(torch.tensor([0.25, 0.75], dtype=torch.float64))

    result = diagnostic.token_distribution_metrics(p, q, chosen_index=1)

    expected_reverse = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    expected_forward = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
    assert float(result["reverse_kl"]) == pytest.approx(expected_reverse)
    assert float(result["forward_kl"]) == pytest.approx(expected_forward)
    assert float(result["chosen_token_nll"]) == pytest.approx(-math.log(0.5))
    assert all(torch.isfinite(value) for value in result.values())


def test_token_distribution_metrics_rejects_nonfinite_support_rows():
    with pytest.raises(FloatingPointError, match="must be finite"):
        diagnostic.token_distribution_metrics(
            torch.tensor([0.0, float("-inf")]),
            torch.log(torch.tensor([0.5, 0.5])),
            chosen_index=0,
        )


_ACTIVE_MATCHER = None


class _FakeMask:
    @property
    def allowed(self):
        return _ACTIVE_MATCHER.allowed

    @property
    def position(self):
        return _ACTIVE_MATCHER.position

    def to(self, _device):
        return self


class _FakeMatcher:
    def __init__(self, allowed):
        global _ACTIVE_MATCHER
        self.allowed = allowed
        self.position = 0
        _ACTIVE_MATCHER = self

    def fill_next_token_bitmask(self, _mask):
        return True

    def accept_token(self, token_id):
        if int(token_id) not in self.allowed[self.position]:
            return False
        self.position += 1
        return True

    def is_completed(self):
        return self.position == len(self.allowed)


class _FakeXgrammar:
    GrammarMatcher = staticmethod(lambda grammar, terminate_without_stop_token=True: _FakeMatcher(grammar))

    @staticmethod
    def allocate_token_bitmask(_batch, _vocab):
        return _FakeMask()

    @staticmethod
    def reset_token_bitmask(_mask):
        return None

    @staticmethod
    def apply_token_bitmask_inplace(scores, mask, *, vocab_size):
        allowed = mask.allowed[mask.position]
        for token_id in range(vocab_size):
            if token_id not in allowed:
                scores[0, token_id] = float("-inf")


def test_build_grammar_supports_records_support_before_each_accept(monkeypatch):
    monkeypatch.setattr(diagnostic.xgr, "GrammarMatcher", _FakeXgrammar.GrammarMatcher)
    monkeypatch.setattr(diagnostic.xgr, "allocate_token_bitmask", lambda _b, _v: _FakeMask())
    monkeypatch.setattr(diagnostic.xgr, "reset_token_bitmask", _FakeXgrammar.reset_token_bitmask)
    monkeypatch.setattr(diagnostic.xgr, "apply_token_bitmask_inplace", _FakeXgrammar.apply_token_bitmask_inplace)

    # The production helper passes the grammar object into GrammarMatcher and
    # the fake treats it as the per-prefix legal-token table.
    supports = diagnostic.build_grammar_supports([[1, 2], [3]], 4, [2, 3])

    assert supports == [[1, 2], [3]]


class _FakeModel:
    class _Inner:
        rope_deltas = torch.tensor([0])

    model = _Inner()


class _FakeAdapter:
    enabled = False


def test_score_fixed_sequence_consumes_opening_once_then_previous_targets(monkeypatch):
    consumed: list[int] = []

    def fake_advance(_model, cache, token_ids):
        consumed.extend(int(value) for value in token_ids)
        # Shape intentionally matches live _advance: [batch, vocab].
        return cache, torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    monkeypatch.setattr(diagnostic, "_advance", fake_advance)
    monkeypatch.setattr(diagnostic, "fork", lambda cache: dict(cache))
    state = {"student": {}, "last_opening_id": 9, "rope_deltas": torch.tensor([0])}

    result = diagnostic.score_fixed_sequence(
        _FakeModel(),
        _FakeAdapter(),
        state,
        [1, 2],
        [[1, 2], [2, 3]],
        adapter_enabled=False,
    )

    assert consumed == [9, 1]
    assert result["chosen_indices"] == [0, 0]
    assert len(result["log_probs"]) == 2


def test_read_eval_step_requires_four_draws_per_image_and_twenty_total(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    rows = []
    for sample_id in ("row-0", "row-1", "row-3", "row-4", "row-5"):
        for draw in range(4):
            rows.append({"sample_id": sample_id, "step": 0, "seed": 1000 + len(rows), "ids": [1, 2]})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    grouped = diagnostic.read_eval_step(path, 0, ("row-0", "row-1", "row-3", "row-4", "row-5"))

    assert sum(len(value) for value in grouped.values()) == 20
    assert all(len(value) == 4 for value in grouped.values())
