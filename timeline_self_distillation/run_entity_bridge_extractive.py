"""Adapter entry point for the source-grounded ``extractive_v3`` bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from timeline_self_distillation import run_entity_bridge_comparison as comparison

from timeline_self_distillation.extractive_entity_bridge import generate_extractive_entity_bridge


BRIDGE_VERSION = "extractive_v3"


def _reasoning_by_expression(path: Path) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = list(comparison.DEFAULT_SAMPLE_IDS)
    if [str(row.get("sample_id")) for row in rows] != expected:
        raise ValueError(f"pilot records must contain fixed IDs in order {expected}")
    result: dict[str, str] = {}
    for row in rows:
        expression = str(row["expression"])
        if expression in result:
            raise ValueError(f"duplicate expression cannot select reasoning text: {expression!r}")
        result[expression] = str(row["reasoning_text"])
    if len(result) != len(rows):
        raise AssertionError("expression-to-reasoning mapping is not one-to-one")
    return result


def make_bridge_wrapper(pilot_records: Path):
    """Return the old callback-shaped wrapper bound to frozen source text."""

    reasoning_by_expression = _reasoning_by_expression(pilot_records)

    def wrapper(
        model: Any,
        tokenizer: Any,
        entity_grammar: Any,
        cache: Any,
        expression: str,
        seed: int,
        *,
        version: str = BRIDGE_VERSION,
    ) -> dict[str, Any]:
        if version != BRIDGE_VERSION:
            raise ValueError(f"unexpected bridge version {version!r}")
        try:
            reasoning_text = reasoning_by_expression[str(expression)]
        except KeyError as error:
            raise KeyError(f"expression is absent from fixed pilot records: {expression!r}") from error
        del entity_grammar
        return generate_extractive_entity_bridge(
            model,
            tokenizer,
            cache,
            str(expression),
            reasoning_text,
            int(seed),
        )

    return wrapper


def run(args):
    wrapper = make_bridge_wrapper(args.pilot_records)
    return comparison.run(args, bridge_generator=wrapper, bridge_version=BRIDGE_VERSION)


def parse_args():
    return comparison.parse_args()


if __name__ == "__main__":
    run(parse_args())
