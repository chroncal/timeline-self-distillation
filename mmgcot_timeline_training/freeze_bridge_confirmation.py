"""Freeze untouched train/dev cases for target-bridge v2 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmgcot_diagnostic.protocol import file_hash


SEED = 20260922
QUOTAS = {("train", "attribute"): 5, ("train", "object"): 5,
          ("dev", "attribute"): 5, ("dev", "object"): 5}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> bytes:
    value = f"{SEED}|bridge-v2-confirmation|{row['sample_id']}"
    return hashlib.sha256(value.encode()).digest()


def freeze(train: Path, dev: Path, prior_review: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    reviewed = {row["sample_id"] for row in _read(prior_review)}
    selected: list[dict[str, Any]] = []
    for split, source in (("train", train), ("dev", dev)):
        rows = _read(source)
        for task in ("attribute", "object"):
            eligible = [row for row in rows
                        if row["task_type"] == task and row["sample_id"] not in reviewed]
            chosen = sorted(eligible, key=_key)[:QUOTAS[(split, task)]]
            if len(chosen) != QUOTAS[(split, task)]:
                raise RuntimeError(f"insufficient untouched rows for {split}/{task}")
            selected.extend(chosen)
    if len({row["image_id"] for row in selected}) != 20:
        raise RuntimeError("confirmation selection reused an image")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "mmgcot_target_bridge_v2_confirmation_selection",
        "seed": SEED,
        "selection_rule": "SHA256 rank after excluding the original frozen 50",
        "quotas": {f"{s}/{t}": n for (s, t), n in QUOTAS.items()},
        "rows": len(selected),
        "output_sha256": file_hash(output),
        "prior_review_sha256": file_hash(prior_review),
        "train_sha256": file_hash(train),
        "dev_sha256": file_hash(dev),
        "independent_confirmation_accessed": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--prior-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.train, args.dev, args.prior_review, args.output)


if __name__ == "__main__":
    main()
