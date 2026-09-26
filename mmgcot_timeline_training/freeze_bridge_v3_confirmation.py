"""Freeze a fresh semantic-confirmation cohort for prompt-only bridge v3p2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmgcot_diagnostic.protocol import file_hash


SEED = 20260922
QUOTAS = {
    ("train", "attribute"): 5,
    ("train", "object"): 5,
    ("dev", "attribute"): 5,
    ("dev", "object"): 5,
}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rank(row: dict[str, Any]) -> bytes:
    return hashlib.sha256(
        f"{SEED}|bridge-v3p2-fresh-confirmation|{row['sample_id']}".encode()
    ).digest()


def freeze(train: Path, dev: Path, exclusions: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    excluded_rows = [row for path in exclusions for row in _read(path)]
    excluded_samples = {row["sample_id"] for row in excluded_rows}
    excluded_images = {str(row["image_id"]) for row in excluded_rows}
    excluded_hashes = {row["image_sha256"] for row in excluded_rows}
    selected: list[dict[str, Any]] = []
    selected_images: set[str] = set()
    selected_hashes: set[str] = set()
    for split, source in (("train", train), ("dev", dev)):
        rows = _read(source)
        for task in ("attribute", "object"):
            eligible = [
                row for row in rows
                if row["task_type"] == task
                and row["sample_id"] not in excluded_samples
                and str(row["image_id"]) not in excluded_images
                and row["image_sha256"] not in excluded_hashes
            ]
            chosen = []
            for row in sorted(eligible, key=_rank):
                image_id, image_hash = str(row["image_id"]), row["image_sha256"]
                if image_id in selected_images or image_hash in selected_hashes:
                    continue
                chosen.append(row)
                selected_images.add(image_id)
                selected_hashes.add(image_hash)
                if len(chosen) == QUOTAS[(split, task)]:
                    break
            if len(chosen) != QUOTAS[(split, task)]:
                raise RuntimeError(f"insufficient fresh rows for {split}/{task}")
            selected.extend(chosen)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "mmgcot_target_bridge_v3p2_fresh_confirmation_selection",
        "seed": SEED,
        "selection_rule": "SHA256 rank after excluding all prior semantic and independent cohorts",
        "quotas": {f"{s}/{t}": n for (s, t), n in QUOTAS.items()},
        "rows": len(selected),
        "output_sha256": file_hash(output),
        "train_sha256": file_hash(train),
        "dev_sha256": file_hash(dev),
        "exclusions": [
            {"path": str(path.absolute()), "sha256": file_hash(path)} for path in exclusions
        ],
        "independent_confirmation_evaluated": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.train, args.dev, args.exclude, args.output)


if __name__ == "__main__":
    main()
