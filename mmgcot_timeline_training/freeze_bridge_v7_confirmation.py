"""Freeze untouched train/dev images for v7 target-bridge confirmation.

This selection reads metadata only.  The independent 48 are exclusions and
remain unevaluated.  The output is immutable and records every input hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mmgcot_diagnostic.protocol import file_hash


SEED = 20260923
QUOTA = 5


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def freeze(train: Path, dev: Path, exclusions: list[Path], output: Path) -> None:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    old = [row for path in exclusions for row in _read(path)]
    excluded_ids = {row["sample_id"] for row in old}
    excluded_images = {str(row["image_id"]) for row in old}
    excluded_hashes = {row["image_sha256"] for row in old}
    selected: list[dict] = []
    for split, source in (("train", train), ("dev", dev)):
        for task in ("attribute", "object"):
            eligible = [row for row in _read(source)
                        if row["task_type"] == task
                        and row["sample_id"] not in excluded_ids
                        and str(row["image_id"]) not in excluded_images
                        and row["image_sha256"] not in excluded_hashes]
            eligible.sort(key=lambda row: hashlib.sha256(
                f"{SEED}|bridge-v7-fresh-confirmation|{row['sample_id']}".encode()
            ).digest())
            chosen = []
            for row in eligible:
                if str(row["image_id"]) in excluded_images or row["image_sha256"] in excluded_hashes:
                    continue
                chosen.append(row)
                excluded_images.add(str(row["image_id"]))
                excluded_hashes.add(row["image_sha256"])
                if len(chosen) == QUOTA:
                    break
            if len(chosen) != QUOTA:
                raise RuntimeError(f"insufficient unused images in {split}/{task}")
            selected.extend(chosen)
    with output.open("x", encoding="utf-8") as stream:
        for row in sorted(selected, key=lambda row: row["sample_id"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "mmgcot_bridge_v7_fresh_confirmation_selection",
        "seed": SEED,
        "quota_per_split_task": QUOTA,
        "rows": len(selected),
        "selection_sha256": file_hash(output),
        "train_sha256": file_hash(train),
        "dev_sha256": file_hash(dev),
        "exclusions": [{"path": str(path.absolute()), "sha256": file_hash(path)}
                       for path in exclusions],
        "independent_48_used_only_for_exclusion": True,
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
