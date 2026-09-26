#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert the frozen pilot JSONL into verl RL dataset parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

GROUNDING_INSTRUCTION = (
    "<image>\nLocate the object described by this referring expression: {expression}\n"
    "Reason naturally, then return exactly "
    '<think>your reasoning</think><answer>{{"bbox":[x1,y1,x2,y2]}}</answer>. '
    "The bbox is xyxy in the original image coordinate system normalized to [0,1000]."
)


def normalize_xywh(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = (float(value) for value in bbox)
    return [1000.0 * x / width, 1000.0 * y / height, 1000.0 * (x + w) / width, 1000.0 * (y + h) / height]


def convert_record(record: dict, index: int) -> dict:
    width, height = int(record["image_width"]), int(record["image_height"])
    target_bbox_normalized = normalize_xywh(record["target_bbox"], width, height)
    extra_info = {
        "index": index,
        "split": record["split"],
        "image_id": record["image_id"],
        "expression": record["expression"],
        "target_ann_id": record["target_ann_id"],
        "image_width": width,
        "image_height": height,
        "target_bbox_normalized": target_bbox_normalized,
        # COCO segmentation alternates between polygon lists and RLE mappings;
        # serializing the full receipt avoids an invalid heterogeneous Arrow
        # nested column while preserving every field losslessly for the loop.
        "routing_record_json": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    }
    return {
        "data_source": "refcocog_umd",
        "prompt": [
            {
                "role": "user",
                "content": GROUNDING_INSTRUCTION.format(expression=record["expression"]),
            }
        ],
        "images": [record["image_path"]],
        "ability": "visual_grounding",
        "reward_model": {"style": "rule", "ground_truth": target_bbox_normalized},
        "extra_info": extra_info,
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(
    manifest_dir: Path,
    output_dir: Path,
    *,
    smoke_size: int = 32,
    seed: int = 260600564,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        records = read_jsonl(manifest_dir / f"{split}.jsonl")
        converted = [convert_record(record, index) for index, record in enumerate(records)]
        pd.DataFrame(converted).to_parquet(output_dir / f"{split}.parquet", index=False)
        if split == "train":
            if smoke_size <= 0 or smoke_size > len(converted):
                raise ValueError("smoke_size must be between 1 and the train manifest size")
            indices = sorted(random.Random(seed).sample(range(len(converted)), smoke_size))
            pd.DataFrame([converted[index] for index in indices]).to_parquet(
                output_dir / "smoke_train.parquet",
                index=False,
            )
            source = manifest_dir / "train.jsonl"
            (output_dir / "smoke_subset.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "size": smoke_size,
                        "train_manifest": str(source.resolve()),
                        "train_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "indices": indices,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260600564)
    args = parser.parse_args()
    prepare(
        args.manifest_dir.resolve(),
        args.output_dir.resolve(),
        smoke_size=args.smoke_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
