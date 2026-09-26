from __future__ import annotations

import json

from mmgcot_timeline_training.freeze_bridge_v3_confirmation import freeze


def test_freeze_excludes_sample_image_and_hash(tmp_path) -> None:
    rows = []
    for split in ("train", "dev"):
        for task in ("attribute", "object"):
            for index in range(8):
                rows.append({
                    "sample_id": f"{split}:{task}:{index}",
                    "split": split,
                    "task_type": task,
                    "image_id": f"{split}:{task}:image:{index}",
                    "image_sha256": f"{split}:{task}:hash:{index}",
                })
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == "train"))
    dev.write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == "dev"))
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text(json.dumps(rows[0]) + "\n")
    output = tmp_path / "out.jsonl"
    freeze(train, dev, [excluded], output)
    chosen = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(chosen) == 20
    assert rows[0]["sample_id"] not in {row["sample_id"] for row in chosen}
    assert rows[0]["image_id"] not in {row["image_id"] for row in chosen}
    assert rows[0]["image_sha256"] not in {row["image_sha256"] for row in chosen}
