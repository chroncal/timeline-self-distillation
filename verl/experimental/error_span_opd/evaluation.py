"""Summarize fixed validation populations without mixing their denominators."""
from __future__ import annotations

import hashlib
import json
import math


def summarize_validation(rows, expected_ids):
    unique = {}
    for row in rows:
        key = (row["sample_id"], row["rollout_index"])
        if key in unique:
            raise ValueError("duplicate validation ID; frozen population needs no framework padding")
        unique[key] = row
    expected_all = {sample for ids in expected_ids.values() for sample in ids}
    if len(expected_all) != sum(len(ids) for ids in expected_ids.values()):
        raise ValueError("validation populations overlap or contain duplicate IDs")
    if set(unique) != {(sample, 0) for sample in expected_all}:
        raise ValueError("validation population differs from its frozen manifest")
    result = {}
    for name, ids in expected_ids.items():
        if not ids:
            continue
        selected = [unique[(sample, 0)] for sample in ids]
        for row in selected:
            if not math.isfinite(row["iou"]) or not 0 <= row["iou"] <= 1:
                raise ValueError("non-finite or out-of-range validation IoU")
            if not row["parse_valid"] and (row["iou"] != 0 or row["hit"]):
                raise ValueError("invalid validation output must remain and score zero")
        result[name] = dict(
            samples=len(selected),
            accuracy=sum(row["hit"] for row in selected) / len(selected),
            mean_iou=sum(row["iou"] for row in selected) / len(selected),
            invalid_rate=sum(not row["parse_valid"] for row in selected) / len(selected),
            sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        )
    return result
