from __future__ import annotations

import json

from mmgcot_timeline_training.analyze_bridge_v2_diagnostic import analyze


def test_geometry_gate_uses_fixed_prefix_r_minus_l(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"sample_id": "s", "image_id": "i"}) + "\n")
    records = tmp_path / "records"
    records.mkdir()
    rows = []
    for arm, value in (("L0", .1), ("L", .2), ("E", .25), ("R", .3)):
        for draw in range(4):
            rows.append({"type": "bbox", "sample_id": "s", "stage": "A", "mode": "random",
                         "prefix_coordinates": None, "arm": arm, "draw": draw, "iou": value})
        rows.append({"type": "bbox", "sample_id": "s", "stage": "A", "mode": "greedy",
                     "prefix_coordinates": None, "arm": arm, "draw": 0, "iou": value})
    for prefix in (1, 2):
        for arm, value in (("L", .2), ("E", .25), ("R", .3)):
            for draw in range(4):
                rows.append({"type": "bbox", "sample_id": "s", "stage": "B", "mode": "random",
                             "prefix_coordinates": prefix, "arm": arm, "draw": draw, "iou": value})
    (records / "s.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = analyze(manifest, records)
    assert summary["opd_calibration_gate_pass"] is True
    assert summary["panels"]["B_prefix1"]["contrasts"]["R-L"]["mean"] > 0
