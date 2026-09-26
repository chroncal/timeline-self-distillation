"""The sealed confirmation cohort is unreachable before checkpoint freeze."""

from __future__ import annotations

import json

import pytest

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training import formal_final_eval_v2 as final


def test_checkpoint_freeze_is_required_and_integrity_checked(tmp_path) -> None:
    config = {"training": {"formal_seeds": [1, 2, 3]},
              "arms": ["bbox_sft", "r_opd", "e_opd"]}
    with pytest.raises(FileNotFoundError):
        final._frozen_checkpoints(tmp_path, config)
    formal = tmp_path / "formal"
    formal.mkdir()
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    selection = calibration / "selection.json"
    selection.write_text("{}")
    sources = {"mmgcot_timeline_training/formal_train_v2.py": file_hash(
        final.ROOT / "mmgcot_timeline_training/formal_train_v2.py")}
    checkpoint_manifest = {}
    for seed in config["training"]["formal_seeds"]:
        for arm in config["arms"]:
            label = f"{arm}_seed{seed}"
            directory = formal / label
            directory.mkdir()
            checkpoint = directory / "step_0200.pt"
            checkpoint.write_bytes(label.encode())
            (directory / "config.json").write_text(json.dumps({
                "reproducibility": {"source_sha256": sources}}))
            checkpoint_manifest[label] = {"path": str(checkpoint),
                                          "sha256": file_hash(checkpoint)}
    (formal / "frozen_checkpoints.json").write_text(json.dumps({
        "selection_sha256": file_hash(selection),
        "checkpoints": checkpoint_manifest}))
    actual = final._frozen_checkpoints(tmp_path, config)
    assert len(actual) == 9
    (formal / "r_opd_seed2" / "step_0200.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        final._frozen_checkpoints(tmp_path, config)
