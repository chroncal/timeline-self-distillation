"""Read-only B-mask audit on all three existing pure-E training runs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from mmgcot_diagnostic.protocol import MODEL, file_hash
from mmgcot_timeline_training.coordinate_decision import annotate


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/formal"
OUTPUT = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/mask_audit.json"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    counts = Counter()
    inputs = {}
    for seed in (20260921, 20260922, 20260923):
        path = OLD / f"e_opd_pure_seed{seed}/rollouts.jsonl"
        inputs[str(path)] = file_hash(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ann = annotate(row["token_ids"], row["support_ids"], tokenizer)
            if ann["numeric_mask"] != row["numeric_mask"]:
                raise RuntimeError(f"old numeric mask differs at seed {seed}, step {row['step']}")
            counts["rollouts"] += 1
            counts["old_numeric_positions"] += sum(row["numeric_mask"])
            counts["new_decision_positions"] += sum(ann["coordinate_decision_mask"])
            missed = sum(d and not n for d, n in zip(
                ann["coordinate_decision_mask"], row["numeric_mask"], strict=True))
            counts["new_end_decision_positions"] += missed
            counts["rollouts_with_new_end_decision"] += int(missed > 0)
            counts.update(ann["decision_type"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"counts": counts, "inputs_sha256": inputs}, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
