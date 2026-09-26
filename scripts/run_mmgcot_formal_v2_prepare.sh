#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/sujingyang/research/routed-grounding-repair-verl
export PYTHONNOUSERSITE=1
PY=.venv/bin/python
DATA=/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1
OUT=outputs/research_experiments/mmgcot_timeline_training/formal_v2_prepared
mkdir -p "$OUT/logs"

# Four independent model processes; each owns a deterministic modulo shard.
pids=()
for shard in 0 1 2 3; do
    device=$((shard + 4))
    (
        if [[ ! -f "$OUT/train/shard${shard}.receipt.json" ]]; then
            "$PY" -m mmgcot_timeline_training.prepare_v2 \
                --selection "$DATA/train_frozen.jsonl" \
                --output-dir "$OUT/train" --device "$device" \
                --shard-index "$shard" --num-shards 4 --trajectories 1 --resume
        fi
        if [[ ! -f "$OUT/dev/shard${shard}.receipt.json" ]]; then
            "$PY" -m mmgcot_timeline_training.prepare_v2 \
                --selection "$DATA/dev_frozen.jsonl" \
                --output-dir "$OUT/dev" --device "$device" \
                --shard-index "$shard" --num-shards 4 --trajectories 1 --resume
        fi
    ) > "$OUT/logs/shard${shard}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
