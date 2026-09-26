#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/sujingyang/research/routed-grounding-repair-verl
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=.venv/bin/python
DATA=/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1
PREP=outputs/research_experiments/mmgcot_timeline_training/formal_v2_prepared
OUT=outputs/research_experiments/mmgcot_timeline_training/formal_v2_run
mkdir -p "$OUT/logs"

# The preparation may have started before this supervisor. Resume missing
# immutable records if its workers disappear; never run duplicate shards.
while :; do
    ready=1
    for cohort in train dev; do
        for shard in 0 1 2 3; do
            if [[ ! -f "$PREP/$cohort/shard${shard}.receipt.json" ]]; then
                ready=0
            fi
        done
    done
    if [[ "$ready" == 1 ]]; then
        break
    fi
    if ! pgrep -f 'mmgcot_timeline_training.prepare_v2.*formal_v2_prepared' >/dev/null; then
        echo "PREPARATION_RESTART $(date -u +%FT%TZ)" >&2
        bash scripts/run_mmgcot_formal_v2_prepare.sh
    else
        echo "PREPARATION_WAIT $(date -u +%FT%TZ)" >&2
        sleep 60
    fi
done

"$PY" -m mmgcot_timeline_training.formal_schedule_v2 \
    --stage preflight --prepared "$PREP" --output "$OUT"

PILOT="$OUT/pilot_dev20.jsonl"
if [[ ! -f "${PILOT%.jsonl}.summary.json" ]]; then
    if [[ -f "$PILOT" ]]; then
        echo "Pilot output exists without completion summary: $PILOT" >&2
        exit 1
    fi
    "$PY" -m mmgcot_timeline_training.formal_pilot_v2 \
        --selection "$DATA/dev_frozen.jsonl" --prepared "$PREP/dev" \
        --output "$PILOT" --images 20 --device 4 \
        > "$OUT/logs/pilot_dev20.log" 2>&1
fi

"$PY" -m mmgcot_timeline_training.formal_schedule_v2 \
    --stage all --prepared "$PREP" --output "$OUT" \
    --devices 4 5 6 7 \
    > "$OUT/logs/calibration_and_formal.log" 2>&1

"$PY" -m mmgcot_timeline_training.formal_final_eval_v2 \
    --formal-root "$OUT" --output "$OUT/final_evaluation" \
    --devices 4 5 6 7 \
    > "$OUT/logs/final_evaluation.log" 2>&1

echo "MMGCOT_FORMAL_V2_COMPLETE $(date -u +%FT%TZ)"
