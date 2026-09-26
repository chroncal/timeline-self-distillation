#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/sujingyang/research/routed-grounding-repair-verl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="$PWD/.venv/bin/python"
selection=/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1/train_frozen.jsonl
prepared="$PWD/outputs/research_experiments/mmgcot_timeline_training/formal_v2_prepared/train"
root="$PWD/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/smoke"
mkdir -p "$root"
trap 'status=$?; printf "SMOKE_FAILED exit=%s time=%s\n" "$status" "$(date -u +%FT%TZ)" > "$root/failed.txt"' ERR

# Existing independent48 evaluation owns GPUs 4-7. Never overlap it.
while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '5p' | tr -d ' ')"
    if ! tmux has-session -t mmgcot_v2_final_eval_0924 2>/dev/null && \
       [[ "$used" =~ ^[0-9]+$ ]] && (( used < 500 )); then
        break
    fi
    printf 'WAIT_GPU4 %s memory=%sMiB\n' "$(date -u +%FT%TZ)" "$used"
    sleep 30
done

"$python_bin" -m mmgcot_timeline_training.formal_pure_opd \
    --mode train --arm e_opd_pure --device 4 --seed 20260921 \
    --selection "$selection" --prepared "$prepared" --cache-root "$prepared/prefix_cache" \
    --output "$root/a_old" --steps 1 --batch 1 --lr 1e-4 --lambda-opd 0.1
"$python_bin" -m mmgcot_timeline_training.formal_pure_e_decision \
    --mode train --supervision-scope numeric --adapter-scope terminal_q --device 4 \
    --seed 20260921 --selection "$selection" --prepared "$prepared" \
    --cache-root "$prepared/prefix_cache" --output "$root/a_new" \
    --steps 1 --batch 1 --lr 1e-4 --lambda-opd 0.1
"$python_bin" - "$root" <<'PY'
import json, sys, torch
from pathlib import Path
root = Path(sys.argv[1])
old = json.loads((root/'a_old/rollouts.jsonl').read_text().splitlines()[0])
new = json.loads((root/'a_new/rollouts.jsonl').read_text().splitlines()[0])
for key in ('sample_id','seed','token_ids','support_ids','numeric_mask','teacher_support_logits','bbox','valid'):
    if old[key] != new[key]:
        raise RuntimeError(f'A compatibility mismatch: {key}')
old_weights = torch.load(root/'a_old/step_0001.pt', map_location='cpu', weights_only=False)
new_weights = torch.load(root/'a_new/step_0001.pt', map_location='cpu', weights_only=False)
for key in ('down','up'):
    torch.testing.assert_close(old_weights[key], new_weights['adapter_weights']['23.q'][key], atol=0, rtol=0)
(root/'a_compatible.json').write_text(json.dumps({'seed': 20260921,
    'sample_id': old['sample_id'], 'token_ids': old['token_ids'],
    'same_raw_tokens': True, 'same_supports': True, 'same_teacher_logits': True,
    'same_updated_weights': True}, indent=2) + '\n')
print('A_COMPATIBLE', flush=True)
PY

for adapter in terminal_q last_two_qvo; do
    "$python_bin" -m mmgcot_timeline_training.formal_pure_e_decision \
        --mode train --supervision-scope coordinate_decision \
        --adapter-scope "$adapter" --device 4 --seed 20260921 \
        --selection "$selection" --prepared "$prepared" \
        --cache-root "$prepared/prefix_cache" --output "$root/$adapter" \
        --steps 2 --batch 1 --lr 1e-4 --lambda-opd 0.1
done
"$python_bin" -m mmgcot_timeline_training.formal_pure_e_decision \
    --mode audit --supervision-scope coordinate_decision \
    --adapter-scope last_two_qvo --device 4 --seed 20260921 \
    --selection "$selection" --prepared "$prepared" \
    --cache-root "$prepared/prefix_cache" \
    --checkpoint "$root/last_two_qvo/step_0002.pt" \
    --output "$root/bf16_continuation_audit.json"
printf 'SMOKE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
