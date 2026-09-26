#!/usr/bin/env bash
set -euo pipefail

# Read-only preflight for the frozen RefCOCOg routed-grounding pilot.
# Every tunable value is OPD_-prefixed; CUDA_VISIBLE_DEVICES is only inspected,
# never assigned a default by this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${OPD_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON_BIN=python3
fi
PYTHON_BIN="${OPD_PYTHON:-$DEFAULT_PYTHON_BIN}"

TRAIN_FILE="${OPD_TRAIN_FILE:-${REPO_ROOT}/data/refcocog_umd_pilot/train.parquet}"
VAL_FILE="${OPD_VAL_FILE:-${REPO_ROOT}/data/refcocog_umd_pilot/val.parquet}"
STUDENT_MODEL="${OPD_STUDENT_MODEL:-/mnt/sda/sujingyang/models/Qwen3.5-0.8B}"
TEACHER_MODEL="${OPD_TEACHER_MODEL:-/mnt/sda/sujingyang/models/Qwen3.5-4B}"
TRAIN_GPUS="${OPD_N_GPUS_PER_NODE:-4}"
TEACHER_GPUS="${OPD_TEACHER_N_GPUS_PER_NODE:-1}"
COLOCATE_WITH_ACTOR="${OPD_DISTILLATION_COLOCATE_WITH_ACTOR:-${OPD_COLOCATE_WITH_ACTOR:-false}}"
SINGLE_GPU="${OPD_SINGLE_GPU:-0}"
if [[ "$SINGLE_GPU" == "1" || "$SINGLE_GPU" == "true" || "$SINGLE_GPU" == "True" ]]; then
  TRAIN_GPUS=1
  TEACHER_GPUS=1
  COLOCATE_WITH_ACTOR=true
fi
WANDB_MODE_VALUE="${OPD_WANDB_MODE:-${WANDB_MODE:-online}}"
REQUIRE_WANDB="${OPD_REQUIRE_WANDB:-1}"
MAX_USED_GPU_MEMORY_MIB="${OPD_MAX_USED_GPU_MEMORY_MIB:-1024}"

failures=0

ok() {
  printf '[preflight] OK   %s\n' "$1"
}

warn() {
  printf '[preflight] WARN %s\n' "$1"
}

fail() {
  printf '[preflight] FAIL %s\n' "$1" >&2
  failures=$((failures + 1))
}

check_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    ok "file exists: ${path}"
  else
    fail "missing file: ${path}"
  fi
}

check_model() {
  local path="$1"
  local label="$2"
  local weight_file
  if [[ ! -d "$path" ]]; then
    fail "${label} model directory missing: ${path}"
    return
  fi
  if [[ ! -f "${path}/config.json" ]]; then
    fail "${label} model has no config.json: ${path}"
    return
  fi
  weight_file="$(find "$path" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.safetensors.index.json' \) -print -quit)"
  if [[ -n "$weight_file" ]]; then
    ok "${label} model metadata and weights: ${path}"
  else
    fail "${label} model has no local weight/index file: ${path}"
  fi
}

printf '[preflight] repo=%s\n' "$REPO_ROOT"
printf '[preflight] train_file=%s\n' "$TRAIN_FILE"
printf '[preflight] val_file=%s\n' "$VAL_FILE"
printf '[preflight] student_model=%s\n' "$STUDENT_MODEL"
printf '[preflight] teacher_model=%s\n' "$TEACHER_MODEL"
printf '[preflight] colocate_with_actor=%s\n' "$COLOCATE_WITH_ACTOR"
printf '[preflight] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset; no default GPU claim>}"

if [[ -d "$REPO_ROOT" ]]; then
  ok "repository directory: ${REPO_ROOT}"
else
  fail "repository directory missing: ${REPO_ROOT}"
fi

check_file "$TRAIN_FILE"
check_file "$VAL_FILE"
check_model "$STUDENT_MODEL" "Student"
check_model "$TEACHER_MODEL" "Teacher"

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  ok "Python executable: ${PYTHON_BIN}"
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("wandb") is not None else 1)
PY
  then
    ok "W&B Python package importable"
  elif [[ "$REQUIRE_WANDB" == "1" ]]; then
    fail "W&B Python package is not importable with ${PYTHON_BIN}"
  else
    warn "W&B Python package is not importable with ${PYTHON_BIN}"
  fi
else
  fail "Python executable is unavailable: ${PYTHON_BIN}"
fi

case "$WANDB_MODE_VALUE" in
  offline|disabled)
    warn "W&B mode=${WANDB_MODE_VALUE}; no online credential check requested"
    ;;
  *)
    if [[ -n "${WANDB_API_KEY:-}" || -f "${NETRC:-${HOME:-}/.netrc}" ]]; then
      ok "W&B online credential location detected (value not printed)"
      if [[ "$REQUIRE_WANDB" == "1" ]]; then
        if WANDB_SILENT=true timeout 20 "$PYTHON_BIN" -c 'import wandb; wandb.Api().viewer' >/dev/null 2>&1; then
          ok "W&B online credential verified"
        else
          fail "W&B online credential could not be verified; run .venv/bin/wandb login"
        fi
      fi
    else
      if [[ "$REQUIRE_WANDB" == "1" ]]; then
        fail "W&B mode=${WANDB_MODE_VALUE} but no API key/netrc was detected"
      else
        warn "W&B mode=${WANDB_MODE_VALUE} but no API key/netrc was detected"
      fi
    fi
    ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_report="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$gpu_report" ]]; then
    ok "nvidia-smi can enumerate visible physical GPUs"
    printf '%s\n' "$gpu_report"
    physical_gpu_count="$(printf '%s\n' "$gpu_report" | awk 'NF {count += 1} END {print count + 0}')"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
      visible_gpu_count="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{count=0; for (i=1; i<=NF; i++) if ($i != "" && $i != "-1") count += 1; print count}')"
    else
      visible_gpu_count="$physical_gpu_count"
    fi
    printf '[preflight] visible_gpu_count=%s requested_train=%s requested_teacher=%s\n' \
      "$visible_gpu_count" "$TRAIN_GPUS" "$TEACHER_GPUS"
    if [[ "$COLOCATE_WITH_ACTOR" == "1" || "$COLOCATE_WITH_ACTOR" == "true" || "$COLOCATE_WITH_ACTOR" == "True" ]]; then
      if (( TRAIN_GPUS > TEACHER_GPUS )); then
        required_gpu_count="$TRAIN_GPUS"
      else
        required_gpu_count="$TEACHER_GPUS"
      fi
    else
      required_gpu_count=$((TRAIN_GPUS + TEACHER_GPUS))
    fi
    if (( visible_gpu_count >= required_gpu_count )); then
      ok "visible GPU count satisfies configured resource pools (${required_gpu_count})"
    else
      fail "visible GPU count ${visible_gpu_count} is below configured resource pools ${required_gpu_count}"
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
      free_gpu_count=0
      IFS=',' read -r -a selected_devices <<<"$CUDA_VISIBLE_DEVICES"
      for selected in "${selected_devices[@]}"; do
        used_mib="$(printf '%s\n' "$gpu_report" | awk -F, -v idx="$selected" '$1 + 0 == idx + 0 {gsub(/[^0-9.]/, "", $4); print $4; exit}')"
        if [[ -n "$used_mib" ]] && awk -v used="$used_mib" -v limit="$MAX_USED_GPU_MEMORY_MIB" 'BEGIN {exit !(used <= limit)}'; then
          free_gpu_count=$((free_gpu_count + 1))
        fi
      done
    else
      free_gpu_count="$(printf '%s\n' "$gpu_report" | awk -F, -v limit="$MAX_USED_GPU_MEMORY_MIB" '{gsub(/[^0-9.]/, "", $4); if (($4 + 0) <= limit) count += 1} END {print count + 0}')"
    fi
    printf '[preflight] low-use_gpu_count=%s threshold_mib=%s required=%s\n' \
      "$free_gpu_count" "$MAX_USED_GPU_MEMORY_MIB" "$required_gpu_count"
    if (( free_gpu_count >= required_gpu_count )); then
      ok "low-use GPU count satisfies configured resource pools (${required_gpu_count})"
    else
      fail "only ${free_gpu_count} visible GPU(s) use <=${MAX_USED_GPU_MEMORY_MIB} MiB; ${required_gpu_count} required"
    fi
  else
    fail "nvidia-smi returned no GPU rows"
  fi
else
  fail "nvidia-smi is unavailable; training requires visible CUDA GPUs"
fi

if (( failures > 0 )); then
  printf '[preflight] %d blocking check(s) failed\n' "$failures" >&2
  exit 1
fi

printf '[preflight] all checks passed; no files or processes were modified\n'
