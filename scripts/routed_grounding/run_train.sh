#!/usr/bin/env bash
set -euo pipefail

# Shared legacy-RayPPO launcher for all three arms.  The first positional
# argument selects the arm; all remaining arguments are Hydra overrides/options.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ $# -lt 1 ]]; then
  printf 'usage: run_train.sh standard_opd|generic_sopd|routed_repair [hydra overrides...]\n' >&2
  exit 2
fi
ARM="$1"
shift

case "$ARM" in
  standard_opd|generic_sopd|routed_repair)
    ;;
  *)
    printf 'unknown routed-grounding arm: %s\n' "$ARM" >&2
    exit 2
    ;;
esac

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON_BIN=python3
fi
PYTHON_BIN="${OPD_PYTHON:-$DEFAULT_PYTHON_BIN}"
SMOKE="${OPD_SMOKE:-0}"
if [[ "$SMOKE" == "1" || "$SMOKE" == "true" || "$SMOKE" == "True" ]]; then
  CONFIG_NAME="${ARM}_smoke"
  DEFAULT_TEACHER_VIEW=gt_overlay
  DEFAULT_TRAIN_FILE=data/refcocog_umd_pilot/smoke_train.parquet
  DEFAULT_VAL_MAX_SAMPLES=200
  # smoke_train.parquet is the fixed 32-row subset; do not resample it.
  DEFAULT_MAX_SAMPLES=-1
  DEFAULT_TOTAL_STEPS=20
  DEFAULT_TOTAL_EPOCHS=20
  DEFAULT_EXPERIMENT="${ARM}_smoke"
  DEFAULT_AGENT_LOOP="routed_grounding_${ARM}_smoke"
else
  CONFIG_NAME="$ARM"
  DEFAULT_TEACHER_VIEW=gt_crop
  DEFAULT_TRAIN_FILE=data/refcocog_umd_pilot/train.parquet
  DEFAULT_MAX_SAMPLES=-1
  DEFAULT_VAL_MAX_SAMPLES=-1
  DEFAULT_TOTAL_STEPS=100
  DEFAULT_TOTAL_EPOCHS=100
  DEFAULT_EXPERIMENT="$ARM"
  DEFAULT_AGENT_LOOP="routed_grounding_${ARM}"
fi
CONFIG_NAME="${OPD_CONFIG_NAME:-$CONFIG_NAME}"

TRAIN_FILE="${OPD_TRAIN_FILE:-$DEFAULT_TRAIN_FILE}"
VAL_FILE="${OPD_VAL_FILE:-data/refcocog_umd_pilot/val.parquet}"
STUDENT_MODEL="${OPD_STUDENT_MODEL:-/mnt/sda/sujingyang/models/Qwen3.5-0.8B}"
TEACHER_MODEL="${OPD_TEACHER_MODEL:-/mnt/sda/sujingyang/models/Qwen3.5-4B}"
SEED="${OPD_SEED:-260600564}"
LR="${OPD_LR:-1e-6}"
WEIGHT_DECAY="${OPD_WEIGHT_DECAY:-1e-2}"
MAX_PROMPT="${OPD_MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE="${OPD_MAX_RESPONSE_LENGTH:-4096}"
TEMPERATURE="${OPD_ROLLOUT_TEMPERATURE:-0.8}"
TOP_P="${OPD_ROLLOUT_TOP_P:-0.95}"
TOP_K="${OPD_ROLLOUT_TOP_K:-0}"
TRAIN_BATCH_SIZE="${OPD_TRAIN_BATCH_SIZE:-32}"
TRAIN_MAX_SAMPLES="${OPD_TRAIN_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
VAL_MAX_SAMPLES="${OPD_VAL_MAX_SAMPLES:-$DEFAULT_VAL_MAX_SAMPLES}"
TOTAL_STEPS="${OPD_TOTAL_TRAINING_STEPS:-$DEFAULT_TOTAL_STEPS}"
TOTAL_EPOCHS="${OPD_TOTAL_EPOCHS:-$DEFAULT_TOTAL_EPOCHS}"
N_GPUS="${OPD_N_GPUS_PER_NODE:-4}"
TEACHER_N_GPUS="${OPD_TEACHER_N_GPUS_PER_NODE:-1}"
COLOCATE_WITH_ACTOR="${OPD_DISTILLATION_COLOCATE_WITH_ACTOR:-false}"
SINGLE_GPU="${OPD_SINGLE_GPU:-0}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${OPD_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
TEACHER_GPU_MEMORY_UTILIZATION="${OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.4}"
ROLLOUT_MAX_NUM_SEQS="${OPD_ROLLOUT_MAX_NUM_SEQS:-64}"
TEACHER_MAX_NUM_SEQS="${OPD_TEACHER_MAX_NUM_SEQS:-64}"
if [[ "$SINGLE_GPU" == "1" || "$SINGLE_GPU" == "true" || "$SINGLE_GPU" == "True" ]]; then
  N_GPUS=1
  TEACHER_N_GPUS=1
  COLOCATE_WITH_ACTOR=true
  ROLLOUT_GPU_MEMORY_UTILIZATION="${OPD_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2}"
  TEACHER_GPU_MEMORY_UTILIZATION="${OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.4}"
  ROLLOUT_MAX_NUM_SEQS="${OPD_ROLLOUT_MAX_NUM_SEQS:-8}"
  TEACHER_MAX_NUM_SEQS="${OPD_TEACHER_MAX_NUM_SEQS:-8}"
fi
TEACHER_VIEW="${OPD_TEACHER_VIEW:-$DEFAULT_TEACHER_VIEW}"
TEACHER_TEMPERATURE="${OPD_TEACHER_TEMPERATURE:-0.0}"
TEACHER_TOP_P="${OPD_TEACHER_TOP_P:-1.0}"
TEACHER_TOP_K="${OPD_TEACHER_TOP_K:-0}"
MIN_PIXELS="${OPD_MIN_PIXELS:-3136}"
MAX_PIXELS="${OPD_MAX_PIXELS:-262144}"
PROJECT_NAME="${OPD_WANDB_PROJECT:-routed_grounding_refcocog}"
EXPERIMENT_NAME="${OPD_EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT}"
OUTPUT_DIR="${OPD_OUTPUT_DIR:-outputs/routed_grounding/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
AGENT_LOOP_CONFIG="${OPD_AGENT_LOOP_CONFIG:-configs/routed_grounding/agent_loops.yaml}"
AGENT_LOOP_NAME="${OPD_AGENT_LOOP_NAME:-$DEFAULT_AGENT_LOOP}"
AGENT_LOOP_MANAGER_FQCN="${OPD_AGENT_LOOP_MANAGER_FQCN:-verl.experimental.routed_grounding.agent_loop.RoutedGroundingAgentLoopManager}"
CUSTOM_LOSS_MODULE="${OPD_CUSTOM_LOSS_MODULE:-verl.experimental.routed_grounding.losses}"

if [[ -n "${OPD_CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Explicit opt-in only; no CUDA_VISIBLE_DEVICES default is assigned here.
  export CUDA_VISIBLE_DEVICES="$OPD_CUDA_VISIBLE_DEVICES"
fi
if [[ -n "${OPD_WANDB_MODE:-}" ]]; then
  export WANDB_MODE="$OPD_WANDB_MODE"
fi

cd "$REPO_ROOT"

audit_required_files() {
  local agent_config_path="$1"
  [[ -f "$agent_config_path" ]] || {
    printf 'required agent-loop YAML is missing: %s\n' "$agent_config_path" >&2
    return 1
  }
  [[ -f "verl/experimental/routed_grounding/agent_loop.py" ]] || {
    printf 'required custom AgentLoopManager source is missing\n' >&2
    return 1
  }
  if ! rg -q "RoutedGroundingAgentLoopManager" verl/experimental/routed_grounding/agent_loop.py; then
    printf 'custom AgentLoopManager symbol is missing from source\n' >&2
    return 1
  fi
  if [[ "$ARM" != "standard_opd" ]]; then
    [[ -f "verl/experimental/routed_grounding/losses.py" ]] || {
      printf 'required routed loss source is missing for %s\n' "$ARM" >&2
      return 1
    }
    rg -q "routed_forward_kl_ce" verl/experimental/routed_grounding/losses.py || {
      printf 'routed_forward_kl_ce registration is missing for %s\n' "$ARM" >&2
      return 1
    }
  fi
}

if [[ "$AGENT_LOOP_CONFIG" = /* ]]; then
  AGENT_CONFIG_PATH="$AGENT_LOOP_CONFIG"
else
  AGENT_CONFIG_PATH="$REPO_ROOT/$AGENT_LOOP_CONFIG"
fi
audit_required_files "$AGENT_CONFIG_PATH"

if [[ "${OPD_SKIP_PREFLIGHT:-0}" != "1" && "${OPD_DRY_RUN:-0}" != "1" ]]; then
  OPD_REPO_ROOT="$REPO_ROOT" \
    OPD_TRAIN_FILE="$TRAIN_FILE" \
    OPD_VAL_FILE="$VAL_FILE" \
    OPD_STUDENT_MODEL="$STUDENT_MODEL" \
    OPD_TEACHER_MODEL="$TEACHER_MODEL" \
    OPD_N_GPUS_PER_NODE="$N_GPUS" \
    OPD_TEACHER_N_GPUS_PER_NODE="$TEACHER_N_GPUS" \
    OPD_DISTILLATION_COLOCATE_WITH_ACTOR="$COLOCATE_WITH_ACTOR" \
    "$SCRIPT_DIR/run_preflight.sh"
fi

OVERRIDES=(
  "routed_grounding.arm=${ARM}"
  "routed_grounding.teacher_view=${TEACHER_VIEW}"
  "routed_grounding.json_bbox_phase=per_request"
  "routed_grounding.seed=${SEED}"
  "routed_grounding.image_processor.min_pixels=${MIN_PIXELS}"
  "routed_grounding.image_processor.max_pixels=${MAX_PIXELS}"
  "actor_rollout_ref.model.path=${STUDENT_MODEL}"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG}"
  "actor_rollout_ref.rollout.agent.agent_loop_manager_class=${AGENT_LOOP_MANAGER_FQCN}"
  "actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP_NAME}"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.top_p=${TOP_P}"
  "actor_rollout_ref.rollout.top_k=${TOP_K}"
  "actor_rollout_ref.rollout.seed=${SEED}"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE}"
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESPONSE + 1))"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=true"
  "data.train_files=['${TRAIN_FILE}']"
  "data.val_files=['${VAL_FILE}']"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
  "data.val_max_samples=${VAL_MAX_SAMPLES}"
  "data.max_prompt_length=${MAX_PROMPT}"
  "data.max_response_length=${MAX_RESPONSE}"
  "data.shuffle=false"
  "data.seed=${SEED}"
  "data.mm_processor_kwargs.min_pixels=${MIN_PIXELS}"
  "data.mm_processor_kwargs.max_pixels=${MAX_PIXELS}"
  "trainer.use_v1=false"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.validation_data_dir=${OUTPUT_DIR}/validation"
  "trainer.rollout_data_dir=${OUTPUT_DIR}/rollouts"
  "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}"
  "distillation.teacher_models.teacher_model.inference.prompt_length=${MAX_PROMPT}"
  "distillation.teacher_models.teacher_model.inference.response_length=${MAX_RESPONSE}"
  "distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT + MAX_RESPONSE + 1))"
  "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}"
  "distillation.teacher_models.teacher_model.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS}"
  "distillation.teacher_models.teacher_model.inference.temperature=${TEACHER_TEMPERATURE}"
  "distillation.teacher_models.teacher_model.inference.top_p=${TEACHER_TOP_P}"
  "distillation.teacher_models.teacher_model.inference.top_k=${TEACHER_TOP_K}"
  "distillation.teacher_models.teacher_model.inference.do_sample=false"
  "distillation.teacher_models.teacher_model.inference.seed=${SEED}"
  "distillation.n_gpus_per_node=${TEACHER_N_GPUS}"
  "distillation.colocate_with_actor=${COLOCATE_WITH_ACTOR}"
  "distillation.distillation_loss.topk=64"
  "distillation.distillation_loss.use_task_rewards=false"
  "distillation.distillation_loss.use_policy_gradient=false"
  "transfer_queue.enable=false"
)

if [[ "$ARM" = "standard_opd" ]]; then
  OVERRIDES+=("distillation.distillation_loss.custom_loss_module=null")
  OVERRIDES+=("distillation.distillation_loss.loss_mode=forward_kl_topk")
else
  OVERRIDES+=("distillation.distillation_loss.custom_loss_module=${CUSTOM_LOSS_MODULE}")
  OVERRIDES+=("distillation.distillation_loss.loss_mode=routed_forward_kl_ce")
fi

printf '[launch] arm=%s config=%s smoke=%s\n' "$ARM" "$CONFIG_NAME" "$SMOKE"
printf '[launch] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset; no default GPU claim>}"
printf '[launch] trainer.use_v1=false; entrypoint=verl.trainer.main_ppo -> legacy main_ppo_v0.TaskRunner/RayPPOTrainer\n'
printf '[launch] AgentLoopManager=%s; TransferQueue/V1 disabled\n' "$AGENT_LOOP_MANAGER_FQCN"
printf '[launch] single_gpu=%s; teacher_colocated=%s; rollout_gpu_memory=%s; teacher_gpu_memory=%s\n' \
  "$SINGLE_GPU" "$COLOCATE_WITH_ACTOR" "$ROLLOUT_GPU_MEMORY_UTILIZATION" "$TEACHER_GPU_MEMORY_UTILIZATION"

if [[ "${OPD_DRY_RUN:-0}" == "1" ]]; then
  printf '[launch] dry-run command: '
  printf '%q ' "$PYTHON_BIN" -m verl.trainer.main_ppo \
    --config-path "$REPO_ROOT/configs/routed_grounding" \
    --config-name "$CONFIG_NAME" "${OVERRIDES[@]}" "$@"
  printf '\n'
  exit 0
fi

LOG_DIR="${OPD_LOG_DIR:-logs/routed_grounding}"
mkdir -p "$LOG_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${ARM}_${RUN_STAMP}.log"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path "$REPO_ROOT/configs/routed_grounding" \
  --config-name "$CONFIG_NAME" \
  "${OVERRIDES[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/routed_grounding/convert_validation_dump.py \
  --validation-dump "${OUTPUT_DIR}/validation" \
  --manifest data/manifests/refcocog_umd_pilot/val.jsonl \
  --output "${OUTPUT_DIR}/monitor_results.csv" \
  --overwrite
