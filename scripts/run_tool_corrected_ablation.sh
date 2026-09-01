#!/usr/bin/env bash
set -euo pipefail

# Corrected post-training ablation after the FP32/log-prob/cold-start fixes.
# DAPO was run separately because its dynamic sampling has a different rollout
# cost.  This script fills the GRPO/CISPO/GSPO arms from the exact same init.

REPO_ROOT="${MM_REPO_ROOT:-/root/minimind}"
PYTHON_BIN="${MM_PYTHON_BIN:-/root/miniconda3/envs/minimind_opt/bin/python}"
SAVE_DIR="${MM_SAVE_DIR:-/root/autodl-tmp/minimind_optimized_weights}"
SEED="${MM_SEED:-242}"
MAX_UPDATES="${MM_MAX_UPDATES:-30}"

mkdir -p "${SAVE_DIR}" "${REPO_ROOT}/out/optimization/metrics" \
  "${REPO_ROOT}/out/optimization/logs"

for ALGORITHM in grpo cispo gspo; do
  WEIGHT="opt_tool_${ALGORITHM}_strict_s${SEED}"
  METRICS="${REPO_ROOT}/out/optimization/metrics/${WEIGHT}.jsonl"
  LOG="${REPO_ROOT}/out/optimization/logs/${WEIGHT}.log"
  CHECKPOINT="${SAVE_DIR}/${WEIGHT}_768.pth"

  if [[ -e "${CHECKPOINT}" || -e "${METRICS}" ]]; then
    echo "Refusing to overwrite an existing run: ${CHECKPOINT} or ${METRICS}" >&2
    exit 1
  fi

  EXTRA_ARGS=()
  if [[ "${ALGORITHM}" == "cispo" ]]; then
    EXTRA_ARGS+=(--epsilon_high 5.0)
  elif [[ "${ALGORITHM}" == "gspo" ]]; then
    EXTRA_ARGS+=(--gspo_epsilon_low 0.0003 --gspo_epsilon_high 0.0004)
  fi

  cd "${REPO_ROOT}/trainer"
  export PYTHONUNBUFFERED=1
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node 1 \
    train_agent.py \
    --loss_type "${ALGORITHM}" \
    --save_dir "${SAVE_DIR}" \
    --save_weight "${WEIGHT}" \
    --checkpoint_dtype float32 \
    --from_weight agent_sft_tool_curriculum96 \
    --from_resume 0 \
    --data_path ../dataset/agent_rl_tool_challenge_train.jsonl \
    --metrics_path "${METRICS}" \
    --seed "${SEED}" \
    --epochs 1 \
    --max_updates "${MAX_UPDATES}" \
    --batch_size 1 \
    --num_generations 4 \
    --rollout_temperature 0.9 \
    --policy_update_epochs 2 \
    --learning_rate 1e-7 \
    --rollout_sync_interval 1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_gen_len 128 \
    --max_turns 3 \
    --max_total_len 1024 \
    --thinking_ratio 0 \
    --beta 0 \
    --weight_decay 0 \
    --reward_mode strict \
    --use_reward_model 0 \
    --require_tool_call_for_success 1 \
    --dtype bfloat16 \
    --num_workers 2 \
    --log_interval 10 \
    --save_interval "${MAX_UPDATES}" \
    --save_resume 0 \
    --max_rollout_logprob_mae 0.02 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"

  ln -sfn "${CHECKPOINT}" "${REPO_ROOT}/out/${WEIGHT}_768.pth"
  sha256sum "${CHECKPOINT}" > "${REPO_ROOT}/out/optimization/${WEIGHT}.sha256"
done

