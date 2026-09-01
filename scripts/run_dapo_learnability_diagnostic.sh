#!/usr/bin/env bash
set -euo pipefail

# One learnability selection pass followed by two short DAPO runs.  The
# checkpoint/evaluation directories are intentionally unique per run.
PYTHON_BIN="${MM_PYTHON_BIN:-python}"
INIT_WEIGHT="${MM_INIT_WEIGHT:-agent_sft_tool_curriculum96}"
SAVE_DIR="${MM_SAVE_DIR:-out/diagnostic_weights}"
TRAIN_DATA="${MM_DIAGNOSTIC_DATA:-out/learnability/diagnostic_train.jsonl}"
HOLDOUT_DATA="${MM_HOLDOUT_DATA:-dataset/agent_rl_tool_challenge_eval.jsonl}"
INSPECTION_DIR="${MM_INSPECTION_DIR:-out/learnability}"
MAX_GROUPS="${MM_MAX_EFFECTIVE_GROUPS:-200}"
MAX_TOKENS="${MM_MAX_GENERATED_TOKENS:-500000}"

mkdir -p "${SAVE_DIR}" "out/diagnostic_metrics"

if [[ "${MM_SKIP_INSPECTION:-0}" == "1" && -s "${TRAIN_DATA}" ]]; then
  echo "reusing existing learnability inspection: ${TRAIN_DATA}"
else
  "${PYTHON_BIN}" scripts/inspect_rl_learnability.py \
    --weight "${INIT_WEIGHT}" \
    --save_dir "${MM_BASE_SAVE_DIR:-out}" \
    --output_dir "${INSPECTION_DIR}" \
    --data_path "${MM_CHALLENGE_TRAIN:-dataset/agent_rl_tool_challenge_train.jsonl}" \
    --sft_data_path "${MM_SFT_CURRICULUM:-dataset/agent_sft_tool_challenge_curriculum_96.jsonl}" \
    --rollout_temperature 1 --rollout_top_k 0 --rollout_top_p 1 \
    --num_generations 8 --num_questions 96 --diagnostic_questions 32
fi

test -s "${TRAIN_DATA}"
for LR_TAG in 3e-7 1e-6; do
  SAFE_TAG="${LR_TAG//-/m}"
  WEIGHT="dapo_learnability_lr${SAFE_TAG}"
  METRICS="out/diagnostic_metrics/${WEIGHT}.jsonl"
  test ! -e "${METRICS}"
  "${PYTHON_BIN}" trainer/train_agent.py \
    --loss_type dapo --dynamic_sampling \
    --from_weight "${INIT_WEIGHT}" --save_dir "${SAVE_DIR}" \
    --from_save_dir "${MM_BASE_SAVE_DIR:-out}" \
    --tokenizer_path model \
    --save_weight "${WEIGHT}" --data_path "${TRAIN_DATA}" \
    --metrics_path "${METRICS}" --epochs 1 --batch_size 1 \
    --num_generations 8 --learning_rate "${LR_TAG}" \
    --policy_update_epochs 1 --beta 0 --weight_decay 0 \
    --reward_mode strict --use_reward_model 0 \
    --rollout_temperature 1 --rollout_top_k 0 --rollout_top_p 1 \
    --max_effective_groups "${MAX_GROUPS}" \
    --max_generated_tokens "${MAX_TOKENS}" \
    --checkpoint_groups 0,50,100,200 --save_interval "${MAX_GROUPS}" \
    --checkpoint_dtype float32 --save_resume 0 \
    --max_gen_len 128 --max_turns 3 --max_total_len 1024 \
    --thinking_ratio 0 --num_workers 2 --max_rollout_logprob_mae 0.1

  EVAL_WEIGHTS=()
  for GROUP in 0 50 100 200; do
    if [[ -f "${SAVE_DIR}/${WEIGHT}_768_groups${GROUP}.pth" ]]; then
      EVAL_WEIGHTS+=("${WEIGHT}_groups${GROUP}")
    fi
  done
  WEIGHT_CSV=$(IFS=,; echo "${EVAL_WEIGHTS[*]}")
  "${PYTHON_BIN}" scripts/eval_agent_rlvr.py \
    --weights "${WEIGHT_CSV}" \
      --reference_weight "${INIT_WEIGHT}" --reference_save_dir "${MM_BASE_SAVE_DIR:-out}" --save_dir "${SAVE_DIR}" \
    --data_path "${TRAIN_DATA}" --output_dir "out/eval_rlvr/${WEIGHT}/train" \
    --seeds 20260903 --limit 32 --batch_size 1 --num_generations 8 \
    --max_turns 3 --max_gen_len 128 --max_total_len 1024 \
    --rollout_temperature 1 --rollout_top_k 0 --rollout_top_p 1 \
    --thinking_ratio 0 --reward_mode strict

  if [[ -f "${HOLDOUT_DATA}" ]]; then
    "${PYTHON_BIN}" scripts/eval_agent_rlvr.py \
      --weights "${WEIGHT_CSV}" \
      --reference_weight "${INIT_WEIGHT}" --reference_save_dir "${MM_BASE_SAVE_DIR:-out}" --save_dir "${SAVE_DIR}" \
      --data_path "${HOLDOUT_DATA}" --output_dir "out/eval_rlvr/${WEIGHT}/holdout" \
      --seeds 20260904 --limit 96 --batch_size 1 --num_generations 8 \
      --max_turns 3 --max_gen_len 128 --max_total_len 1024 \
      --rollout_temperature 1 --rollout_top_k 0 --rollout_top_p 1 \
      --thinking_ratio 0 --reward_mode strict
    "${PYTHON_BIN}" scripts/plot_dapo_learnability_curve.py \
      --metrics "${METRICS}" \
      --train_eval "out/eval_rlvr/${WEIGHT}/train/summary.csv" \
      --holdout_eval "out/eval_rlvr/${WEIGHT}/holdout/summary.csv" \
      --output "out/eval_rlvr/${WEIGHT}/learnability_curve.csv" \
      --plot "out/eval_rlvr/${WEIGHT}/learnability_curve.png"
  else
    echo "holdout data missing; train-curve evaluation completed without holdout: ${HOLDOUT_DATA}"
  fi
done
