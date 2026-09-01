#!/usr/bin/env bash
set -euo pipefail

# Reproducible GRPO/CISPO/DAPO/GSPO ablation on MiniMind Agent RLVR data.
# Run from the repository root. Override any MM_* variable as needed, e.g.:
# MM_GPUS=2 MM_SEEDS="42 43 44" bash scripts/run_rlvr_ablation.sh

MM_GPUS="${MM_GPUS:-1}"
MM_SEEDS="${MM_SEEDS:-42 43 44}"
MM_TASK="${MM_TASK:-math}"
if [[ "${MM_TASK}" == "tool" ]]; then
  MM_DEFAULT_DATA_PATH="../dataset/agent_rl_tool_challenge_train.jsonl"
else
  MM_DEFAULT_DATA_PATH="../dataset/agent_rl_${MM_TASK}_train.jsonl"
fi
MM_DATA_PATH="${MM_DATA_PATH:-${MM_DEFAULT_DATA_PATH}}"
MM_HIDDEN_SIZE="${MM_HIDDEN_SIZE:-768}"
MM_LAYERS="${MM_LAYERS:-8}"
MM_BATCH_SIZE="${MM_BATCH_SIZE:-2}"
MM_GENERATIONS="${MM_GENERATIONS:-4}"
MM_POLICY_EPOCHS="${MM_POLICY_EPOCHS:-2}"
MM_LEARNING_RATE="${MM_LEARNING_RATE:-1e-7}"
MM_EPOCHS="${MM_EPOCHS:-1}"
MM_MAX_UPDATES="${MM_MAX_UPDATES:-30}"
MM_MAX_TURNS="${MM_MAX_TURNS:-3}"
MM_MAX_GEN_LEN="${MM_MAX_GEN_LEN:-128}"
MM_MAX_TOTAL_LEN="${MM_MAX_TOTAL_LEN:-1536}"
MM_THINKING_RATIO="${MM_THINKING_RATIO:-0}"
MM_INIT_WEIGHT="${MM_INIT_WEIGHT:-agent_sft}"
MM_DYNAMIC_SAMPLING_ROUNDS="${MM_DYNAMIC_SAMPLING_ROUNDS:-50}"

MM_DATA_CHECK="${MM_DATA_PATH#../}"
test -f "${MM_DATA_CHECK}" || {
  echo "Missing ${MM_DATA_CHECK}. Create the persisted train/eval split first." >&2
  exit 1
}
test -f "out/${MM_INIT_WEIGHT}_${MM_HIDDEN_SIZE}.pth" || {
  echo "Missing out/${MM_INIT_WEIGHT}_${MM_HIDDEN_SIZE}.pth. Finish the common Agent SFT cold start first." >&2
  exit 1
}

mkdir -p out/metrics

for MM_SEED in ${MM_SEEDS}; do
  for MM_ALGORITHM in grpo cispo dapo gspo; do
    MM_METRICS_FILE="out/metrics/${MM_TASK}_${MM_ALGORITHM}_s${MM_SEED}.jsonl"
    MM_WEIGHT_FILE="out/agent_${MM_TASK}_${MM_ALGORITHM}_s${MM_SEED}_${MM_HIDDEN_SIZE}.pth"
    MM_RESUME_FILE="checkpoints/agent_${MM_TASK}_${MM_ALGORITHM}_s${MM_SEED}_${MM_HIDDEN_SIZE}_resume.pth"
    if [[ -e "${MM_METRICS_FILE}" || -e "${MM_WEIGHT_FILE}" || -e "${MM_RESUME_FILE}" ]]; then
      echo "Refusing to mix/overwrite an existing run: ${MM_METRICS_FILE} or ${MM_WEIGHT_FILE}" >&2
      echo "Archive it under a new experiment directory or remove it explicitly, then rerun." >&2
      exit 1
    fi
    MM_EXTRA_ARGS=()
    if [[ "${MM_ALGORITHM}" == "dapo" ]]; then
      MM_EXTRA_ARGS+=(--dynamic_sampling --dynamic_sampling_rounds "${MM_DYNAMIC_SAMPLING_ROUNDS}")
      MM_EXTRA_ARGS+=(--dapo_epsilon_low 0.2 --dapo_epsilon_high 0.28)
      MM_EXTRA_ARGS+=(--overlong_cache_len 128 --overlong_penalty_coef 1.0)
    fi
    if [[ "${MM_ALGORITHM}" == "gspo" ]]; then
      MM_EXTRA_ARGS+=(--gspo_epsilon_low 0.0003 --gspo_epsilon_high 0.0004)
    fi

    (
      cd trainer
      torchrun --standalone --nproc_per_node "${MM_GPUS}" train_agent.py \
        --loss_type "${MM_ALGORITHM}" \
        --save_weight "agent_${MM_TASK}_${MM_ALGORITHM}_s${MM_SEED}" \
        --from_weight "${MM_INIT_WEIGHT}" \
        --data_path "${MM_DATA_PATH}" \
        --metrics_path "../out/metrics/${MM_TASK}_${MM_ALGORITHM}_s${MM_SEED}.jsonl" \
        --seed "${MM_SEED}" \
        --epochs "${MM_EPOCHS}" \
        --max_updates "${MM_MAX_UPDATES}" \
        --batch_size "${MM_BATCH_SIZE}" \
        --num_generations "${MM_GENERATIONS}" \
        --policy_update_epochs "${MM_POLICY_EPOCHS}" \
        --learning_rate "${MM_LEARNING_RATE}" \
        --rollout_sync_interval 1 \
        --hidden_size "${MM_HIDDEN_SIZE}" \
        --num_hidden_layers "${MM_LAYERS}" \
        --max_gen_len "${MM_MAX_GEN_LEN}" \
        --max_turns "${MM_MAX_TURNS}" \
        --max_total_len "${MM_MAX_TOTAL_LEN}" \
        --thinking_ratio "${MM_THINKING_RATIO}" \
        --beta 0 \
        --weight_decay 0 \
        --reward_mode strict \
        --use_reward_model 0 \
        --require_tool_call_for_success 1 \
        --dtype bfloat16 \
        --save_interval 50 \
        --save_resume 0 \
        "${MM_EXTRA_ARGS[@]}"
    )
  done
done

python scripts/summarize_rl_metrics.py "out/metrics/${MM_TASK}_*.jsonl" \
  --last_n 20 \
  --output "out/metrics/algorithm_comparison_${MM_TASK}.csv"

echo "Training comparison: out/metrics/algorithm_comparison_${MM_TASK}.csv"
echo "Next run the fixed-set evaluator; see docs/foundation_model_interview/00_END_TO_END_GUIDE.md."
