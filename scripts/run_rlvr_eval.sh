#!/usr/bin/env bash
set -euo pipefail

MM_TASK="${MM_TASK:-math}"
MM_TRAIN_SEEDS="${MM_TRAIN_SEEDS:-42 43 44}"
MM_DECODE_SEEDS="${MM_DECODE_SEEDS:-101,102,103}"
MM_LIMIT="${MM_LIMIT:-128}"
MM_BATCH_SIZE="${MM_BATCH_SIZE:-2}"
MM_GENERATIONS="${MM_GENERATIONS:-1}"
MM_MAX_TURNS="${MM_MAX_TURNS:-3}"
MM_MAX_GEN_LEN="${MM_MAX_GEN_LEN:-128}"
MM_MAX_TOTAL_LEN="${MM_MAX_TOTAL_LEN:-1536}"
if [[ "${MM_TASK}" == "tool" ]]; then
  MM_EVAL_DATA_PATH="${MM_EVAL_DATA_PATH:-dataset/agent_rl_tool_challenge_eval.jsonl}"
else
  MM_EVAL_DATA_PATH="${MM_EVAL_DATA_PATH:-dataset/agent_rl_${MM_TASK}_eval.jsonl}"
fi

# The shared Agent-SFT initialization is evaluated on the identical holdout
# and decode seeds.  This makes "improvement over baseline" a measured delta,
# not an inference from a separate readiness configuration.
weights=("agent_sft")
for seed in ${MM_TRAIN_SEEDS}; do
  for algorithm in grpo cispo dapo gspo; do
    weight="agent_${MM_TASK}_${algorithm}_s${seed}"
    test -f "out/${weight}_768.pth" || {
      echo "Missing out/${weight}_768.pth" >&2
      exit 1
    }
    weights+=("${weight}")
  done
done
weight_csv=$(IFS=,; echo "${weights[*]}")
output_dir="out/eval_rlvr/${MM_TASK}"
test ! -e "${output_dir}" || {
  echo "Refusing to overwrite ${output_dir}; archive it first." >&2
  exit 1
}

python scripts/eval_agent_rlvr.py \
  --weights "${weight_csv}" \
  --reference_weight agent_sft \
  --data_path "${MM_EVAL_DATA_PATH}" \
  --output_dir "${output_dir}" \
  --seeds "${MM_DECODE_SEEDS}" \
  --limit "${MM_LIMIT}" \
  --batch_size "${MM_BATCH_SIZE}" \
  --num_generations "${MM_GENERATIONS}" \
  --max_turns "${MM_MAX_TURNS}" \
  --max_gen_len "${MM_MAX_GEN_LEN}" \
  --max_total_len "${MM_MAX_TOTAL_LEN}" \
  --thinking_ratio 0 \
  --rollout_temperature 1 --rollout_top_k 0 --rollout_top_p 1 \
  --reward_mode strict \
  --require_tool_call_for_success 1 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda:0

python scripts/summarize_eval_results.py "${output_dir}/summary.csv" \
  --output "${output_dir}/algorithm_comparison.csv"
