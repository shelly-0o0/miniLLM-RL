#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MM_REPO_ROOT:-/root/minimind}"
PYTHON_BIN="${MM_PYTHON_BIN:-/root/miniconda3/envs/minimind_opt/bin/python}"
SEED="${MM_SEED:-242}"
OUTPUT_DIR="${REPO_ROOT}/out/optimization/eval/tool_corrected_ablation_3seed"
LOG="${REPO_ROOT}/out/optimization/logs/tool_corrected_ablation_3seed_eval.log"
WEIGHTS="agent_sft,agent_sft_tool_curriculum96,opt_tool_grpo_strict_s${SEED},opt_tool_cispo_strict_s${SEED},opt_tool_dapo_strict_s${SEED},opt_tool_gspo_strict_s${SEED}"

for WEIGHT in opt_tool_grpo_strict_s${SEED} opt_tool_cispo_strict_s${SEED} \
  opt_tool_dapo_strict_s${SEED} opt_tool_gspo_strict_s${SEED}; do
  if [[ ! -e "${REPO_ROOT}/out/${WEIGHT}_768.pth" ]]; then
    echo "Missing checkpoint: ${REPO_ROOT}/out/${WEIGHT}_768.pth" >&2
    exit 1
  fi
done
if [[ -e "${OUTPUT_DIR}/summary.csv" ]]; then
  echo "Refusing to overwrite existing evaluation: ${OUTPUT_DIR}/summary.csv" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG}")"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" scripts/eval_agent_rlvr.py \
  --weights "${WEIGHTS}" \
  --reference_weight agent_sft_tool_curriculum96 \
  --data_path dataset/agent_rl_tool_challenge_eval.jsonl \
  --save_dir out \
  --output_dir "${OUTPUT_DIR}" \
  --seeds 101,102,103 \
  --limit 96 \
  --batch_size 1 \
  --num_generations 1 \
  --max_turns 3 \
  --max_gen_len 128 \
  --max_total_len 1024 \
  --thinking_ratio 0 \
  --rollout_temperature 1 --rollout_top_k 0 --rollout_top_p 1 \
  --require_tool_call_for_success 1 \
  --reward_mode strict \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda:0 \
  2>&1 | tee "${LOG}"

"${PYTHON_BIN}" scripts/summarize_eval_results.py \
  "${OUTPUT_DIR}/summary.csv" \
  --output "${OUTPUT_DIR}/checkpoint_comparison.csv"

"${PYTHON_BIN}" scripts/summarize_eval_results.py \
  "${OUTPUT_DIR}/summary.csv" \
  --baseline agent_sft_tool_curriculum96 \
  --output "${OUTPUT_DIR}/checkpoint_comparison_vs_curriculum.csv"
