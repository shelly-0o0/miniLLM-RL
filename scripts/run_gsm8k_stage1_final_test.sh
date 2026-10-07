#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_gsm8k_stage1_final_test.sh {agent_sft|dapo} [TRAIN_SEED]" >&2
  exit 2
fi

target="$1"
train_seed="${2:-}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

case "$target" in
  agent_sft)
    [[ -z "$train_seed" ]] || { echo "agent_sft does not take TRAIN_SEED" >&2; exit 2; }
    weight="agent_sft"
    save_dir="out"
    output_dir="out/eval_test/stage1/agent_sft"
    log_path="out/logs/stage1_test_agent_sft.log"
    ;;
  dapo)
    [[ "$train_seed" =~ ^(42|43|44)$ ]] || {
      echo "DAPO final test requires TRAIN_SEED 42, 43, or 44" >&2
      exit 2
    }
    weight="gsm8k_dapo_full_s${train_seed}"
    save_dir="out/rl_full/dapo/seed_${train_seed}"
    output_dir="out/eval_test/stage1/dapo/train_seed_${train_seed}"
    log_path="out/logs/stage1_test_dapo_s${train_seed}.log"
    ;;
  *)
    echo "unsupported target: $target" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output_dir")" out/logs

PYTHONUNBUFFERED=1 python -u scripts/eval_agent_rlvr.py \
  --weights "$weight" \
  --reference_weight agent_sft \
  --reference_save_dir out \
  --save_dir "$save_dir" \
  --data_path data/processed/gsm8k_agent/test_rl.jsonl \
  --tokenizer_path models/minimind-3 \
  --output_dir "$output_dir" \
  --seeds 42,43,44 \
  --limit 1319 \
  --batch_size 1 \
  --num_generations 1 \
  --max_turns 3 \
  --max_gen_len 384 \
  --max_total_len 2500 \
  --thinking_ratio 0 \
  --rollout_temperature 1.0 \
  --rollout_top_k 0 \
  --rollout_top_p 1.0 \
  --require_tool_call_for_success 1 \
  --reward_mode strict \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device "${STAGE1_DEVICE:-cuda:0}" \
  2>&1 | tee "$log_path"
