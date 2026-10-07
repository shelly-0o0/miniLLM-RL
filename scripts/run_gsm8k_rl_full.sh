#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: bash scripts/run_gsm8k_rl_full.sh {grpo|cispo|dapo|gspo} SEED [--resume]" >&2
  exit 2
fi

algorithm="$1"
seed="$2"
resume=0
if [[ ${3:-} == "--resume" ]]; then resume=1; elif [[ $# -eq 3 ]]; then echo "third argument must be --resume" >&2; exit 2; fi
case "$algorithm" in
  grpo|cispo|dapo|gspo) ;;
  *) echo "unsupported algorithm: $algorithm" >&2; exit 2 ;;
esac
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

run_name="gsm8k_${algorithm}_full_s${seed}"
save_dir="out/rl_full/${algorithm}/seed_${seed}"
checkpoint_dir="checkpoints/rl_full/${algorithm}/seed_${seed}"
metrics_path="out/metrics/${run_name}.jsonl"
log_path="out/logs/${run_name}.log"

mkdir -p "$save_dir" "$checkpoint_dir" out/metrics out/logs
if [[ "$resume" -eq 0 && ( -e "$metrics_path" || -e "$save_dir/${run_name}_768.pth" ) ]]; then
  echo "refusing to overwrite an existing run: $run_name" >&2
  exit 1
fi

extra_args=()
if [[ "$algorithm" == "dapo" ]]; then
  extra_args+=(--dynamic_sampling --dynamic_sampling_rounds 2)
fi

PYTHONUNBUFFERED=1 python -u trainer/train_agent.py \
  --data_path data/processed/gsm8k_agent/train_rl.jsonl \
  --tokenizer_path models/minimind-3 \
  --from_weight agent_sft \
  --from_save_dir out \
  --save_dir "$save_dir" \
  --checkpoint_dir "$checkpoint_dir" \
  --save_weight "$run_name" \
  --metrics_path "$metrics_path" \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 1024 \
  --max_gen_len 384 \
  --max_total_len 2500 \
  --max_turns 3 \
  --batch_size 1 \
  --num_generations 8 \
  --epochs 1 \
  --max_updates 0 \
  --max_candidate_groups 6726 \
  --max_effective_groups 0 \
  --max_generated_tokens 0 \
  --learning_rate 3e-7 \
  --weight_decay 0 \
  --accumulation_steps 1 \
  --policy_update_epochs 2 \
  --loss_type "$algorithm" \
  --beta 0.1 \
  --epsilon 0.2 \
  --epsilon_high 5.0 \
  --dapo_epsilon_low 0.2 \
  --dapo_epsilon_high 0.28 \
  --gspo_epsilon_low 0.0003 \
  --gspo_epsilon_high 0.0004 \
  --overlong_cache_len 128 \
  --overlong_penalty_coef 1.0 \
  --rollout_temperature 1.0 \
  --rollout_top_k 0 \
  --rollout_top_p 1.0 \
  --thinking_ratio 0 \
  --require_tool_call_for_success 1 \
  --reward_mode strict \
  --use_reward_model 0 \
  --checkpoint_dtype float32 \
  --checkpoint_groups 0 \
  --save_resume 1 \
  --from_resume "$resume" \
  --save_interval 100 \
  --log_interval 10 \
  --num_workers 0 \
  --rollout_engine torch \
  --rollout_sync_interval 1 \
  --max_rollout_logprob_mae 0.1 \
  --dtype bfloat16 \
  --device cuda:0 \
  --use_compile 0 \
  --seed "$seed" \
  "${extra_args[@]}" \
  2>&1 | tee "$log_path"
