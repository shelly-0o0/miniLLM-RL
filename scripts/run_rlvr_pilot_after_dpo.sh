#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/metrics out/eval_rlvr/pilot out/run_meta
exec > >(tee -a out/logs/rlvr_pilot.log) 2>&1

echo "[$(date -Is)] waiting for DPO branch so the single GPU is not contended"
while ! grep -q 'DPO holdout pipeline complete' out/logs/dpo_holdout_pipeline.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_dpo_holdout'; then
    echo "DPO pipeline exited without a success marker" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] preparing and auditing non-saturated compositional Tool-Use challenge"
/root/miniconda3/bin/python scripts/prepare_tool_rlvr_challenge_data.py
/root/miniconda3/bin/python scripts/audit_tool_rlvr_dataset.py \
  --train dataset/agent_rl_tool_challenge_train.jsonl \
  --eval dataset/agent_rl_tool_challenge_eval.jsonl \
  --output out/run_meta/tool_rlvr_challenge_execution_audit.json

challenge_readiness="out/eval_rlvr/readiness_tool_challenge_train"
if [[ ! -f "${challenge_readiness}/summary.csv" ]]; then
  echo "[$(date -Is)] calibrating Agent-SFT on challenge training prompts (not held-out test)"
  /root/miniconda3/bin/python scripts/eval_agent_rlvr.py \
    --weights agent_sft \
    --reference_weight agent_sft \
    --data_path dataset/agent_rl_tool_challenge_train.jsonl \
    --output_dir "${challenge_readiness}" \
    --seeds 101 \
    --limit 32 \
    --batch_size 2 \
    --num_generations 4 \
    --max_turns 3 \
    --max_gen_len 128 \
    --max_total_len 1536 \
    --thinking_ratio 0 \
    --reward_mode strict \
    --require_tool_call_for_success 1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --use_moe 0 \
    --device cuda:0
fi

for task in math tool; do
  if [[ "${task}" == "tool" ]]; then
    readiness="${challenge_readiness}/summary.csv"
  else
    readiness="out/eval_rlvr/readiness_${task}/summary.csv"
  fi
  effective=$(/root/miniconda3/bin/python -c 'import csv,sys; print(float(next(csv.DictReader(open(sys.argv[1], encoding="utf-8")))["dapo_effective_group_rate"]))' "${readiness}")
  accuracy=$(/root/miniconda3/bin/python -c 'import csv,sys; print(float(next(csv.DictReader(open(sys.argv[1], encoding="utf-8")))["task_accuracy"]))' "${readiness}")
  echo "task=${task}, baseline_accuracy=${accuracy}, dapo_effective_group_rate=${effective}"
  /root/miniconda3/bin/python -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)' "${effective}" || {
    echo "SKIP ${task}: no mixed-success group; cold-start SFT/curriculum is required before policy optimization"
    continue
  }

  if [[ "${task}" == "tool" ]]; then
    train_data="../dataset/agent_rl_tool_challenge_train.jsonl"
    eval_data="dataset/agent_rl_tool_challenge_eval.jsonl"
  else
    train_data="../dataset/agent_rl_math_train.jsonl"
    eval_data="dataset/agent_rl_math_eval.jsonl"
  fi
  # Evaluate the untouched common initialization on the exact same prompts
  # and decode seed, so a pilot delta is not inferred from the differently
  # configured readiness run.
  weights=("agent_sft")
  for algorithm in grpo cispo dapo gspo; do
    weight="agent_pilot_${task}_${algorithm}_s42"
    metrics="../out/metrics/pilot_${task}_${algorithm}_s42.jsonl"
    test ! -e "out/${weight}_768.pth" || {
      echo "Refusing to overwrite out/${weight}_768.pth" >&2
      exit 1
    }
    extra=()
    if [[ "${algorithm}" == "dapo" ]]; then
      extra+=(--dynamic_sampling --dynamic_sampling_rounds 50)
      extra+=(--dapo_epsilon_low 0.2 --dapo_epsilon_high 0.28)
      extra+=(--overlong_cache_len 96 --overlong_penalty_coef 1.0)
    elif [[ "${algorithm}" == "gspo" ]]; then
      extra+=(--gspo_epsilon_low 0.0003 --gspo_epsilon_high 0.0004)
    fi
    echo "[$(date -Is)] pilot ${task}/${algorithm}"
    (
      cd trainer
      torchrun --standalone --nproc_per_node 1 train_agent.py \
        --loss_type "${algorithm}" \
        --save_weight "${weight}" \
        --from_weight agent_sft \
        --from_resume 0 \
        --data_path "${train_data}" \
        --metrics_path "${metrics}" \
        --seed 42 \
        --epochs 1 \
        --max_updates 3 \
        --batch_size 1 \
        --num_generations 4 \
        --policy_update_epochs 2 \
        --learning_rate 1e-7 \
        --rollout_sync_interval 1 \
        --hidden_size 768 \
        --num_hidden_layers 8 \
        --max_gen_len 96 \
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
        --log_interval 1 \
        --save_interval 3 \
        --save_resume 0 \
        --max_rollout_logprob_mae 0.1 \
        "${extra[@]}"
    ) 2>&1 | tee "out/logs/pilot_${task}_${algorithm}_s42.log"
    weights+=("${weight}")
  done

  weight_csv=$(IFS=,; echo "${weights[*]}")
  output_dir="out/eval_rlvr/pilot/${task}"
  test ! -e "${output_dir}/summary.csv" || {
    echo "Refusing to overwrite ${output_dir}" >&2
    exit 1
  }
  /root/miniconda3/bin/python scripts/eval_agent_rlvr.py \
    --weights "${weight_csv}" \
    --reference_weight agent_sft \
    --data_path "${eval_data}" \
    --output_dir "${output_dir}" \
    --seeds 201 \
    --limit 16 \
    --batch_size 2 \
    --num_generations 1 \
    --max_turns 3 \
    --max_gen_len 96 \
    --max_total_len 1024 \
    --thinking_ratio 0 \
    --reward_mode strict \
    --require_tool_call_for_success 1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --use_moe 0 \
    --device cuda:0
done

sha256sum out/metrics/pilot_*.jsonl out/eval_rlvr/pilot/*/summary.csv \
  dataset/agent_rl_tool_challenge_train.jsonl \
  dataset/agent_rl_tool_challenge_eval.jsonl \
  out/run_meta/agent_tool_challenge_manifest.json \
  out/run_meta/tool_rlvr_challenge_execution_audit.json \
  > out/run_meta/rlvr_pilot_artifacts.sha256 2>/dev/null || true
echo "[$(date -Is)] RLVR pilot complete"
