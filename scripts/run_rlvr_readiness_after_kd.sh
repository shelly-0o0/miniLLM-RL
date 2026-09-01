#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/eval_rlvr out/run_meta
exec > >(tee -a out/logs/rlvr_readiness.log) 2>&1

echo "[$(date -Is)] waiting for KD pipeline so the single GPU is not contended"
while ! grep -q 'KD pipeline complete' out/logs/kd_pipeline.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_kd_pipeline'; then
    echo "KD pipeline exited without a success marker; RLVR readiness was not started" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] regenerating oracle-backed training-only Agent SFT data"
/root/miniconda3/bin/python scripts/prepare_tool_rlvr_data.py
/root/miniconda3/bin/python scripts/prepare_agent_sft_data.py
/root/miniconda3/bin/python scripts/audit_tool_rlvr_dataset.py
/root/miniconda3/bin/python scripts/audit_sft_mask.py \
  --data_path dataset/agent_sft_coldstart.jsonl \
  --max_seq_len 1024 --samples 128 --examples 3 \
  --output out/run_meta/agent_sft_mask_audit_128.json

if [[ -f out/agent_sft_768.pth ]] \
   && grep -q 'agent_sft_exit_code=0' out/logs/agent_sft_coldstart.log 2>/dev/null; then
  echo "[$(date -Is)] reusing completed common Agent SFT cold-start checkpoint"
else
  echo "[$(date -Is)] training common Agent SFT cold-start checkpoint"
  scripts/run_agent_sft.sh
fi

for task in math tool; do
  if [[ "${task}" == "tool" ]]; then
    data_path="dataset/agent_rl_tool_verified_eval.jsonl"
  else
    data_path="dataset/agent_rl_math_eval.jsonl"
  fi
  for stage in pre_sft post_sft; do
    if [[ "${stage}" == "pre_sft" ]]; then
      weight=full_sft
      reference=full_sft
      output_dir="out/eval_rlvr/readiness_pre_sft_${task}"
    else
      weight=agent_sft
      reference=agent_sft
      output_dir="out/eval_rlvr/readiness_${task}"
    fi
    test ! -e "${output_dir}" || {
      echo "Refusing to overwrite ${output_dir}" >&2
      exit 1
    }
    echo "[$(date -Is)] evaluating ${weight} readiness on ${task}"
    /root/miniconda3/bin/python scripts/eval_agent_rlvr.py \
      --weights "${weight}" \
      --reference_weight "${reference}" \
      --data_path "${data_path}" \
      --output_dir "${output_dir}" \
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
  done
done

sha256sum \
  out/eval_rlvr/readiness_math/summary.csv \
  out/eval_rlvr/readiness_math/trajectories.jsonl \
  out/eval_rlvr/readiness_tool/summary.csv \
  out/eval_rlvr/readiness_tool/trajectories.jsonl \
  out/eval_rlvr/readiness_pre_sft_math/summary.csv \
  out/eval_rlvr/readiness_pre_sft_tool/summary.csv \
  out/agent_sft_768.pth \
  dataset/agent_sft_coldstart.jsonl \
  > out/run_meta/rlvr_readiness.sha256
echo "[$(date -Is)] RLVR readiness complete"
