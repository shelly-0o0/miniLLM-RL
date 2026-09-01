#!/usr/bin/env bash
set -u
set -o pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}/trainer" || exit 1
mkdir -p ../out/logs ../out/run_meta
export PYTHONUNBUFFERED=1

SECONDS=0
torchrun --standalone --nproc_per_node 1 train_full_sft.py \
  --data_path ../dataset/agent_sft_coldstart.jsonl \
  --save_weight agent_sft \
  --from_weight full_sft \
  --from_resume 0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 1024 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --learning_rate 1e-5 \
  --dtype bfloat16 \
  --epochs 3 \
  --num_workers 4 \
  --grad_clip 1.0 \
  --log_interval 20 \
  --save_interval 1000 \
  --use_compile 0 \
  2>&1 | tee ../out/logs/agent_sft_coldstart.log
status=${PIPESTATUS[0]}
printf 'agent_sft_exit_code=%s, elapsed_seconds=%s\n' "${status}" "${SECONDS}" \
  | tee -a ../out/logs/agent_sft_coldstart.log
exit "${status}"
