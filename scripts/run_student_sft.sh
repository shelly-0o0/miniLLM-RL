#!/usr/bin/env bash
set -u
set -o pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}/trainer" || exit 1
mkdir -p ../out/logs ../out/run_meta
export PYTHONUNBUFFERED=1
SECONDS=0
FROM_RESUME="${FROM_RESUME:-0}"

torchrun \
  --standalone \
  --nproc_per_node 1 \
  train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --save_weight full_sft_student \
  --from_weight pretrain_student \
  --from_resume "${FROM_RESUME}" \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 768 \
  --batch_size 16 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --dtype bfloat16 \
  --epochs 1 \
  --num_workers 8 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --save_interval 5000 \
  --use_compile 0 \
  2>&1 | tee -a ../out/logs/full_sft_student_512.log

status=${PIPESTATUS[0]}
printf 'student_sft_exit_code=%s, elapsed_seconds=%s\n' "${status}" "${SECONDS}" \
  | tee -a ../out/logs/full_sft_student_512.log
exit "${status}"
