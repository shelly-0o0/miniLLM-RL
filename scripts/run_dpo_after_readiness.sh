#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/eval out/run_meta
exec > >(tee -a out/logs/dpo_holdout_pipeline.log) 2>&1

echo "[$(date -Is)] waiting for RLVR readiness evaluation"
while ! grep -q 'RLVR readiness complete' out/logs/rlvr_readiness.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_rlvr_readiness'; then
    echo "readiness evaluation exited without a success marker" >&2
    exit 1
  fi
  sleep 30
done

/root/miniconda3/bin/python scripts/prepare_dpo_data.py

cd trainer
SECONDS=0
set -o pipefail
torchrun --standalone --nproc_per_node 1 train_dpo.py \
  --data_path ../dataset/dpo_train.jsonl \
  --save_weight dpo_holdout \
  --from_weight full_sft \
  --from_resume 0 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 768 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --learning_rate 4e-8 \
  --beta 0.15 \
  --dtype bfloat16 \
  --epochs 1 \
  --num_workers 8 \
  --grad_clip 1.0 \
  --log_interval 50 \
  --save_interval 500 \
  --use_compile 0 \
  2>&1 | tee ../out/logs/dpo_holdout_dense.log
status=${PIPESTATUS[0]}
printf 'dpo_exit_code=%s, elapsed_seconds=%s\n' "${status}" "${SECONDS}" | tee -a ../out/logs/dpo_holdout_dense.log
test "${status}" -eq 0

cd ..
/root/miniconda3/bin/python scripts/eval_dpo_holdout.py \
  --data_path dataset/dpo_eval.jsonl \
  --reference_weight full_sft \
  --policy_weight dpo_holdout \
  --max_seq_len 768 \
  --batch_size 16 \
  --limit 1000 \
  --device cuda:0 \
  --output out/eval/dpo_holdout_metrics.json \
  2>&1 | tee out/logs/dpo_holdout_eval.log

sha256sum out/dpo_holdout_768.pth out/eval/dpo_holdout_metrics.json \
  dataset/dpo_train.jsonl dataset/dpo_eval.jsonl \
  > out/run_meta/dpo_holdout_artifacts.sha256
echo "[$(date -Is)] DPO holdout pipeline complete"
