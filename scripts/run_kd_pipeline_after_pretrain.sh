#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/eval out/run_meta
exec > >(tee -a out/logs/kd_pipeline.log) 2>&1

echo "[$(date -Is)] waiting for successful student pretraining"
while ! grep -q 'student_pretrain_exit_code=0' out/logs/pretrain_student_512.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_student_pretrain'; then
    echo "student pretraining screen exited without a success marker" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] student pretraining complete; starting SFT"
if grep -q 'student_sft_exit_code=0' out/logs/full_sft_student_512.log 2>/dev/null; then
  echo "student SFT already has a success marker; reusing completed checkpoint"
elif [[ -f checkpoints/full_sft_student_512_resume.pth ]]; then
  echo "resuming interrupted student SFT from its persisted checkpoint"
  FROM_RESUME=1 scripts/run_student_sft.sh
else
  scripts/run_student_sft.sh
fi

echo "[$(date -Is)] preparing controlled KD subsets"
/root/miniconda3/bin/python scripts/prepare_kd_data.py

echo "[$(date -Is)] running CE versus KD controlled continuation"
scripts/run_kd_ablation.sh

echo "[$(date -Is)] evaluating teacher and all student variants"
/root/miniconda3/bin/python scripts/eval_distillation.py \
  --eval_data dataset/kd_rlaif_eval_2000.jsonl \
  --teacher out/full_sft_768.pth \
  --student_init out/full_sft_student_512.pth \
  --ce_student out/full_ce_student_512.pth \
  --kd_student out/full_dist_student_512.pth \
  --max_seq_len 512 \
  --batch_size 16 \
  --num_workers 4 \
  --seed 42 \
  --device cuda:0 \
  --dtype bfloat16 \
  --output out/eval/kd_comparison.json \
  2>&1 | tee out/logs/kd_evaluation.log

sha256sum \
  out/pretrain_student_512.pth \
  out/full_sft_student_512.pth \
  out/full_ce_student_512.pth \
  out/full_dist_student_512.pth \
  out/eval/kd_comparison.json \
  > out/run_meta/kd_artifacts.sha256

echo "[$(date -Is)] KD pipeline complete"
