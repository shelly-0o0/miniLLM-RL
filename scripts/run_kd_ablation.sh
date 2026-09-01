#!/usr/bin/env bash
set -u
set -o pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}/trainer" || exit 1
mkdir -p ../out/logs ../out/run_meta
export PYTHONUNBUFFERED=1

run_variant() {
  local name=$1
  shift
  SECONDS=0
  torchrun \
    --standalone \
    --nproc_per_node 1 \
    train_distillation.py \
    --data_path ../dataset/kd_sft_train_50000.jsonl \
    --student_hidden_size 512 \
    --student_num_layers 8 \
    --student_use_moe 0 \
    --teacher_hidden_size 768 \
    --teacher_num_layers 8 \
    --teacher_use_moe 0 \
    --from_student_weight full_sft_student \
    --from_teacher_weight full_sft \
    --from_resume 0 \
    --max_seq_len 512 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --learning_rate 5e-6 \
    --dtype bfloat16 \
    --epochs 1 \
    --num_workers 8 \
    --grad_clip 1.0 \
    --log_interval 50 \
    --save_interval 1000 \
    --seed 42 \
    --use_compile 0 \
    "$@" \
    2>&1 | tee "../out/logs/${name}.log"
  local status=${PIPESTATUS[0]}
  printf '%s_exit_code=%s, elapsed_seconds=%s\n' "${name}" "${status}" "${SECONDS}" \
    | tee -a "../out/logs/${name}.log"
  return "${status}"
}

# Same initial checkpoint, samples, ordering, seed, optimizer budget and
# effective batch size.  Only the objective differs.
run_variant full_ce_student_512 \
  --save_weight full_ce_student \
  --disable_teacher 1 \
  --alpha 1.0 \
  --temperature 1.5 || exit $?

run_variant full_dist_student_512 \
  --save_weight full_dist_student \
  --disable_teacher 0 \
  --alpha 0.5 \
  --temperature 1.5
