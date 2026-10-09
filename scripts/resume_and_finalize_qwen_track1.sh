#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${TRACK1_PYTHON:-python}"
pure_gpu="${TRACK1_PURE_GPU:-3}"
mkdir -p out/logs

aux_pids=()

start_shard() {
  local split="$1"
  local label="$2"
  local gpu="$3"
  local limit_arg="${4:-}"
  local shard_dir="out/stage2/qwen3_4b/evaluation/${split}_shards/${label}"
  local summary_path="${shard_dir}/summary.json"
  local log_path="out/logs/qwen3_4b_eval_${split}_${label}.log"

  if [[ -f "$summary_path" ]]; then
    echo "[$(date -Is)] reuse completed shard: ${split}/${label}"
    return
  fi
  if [[ -e "$shard_dir" ]]; then
    echo "partial shard exists; inspect/archive before retrying: $shard_dir" >&2
    exit 1
  fi

  echo "[$(date -Is)] launch ${split}/${label} on physical GPU ${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    command=(
      "$python_bin" -u scripts/evaluate/evaluate_qwen_stage2.py
      --config configs/qwen3_4b/eval_matrix.yaml
      --split "$split"
      --run-label "$label"
    )
    [[ -n "$limit_arg" ]] && command+=(--limit "$limit_arg")
    "${command[@]}" 2>&1 | tee "$log_path"
  ) &
  aux_pids+=("$!")
}

run_pure_shard() {
  local split="$1"
  local limit_arg="${2:-}"
  local shard_dir="out/stage2/qwen3_4b/evaluation/${split}_shards/pure_grpo"
  local summary_path="${shard_dir}/summary.json"
  local log_path="out/logs/qwen3_4b_eval_${split}_pure_grpo.log"

  if [[ -f "$summary_path" ]]; then
    echo "[$(date -Is)] reuse completed shard: ${split}/pure_grpo"
    return
  fi
  if [[ -e "$shard_dir" ]]; then
    echo "partial shard exists; inspect/archive before retrying: $shard_dir" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="$pure_gpu"
  command=(
    "$python_bin" -u scripts/evaluate/evaluate_qwen_stage2.py
    --config configs/qwen3_4b/eval_matrix.yaml
    --split "$split"
    --run-label pure_grpo
  )
  [[ -n "$limit_arg" ]] && command+=(--limit "$limit_arg")
  "${command[@]}" 2>&1 | tee "$log_path"
}

# Finish every arm that does not depend on the Pure terminal adapter while the
# resumed Pure run uses GPU 3. The existing completed sft_grpo test shard is
# reused after its manifest is revalidated by the atomic merger.
start_shard validation base 4
start_shard validation sft_only 5
start_shard validation sft_grpo 6
start_shard test base 1 0
start_shard test sft_only 2 0

export CUDA_VISIBLE_DEVICES="$pure_gpu"
bash scripts/run_qwen_stage2.sh pure_grpo --resume

run_pure_shard validation
run_pure_shard test 0

for pid in "${aux_pids[@]}"; do
  wait "$pid"
done

"$python_bin" -u scripts/evaluate/merge_qwen_stage2_shards.py \
  --config configs/qwen3_4b/eval_matrix.yaml --split validation
bash scripts/run_qwen_stage2.sh audit_validation_results

"$python_bin" -u scripts/evaluate/merge_qwen_stage2_shards.py \
  --config configs/qwen3_4b/eval_matrix.yaml --split test --limit 0
bash scripts/run_qwen_stage2.sh audit_results

echo "[$(date -Is)] Track 1 training, four-arm evaluation, merge, and audit complete"
