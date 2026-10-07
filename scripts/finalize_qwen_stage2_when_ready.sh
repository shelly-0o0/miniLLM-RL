#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: bash scripts/finalize_qwen_stage2_when_ready.sh {track1|track2} [gpu_index] [poll_seconds]" >&2
  exit 2
fi

mode="$1"
gpu_index="${2:-2}"
poll_seconds="${3:-60}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

case "$mode" in
  track1)
    sessions=(track1-pure-full-bounded-kl track1-sft-grpo-formal)
    artifacts=(
      out/stage2/qwen3_4b/pure_grpo_s42_adapter/adapter_model.safetensors
      out/stage2/qwen3_4b/sft_grpo_s42_adapter/adapter_model.safetensors
    )
    validation_summary=out/stage2/qwen3_4b/evaluation/validation/summary.json
    test_summary=out/stage2/qwen3_4b/evaluation/test/summary.json
    ;;
  track2)
    sessions=(track2-grpo-a-formal track2-grpo-b-formal)
    artifacts=(
      out/stage2_track2/qwen3_4b/grpo_a_s42_adapter/adapter_model.safetensors
      out/stage2_track2/qwen3_4b/grpo_b_s42_adapter/adapter_model.safetensors
    )
    test_summary=out/stage2_track2/qwen3_4b/evaluation/test/summary.json
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] $mode finalizer started; evaluation GPU=$gpu_index"

while true; do
  waiting=0
  for index in "${!artifacts[@]}"; do
    artifact="${artifacts[$index]}"
    session="${sessions[$index]}"
    if [[ -f "$artifact" ]]; then
      continue
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "[$(date -Is)] ERROR: $session stopped before producing $artifact" >&2
      exit 1
    fi
    waiting=1
  done
  if (( waiting == 0 )); then
    # Adapter saving precedes the terminal metrics write by a short interval.
    # Wait for both owning tmux sessions to close before auditing/evaluating.
    still_running=0
    for session in "${sessions[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        still_running=1
      fi
    done
    if (( still_running == 0 )); then
      break
    fi
  fi
  echo "[$(date -Is)] $mode training is still in progress"
  sleep "$poll_seconds"
done

echo "[$(date -Is)] both $mode training artifacts are complete"

# Track 1 and Track 2 may finish at different times.  This host-local lock keeps
# their long official evaluations from accidentally sharing the selected GPU.
lock_path="/tmp/mini_rl_stage2_eval_gpu_${gpu_index}.lock"
exec 9>"$lock_path"
echo "[$(date -Is)] waiting for evaluation lock $lock_path"
flock 9
export CUDA_VISIBLE_DEVICES="$gpu_index"
echo "[$(date -Is)] acquired evaluation lock"

if [[ "$mode" == track1 ]]; then
  if [[ ! -f "$validation_summary" ]]; then
    if [[ -f "${validation_summary%/*}/manifest.json" ]]; then
      echo "partial validation evaluation exists; inspect/archive it before retrying" >&2
      exit 1
    fi
    bash scripts/run_qwen_stage2.sh eval_validation
  fi
  bash scripts/run_qwen_stage2.sh audit_validation_results
  if [[ ! -f "$test_summary" ]]; then
    if [[ -f "${test_summary%/*}/manifest.json" ]]; then
      echo "partial test evaluation exists; inspect/archive it before retrying" >&2
      exit 1
    fi
    bash scripts/run_qwen_stage2.sh eval_test
  fi
  bash scripts/run_qwen_stage2.sh audit_results
else
  if [[ ! -f "$test_summary" ]]; then
    if [[ -f "${test_summary%/*}/manifest.json" ]]; then
      echo "partial test evaluation exists; inspect/archive it before retrying" >&2
      exit 1
    fi
    bash scripts/run_qwen_track2.sh eval_test
  fi
  bash scripts/run_qwen_track2.sh audit_results
fi

echo "[$(date -Is)] $mode final evaluation and fail-closed audit completed"
