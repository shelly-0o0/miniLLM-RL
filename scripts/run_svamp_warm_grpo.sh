#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_svamp_warm_grpo.sh {download|prepare|audit|probe|grpo|eval_zero_shot|eval_grpo|merge_eval|audit_results} [--resume]" >&2
  exit 2
fi

phase="$1"
resume_flag="${2:-}"
if [[ -n "$resume_flag" && "$resume_flag" != "--resume" ]]; then
  echo "second argument must be --resume" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p out/logs out/metrics out/run_meta out/stage2_svamp checkpoints/stage2_svamp

require_no_resume() {
  [[ -z "$resume_flag" ]] || { echo "$phase does not accept --resume" >&2; exit 2; }
}

case "$phase" in
  download)
    require_no_resume
    command=(python -u scripts/download/download_svamp.py)
    log_path=out/logs/svamp_download.log
    ;;
  prepare)
    require_no_resume
    command=(python -u scripts/prepare/prepare_svamp_agent.py)
    log_path=out/logs/svamp_prepare.log
    ;;
  audit)
    require_no_resume
    command=(python -u scripts/audit_svamp_warm_grpo.py)
    log_path=out/logs/svamp_readiness_audit.log
    ;;
  probe)
    require_no_resume
    command=(python -u scripts/evaluate/probe_qwen_track2.py
      --config configs/qwen3_4b/svamp/pre_grpo_probe.yaml
      --pool svamp)
    log_path=out/logs/svamp_pre_grpo_probe.log
    ;;
  grpo)
    command=(python -u trainer/train_qwen_grpo.py
      --config configs/qwen3_4b/svamp/grpo_from_additional_sft_b.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path=out/logs/svamp_grpo_s42.log
    ;;
  eval_zero_shot)
    require_no_resume
    command=(python -u scripts/evaluate/evaluate_qwen_stage2.py
      --config configs/qwen3_4b/svamp/eval_holdout.yaml
      --split test --limit 0 --run-label additional_sft_b_zero_shot)
    log_path=out/logs/svamp_eval_zero_shot.log
    ;;
  eval_grpo)
    require_no_resume
    [[ -f out/stage2_svamp/qwen3_4b/grpo_s42_adapter/adapter_model.safetensors ]] || {
      echo "missing completed SVAMP GRPO adapter" >&2
      exit 1
    }
    command=(python -u scripts/evaluate/evaluate_qwen_stage2.py
      --config configs/qwen3_4b/svamp/eval_holdout.yaml
      --split test --limit 0 --run-label svamp_grpo)
    log_path=out/logs/svamp_eval_grpo.log
    ;;
  merge_eval)
    require_no_resume
    command=(python -u scripts/evaluate/merge_qwen_stage2_shards.py
      --config configs/qwen3_4b/svamp/eval_holdout.yaml --split test)
    log_path=out/logs/svamp_eval_merge.log
    ;;
  audit_results)
    require_no_resume
    command=(python -u scripts/audit_svamp_warm_grpo.py --require-results)
    log_path=out/logs/svamp_results_audit.log
    ;;
  *)
    echo "unknown SVAMP phase: $phase" >&2
    exit 2
    ;;
esac

PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee "$log_path"
