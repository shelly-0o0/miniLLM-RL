#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_qwen_stage2.sh {smoke_sft|sft|calibrate_lm_head_w2|calibrate_lm_head_w4|calibrate_lm_head_w2_100|calibrate_lm_head_w2_200|audit_lm_head_w2|audit_lm_head_w4|audit_lm_head_w2_100|audit_lm_head_w2_200|lm_head_sft|audit_lm_head_sft|probe_base_smoke|probe_sft_smoke|probe_base|probe_sft|smoke_pure_grpo|smoke_sft_grpo|pilot_pure_grpo|pilot_sft_grpo|pure_grpo|sft_grpo|eval_validation|eval_test|audit_validation_results|audit_results} [--resume]" >&2
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
mkdir -p out/logs out/metrics out/stage2 checkpoints/stage2

require_original_sft() {
  [[ -f out/stage2/qwen3_4b/sft_adapter/adapter_config.json ]] || {
    echo "missing original full SFT adapter: run sft first" >&2
    exit 1
  }
}

require_repaired_sft() {
  [[ -f out/stage2/qwen3_4b/sft_lm_head_w8_s42_adapter/adapter_config.json ]] || {
    echo "missing repaired full SFT adapter: run lm_head_sft first" >&2
    exit 1
  }
}

require_no_resume() {
  [[ -z "$resume_flag" ]] || {
    echo "$phase does not accept --resume" >&2
    exit 2
  }
}

case "$phase" in
  smoke_sft)
    command=(python -u trainer/train_qwen_lora_sft.py --config configs/qwen3_4b/smoke_sft.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_smoke_sft_s42.log"
    ;;
  smoke_pure_grpo)
    command=(python -u trainer/train_qwen_grpo.py --config configs/qwen3_4b/smoke_pure_grpo.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_smoke_pure_grpo_s42.log"
    ;;
  smoke_sft_grpo)
    [[ -f out/stage2/qwen3_4b/smoke_sft_adapter/adapter_config.json ]] || {
      echo "missing smoke SFT adapter: run smoke_sft first" >&2
      exit 1
    }
    command=(python -u trainer/train_qwen_grpo.py --config configs/qwen3_4b/smoke_sft_grpo.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_smoke_sft_grpo_s42.log"
    ;;
  sft)
    command=(python -u trainer/train_qwen_lora_sft.py --config configs/qwen3_4b/lora_sft.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_sft_s42.log"
    ;;
  calibrate_lm_head_w2|calibrate_lm_head_w4)
    require_no_resume
    require_original_sft
    weight="${phase##*_w}"
    tag="calib_w${weight}_50"
    command=(python -u trainer/train_qwen_lora_sft.py
      --config configs/qwen3_4b/lora_sft_lm_head_calibration.yaml
      --max-steps 50
      --structure-token-weight "$weight"
      --output-tag "$tag")
    log_path="out/logs/qwen3_4b_sft_lm_head_${tag}.log"
    ;;
  calibrate_lm_head_w2_100|calibrate_lm_head_w2_200)
    require_no_resume
    require_original_sft
    steps="${phase##*_}"
    tag="calib_w2_${steps}"
    command=(python -u trainer/train_qwen_lora_sft.py
      --config configs/qwen3_4b/lora_sft_lm_head_calibration.yaml
      --max-steps "$steps"
      --structure-token-weight 2
      --output-tag "$tag")
    log_path="out/logs/qwen3_4b_sft_lm_head_${tag}.log"
    ;;
  audit_lm_head_w2|audit_lm_head_w4)
    require_no_resume
    weight="${phase##*_w}"
    tag="calib_w${weight}_50"
    after="out/stage2/qwen3_4b/sft_lm_head_w4_adapter_${tag}"
    [[ -f "$after/adapter_config.json" ]] || {
      echo "missing calibration adapter: $after" >&2
      exit 1
    }
    command=(python -u scripts/audit_qwen_sft_structure.py
      --config configs/qwen3_4b/lora_sft_lm_head_calibration.yaml
      --before-adapter out/stage2/qwen3_4b/sft_adapter
      --after-adapter "$after"
      --output "out/run_meta/qwen3_4b_sft_lm_head_${tag}_audit.json")
    log_path="out/logs/qwen3_4b_sft_lm_head_${tag}_audit.log"
    ;;
  audit_lm_head_w2_100|audit_lm_head_w2_200)
    require_no_resume
    steps="${phase##*_}"
    tag="calib_w2_${steps}"
    after="out/stage2/qwen3_4b/sft_lm_head_w4_adapter_${tag}"
    [[ -f "$after/adapter_config.json" ]] || {
      echo "missing calibration adapter: $after" >&2
      exit 1
    }
    command=(python -u scripts/audit_qwen_sft_structure.py
      --config configs/qwen3_4b/lora_sft_lm_head_calibration.yaml
      --before-adapter out/stage2/qwen3_4b/sft_adapter
      --after-adapter "$after"
      --output "out/run_meta/qwen3_4b_sft_lm_head_${tag}_audit.json")
    log_path="out/logs/qwen3_4b_sft_lm_head_${tag}_audit.log"
    ;;
  lm_head_sft)
    require_original_sft
    command=(python -u trainer/train_qwen_lora_sft.py
      --config configs/qwen3_4b/lora_sft_lm_head.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_sft_lm_head_w8_s42.log"
    ;;
  audit_lm_head_sft)
    require_no_resume
    require_repaired_sft
    command=(python -u scripts/audit_qwen_sft_structure.py
      --config configs/qwen3_4b/lora_sft_lm_head.yaml
      --before-adapter out/stage2/qwen3_4b/sft_adapter
      --after-adapter out/stage2/qwen3_4b/sft_lm_head_w8_s42_adapter
      --output out/run_meta/qwen3_4b_sft_lm_head_w8_s42_audit.json)
    log_path="out/logs/qwen3_4b_sft_lm_head_w8_s42_audit.log"
    ;;
  probe_base_smoke|probe_sft_smoke|probe_base|probe_sft)
    require_no_resume
    probe_args=(--config configs/qwen3_4b/pre_grpo_probe.yaml --pool validation)
    if [[ "$phase" == probe_base* ]]; then
      probe_args+=(--base-model)
      label="base"
    else
      require_repaired_sft
      label="sft"
    fi
    if [[ "$phase" == *_smoke ]]; then
      probe_args+=(--limit 2 --num-generations 2 --output-tag "${label}_smoke2")
      log_path="out/logs/qwen3_4b_probe_${label}_smoke2.log"
    else
      probe_args+=(--limit 128 --output-tag "${label}_128")
      log_path="out/logs/qwen3_4b_probe_${label}_128.log"
    fi
    command=(python -u scripts/evaluate/probe_qwen_track2.py "${probe_args[@]}")
    ;;
  pilot_pure_grpo)
    command=(python -u trainer/train_qwen_grpo.py --config configs/qwen3_4b/pure_grpo.yaml --max-candidate-groups 4 --output-tag pilot4)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_pure_grpo_s42_pilot4.log"
    ;;
  pilot_sft_grpo)
    require_repaired_sft
    command=(python -u trainer/train_qwen_grpo.py --config configs/qwen3_4b/sft_grpo.yaml --max-candidate-groups 4 --output-tag pilot4)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_sft_grpo_s42_pilot4.log"
    ;;
  pure_grpo)
    command=(python -u trainer/train_qwen_grpo.py --config configs/qwen3_4b/pure_grpo.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_pure_grpo_s42.log"
    ;;
  sft_grpo)
    require_repaired_sft
    command=(python -u trainer/train_qwen_grpo.py --config configs/qwen3_4b/sft_grpo.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/qwen3_4b_sft_grpo_s42.log"
    ;;
  eval_validation)
    [[ -z "$resume_flag" ]] || { echo "evaluation does not accept --resume" >&2; exit 2; }
    command=(python -u scripts/evaluate/evaluate_qwen_stage2.py --config configs/qwen3_4b/eval_matrix.yaml --split validation)
    log_path="out/logs/qwen3_4b_eval_validation.log"
    ;;
  eval_test)
    [[ -z "$resume_flag" ]] || { echo "evaluation does not accept --resume" >&2; exit 2; }
    command=(python -u scripts/evaluate/evaluate_qwen_stage2.py --config configs/qwen3_4b/eval_matrix.yaml --split test --limit 0)
    log_path="out/logs/qwen3_4b_eval_test.log"
    ;;
  audit_validation_results)
    require_no_resume
    command=(python -u scripts/audit_qwen_stage2_results.py)
    log_path="out/logs/qwen3_4b_results_audit_validation.log"
    ;;
  audit_results)
    require_no_resume
    command=(python -u scripts/audit_qwen_stage2_results.py --require-test)
    log_path="out/logs/qwen3_4b_results_audit_final.log"
    ;;
  *)
    echo "unknown Stage 2 phase: $phase" >&2
    exit 2
    ;;
esac

PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee "$log_path"
