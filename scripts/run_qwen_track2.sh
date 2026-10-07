#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_qwen_track2.sh {prepare|audit|smoke_sft_a|sft_a|smoke_repair_sft_a|repair_sft_a|audit_repair_sft|smoke_lm_head_sft_a|lm_head_sft_a|audit_lm_head_smoke|audit_lm_head_sft|probe_lm_head_smoke|probe_lm_head|probe_smoke|pre_grpo_probe|probe_repair|smoke_additional_sft_b|additional_sft_b|smoke_shaped_grpo_lm_head_a|smoke_grpo_a|smoke_grpo_b|pilot_grpo_a|pilot_grpo_b|grpo_a|grpo_b|eval_test|audit_results} [--resume]" >&2
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
mkdir -p out/logs out/metrics out/run_meta out/stage2_track2 checkpoints/stage2_track2

require_no_resume() {
  if [[ -n "$resume_flag" ]]; then
    echo "$phase does not accept --resume" >&2
    exit 2
  fi
}

require_sft_a() {
  if [[ ! -f out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter/adapter_config.json ]]; then
    echo "missing Agent-SFT(A) adapter: run sft_a first" >&2
    exit 1
  fi
}

require_smoke_sft_a() {
  if [[ ! -f out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter_smoke/adapter_config.json ]]; then
    echo "missing smoke Agent-SFT(A) adapter: run smoke_sft_a first" >&2
    exit 1
  fi
}

require_repair_sft_a() {
  if [[ ! -f out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter_structure_w8/adapter_config.json ]]; then
    echo "missing structure-weighted Agent-SFT(A) adapter: run repair_sft_a first" >&2
    exit 1
  fi
}

require_lm_head_sft_a() {
  if [[ ! -f out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter/adapter_config.json ]]; then
    echo "missing lm_head Agent-SFT repair adapter: run lm_head_sft_a first" >&2
    exit 1
  fi
}

require_lm_head_smoke() {
  if [[ ! -f out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter_smoke10/adapter_config.json ]]; then
    echo "missing lm_head smoke adapter: run smoke_lm_head_sft_a first" >&2
    exit 1
  fi
}

case "$phase" in
  prepare)
    require_no_resume
    command=(python -u scripts/prepare/prepare_gsm8k_track2.py)
    log_path="out/logs/track2_prepare.log"
    ;;
  audit)
    require_no_resume
    command=(python -u scripts/audit_qwen_track2.py)
    log_path="out/logs/track2_audit.log"
    ;;
  smoke_sft_a)
    command=(python -u trainer/train_qwen_lora_sft.py --config configs/qwen3_4b/track2/agent_sft_a.yaml --max-steps 2 --output-tag smoke)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/track2_agent_sft_a_smoke.log"
    ;;
  sft_a)
    command=(python -u trainer/train_qwen_lora_sft.py --config configs/qwen3_4b/track2/agent_sft_a.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/track2_agent_sft_a_s42.log"
    ;;
  smoke_repair_sft_a|repair_sft_a)
    require_sft_a
    repair_tag="structure_w8"
    repair_extra=()
    if [[ "$phase" == smoke_* ]]; then
      repair_tag="structure_w8_smoke"
      repair_extra=(--max-steps 2)
      log_path="out/logs/track2_agent_sft_a_structure_w8_smoke.log"
    else
      log_path="out/logs/track2_agent_sft_a_structure_w8_s42.log"
    fi
    command=(python -u trainer/train_qwen_lora_sft.py
      --config configs/qwen3_4b/track2/agent_sft_a.yaml
      --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter
      --structure-token-weight 8
      --output-tag "$repair_tag" "${repair_extra[@]}")
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    ;;
  audit_repair_sft)
    require_no_resume
    require_repair_sft_a
    command=(python -u scripts/audit_qwen_sft_structure.py)
    log_path="out/logs/track2_qwen_sft_structure_w8_audit.log"
    ;;
  smoke_lm_head_sft_a|lm_head_sft_a)
    require_sft_a
    command=(python -u trainer/train_qwen_lora_sft.py
      --config configs/qwen3_4b/track2/agent_sft_a_lm_head.yaml)
    if [[ "$phase" == smoke_* ]]; then
      command+=(--max-steps 10 --output-tag smoke10)
      log_path="out/logs/track2_agent_sft_a_lm_head_w8_smoke10.log"
    else
      log_path="out/logs/track2_agent_sft_a_lm_head_w8_s42.log"
    fi
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    ;;
  audit_lm_head_smoke)
    require_no_resume
    require_lm_head_smoke
    command=(python -u scripts/audit_qwen_sft_structure.py
      --after-adapter out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter_smoke10
      --limit 32
      --output out/run_meta/track2_qwen_sft_lm_head_w8_smoke10_comparison.json)
    log_path="out/logs/track2_qwen_sft_lm_head_w8_smoke10_audit.log"
    ;;
  audit_lm_head_sft)
    require_no_resume
    require_lm_head_sft_a
    command=(python -u scripts/audit_qwen_sft_structure.py
      --after-adapter out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter
      --output out/run_meta/track2_qwen_sft_lm_head_w8_comparison.json)
    log_path="out/logs/track2_qwen_sft_lm_head_w8_audit.log"
    ;;
  probe_smoke)
    require_no_resume
    require_smoke_sft_a
    command=(python -u scripts/evaluate/probe_qwen_track2.py --limit 2 --num-generations 2 --output-tag smoke --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter_smoke)
    log_path="out/logs/track2_pre_grpo_probe_smoke.log"
    ;;
  pre_grpo_probe)
    require_no_resume
    require_lm_head_sft_a
    command=(python -u scripts/evaluate/probe_qwen_track2.py)
    log_path="out/logs/track2_pre_grpo_probe_s42.log"
    ;;
  probe_repair)
    require_no_resume
    require_repair_sft_a
    command=(python -u scripts/evaluate/probe_qwen_track2.py
      --pool a --limit 16 --num-generations 4
      --output-tag structure_w8
      --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter_structure_w8)
    log_path="out/logs/track2_probe_structure_w8_s42.log"
    ;;
  probe_lm_head)
    require_no_resume
    require_lm_head_sft_a
    command=(python -u scripts/evaluate/probe_qwen_track2.py
      --pool a --limit 16 --num-generations 4
      --output-tag lm_head_w8
      --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter)
    log_path="out/logs/track2_probe_lm_head_w8_s42.log"
    ;;
  probe_lm_head_smoke)
    require_no_resume
    require_lm_head_sft_a
    command=(python -u scripts/evaluate/probe_qwen_track2.py
      --pool a --limit 2 --num-generations 2
      --output-tag lm_head_w8_smoke2
      --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter)
    log_path="out/logs/track2_probe_lm_head_w8_smoke2_s42.log"
    ;;
  smoke_additional_sft_b)
    require_smoke_sft_a
    command=(python -u trainer/train_qwen_lora_sft.py --config configs/qwen3_4b/track2/additional_sft_b.yaml --max-steps 2 --output-tag smoke --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter_smoke)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/track2_additional_sft_b_smoke.log"
    ;;
  additional_sft_b)
    require_lm_head_sft_a
    command=(python -u trainer/train_qwen_lora_sft.py --config configs/qwen3_4b/track2/additional_sft_b.yaml)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/track2_additional_sft_b_s42.log"
    ;;
  smoke_shaped_grpo_lm_head_a)
    require_no_resume
    require_lm_head_sft_a
    command=(python -u trainer/train_qwen_grpo.py
      --config configs/qwen3_4b/track2/grpo_a.yaml
      --max-candidate-groups 2
      --output-tag lm_head_shaped_pilot2
      --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_lm_head_w8_s42_adapter
      --reward-mode shaped)
    log_path="out/logs/track2_grpo_a_lm_head_shaped_pilot2_s42.log"
    ;;
  smoke_grpo_a|smoke_grpo_b)
    require_smoke_sft_a
    pool="${phase##*_}"
    command=(python -u trainer/train_qwen_grpo.py --config "configs/qwen3_4b/track2/grpo_${pool}.yaml" --max-candidate-groups 2 --output-tag smoke --adapter-path out/stage2_track2/qwen3_4b/agent_sft_a_s42_adapter_smoke)
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/track2_${phase}_s42.log"
    ;;
  pilot_grpo_a|pilot_grpo_b|grpo_a|grpo_b)
    require_lm_head_sft_a
    pool="${phase##*_}"
    command=(python -u trainer/train_qwen_grpo.py --config "configs/qwen3_4b/track2/grpo_${pool}.yaml")
    if [[ "$phase" == pilot_* ]]; then
      command+=(--max-candidate-groups 4 --output-tag pilot4)
    fi
    [[ "$resume_flag" == "--resume" ]] && command+=(--resume)
    log_path="out/logs/track2_${phase}_s42.log"
    ;;
  eval_test)
    require_no_resume
    command=(python -u scripts/evaluate/evaluate_qwen_stage2.py --config configs/qwen3_4b/track2/eval_test.yaml --split test --limit 0)
    log_path="out/logs/track2_eval_test_s42.log"
    ;;
  audit_results)
    require_no_resume
    command=(python -u scripts/audit_qwen_track2_results.py)
    log_path="out/logs/track2_results_audit.log"
    ;;
  *)
    echo "unknown Track 2 phase: $phase" >&2
    exit 2
    ;;
esac

PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee "$log_path"
