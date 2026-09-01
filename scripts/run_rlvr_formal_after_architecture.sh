#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/run_meta
exec > >(tee -a out/logs/rlvr_formal_pipeline.log) 2>&1

echo "[$(date -Is)] waiting for architecture verification and all earlier GPU jobs"
while ! grep -q 'architecture verification complete' out/logs/architecture_verification.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_arch_verify'; then
    echo "architecture pipeline exited without a success marker" >&2
    exit 1
  fi
  sleep 30
done

for task in math tool; do
  pilot_summary="out/eval_rlvr/pilot/${task}/summary.csv"
  if [[ ! -f "${pilot_summary}" ]]; then
    echo "SKIP formal ${task}: the guarded pilot did not produce an evaluation"
    continue
  fi

  echo "[$(date -Is)] formal small-budget cross-seed ablation: ${task}"
  MM_TASK="${task}" \
  MM_SEEDS="42 43 44" \
  MM_BATCH_SIZE=1 \
  MM_GENERATIONS=4 \
  MM_POLICY_EPOCHS=2 \
  MM_MAX_UPDATES=10 \
  MM_MAX_TURNS=3 \
  MM_MAX_GEN_LEN=96 \
  MM_MAX_TOTAL_LEN=1024 \
  MM_DYNAMIC_SAMPLING_ROUNDS=50 \
  MM_INIT_WEIGHT=agent_sft \
    bash scripts/run_rlvr_ablation.sh \
    2>&1 | tee "out/logs/rlvr_formal_${task}_train.log"

  echo "[$(date -Is)] fixed holdout evaluation: ${task}"
  MM_TASK="${task}" \
  MM_TRAIN_SEEDS="42 43 44" \
  MM_DECODE_SEEDS="101,102,103" \
  MM_LIMIT=64 \
  MM_BATCH_SIZE=2 \
  MM_GENERATIONS=1 \
  MM_MAX_TURNS=3 \
  MM_MAX_GEN_LEN=96 \
  MM_MAX_TOTAL_LEN=1024 \
    bash scripts/run_rlvr_eval.sh \
    2>&1 | tee "out/logs/rlvr_formal_${task}_eval.log"
done

find out/metrics out/eval_rlvr -type f \
  \( -name '*.jsonl' -o -name '*.csv' \) -print0 \
  | sort -z | xargs -0 sha256sum > out/run_meta/rlvr_formal_artifacts.sha256
echo "[$(date -Is)] RLVR formal pipeline complete"
