#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/run_meta
exec > >(tee -a out/logs/finalize_pipeline.log) 2>&1

echo "[$(date -Is)] waiting for formal RLVR pipeline"
while ! grep -q 'RLVR formal pipeline complete' out/logs/rlvr_formal_pipeline.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_rlvr_formal'; then
    echo "formal RLVR pipeline exited without a success marker" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] re-auditing Agent SFT assistant-only mask with a fixed seed"
/root/miniconda3/bin/python scripts/audit_sft_mask.py \
  --data_path dataset/agent_sft_coldstart.jsonl \
  --max_seq_len 1024 \
  --samples 128 \
  --examples 3 \
  --seed 42 \
  --output out/run_meta/agent_sft_mask_audit_128.json

echo "[$(date -Is)] running reproducible MoE routing audit"
/root/miniconda3/bin/python scripts/audit_moe_routing.py \
  --weight pretrain_partial --hidden_size 768 --num_hidden_layers 8 \
  --batch_size 2 --seq_len 64 --seed 42 --device cuda:0 \
  --output out/run_meta/moe_routing_audit.json

echo "[$(date -Is)] profiling Flash SDPA through the actual MiniMind model"
/root/miniconda3/bin/python scripts/verify_model_flash_sdpa.py \
  --weight full_sft --hidden_size 768 --num_hidden_layers 8 \
  --seq_len 1024 --device cuda:0 \
  --output out/model_flash_sdpa_verification.json

echo "[$(date -Is)] re-auditing multi-turn serialization and every Tool-RLVR oracle"
/root/miniconda3/bin/python scripts/audit_multiturn_serialization.py \
  --output out/run_meta/multiturn_serialization_audit.json
/root/miniconda3/bin/python scripts/audit_tool_rlvr_dataset.py \
  --output out/run_meta/tool_rlvr_execution_audit.json
/root/miniconda3/bin/python scripts/audit_tool_rlvr_dataset.py \
  --train dataset/agent_rl_tool_challenge_train.jsonl \
  --eval dataset/agent_rl_tool_challenge_eval.jsonl \
  --output out/run_meta/tool_rlvr_challenge_execution_audit.json

echo "[$(date -Is)] compiling all project Python sources"
/root/miniconda3/bin/python -m compileall -q model dataset trainer scripts tests

echo "[$(date -Is)] running final unit-test suite"
PYTHONPATH="${REPO_DIR}" /root/miniconda3/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v \
  2>&1 | tee out/logs/final_unit_tests.log

sha256sum out/run_meta/agent_sft_mask_audit_128.json \
  out/run_meta/moe_routing_audit.json out/model_flash_sdpa_verification.json \
  out/run_meta/multiturn_serialization_audit.json \
  out/run_meta/tool_rlvr_execution_audit.json \
  out/run_meta/tool_rlvr_challenge_execution_audit.json \
  out/logs/final_unit_tests.log \
  > out/run_meta/final_validation.sha256

echo "[$(date -Is)] collecting machine-readable experiment evidence"
/root/miniconda3/bin/python scripts/collect_experiment_evidence.py
echo "[$(date -Is)] final validation complete"
# Rebuild once after the success marker so the evidence index records its own
# final-validation stage as complete.  The first collection above remains a
# gate: the marker is never written if evidence parsing fails.
/root/miniconda3/bin/python scripts/collect_experiment_evidence.py
sha256sum out/run_meta/experiment_evidence.json \
  > out/run_meta/experiment_evidence.sha256
