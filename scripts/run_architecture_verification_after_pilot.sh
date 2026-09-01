#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/minimind}"
cd "${REPO_DIR}"
mkdir -p out/logs out/run_meta
exec > >(tee -a out/logs/architecture_verification.log) 2>&1

echo "[$(date -Is)] waiting for RLVR pilot so the single GPU is not contended"
while ! grep -q 'RLVR pilot complete' out/logs/rlvr_pilot.log 2>/dev/null; do
  if ! screen -ls 2>/dev/null | grep -q 'minimind_rlvr_pilot'; then
    echo "RLVR pilot exited without a success marker" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] forcing and profiling Flash SDPA"
/root/miniconda3/bin/python scripts/verify_flash_sdpa.py \
  --seq_len 1024 --head_dim 96 --repeats 10 \
  --output out/flash_sdpa_verification.json

echo "[$(date -Is)] persisting sparse-MoE routing and gradient audit"
/root/miniconda3/bin/python scripts/audit_moe_routing.py \
  --weight pretrain_partial \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --batch_size 2 \
  --seq_len 64 \
  --seed 42 \
  --device cuda:0 \
  --output out/run_meta/moe_routing_audit.json

output="out/architecture_yarn_4096.csv"
test ! -e "${output}" || {
  echo "Refusing to append to existing ${output}" >&2
  exit 1
}
for yarn in 0 1; do
  echo "[$(date -Is)] long-context functional benchmark yarn=${yarn}"
  /root/miniconda3/bin/python scripts/benchmark_architecture.py \
    --weight full_sft \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --use_moe 0 \
    --flash_attn 1 \
    --inference_rope_scaling "${yarn}" \
    --max_position_embeddings 32768 \
    --prompt_length 4096 \
    --decode_tokens 32 \
    --repeats 2 \
    --device cuda:0 \
    --output "${output}"
done

sha256sum out/flash_sdpa_verification.json out/run_meta/moe_routing_audit.json "${output}" \
  > out/run_meta/architecture_verification.sha256
echo "[$(date -Is)] architecture verification complete"
