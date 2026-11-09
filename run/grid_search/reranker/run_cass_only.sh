#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH="/home/dongseob/miniconda3/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mode=$(nvidia-smi -i 0 --query-gpu=compute_mode --format=csv,noheader | tr -d '[:space:]')
exclusive_args=()
if [[ "$mode" == "Exclusive_Process" || "$mode" == "EXCLUSIVE_PROCESS" ]]; then
    exclusive_args=(--no-exclusive)
fi
exec bash run/run_grid.sh --gpu 0 --wait-gpu --wait-lock --interval 10 \
    --lock-dir "$REPO_ROOT/.runtime/gpu_locks" "${exclusive_args[@]}" \
    "$SCRIPT_DIR/05_jina_top20_cass_nq.yaml" \
    "$SCRIPT_DIR/06_jina_top20_cass_newsqa.yaml" \
    "$SCRIPT_DIR/07_jina_top20_cass_bioasq.yaml" \
    "$SCRIPT_DIR/08_jina_top20_cass_covid.yaml"
