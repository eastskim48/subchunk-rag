#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH="/home/dongseob/miniconda3/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python "$SCRIPT_DIR/check_prepared.py" --report "$REPO_ROOT/outputs/aaa_icde/reranker/preparation/launch_validation.json"
mode=$(nvidia-smi -i 0 --query-gpu=compute_mode --format=csv,noheader | tr -d '[:space:]')
exclusive_args=()
if [[ "$mode" == "Exclusive_Process" || "$mode" == "EXCLUSIVE_PROCESS" ]]; then
    exclusive_args=(--no-exclusive)
fi
exec bash run/run_grid.sh --gpu 0 --wait-gpu --wait-lock --interval 30 \
    --lock-dir "$REPO_ROOT/.runtime/gpu_locks" "${exclusive_args[@]}" \
    "$SCRIPT_DIR/01_jina_top2_nq.yaml" \
    "$SCRIPT_DIR/02_jina_top2_newsqa.yaml" \
    "$SCRIPT_DIR/03_jina_top2_bioasq.yaml" \
    "$SCRIPT_DIR/04_jina_top2_covid.yaml" \
    "$SCRIPT_DIR/05_jina_top20_cass_nq.yaml" \
    "$SCRIPT_DIR/06_jina_top20_cass_newsqa.yaml" \
    "$SCRIPT_DIR/07_jina_top20_cass_bioasq.yaml" \
    "$SCRIPT_DIR/08_jina_top20_cass_covid.yaml" \
    "$SCRIPT_DIR/09_vanilla_top2_nq.yaml" \
    "$SCRIPT_DIR/10_vanilla_top2_newsqa.yaml" \
    "$SCRIPT_DIR/11_vanilla_top2_bioasq.yaml" \
    "$SCRIPT_DIR/12_vanilla_top2_covid.yaml"
