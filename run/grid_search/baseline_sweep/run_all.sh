#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PATH="/home/dongseob/miniconda3/bin:$PATH"
GPU_ID="${GPU_ID:-0}"
# Preserve an existing exclusive mode without requiring a redundant sudo call.
MODE="$(nvidia-smi -i "$GPU_ID" --query-gpu=compute_mode --format=csv,noheader)"
EXCLUSIVE_ARGS=()
if [[ "$MODE" == "Exclusive_Process" ]]; then
    EXCLUSIVE_ARGS=(--no-exclusive)
fi
exec bash run/run_grid.sh --gpu "$GPU_ID" --wait-gpu --wait-lock \
    --lock-dir "$REPO_ROOT/outputs/aaa_icde/baseline_sweep" \
    "${EXCLUSIVE_ARGS[@]}" \
    "$SCRIPT_DIR/09_k20_tau03_nq.yaml" \
    "$SCRIPT_DIR/10_k20_tau03_newsqa.yaml" \
    "$SCRIPT_DIR/11_k20_tau03_covid.yaml" \
    "$SCRIPT_DIR/12_k20_tau03_bioasq.yaml" \
    "$SCRIPT_DIR/13_k20_tau04_nq.yaml" \
    "$SCRIPT_DIR/14_k20_tau04_newsqa.yaml" \
    "$SCRIPT_DIR/15_k20_tau04_covid.yaml" \
    "$SCRIPT_DIR/16_k20_tau04_bioasq.yaml" \
    "$SCRIPT_DIR/02_k100_tau05_newsqa.yaml" \
    "$SCRIPT_DIR/03_k100_tau05_covid.yaml" \
    "$SCRIPT_DIR/04_k100_tau05_bioasq.yaml" \
    "$SCRIPT_DIR/01_k100_tau05_nq.yaml"
