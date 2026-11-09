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
    "$SCRIPT_DIR/01_k100_tau05_nq.yaml"
