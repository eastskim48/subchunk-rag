#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
export PATH="/home/dongseob/miniconda3/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
grids=("$SCRIPT_DIR"/[0-9][0-9]_*.yaml)
[[ ${#grids[@]} -eq 12 ]]
exec bash run/run_grid.sh --gpu 0 --no-exclusive --wait-gpu --wait-lock --interval 30 \
    --lock-dir "$REPO_ROOT/.runtime/gpu_locks" "${grids[@]}"
