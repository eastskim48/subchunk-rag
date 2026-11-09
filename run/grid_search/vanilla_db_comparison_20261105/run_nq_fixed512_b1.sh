#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
cd "$repo_root"
export PATH=/home/dongseob/miniconda3/bin:$PATH
mode=$(nvidia-smi -i 0 --query-gpu=compute_mode --format=csv,noheader | tr -d '[:space:]')
exclusive_args=()
if [[ "$mode" == "Exclusive_Process" || "$mode" == "EXCLUSIVE_PROCESS" ]]; then
    exclusive_args=(--no-exclusive)
fi
bash run/run_grid.sh --gpu 0 --wait-gpu --wait-lock --interval 10 \
    --lock-dir "$repo_root/.runtime/gpu_locks" "${exclusive_args[@]}" \
    "$script_dir/grid_06_nq_fixed512.yaml"
python "$script_dir/finalize_nq_fixed512.py"
