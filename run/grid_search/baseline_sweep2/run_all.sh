#!/usr/bin/env bash
set -euo pipefail
cd /mnt/nvme1/dongseob/subchunk
export PATH=/home/dongseob/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mode=$(nvidia-smi -i 0 --query-gpu=compute_mode --format=csv,noheader | tr -d '[:space:]')
exclusive_args=()
if [[ "$mode" == "Exclusive_Process" || "$mode" == "EXCLUSIVE_PROCESS" ]]; then
    exclusive_args=(--no-exclusive)
fi
exec bash run/run_grid.sh --gpu 0 --wait-gpu --wait-lock --interval 15 --lock-dir /mnt/nvme1/dongseob/subchunk/.runtime/gpu_locks "${exclusive_args[@]}" \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/01_k20_rerank5_tau01_nq.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/02_k20_rerank5_tau01_newsqa.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/03_k20_rerank5_tau01_bioasq.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/04_k20_rerank5_tau01_covid.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/05_k20_rerank5_tau005_nq.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/06_k20_rerank5_tau005_newsqa.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/07_k20_rerank5_tau005_bioasq.yaml \
    /mnt/nvme1/dongseob/subchunk/run/grid_search/baseline_sweep2/08_k20_rerank5_tau005_covid.yaml
