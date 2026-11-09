#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
out=outputs/qasper_fixed120_r035_llama8b_20261025
python -u test/build_qasper_denseon_regions.py --sentence-store /mnt/nvme1/datasets/qasper-test-v0.3/text-evidence-1099-lateon/sentences --output /mnt/nvme1/datasets/qasper-test-v0.3/text-evidence-1099-denseon-region120 --region-token-budget 120 > "$out/build.log" 2>&1
python -u test/run_qasper_paired_fixed180.py --config "$out/config.json" > "$out/run.log" 2>&1
