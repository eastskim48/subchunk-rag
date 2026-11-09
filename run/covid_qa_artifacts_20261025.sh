#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
root=/mnt/nvme1/datasets/covid-qa-full-test
out=outputs/covid_qa_artifacts_20261025
python -u test/build_qasper_lateon_sentences.py --input-format covid-qa --input "$root/test.json" --output "$root/lateon-sentence-2bit-region180" > "$out/lateon.log" 2>&1
python -u test/build_qasper_denseon_regions.py --sentence-store "$root/lateon-sentence-2bit-region180/sentences" --output "$root/denseon-region180" --region-token-budget 180 > "$out/denseon.log" 2>&1
