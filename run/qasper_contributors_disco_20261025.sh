#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for name in lateon_contributors_r035 lateon_disco_stop_r025; do
  out="outputs/qasper_${name}_20261025"
  cp test/run_qasper_paired_fixed180.py "$out/runner_snapshot.py"
  cp test/lateon_marginal_selection.py "$out/marginal_selector_snapshot.py"
  python -u test/run_qasper_paired_fixed180.py --config "$out/config.json" > "$out/run.log" 2>&1
done
