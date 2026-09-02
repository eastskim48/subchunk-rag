#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_PREFIX="${DATASET_PREFIX:-/mnt/nvme1/datasets}"
HOTPOT_FULL_OUTPUT_DIR="${HOTPOT_FULL_OUTPUT_DIR:-$DATASET_PREFIX/hotpotqa-full}"
HOTPOTQA_SOURCE_DIR="${HOTPOTQA_SOURCE_DIR:-$DATASET_PREFIX/hotpotqa-sources}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" dataset/hotpotqa/prepare_full.py \
    --output-dir "$HOTPOT_FULL_OUTPUT_DIR" \
    --source-dir "$HOTPOTQA_SOURCE_DIR" \
    "$@"
