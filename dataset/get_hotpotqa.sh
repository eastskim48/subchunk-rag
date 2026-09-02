#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_PREFIX="${DATASET_PREFIX:-/mnt/nvme1/datasets}"
HOTPOTQA_OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$DATASET_PREFIX/hotpotqa}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" dataset/hotpotqa/prepare.py \
    --output-dir "$HOTPOTQA_OUTPUT_DIR" \
    "$@"
