#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_PREFIX="${DATASET_PREFIX:-/mnt/nvme1/datasets}"
NQ_OUTPUT_DIR="${NQ_OUTPUT_DIR:-$DATASET_PREFIX/dapr-nq-open}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" dataset/nq/prepare.py \
    --output-dir "$NQ_OUTPUT_DIR" \
    "$@"
