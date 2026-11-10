#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
DATASET_PREFIX="${DATASET_PREFIX:-/mnt/nvme1/datasets}"
exec "${PYTHON_BIN:-python}" -m dataset.bioasq.prepare \
    --output-dir "$DATASET_PREFIX/bioasq14b-linked-documents" "$@"
