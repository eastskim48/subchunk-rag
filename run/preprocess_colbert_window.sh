#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET="${DATASET:-longbench-hotpotqa}"
if [ -n "${DATASET_PREFIX:-}" ]; then
    export DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    export DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
    export DATASET_PATH="$DATASET"
fi

export PREPROCESS_SUBDIR="${PREPROCESS_SUBDIR:-sent}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/db}"
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/colbert_window}"
export COLBERT_MODEL_NAME="${COLBERT_MODEL_NAME:-colbert-ir/colbertv2.0}"
export COLBERT_WINDOW_BATCH_SIZE="${COLBERT_WINDOW_BATCH_SIZE:-1024}"
export COLBERT_WINDOW_TOKEN_BUDGET="${COLBERT_WINDOW_TOKEN_BUDGET:-180}"
export COLBERT_WINDOW_OVERWRITE="${COLBERT_WINDOW_OVERWRITE:-False}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export COLBERT_DISABLE_CPU_EXTENSION="${COLBERT_DISABLE_CPU_EXTENSION:-True}"
export COLBERT_VALIDATE_AGAINST_DB="${COLBERT_VALIDATE_AGAINST_DB:-True}"
export COLBERT_VALIDATION_BATCH_SIZE="${COLBERT_VALIDATION_BATCH_SIZE:-2048}"
export COLBERT_WINDOW_PREFIX_TITLE="${COLBERT_WINDOW_PREFIX_TITLE:-False}"
export COLBERT_WINDOW_TITLE_SEPARATOR="${COLBERT_WINDOW_TITLE_SEPARATOR:-[SEP]}"
export COLBERT_WINDOW_CENTER_UNIT="${COLBERT_WINDOW_CENTER_UNIT:-sentence}"
export COLBERT_WINDOW_FIXED_CHUNK_SIZE="${COLBERT_WINDOW_FIXED_CHUNK_SIZE:-}"

mkdir -p "$COLBERT_WINDOW_DIR"

args=(
    --docs_dir "$DATASET_PATH/documents"
    --output_dir "$COLBERT_WINDOW_DIR"
    --db_dir "$DB_DIR"
    --model_name "$COLBERT_MODEL_NAME"
    --batch_size "$COLBERT_WINDOW_BATCH_SIZE"
    --window_token_budget "$COLBERT_WINDOW_TOKEN_BUDGET"
    --overwrite "$COLBERT_WINDOW_OVERWRITE"
    --repo_path "$COLBERT_REPO_PATH"
    --disable_cpu_extension "$COLBERT_DISABLE_CPU_EXTENSION"
    --validate_against_db "$COLBERT_VALIDATE_AGAINST_DB"
    --validation_batch_size "$COLBERT_VALIDATION_BATCH_SIZE"
    --prefix_title "$COLBERT_WINDOW_PREFIX_TITLE"
    --title_separator "$COLBERT_WINDOW_TITLE_SEPARATOR"
    --center_unit "$COLBERT_WINDOW_CENTER_UNIT"
)

if [ -n "$COLBERT_WINDOW_FIXED_CHUNK_SIZE" ]; then
    args+=(--fixed_chunk_size "$COLBERT_WINDOW_FIXED_CHUNK_SIZE")
fi

python src/entrypoint/preprocess_colbert_window.py "${args[@]}"
