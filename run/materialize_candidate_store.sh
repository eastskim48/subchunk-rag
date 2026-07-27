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
export CANDIDATE_STORE_BACKEND="${CANDIDATE_STORE_BACKEND:-colbert}"
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/colbert_window}"
export COLBERT_MODEL_NAME="${COLBERT_MODEL_NAME:-colbert-ir/colbertv2.0}"
export COLBERT_WINDOW_BATCH_SIZE="${COLBERT_WINDOW_BATCH_SIZE:-1024}"
export COLBERT_WINDOW_TOKEN_BUDGET="${COLBERT_WINDOW_TOKEN_BUDGET:-180}"
export COLBERT_WINDOW_OVERWRITE="${COLBERT_WINDOW_OVERWRITE:-False}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export COLBERT_DISABLE_CPU_EXTENSION="${COLBERT_DISABLE_CPU_EXTENSION:-True}"
export COLBERT_VALIDATION_BATCH_SIZE="${COLBERT_VALIDATION_BATCH_SIZE:-2048}"
export COLBERT_WINDOW_CENTER_UNIT="${COLBERT_WINDOW_CENTER_UNIT:-subchunk}"
export COLBERT_WINDOW_FIXED_CHUNK_SIZE="${COLBERT_WINDOW_FIXED_CHUNK_SIZE:-}"
export DENSE_EMBED_DIR="${DENSE_EMBED_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/dense_embed}"
export DENSE_EMBED_MODEL="${DENSE_EMBED_MODEL:-BAAI/bge-m3}"
export DENSE_EMBED_BATCH_SIZE="${DENSE_EMBED_BATCH_SIZE:-128}"
export DENSE_DB_BATCH_SIZE="${DENSE_DB_BATCH_SIZE:-2048}"
export DENSE_EMBED_OVERWRITE="${DENSE_EMBED_OVERWRITE:-False}"

case "$CANDIDATE_STORE_BACKEND" in
    colbert)
        mkdir -p "$COLBERT_WINDOW_DIR"
        args=(
            --backend colbert
            --docs_dir "$DATASET_PATH/documents"
            --output_dir "$COLBERT_WINDOW_DIR"
            --db_dir "$DB_DIR"
            --model_name "$COLBERT_MODEL_NAME"
            --batch_size "$COLBERT_WINDOW_BATCH_SIZE"
            --window_token_budget "$COLBERT_WINDOW_TOKEN_BUDGET"
            --overwrite "$COLBERT_WINDOW_OVERWRITE"
            --repo_path "$COLBERT_REPO_PATH"
            --disable_cpu_extension "$COLBERT_DISABLE_CPU_EXTENSION"
            --validation_batch_size "$COLBERT_VALIDATION_BATCH_SIZE"
            --center_unit "$COLBERT_WINDOW_CENTER_UNIT"
        )
        if [ -n "$COLBERT_WINDOW_FIXED_CHUNK_SIZE" ]; then
            args+=(--fixed_chunk_size "$COLBERT_WINDOW_FIXED_CHUNK_SIZE")
        fi
        ;;
    dense)
        mkdir -p "$DENSE_EMBED_DIR"
        args=(
            --backend dense
            --output_dir "$DENSE_EMBED_DIR"
            --db_dir "$DB_DIR"
            --model_name "$DENSE_EMBED_MODEL"
            --batch_size "$DENSE_EMBED_BATCH_SIZE"
            --db_batch_size "$DENSE_DB_BATCH_SIZE"
            --overwrite "$DENSE_EMBED_OVERWRITE"
        )
        ;;
    *)
        echo "unsupported CANDIDATE_STORE_BACKEND=$CANDIDATE_STORE_BACKEND" >&2
        exit 1
        ;;
esac

python src/entrypoint/materialize_candidate_store.py "${args[@]}"
