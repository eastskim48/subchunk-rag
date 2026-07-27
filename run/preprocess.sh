#!/bin/bash

export DATASET="${DATASET:-longbench-hotpotqa}"
if [ -n "$DATASET_PREFIX" ]; then
    export DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    export DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
    export DATASET_PATH="$DATASET"
fi
export MATERIALIZE_CACHE="${MATERIALIZE_CACHE:-True}"
export MATERIALIZE_DB="${MATERIALIZE_DB:-True}"
export PREPROCESS_SUBDIR="${PREPROCESS_SUBDIR:-sent}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cuda}"
export CHROMA_EMBED_BATCH_SIZE="${CHROMA_EMBED_BATCH_SIZE:-128}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export BATCH_SIZE="${BATCH_SIZE:-36}"
export DB_BATCH_SIZE="${DB_BATCH_SIZE:-256}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-False}"
export SPLITTER="${SPLITTER:-sentence}"
export MERGER="${MERGER-}"
export DEDUPLICATE_DOCUMENTS_BY_HASH="${DEDUPLICATE_DOCUMENTS_BY_HASH:-False}"
export MAX_SUBCHUNK_TOKENS="${MAX_SUBCHUNK_TOKENS:-180}"
export CACHEABLE_CHUNK_SIZE="${CACHEABLE_CHUNK_SIZE:-None}"
export RETRIEVABLE_CHUNK_SIZE="${RETRIEVABLE_CHUNK_SIZE:-1024}"
export CACHE_SUBDIR="${CACHE_SUBDIR:-cache}"
export CACHE_DIR="${CACHE_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/$CACHE_SUBDIR}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/db}"
export TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"

is_enabled() {
    case "${1,,}" in
        true|1|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

if is_enabled "$MATERIALIZE_CACHE"; then
    mkdir -p "$CACHE_DIR"
fi
if is_enabled "$MATERIALIZE_DB"; then
    mkdir -p "$DB_DIR"
fi

MERGER_ARGS=()
if [ -n "$MERGER" ]; then
    MERGER_ARGS=(--merger "$MERGER")
fi

torchrun --nproc_per_node 1 --master_port "$TORCHRUN_MASTER_PORT" src/entrypoint/preprocess.py \
    --db_dir="$DB_DIR" \
    --cache_dir="$CACHE_DIR" \
    --model_name "$MODEL_NAME" \
    --docs_dir="$DATASET_PATH/documents" \
    --cacheable_chunk_size="$CACHEABLE_CHUNK_SIZE" \
    --retrievable_chunk_size="$RETRIEVABLE_CHUNK_SIZE" \
    --splitter "$SPLITTER" \
    "${MERGER_ARGS[@]}" \
    --deduplicate_documents_by_hash "$DEDUPLICATE_DOCUMENTS_BY_HASH" \
    --max_subchunk_tokens "$MAX_SUBCHUNK_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --db_batch_size "$DB_BATCH_SIZE" \
    --chroma_embed_device "$CHROMA_EMBED_DEVICE" \
    --chroma_embed_batch_size "$CHROMA_EMBED_BATCH_SIZE" \
    --materialize_cache "$MATERIALIZE_CACHE" \
    --materialize_db "$MATERIALIZE_DB" \
    --resume_from_cache "$RESUME_FROM_CACHE" \
    --dummy_bos_count 4
