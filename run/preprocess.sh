#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
export MATERIALIZE_COMPARE_EMBEDS="${MATERIALIZE_COMPARE_EMBEDS:-True}"
export MATERIALIZE_COLBERT_WINDOW="${MATERIALIZE_COLBERT_WINDOW:-False}"
export PREPROCESS_SUBDIR="${PREPROCESS_SUBDIR:-sent}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cuda}"
export COMPARE_EMBED_OVERWRITE="${COMPARE_EMBED_OVERWRITE:-False}"
export COMPARE_EMBED_MODEL="${COMPARE_EMBED_MODEL:-BAAI/bge-m3}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export COLBERT_WINDOW_MODEL="${COLBERT_WINDOW_MODEL:-colbert-ir/colbertv2.0}"
export COLBERT_WINDOW_DEVICE="${COLBERT_WINDOW_DEVICE:-cuda}"
export COLBERT_WINDOW_BATCH_SIZE="${COLBERT_WINDOW_BATCH_SIZE:-32}"
export COLBERT_WINDOW_TOKEN_BUDGET="${COLBERT_WINDOW_TOKEN_BUDGET:-180}"
export COLBERT_WINDOW_OVERWRITE="${COLBERT_WINDOW_OVERWRITE:-False}"
export COLBERT_SOURCE_TOKENIZER_NAME="${COLBERT_SOURCE_TOKENIZER_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export COLBERT_DISABLE_CPU_EXTENSION="${COLBERT_DISABLE_CPU_EXTENSION:-True}"
export COLBERT_VERIFY_TENSORIZATION="${COLBERT_VERIFY_TENSORIZATION:-True}"
export COLBERT_WINDOW_CENTER_UNIT="${COLBERT_WINDOW_CENTER_UNIT:-sentence}"
export COLBERT_WINDOW_FIXED_CHUNK_SIZE="${COLBERT_WINDOW_FIXED_CHUNK_SIZE:-None}"
export BATCH_SIZE="${BATCH_SIZE:-36}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-False}"
export SPLITTER="${SPLITTER:-semantic}"
export MERGER="${MERGER-}"
export SENTENCE_RESOLVER="${SENTENCE_RESOLVER:-openai}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
export FASTCOREF_MODEL_NAME="${FASTCOREF_MODEL_NAME:-biu-nlp/f-coref}"
export CACHEABLE_CHUNK_SIZE="${CACHEABLE_CHUNK_SIZE:-None}"
export RETRIEVABLE_CHUNK_SIZE="${RETRIEVABLE_CHUNK_SIZE:-1024}"
export CACHE_SUBDIR="${CACHE_SUBDIR:-cache}"
export CACHE_DIR="${CACHE_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/$CACHE_SUBDIR}"
export COMPARE_EMBED_DIR="${COMPARE_EMBED_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/compare_embed}"
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/colbert_window}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$PREPROCESS_SUBDIR/db}"
export TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"

mkdir -p "$CACHE_DIR"
mkdir -p "$COMPARE_EMBED_DIR"
mkdir -p "$DB_DIR"
if [ "$MATERIALIZE_COLBERT_WINDOW" = "True" ]; then
    mkdir -p "$COLBERT_WINDOW_DIR"
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
    --sentence_resolver "$SENTENCE_RESOLVER" \
    --openai_model "$OPENAI_MODEL" \
    --fastcoref_model_name "$FASTCOREF_MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --materialize_cache "$MATERIALIZE_CACHE" \
    --materialize_db "$MATERIALIZE_DB" \
    --materialize_compare_embeds "$MATERIALIZE_COMPARE_EMBEDS" \
    --compare_embed_dir "$COMPARE_EMBED_DIR" \
    --compare_embed_model "$COMPARE_EMBED_MODEL" \
    --compare_embed_overwrite "$COMPARE_EMBED_OVERWRITE" \
    --materialize_colbert_window "$MATERIALIZE_COLBERT_WINDOW" \
    --colbert_window_dir "$COLBERT_WINDOW_DIR" \
    --colbert_window_model "$COLBERT_WINDOW_MODEL" \
    --colbert_window_device "$COLBERT_WINDOW_DEVICE" \
    --colbert_window_batch_size "$COLBERT_WINDOW_BATCH_SIZE" \
    --colbert_window_token_budget "$COLBERT_WINDOW_TOKEN_BUDGET" \
    --colbert_window_overwrite "$COLBERT_WINDOW_OVERWRITE" \
    --colbert_source_tokenizer_name "$COLBERT_SOURCE_TOKENIZER_NAME" \
    --colbert_repo_path "$COLBERT_REPO_PATH" \
    --colbert_disable_cpu_extension "$COLBERT_DISABLE_CPU_EXTENSION" \
    --colbert_verify_tensorization "$COLBERT_VERIFY_TENSORIZATION" \
    --colbert_window_center_unit "$COLBERT_WINDOW_CENTER_UNIT" \
    --colbert_window_fixed_chunk_size "$COLBERT_WINDOW_FIXED_CHUNK_SIZE" \
    --resume_from_cache "$RESUME_FROM_CACHE" \
    --dummy_bos_count 4
