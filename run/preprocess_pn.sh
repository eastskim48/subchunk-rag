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
export MATERIALIZE_COMPARE_EMBEDS="${MATERIALIZE_COMPARE_EMBEDS:-True}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cuda}"
export COMPARE_EMBED_OVERWRITE="${COMPARE_EMBED_OVERWRITE:-False}"
export COMPARE_EMBED_MODEL="${COMPARE_EMBED_MODEL:-BAAI/bge-m3}"
export BATCH_SIZE="${BATCH_SIZE:-36}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-False}"
export PREPROCESS_SUBDIR="${PREPROCESS_SUBDIR:-sent-pn}"
export PN_MAPPING_SUBDIR="${PN_MAPPING_SUBDIR:-pn_mapping}"
export SPLITTER="${SPLITTER:-pn_sentence}"
export CACHEABLE_CHUNK_SIZE="${CACHEABLE_CHUNK_SIZE:-None}"
export RETRIEVABLE_CHUNK_SIZE="${RETRIEVABLE_CHUNK_SIZE:-None}"
export CACHE_DIR="$DATASET_PATH/$PREPROCESS_SUBDIR/cache"
export COMPARE_EMBED_DIR="$DATASET_PATH/$PREPROCESS_SUBDIR/compare_embed"
export DB_DIR="$DATASET_PATH/$PREPROCESS_SUBDIR/db"
export DOCS_DIR="$DATASET_PATH/documents"
export PN_MAPPING_DIR="$DATASET_PATH/$PN_MAPPING_SUBDIR"

mkdir -p "$CACHE_DIR"
mkdir -p "$COMPARE_EMBED_DIR"
mkdir -p "$DB_DIR"

torchrun --nproc_per_node 1 src/entrypoint/preprocess.py \
    --db_dir="$DB_DIR" \
    --cache_dir="$CACHE_DIR" \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --docs_dir="$DOCS_DIR" \
    --cacheable_chunk_size="$CACHEABLE_CHUNK_SIZE" \
    --retrievable_chunk_size="$RETRIEVABLE_CHUNK_SIZE" \
    --splitter "$SPLITTER" \
    --pn_mapping_dir "$PN_MAPPING_DIR" \
    --batch_size "$BATCH_SIZE" \
    --materialize_cache "$MATERIALIZE_CACHE" \
    --materialize_compare_embeds "$MATERIALIZE_COMPARE_EMBEDS" \
    --compare_embed_dir "$COMPARE_EMBED_DIR" \
    --compare_embed_model "$COMPARE_EMBED_MODEL" \
    --compare_embed_overwrite "$COMPARE_EMBED_OVERWRITE" \
    --resume_from_cache "$RESUME_FROM_CACHE" \
    --dummy_bos_count 4
