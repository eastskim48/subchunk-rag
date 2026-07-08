#!/bin/bash
export DATASET="${DATASET:-longbench-hotpotqa}"
if [ -n "$DATASET_PREFIX" ]; then
    export DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    export DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
    export DATASET_PATH="$DATASET"
fi
export CHUNK_SIZE="${CHUNK_SIZE:-1024}"
export MATKV_SUBDIR="${MATKV_SUBDIR:-matkv_${CHUNK_SIZE}}"
export MATERIALIZE_CACHE="${MATERIALIZE_CACHE:-True}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-False}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export CACHE_SUBDIR="${CACHE_SUBDIR:-cache}"
export CACHE_DIR="${CACHE_DIR:-$DATASET_PATH/$MATKV_SUBDIR/$CACHE_SUBDIR}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$MATKV_SUBDIR/db}"
export TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"

mkdir -p "$CACHE_DIR"

torchrun --nproc_per_node 1 --master_port "$TORCHRUN_MASTER_PORT" src/entrypoint/preprocess.py \
    --db_dir="$DB_DIR" \
    --cache_dir="$CACHE_DIR" \
    --model_name "$MODEL_NAME" \
    --docs_dir="$DATASET_PATH/documents" \
    --cacheable_chunk_size="$CHUNK_SIZE" \
    --retrievable_chunk_size="$CHUNK_SIZE" \
    --batch_size 64 \
    --splitter fixed_size \
    --materialize_cache "$MATERIALIZE_CACHE" \
    --resume_from_cache "$RESUME_FROM_CACHE" \
    --dummy_bos_count 4
