#!/bin/bash

export DATASET="${DATASET:-longbench-hotpotqa}"
#export DATASET="LongBench-TriviaQA"
if [ -n "$DATASET_PREFIX" ]; then
    export DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    export DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
    export DATASET_PATH="$DATASET"
fi
export CHUNK_SIZE="${CHUNK_SIZE:-1024}"
export VANILLA_SUBDIR="${VANILLA_SUBDIR:-vanilla-default-${CHUNK_SIZE}}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cpu}"
export COMPARE_EMBED_DEVICE="${COMPARE_EMBED_DEVICE:-cpu}"
export DEFAULT_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
export CACHE_SUBDIR="${CACHE_SUBDIR:-cache}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$VANILLA_SUBDIR/db}"
export CACHE_DIR="${CACHE_DIR:-$DATASET_PATH/$VANILLA_SUBDIR/$CACHE_SUBDIR}"
export EVAL_USE_PAST_CACHE="${EVAL_USE_PAST_CACHE:-False}"
export DISABLE_ROPE="${DISABLE_ROPE:-False}"
export USE_FRONT_BOS_CACHE="${USE_FRONT_BOS_CACHE:-False}"
export MODEL_LOAD_IN_4BIT="${MODEL_LOAD_IN_4BIT:-False}"
export USE_CLEANER="${USE_CLEANER:-False}"
export EVAL_BSZ="${EVAL_BSZ:-1}"
export TOP_K="${TOP_K:-20}"
if [ "$EVAL_USE_PAST_CACHE" = "True" ]; then
    OUTPUT_SUFFIX="cacheon"
else
    OUTPUT_SUFFIX="cacheoff"
fi
if [ "$MODEL_LOAD_IN_4BIT" = "True" ] || [ "$MODEL_LOAD_IN_4BIT" = "true" ] || [ "$MODEL_LOAD_IN_4BIT" = "1" ]; then
    OUTPUT_SUFFIX="llm4bit-$OUTPUT_SUFFIX"
else
    OUTPUT_SUFFIX="llmfp16-$OUTPUT_SUFFIX"
fi
if [ "$MODEL_NAME" != "$DEFAULT_MODEL_NAME" ]; then
    MODEL_TAG="${MODEL_OUTPUT_TAG:-$MODEL_NAME}"
    MODEL_TAG="${MODEL_TAG//\//_}"
    MODEL_TAG="${MODEL_TAG// /_}"
    OUTPUT_SUFFIX="$MODEL_TAG-$OUTPUT_SUFFIX"
fi

torchrun --nproc_per_node 1 src/entrypoint/eval.py \
    --dataset="$DATASET" \
    --db_dir="$DB_DIR" \
    --cache_dir="$CACHE_DIR" \
    --query_file="$DATASET_PATH/questions/query.jsonl" \
    --top_k "$TOP_K" --bsz "$EVAL_BSZ" --max_new_tokens 20 --total_num 200 \
    --model_name "$MODEL_NAME" \
    --use_past_cache="$EVAL_USE_PAST_CACHE" \
    --disable_rope "$DISABLE_ROPE" \
    --use_front_bos_cache "$USE_FRONT_BOS_CACHE" \
    --model_load_in_4bit "$MODEL_LOAD_IN_4BIT" \
    --use_cleaner "$USE_CLEANER" \
    --output_file "./outputs/eval-vanilla-$DATASET-topk${TOP_K}-$OUTPUT_SUFFIX.jsonl" \
    --answer_file "$DATASET_PATH/answers/answer.jsonl"
