#!/bin/bash

# torchrun --nproc_per_node 1 --master_port=29500 eval_batch.py --db_dir=/mnt/raid0/kunwooshin/data/db --cache_dir=/mnt/raid0/kunwooshin/data/cache --query_file=./questions/query.jsonl --top_k 2 --use_past_cache=True --bsz 32 --max_new_tokens 30 --total_num 128

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

export DEFAULT_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
export DATA_SUBDIR="${DATA_SUBDIR:-sent}"
export CACHE_SUBDIR="${CACHE_SUBDIR:-cache}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$DATA_SUBDIR/db}"
export CACHE_DIR="${CACHE_DIR:-$DATASET_PATH/$DATA_SUBDIR/$CACHE_SUBDIR}"
export CHROMA_EMBED_BACKEND="${CHROMA_EMBED_BACKEND:-bge_small_v1_5}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cpu}"
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$DATA_SUBDIR/colbert_window}"
export COLBERT_MODEL_NAME="${COLBERT_MODEL_NAME:-colbert-ir/colbertv2.0}"
export COLBERT_BATCH_SIZE="${COLBERT_BATCH_SIZE:-32}"
export COLBERT_REGION_GROUP_ORDER="${COLBERT_REGION_GROUP_ORDER:-retrieval}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export EVAL_USE_PAST_CACHE="${EVAL_USE_PAST_CACHE:-False}"
export DISABLE_ROPE="${DISABLE_ROPE:-False}"
export USE_FRONT_BOS_CACHE="${USE_FRONT_BOS_CACHE:-False}"
export MODEL_LOAD_IN_4BIT="${MODEL_LOAD_IN_4BIT:-False}"
export MEASURE_PROMPT_STATS="${MEASURE_PROMPT_STATS:-True}"
export RETRIEVAL_INCLUDE_DOCUMENTS="${RETRIEVAL_INCLUDE_DOCUMENTS:-True}"
export USE_CLEANER="${USE_CLEANER:-False}"
export PROMPT_FORMAT="${PROMPT_FORMAT:-raw_chunk_first}"
export EVAL_BSZ="${EVAL_BSZ:-4}"
export TOP_K="${TOP_K:-20}"
export TOTAL_NUM="${TOTAL_NUM:-}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
export COMPRESS_METHOD="${COMPRESS_METHOD-colbert_sliding_region}"
export TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"

USES_TOKEN_BUDGET=False
case "$COMPRESS_METHOD" in
    dense|colbert_subchunk|colbert_sliding_region|rerank_and_region)
        USES_TOKEN_BUDGET=True
        ;;
esac

if [ "$USES_TOKEN_BUDGET" = "True" ]; then
    if { [ -n "${RETAIN_TOKEN_RATIO:-}" ] && [ -n "${FINAL_TOKEN_BUDGET:-}" ]; } || \
       { [ -z "${RETAIN_TOKEN_RATIO:-}" ] && [ -z "${FINAL_TOKEN_BUDGET:-}" ]; }; then
        echo "exactly one of RETAIN_TOKEN_RATIO or FINAL_TOKEN_BUDGET must be set for $COMPRESS_METHOD" >&2
        exit 1
    fi
fi

if [ "$EVAL_USE_PAST_CACHE" = "True" ]; then
    OUTPUT_SUFFIX="cacheon"
else
    OUTPUT_SUFFIX="cacheoff"
fi
if [ "$PROMPT_FORMAT" != "raw_chunk_first" ]; then
    PROMPT_FORMAT_TAG="${PROMPT_FORMAT//\//_}"
    PROMPT_FORMAT_TAG="${PROMPT_FORMAT_TAG// /_}"
    OUTPUT_SUFFIX="pf${PROMPT_FORMAT_TAG}-$OUTPUT_SUFFIX"
fi
if [ "$USES_TOKEN_BUDGET" = "True" ]; then
    if [ -n "${RETAIN_TOKEN_RATIO:-}" ]; then
        OUTPUT_SUFFIX="rtr${RETAIN_TOKEN_RATIO}-$OUTPUT_SUFFIX"
    elif [ -n "${FINAL_TOKEN_BUDGET:-}" ]; then
        OUTPUT_SUFFIX="ftb${FINAL_TOKEN_BUDGET}-$OUTPUT_SUFFIX"
    fi
fi
if [ "$COMPRESS_METHOD" = "colbert_chunk_rerank" ] && [ -n "${COLBERT_CHUNK_RERANK_KEEP:-}" ]; then
    OUTPUT_SUFFIX="ccrk${COLBERT_CHUNK_RERANK_KEEP}-$OUTPUT_SUFFIX"
fi
if { [ "$COMPRESS_METHOD" = "colbert_rerank" ] || [ "$COMPRESS_METHOD" = "rerank_and_region" ]; } && [ -n "${COLBERT_RERANK_KEEP:-}" ]; then
    OUTPUT_SUFFIX="crk${COLBERT_RERANK_KEEP}-$OUTPUT_SUFFIX"
fi
if { [ "$COMPRESS_METHOD" = "colbert_sliding_region" ] || [ "$COMPRESS_METHOD" = "rerank_and_region" ]; } && [ "$COLBERT_REGION_GROUP_ORDER" != "retrieval" ]; then
    OUTPUT_SUFFIX="rgo${COLBERT_REGION_GROUP_ORDER}-$OUTPUT_SUFFIX"
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

COMPRESS_ARGS=()
if [ -n "$COMPRESS_METHOD" ]; then
    COMPRESS_ARGS=(--compress_method "$COMPRESS_METHOD")
    METHOD_SUFFIX="$(echo "$COMPRESS_METHOD" | tr '/' '_')"
    OUTPUT_PATH_SUFFIX="-${METHOD_SUFFIX}-topk${TOP_K}"
    OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX}-$OUTPUT_SUFFIX"
else
    OUTPUT_PATH_SUFFIX="-topk${TOP_K}-$OUTPUT_SUFFIX"
fi
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/eval-$DATASET-$DATA_SUBDIR${OUTPUT_PATH_SUFFIX}.jsonl}"

TOTAL_NUM_ARGS=()
if [ -n "$TOTAL_NUM" ]; then
    TOTAL_NUM_ARGS=(--total_num "$TOTAL_NUM")
fi

torchrun --nproc_per_node 1 --master_port "$TORCHRUN_MASTER_PORT" src/entrypoint/eval.py \
    --dataset="$DATASET" \
    --db_dir="$DB_DIR" \
    --cache_dir="$CACHE_DIR" \
    --query_file="$DATASET_PATH/questions/query.jsonl" \
    --top_k "$TOP_K" --max_new_tokens "$MAX_NEW_TOKENS" \
    --model_name "$MODEL_NAME" \
    --use_past_cache="$EVAL_USE_PAST_CACHE" \
    --disable_rope "$DISABLE_ROPE" \
    --use_front_bos_cache "$USE_FRONT_BOS_CACHE" \
    --model_load_in_4bit "$MODEL_LOAD_IN_4BIT" \
    --prompt_format "$PROMPT_FORMAT" \
    --output_file "$OUTPUT_FILE" \
    --answer_file "$DATASET_PATH/answers/answer.jsonl" \
    --bsz "$EVAL_BSZ" \
    --use_cleaner "$USE_CLEANER" \
    "${TOTAL_NUM_ARGS[@]}" \
    "${COMPRESS_ARGS[@]}"
## HOTPOTQA 실험 ##
# BOS 유무
# Instruct 여부 (V)
# Prompting 방식
# Data Preprocessing 방식
# Top-K (V)
# meta-llama/llama-3.1-8B-Instruct
