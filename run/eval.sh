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
export PROVENCE_MODEL_NAME="${PROVENCE_MODEL_NAME:-naver/provence-reranker-debertav3-v1}"
export PROVENCE_THRESHOLD="${PROVENCE_THRESHOLD:-0.05}"

export DEFAULT_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
export EXIT_MODEL_NAME="${EXIT_MODEL_NAME:-doubleyyh/exit-gemma-2b}"
export EXIT_BASE_MODEL_NAME="${EXIT_BASE_MODEL_NAME:-google/gemma-2b-it}"
export EXIT_THRESHOLD="${EXIT_THRESHOLD:-0.1}"
export DATA_SUBDIR="${DATA_SUBDIR:-sent}"
export CACHE_SUBDIR="${CACHE_SUBDIR:-cache}"
export DB_DIR="${DB_DIR:-$DATASET_PATH/$DATA_SUBDIR/db}"
export CACHE_DIR="${CACHE_DIR:-$DATASET_PATH/$DATA_SUBDIR/$CACHE_SUBDIR}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cpu}"
export COMPARE_EMBED_DEVICE="${COMPARE_EMBED_DEVICE:-cpu}"
export COMPARE_EMBED_DIR="${COMPARE_EMBED_DIR:-$DATASET_PATH/$DATA_SUBDIR/compare_embed}"
export TITLE_COMPARE_EMBED_DIR="${TITLE_COMPARE_EMBED_DIR:-$DATASET_PATH/sent-title-test/compare_embed}"
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$DATA_SUBDIR/colbert_window}"
export COLBERT_MODEL_NAME="${COLBERT_MODEL_NAME:-colbert-ir/colbertv2.0}"
export COLBERT_DEVICE="${COLBERT_DEVICE:-cpu}"
export COLBERT_BATCH_SIZE="${COLBERT_BATCH_SIZE:-32}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export COLBERT_DISABLE_CPU_EXTENSION="${COLBERT_DISABLE_CPU_EXTENSION:-True}"
export COLBERT_USE_COMPACT_ARTIFACT="${COLBERT_USE_COMPACT_ARTIFACT:-True}"
export COLBERT_CLEAR_INTER_BATCH_CACHE="${COLBERT_CLEAR_INTER_BATCH_CACHE:-True}"
export COLBERT_WARMUP_QUERY_ENCODER="${COLBERT_WARMUP_QUERY_ENCODER:-True}"
export EVAL_USE_PAST_CACHE="${EVAL_USE_PAST_CACHE:-False}"
export DISABLE_ROPE="${DISABLE_ROPE:-False}"
export USE_FRONT_BOS_CACHE="${USE_FRONT_BOS_CACHE:-False}"
export MODEL_LOAD_IN_4BIT="${MODEL_LOAD_IN_4BIT:-False}"
export MEASURE_PROMPT_STATS="${MEASURE_PROMPT_STATS:-True}"
export RETRIEVAL_INCLUDE_DOCUMENTS="${RETRIEVAL_INCLUDE_DOCUMENTS:-True}"
export USE_CLEANER="${USE_CLEANER:-False}"
export EVAL_BSZ="${EVAL_BSZ:-4}"
export TOP_K="${TOP_K:-20}"
export TOTAL_NUM="${TOTAL_NUM:-200}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
export COMPRESS_METHOD="${COMPRESS_METHOD-compare_all_materialized}"
export GLOBAL_TOP_R="${GLOBAL_TOP_R:-0.1}"
export TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"
if [ "$EVAL_USE_PAST_CACHE" = "True" ]; then
    OUTPUT_SUFFIX="cacheon"
else
    OUTPUT_SUFFIX="cacheoff"
fi
if [ -n "$HYBRID_BGE_WEIGHT" ]; then
    OUTPUT_SUFFIX="hbw${HYBRID_BGE_WEIGHT}-$OUTPUT_SUFFIX"
fi
if [ -n "$TITLE_RRF_K" ]; then
    OUTPUT_SUFFIX="trrfk${TITLE_RRF_K}-$OUTPUT_SUFFIX"
fi
if [ -n "${COLBERT_FINAL_TOKEN_BUDGET:-}" ]; then
    OUTPUT_SUFFIX="cfb${COLBERT_FINAL_TOKEN_BUDGET}-$OUTPUT_SUFFIX"
fi
if [ -n "${COLBERT_CHUNK_RERANK_KEEP:-}" ]; then
    OUTPUT_SUFFIX="ccrk${COLBERT_CHUNK_RERANK_KEEP}-$OUTPUT_SUFFIX"
fi
if [ -n "${RETAIN_TOKEN_RATIO:-}" ]; then
    OUTPUT_SUFFIX="rtr${RETAIN_TOKEN_RATIO}-$OUTPUT_SUFFIX"
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
    if [ "$COMPRESS_METHOD" = "provence" ]; then
        METHOD_SUFFIX="${METHOD_SUFFIX}-pth${PROVENCE_THRESHOLD}"
    elif [ "$COMPRESS_METHOD" = "exit" ]; then
        METHOD_SUFFIX="${METHOD_SUFFIX}-eth${EXIT_THRESHOLD}"
    fi
    OUTPUT_PATH_SUFFIX="-${METHOD_SUFFIX}-topk${TOP_K}-gtr${GLOBAL_TOP_R}-$OUTPUT_SUFFIX"
else
    OUTPUT_PATH_SUFFIX="-topk${TOP_K}-$OUTPUT_SUFFIX"
fi
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/eval-$DATASET-$DATA_SUBDIR${OUTPUT_PATH_SUFFIX}.jsonl}"

torchrun --nproc_per_node 1 --master_port "$TORCHRUN_MASTER_PORT" src/entrypoint/eval.py \
    --dataset="$DATASET" \
    --db_dir="$DB_DIR" \
    --cache_dir="$CACHE_DIR" \
    --query_file="$DATASET_PATH/questions/query.jsonl" \
    --top_k "$TOP_K" --max_new_tokens "$MAX_NEW_TOKENS" --total_num "$TOTAL_NUM" \
    --model_name "$MODEL_NAME" \
    --use_past_cache="$EVAL_USE_PAST_CACHE" \
    --disable_rope "$DISABLE_ROPE" \
    --use_front_bos_cache "$USE_FRONT_BOS_CACHE" \
    --model_load_in_4bit "$MODEL_LOAD_IN_4BIT" \
    --output_file "$OUTPUT_FILE" \
    --answer_file "$DATASET_PATH/answers/answer.jsonl" \
    --bsz "$EVAL_BSZ" \
    --use_cleaner "$USE_CLEANER" \
    "${COMPRESS_ARGS[@]}"
## HOTPOTQA 실험 ##
# BOS 유무
# Instruct 여부 (V)
# Prompting 방식
# Data Preprocessing 방식
# Top-K (V)
# meta-llama/llama-3.1-8B-Instruct
