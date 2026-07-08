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

export DATA_SUBDIR="${DATA_SUBDIR:-sent}"
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
export TOP_K="${TOP_K:-20}"
export COMPRESS_METHOD="${COMPRESS_METHOD-compare_all_materialized}"
export GLOBAL_TOP_R="${GLOBAL_TOP_R:-0.1}"
export EVAL_BSZ="${EVAL_BSZ:-8}"
export TOTAL_NUM="${TOTAL_NUM:-200}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export GOLD_FIELD="${GOLD_FIELD:-llm_valid_evidence_spans}"

OUTPUT_SUFFIX="topk${TOP_K}-gtr${GLOBAL_TOP_R}"
if [ -n "${HYBRID_BGE_WEIGHT:-}" ]; then
    OUTPUT_SUFFIX="hbw${HYBRID_BGE_WEIGHT}-${OUTPUT_SUFFIX}"
fi
if [ -n "${HYBRID_RRF_K:-}" ]; then
    OUTPUT_SUFFIX="rrfk${HYBRID_RRF_K}-${OUTPUT_SUFFIX}"
fi
if [ -n "${TITLE_RRF_K:-}" ]; then
    OUTPUT_SUFFIX="trrfk${TITLE_RRF_K}-${OUTPUT_SUFFIX}"
fi

METHOD_SUFFIX="${COMPRESS_METHOD:-none}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/eval-retrieval-compression-$DATASET-$DATA_SUBDIR-$METHOD_SUFFIX-$OUTPUT_SUFFIX-summary.json}"
DETAILS_FILE="${DETAILS_FILE:-./outputs/eval-retrieval-compression-$DATASET-$DATA_SUBDIR-$METHOD_SUFFIX-$OUTPUT_SUFFIX-details.jsonl}"

COMPRESS_ARGS=()
if [ -n "$COMPRESS_METHOD" ] && [ "$COMPRESS_METHOD" != "none" ]; then
    COMPRESS_ARGS=(--compress_method "$COMPRESS_METHOD")
fi
if [ -n "${SAMPLE_FILE:-}" ]; then
    SAMPLE_ARGS=(--sample_file "$SAMPLE_FILE")
else
    SAMPLE_ARGS=()
fi

python test/eval_retrieval_and_compression.py \
    --dataset "$DATASET" \
    --db_dir "$DATASET_PATH/$DATA_SUBDIR/db" \
    --query_file "$DATASET_PATH/questions/query.jsonl" \
    --top_k "$TOP_K" \
    --total_num "$TOTAL_NUM" \
    --bsz "$EVAL_BSZ" \
    --gold_field "$GOLD_FIELD" \
    --model_name "$MODEL_NAME" \
    --output_file "$OUTPUT_FILE" \
    --details_file "$DETAILS_FILE" \
    "${COMPRESS_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}"
