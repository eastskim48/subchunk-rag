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
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$DATA_SUBDIR/colbert_window}"
export COLBERT_MODEL_NAME="${COLBERT_MODEL_NAME:-colbert-ir/colbertv2.0}"
export COLBERT_BATCH_SIZE="${COLBERT_BATCH_SIZE:-32}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export TOP_K="${TOP_K:-20}"
export COMPRESS_METHOD="${COMPRESS_METHOD-dense}"
export GLOBAL_TOP_R="${GLOBAL_TOP_R:-0.1}"
export EVAL_BSZ="${EVAL_BSZ:-8}"
export TOTAL_NUM="${TOTAL_NUM:-200}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export GOLD_FIELD="${GOLD_FIELD:-llm_valid_evidence_spans}"

OUTPUT_SUFFIX="topk${TOP_K}"
case "$COMPRESS_METHOD" in
    dense)
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-gtr${GLOBAL_TOP_R}"
        ;;
    colbert_subchunk|colbert_sliding_region|rerank_and_region)
        if [ -n "${RETAIN_TOKEN_RATIO:-}" ]; then
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-rtr${RETAIN_TOKEN_RATIO}"
        elif [ -n "${COLBERT_FINAL_TOKEN_BUDGET:-}" ]; then
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-cfb${COLBERT_FINAL_TOKEN_BUDGET}"
        else
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-gtr${GLOBAL_TOP_R}"
        fi
        ;;
    colbert_window_budget)
        if [ -n "${RETAIN_TOKEN_RATIO:-}" ]; then
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-rtr${RETAIN_TOKEN_RATIO}"
        elif [ -n "${COLBERT_FINAL_TOKEN_BUDGET:-}" ]; then
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-cfb${COLBERT_FINAL_TOKEN_BUDGET}"
        fi
        ;;
    colbert_rerank)
        if [ -n "${COLBERT_RERANK_KEEP:-}" ]; then
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-crk${COLBERT_RERANK_KEEP}"
        fi
        ;;
    colbert_chunk_rerank)
        if [ -n "${COLBERT_CHUNK_RERANK_KEEP:-}" ]; then
            OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-ccrk${COLBERT_CHUNK_RERANK_KEEP}"
        fi
        ;;
esac
if [ "$COMPRESS_METHOD" = "rerank_and_region" ] && [ -n "${COLBERT_RERANK_KEEP:-}" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}-crk${COLBERT_RERANK_KEEP}"
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
