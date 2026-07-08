#!/bin/bash
set -euo pipefail

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
export TOP_K="${TOP_K:-20}"
export COMPRESS_METHOD_A="${COMPRESS_METHOD_A:-compare_all_materialized}"
export COMPRESS_METHOD_B="${COMPRESS_METHOD_B:-bm25_global}"
export GLOBAL_TOP_R="${GLOBAL_TOP_R:-0.1}"
export EVAL_BSZ="${EVAL_BSZ:-4}"
export TOTAL_NUM="${TOTAL_NUM:-200}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export GOLD_FIELD="${GOLD_FIELD:-supporting_facts}"

OUTPUT_SUFFIX="topk${TOP_K}-gtr${GLOBAL_TOP_R}"
if [ -n "${HYBRID_BGE_WEIGHT:-}" ]; then
    OUTPUT_SUFFIX="hbw${HYBRID_BGE_WEIGHT}-${OUTPUT_SUFFIX}"
fi
if [ -n "${HYBRID_RRF_K:-}" ]; then
    OUTPUT_SUFFIX="rrfk${HYBRID_RRF_K}-${OUTPUT_SUFFIX}"
fi

METHOD_A_SUFFIX="${COMPRESS_METHOD_A:-none}"
METHOD_B_SUFFIX="${COMPRESS_METHOD_B:-none}"
OUTPUT_FILE="${OUTPUT_FILE:-./outputs/eval-retrieval-compression-pair-$DATASET-$DATA_SUBDIR-${METHOD_A_SUFFIX}_vs_${METHOD_B_SUFFIX}-$OUTPUT_SUFFIX-summary.json}"
DETAILS_FILE="${DETAILS_FILE:-./outputs/eval-retrieval-compression-pair-$DATASET-$DATA_SUBDIR-${METHOD_A_SUFFIX}_vs_${METHOD_B_SUFFIX}-$OUTPUT_SUFFIX-details.jsonl}"

if [ -n "${SAMPLE_FILE:-}" ]; then
    SAMPLE_ARGS=(--sample_file "$SAMPLE_FILE")
else
    SAMPLE_ARGS=()
fi

python test/eval_retrieval_compression_pair.py \
    --dataset "$DATASET" \
    --db_dir "$DATASET_PATH/$DATA_SUBDIR/db" \
    --query_file "$DATASET_PATH/questions/query.jsonl" \
    --top_k "$TOP_K" \
    --total_num "$TOTAL_NUM" \
    --bsz "$EVAL_BSZ" \
    --method_a "$COMPRESS_METHOD_A" \
    --method_b "$COMPRESS_METHOD_B" \
    --gold_field "$GOLD_FIELD" \
    --model_name "$MODEL_NAME" \
    --output_file "$OUTPUT_FILE" \
    --details_file "$DETAILS_FILE" \
    "${SAMPLE_ARGS[@]}"
