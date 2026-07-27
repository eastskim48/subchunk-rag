#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET="${DATASET:-hotpotqa-lb200-original}"
if [ -n "${DATASET_PREFIX:-}" ]; then
    export DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    export DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
    export DATASET_PATH="$DATASET"
fi

export DATA_SUBDIR="${DATA_SUBDIR:-sent-original-512}"
export CHROMA_EMBED_DEVICE="${CHROMA_EMBED_DEVICE:-cpu}"
export DENSE_EMBED_DEVICE="${DENSE_EMBED_DEVICE:-cpu}"
export DENSE_EMBED_DIR="${DENSE_EMBED_DIR:-$DATASET_PATH/$DATA_SUBDIR/dense_embed}"
export COLBERT_WINDOW_DIR="${COLBERT_WINDOW_DIR:-$DATASET_PATH/$DATA_SUBDIR/colbert_window}"
export COLBERT_MODEL_NAME="${COLBERT_MODEL_NAME:-colbert-ir/colbertv2.0}"
export COLBERT_BATCH_SIZE="${COLBERT_BATCH_SIZE:-32}"
export COLBERT_REPO_PATH="${COLBERT_REPO_PATH:-$REPO_ROOT/third_party/ColBERT}"
export COLBERT_QUERY_MAXLEN="${COLBERT_QUERY_MAXLEN:-}"
export COLBERT_QUERY_MINLEN="${COLBERT_QUERY_MINLEN:-}"
export COLBERT_QUERY_TRUNCATION_SIDE="${COLBERT_QUERY_TRUNCATION_SIDE:-right}"
export TOP_K="${TOP_K:-20}"
export COMPRESS_METHOD="${COMPRESS_METHOD:-none}"
export EVAL_BSZ="${EVAL_BSZ:-8}"
export TOTAL_NUM="${TOTAL_NUM:-200}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export SAMPLE_FILE="${SAMPLE_FILE:-$DATASET_PATH/evidence_labels.json}"
export PASSAGE_RECALL_THRESHOLD="${PASSAGE_RECALL_THRESHOLD:-0.8}"

if [ ! -f "$SAMPLE_FILE" ]; then
    echo "missing SAMPLE_FILE=$SAMPLE_FILE" >&2
    exit 1
fi

OUTPUT_SUFFIX="topk${TOP_K}"

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
    if [ -n "${RETAIN_TOKEN_RATIO:-}" ]; then
        OUTPUT_SUFFIX="rtr${RETAIN_TOKEN_RATIO}-$OUTPUT_SUFFIX"
    else
        OUTPUT_SUFFIX="ftb${FINAL_TOKEN_BUDGET}-$OUTPUT_SUFFIX"
    fi
fi
if [ "$COMPRESS_METHOD" = "colbert_chunk_rerank" ] && [ -n "${COLBERT_CHUNK_RERANK_KEEP:-}" ]; then
    OUTPUT_SUFFIX="ccrk${COLBERT_CHUNK_RERANK_KEEP}-$OUTPUT_SUFFIX"
fi
if { [ "$COMPRESS_METHOD" = "colbert_rerank" ] || [ "$COMPRESS_METHOD" = "rerank_and_region" ]; } && [ -n "${COLBERT_RERANK_KEEP:-}" ]; then
    OUTPUT_SUFFIX="crk${COLBERT_RERANK_KEEP}-$OUTPUT_SUFFIX"
fi

METHOD_SUFFIX="${COMPRESS_METHOD:-none}"
if [[ "$COMPRESS_METHOD" == colbert_* || "$COMPRESS_METHOD" == "rerank_and_region" ]]; then
    if [ -n "$COLBERT_QUERY_MAXLEN" ]; then
        OUTPUT_SUFFIX="qmax${COLBERT_QUERY_MAXLEN}-$OUTPUT_SUFFIX"
    fi
    if [ -n "$COLBERT_QUERY_MINLEN" ]; then
        OUTPUT_SUFFIX="qmin${COLBERT_QUERY_MINLEN}-$OUTPUT_SUFFIX"
    fi
    OUTPUT_SUFFIX="qside${COLBERT_QUERY_TRUNCATION_SIDE}-$OUTPUT_SUFFIX"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/retrieval_eval/$DATASET}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/$DATA_SUBDIR-$METHOD_SUFFIX-$OUTPUT_SUFFIX-summary.json}"
DETAILS_FILE="${DETAILS_FILE:-$OUTPUT_DIR/$DATA_SUBDIR-$METHOD_SUFFIX-$OUTPUT_SUFFIX-details.jsonl}"

COMPRESS_ARGS=()
if [ -n "$COMPRESS_METHOD" ] && [ "$COMPRESS_METHOD" != "none" ]; then
    COMPRESS_ARGS=(--compress_method "$COMPRESS_METHOD")
fi

python "$REPO_ROOT/src/entrypoint/eval_retrieval_only.py" \
    --dataset "$DATASET" \
    --db_dir "$DATASET_PATH/$DATA_SUBDIR/db" \
    --query_file "$DATASET_PATH/questions/query.jsonl" \
    --top_k "$TOP_K" \
    --total_num "$TOTAL_NUM" \
    --bsz "$EVAL_BSZ" \
    --model_name "$MODEL_NAME" \
    --passage_recall_threshold "$PASSAGE_RECALL_THRESHOLD" \
    --sample_file "$SAMPLE_FILE" \
    --output_file "$OUTPUT_FILE" \
    --details_file "$DETAILS_FILE" \
    "${COMPRESS_ARGS[@]}"
