#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET="${DATASET:-dapr-nq-open}"
if [ -n "${DATASET_PREFIX:-}" ]; then
    DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
    DATASET_PATH="$DATASET"
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_LOAD_IN_4BIT="${MODEL_LOAD_IN_4BIT:-False}"
EVAL_BSZ="${EVAL_BSZ:-64}"
TOTAL_NUM="${TOTAL_NUM:-2390}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
MEASURE_PROMPT_STATS="${MEASURE_PROMPT_STATS:-True}"
USE_CLEANER="${USE_CLEANER:-False}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw_chunk_first}"
TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"
GOLD_EVIDENCE_FILE="${GOLD_EVIDENCE_FILE:-$DATASET_PATH/dataset_info/evidence_labels.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$REPO_ROOT/outputs/eval-$DATASET-gold-evidence.jsonl}"

export MEASURE_PROMPT_STATS

torchrun --nproc_per_node 1 --master_port "$TORCHRUN_MASTER_PORT" \
    src/entrypoint/eval.py \
    --dataset "$DATASET" \
    --query_file "$DATASET_PATH/questions/query.jsonl" \
    --answer_file "$DATASET_PATH/answers/answer.jsonl" \
    --gold_evidence_file "$GOLD_EVIDENCE_FILE" \
    --db_dir "$DATASET_PATH" \
    --cache_dir "$DATASET_PATH" \
    --top_k 1 \
    --use_past_cache False \
    --bsz "$EVAL_BSZ" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --total_num "$TOTAL_NUM" \
    --output_file "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --model_load_in_4bit "$MODEL_LOAD_IN_4BIT" \
    --prompt_format "$PROMPT_FORMAT" \
    --use_cleaner "$USE_CLEANER"
