#!/bin/bash

export DATASET="${DATASET:-longbench-hotpotqa}"
export DATASET_PATH="$DATASET_PREFIX/$DATASET"
export SENTENCE_DB_SUBDIR="${SENTENCE_DB_SUBDIR:-sent_sentence/db}"
export EVAL_USE_PAST_CACHE="${EVAL_USE_PAST_CACHE:-True}"
export EVAL_TOP_K="${EVAL_TOP_K:-10}"
export DISABLE_ROPE="${DISABLE_ROPE:-False}"

torchrun --nproc_per_node 1 src/entrypoint/eval.py \
    --db_dir="$DATASET_PATH/$SENTENCE_DB_SUBDIR" \
    --cache_dir="$DATASET_PATH/sent/cache" \
    --query_file="$DATASET_PATH/questions/query.jsonl" \
    --top_k "$EVAL_TOP_K" --bsz 1 --max_new_tokens 20 --total_num 200 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --use_past_cache=False \
    --disable_rope "$DISABLE_ROPE" \
    --output_file "./outputs/eval-sentence-retrieval-$DATASET.jsonl" \
    --answer_file "$DATASET_PATH/answers/answer.jsonl" \
    --compress_method "compare_all"
