#!/bin/bash

export DATASET="${DATASET:-longbench-hotpotqa}"
if [ -n "$DATASET_PREFIX" ]; then
    export DATASET_PATH="$DATASET_PREFIX/$DATASET"
elif [ -d "/mnt/nvme1/datasets/$DATASET" ]; then
    export DATASET_PATH="/mnt/nvme1/datasets/$DATASET"
else
export DATASET_PATH="$DATASET"
fi

export INPUT_DOCS_SUBDIR="${INPUT_DOCS_SUBDIR:-documents}"
export OUTPUT_MAPPING_SUBDIR="${OUTPUT_MAPPING_SUBDIR:-${OUTPUT_DOCS_SUBDIR:-pn_mapping}}"
export SENTENCE_RESOLVER="${SENTENCE_RESOLVER:-openai}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
export FASTCOREF_MODEL_NAME="${FASTCOREF_MODEL_NAME:-biu-nlp/f-coref}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export RETRIEVABLE_CHUNK_SIZE="${RETRIEVABLE_CHUNK_SIZE:-1024}"
export OVERWRITE="${OVERWRITE:-False}"
export NUM_WORKERS="${NUM_WORKERS:-1}"

export INPUT_DOCS_DIR="$DATASET_PATH/$INPUT_DOCS_SUBDIR"
export OUTPUT_MAPPING_DIR="$DATASET_PATH/$OUTPUT_MAPPING_SUBDIR"

mkdir -p "$OUTPUT_MAPPING_DIR"

python src/entrypoint/pronoun_resolve.py \
    --input_docs_dir "$INPUT_DOCS_DIR" \
    --output_mapping_dir "$OUTPUT_MAPPING_DIR" \
    --resolver "$SENTENCE_RESOLVER" \
    --openai_model "$OPENAI_MODEL" \
    --fastcoref_model_name "$FASTCOREF_MODEL_NAME" \
    --tokenizer_model "$TOKENIZER_MODEL" \
    --retrievable_chunk_size "$RETRIEVABLE_CHUNK_SIZE" \
    --overwrite "$OVERWRITE" \
    --num_workers "$NUM_WORKERS"








