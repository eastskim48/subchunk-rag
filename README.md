# Subchunk-RAG

Subchunk-RAG is a research codebase for optimizing retrieval-augmented
generation (RAG). It retrieves coarse chunks from a vector database, then uses
lightweight subchunk selection to construct a compact, query-relevant prompt.

## Setup

After cloning, initialize the ColBERT submodule:

```bash
git submodule update --init --recursive
```

Install Python dependencies in your experiment environment:

```bash
pip install -r requirements.txt
```

Run every command below from the repository root. With `DATASET_PREFIX=.`, the
dataset and its preprocessing artifacts are stored under `./dapr-nq-open`.

## Dataset Construction

Build the project's custom Document-Aware Passage Retrieval Natural Questions
(DAPR-NQ)/Natural Questions Open (NQ-open) dataset:

```bash
DATASET_PREFIX=. ./dataset/get_nq.sh
```

`get_nq.sh` downloads and verifies the pinned DAPR-NQ and NQ-open source files,
then constructs the 108,626-document corpus with all 2,390 custom intersection
questions and answers under `./dapr-nq-open`.

This is a custom dataset constructed by this project by exact question matching
between the DAPR-NQ test split and the NQ-open development split; it is not an
official Natural Questions benchmark setting. Dataset construction does not
build the vector database or ColBERT artifact.

`DATASET_PREFIX` selects the parent directory that contains the
`dapr-nq-open` dataset directory. The examples below use `DATASET_PREFIX=.`
consistently; replace `.` with another parent directory when needed.

## Preprocess

Database preprocessing and ColBERT artifact construction are separate stages.
First, build the representative BGE-small-en-v1.5 retrieval database of
512-token chunks with sentence-based encoding units. Under this custom
preprocessing configuration, sentences longer than 180 tokens are split:

```bash
DATASET_PREFIX=. \
DATASET=dapr-nq-open \
PREPROCESS_SUBDIR=sent-bge-small-v1.5-512-splitlong180 \
SPLITTER=sentence \
RETRIEVABLE_CHUNK_SIZE=512 \
CACHEABLE_CHUNK_SIZE=None \
MAX_SUBCHUNK_TOKENS=180 \
CHROMA_EMBED_BACKEND=bge_small_v1_5 \
CHROMA_EMBED_DEVICE=cuda \
CHROMA_EMBED_BATCH_SIZE=5461 \
MATERIALIZE_CACHE=False \
MATERIALIZE_DB=True \
DB_BATCH_SIZE=5461 \
./run/preprocess.sh
```

This writes the database to
`./dapr-nq-open/sent-bge-small-v1.5-512-splitlong180/db`.

Then build the matching ColBERT artifact:

```bash
DATASET_PREFIX=. \
DATASET=dapr-nq-open \
PREPROCESS_SUBDIR=sent-bge-small-v1.5-512-splitlong180 \
CANDIDATE_STORE_BACKEND=colbert \
COLBERT_WINDOW_BATCH_SIZE=6144 \
COLBERT_WINDOW_TOKEN_BUDGET=180 \
COLBERT_WINDOW_CENTER_UNIT=subchunk_only \
bash ./run/materialize_candidate_store.sh
```

This writes the artifact to
`./dapr-nq-open/sent-bge-small-v1.5-512-splitlong180/colbert_window`.

## Representative Experiment

Run the cache-off Subchunk-RAG experiment over all 2,390 questions:

```bash
DATASET_PREFIX=. \
DATASET=dapr-nq-open \
DATA_SUBDIR=sent-bge-small-v1.5-512-splitlong180 \
COMPRESS_METHOD=colbert_sliding_region \
CHROMA_EMBED_BACKEND=bge_small_v1_5 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
EVAL_USE_PAST_CACHE=False \
TOP_K=20 \
RETAIN_TOKEN_RATIO=0.25 \
TOTAL_NUM=2390 \
./run/eval.sh
```

With `DATASET_PREFIX=.`, `run/eval.sh` reads the files produced by the
dataset and preprocessing commands above:

```text
./dapr-nq-open/questions/query.jsonl
./dapr-nq-open/answers/answer.jsonl
./dapr-nq-open/sent-bge-small-v1.5-512-splitlong180/db
./dapr-nq-open/sent-bge-small-v1.5-512-splitlong180/colbert_window
```

All remaining evaluation settings use the defaults in `run/eval.sh`.
Evaluation outputs are written to `outputs/` unless `OUTPUT_FILE` is set.
