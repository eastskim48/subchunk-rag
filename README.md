# Subchunk-RAG

Subchunk-RAG is a research codebase for optimizing retrieval-augmented
generation (RAG). It retrieves coarse chunks from a vector database, then uses
lightweight subchunk selection to construct a compact, query-relevant prompt.

## Setup

The default build and evaluation commands target Python 3.13 and an NVIDIA driver compatible with PyTorch's CUDA 12.4 runtime. 
Retrieval-vector construction and representation payload construction both use the GPU.

After cloning, initialize the ColBERT submodule:

```bash
git submodule update --init --recursive
```

Install Python dependencies in a separate experiment environment:

```bash
python -m pip install -r requirements.txt
```

The requirements include GPU-enabled FAISS for 2-bit residual compression and
GitPython for the source ColBERT adapter. The pinned [`faiss-gpu-cu12` wheel](https://github.com/Di-Is/faiss-gpu-wheels) supports
NVIDIA compute capabilities 7.0–8.9 and uses the CUDA 12 libraries installed with
PyTorch.

```

Before the first DB build or evaluation, request access to
[`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
on Hugging Face and authenticate with an account that has access:

```bash
hf auth login
```

Run every command below from the repository root. With `DATASET_PREFIX=.`, the
dataset and its preprocessing artifacts are stored under `./dapr-nq-open`.

## Dataset Construction

See [dataset preparation](dataset/README.md#four-baseline-datasets) for all commands, BioASQ manual download steps, and exact construction scope.

```bash
DATASET_PREFIX=. ./dataset/get_nq.sh
```

`get_nq.sh` downloads and verifies the pinned DAPR-NQ and NQ-open source files,
then constructs the corpus under `./dapr-nq-open`.

`DATASET_PREFIX` selects the parent directory that contains the
`dapr-nq-open` dataset directory. The examples below use `DATASET_PREFIX=.`
consistently; replace `.` with another parent directory when needed.

## Preprocess

The common launcher runs the existing three construction steps: build retrieval DB, build subchunk catalog with
LateOn, then encode the sentences and store vectors.

For example, after constructing NQ:

```bash
bash run/preprocess.sh \
    --dataset nq \
    --dataset-root ./dapr-nq-open
```

The other dataset roots use the same launcher:

```bash
bash run/preprocess.sh --dataset bioasq --dataset-root ./bioasq14b-linked-documents
bash run/preprocess.sh --dataset newsqa --dataset-root ./newsqa
bash run/preprocess.sh --dataset covid-qa --dataset-root ./covid-qa-full-test
```

Inspect the exact commands without loading models or creating dataset artifacts:

```bash
bash run/preprocess.sh \
    --dataset nq --dataset-root ./dapr-nq-open --dry-run
```

The final DB and LateOn artifact are written beneath each dataset root:

```text
sent-bge-small-v1.5-512-splitlong100-lateon/db
sent-bge-small-v1.5-512-splitlong100-lateon/colbert_window
```

Use `--stage retrieval|subchunks|embeddings` to run one stage, or supply
`--source-db /path/to/an/existing/idle/db` to reuse a retrieval DB explicitly.

## Representative Experiment

Run the Subchunk-RAG experiment over all 2,390 questions:

```bash
DATASET_PREFIX=. \
DATASET=dapr-nq-open \
DATA_SUBDIR=sent-bge-small-v1.5-512-splitlong100-lateon \
COMPRESS_METHOD=colbert_sliding_region \
COLBERT_MODEL_NAME=lightonai/LateOn \
CHROMA_EMBED_BACKEND=bge_small_v1_5 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
TOP_K=20 \
FINAL_TOKEN_BUDGET=1000 \
TOTAL_NUM=2390 \
./run/eval.sh
```

With `DATASET_PREFIX=.`, `run/eval.sh` reads the files produced by the
dataset and preprocessing commands above:

```text
./dapr-nq-open/questions/query.jsonl
./dapr-nq-open/answers/answer.jsonl
./dapr-nq-open/sent-bge-small-v1.5-512-splitlong100-lateon/db
./dapr-nq-open/sent-bge-small-v1.5-512-splitlong100-lateon/colbert_window
```

`FINAL_TOKEN_BUDGET` is required when compression is enabled. It specifies a
fixed context-token budget; the example above uses 1,000 tokens.
All remaining evaluation settings use the defaults in `run/eval.sh`.
Evaluation outputs are written to `outputs/` unless `OUTPUT_FILE` is set.
