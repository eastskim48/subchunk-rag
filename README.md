# Subchunk-RAG

Subchunk-RAG is a research codebase for optimizing retrieval-augmented
generation (RAG). It retrieves coarse chunks from a vector database, then uses
lightweight subchunk selection to construct a compact, query-relevant prompt.

The default evidence encoder is `lightonai/LateOn`. The representative custom
configuration uses a **100-token sentence-splitting limit** and a **180-token
selection window**, both measured with the LateOn tokenizer. These are separate
settings: 100 limits individual sentence units, while 180 limits the surrounding
sentence region used for selection. Retrieval chunks and the final prompt budget
are measured with the Llama tokenizer.

## Setup

The default build and evaluation commands target Linux x86_64 with Python 3.13
and an NVIDIA driver compatible with PyTorch's CUDA 12.4 runtime. LateOn artifact
construction requires a GPU even when retrieval embeddings use `--embedding-device cpu`.

After cloning, initialize the ColBERT submodule:

```bash
git submodule update --init --recursive
```

Install Python dependencies in a separate experiment environment:

```bash
python -m pip install -r requirements.txt
```

The requirements include GPU-enabled FAISS for 2-bit residual compression and
GitPython for the source ColBERT adapter. The pinned
[`faiss-gpu-cu12` wheel](https://github.com/Di-Is/faiss-gpu-wheels) supports
NVIDIA compute capabilities 7.0–8.9 and uses the CUDA 12 libraries installed with
PyTorch. A CPU-only FAISS package cannot run the default artifact builder.

Check GPU availability in that environment:

```bash
python - <<'PY'
import faiss
import torch

assert torch.cuda.is_available(), "PyTorch cannot access an NVIDIA GPU"
assert hasattr(faiss, "StandardGpuResources"), "GPU-enabled FAISS is required"
resources = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(resources, 128)
print(f"PyTorch {torch.__version__}; GPU FAISS {faiss.__version__} ready")
PY
```

Before the first DB build or evaluation, request access to
[`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
on Hugging Face and authenticate with an account that has access:

```bash
hf auth login
```

DB construction downloads the Llama tokenizer; evaluation also downloads the
model weights. Model files already present in the local cache can be reused.

Run every command below from the repository root. With `DATASET_PREFIX=.`, the
dataset and its preprocessing artifacts are stored under `./dapr-nq-open`.

## Dataset Construction

The four baseline datasets have separate download/construction launchers and
share one preprocessing launcher. All four configurations are custom:

| Dataset | Launcher | Documents / questions | Required local input |
| --- | --- | --- | --- |
| NQ | `dataset/get_nq.sh` | 108,626 / 2,390 | None; pinned public sources are downloaded |
| BioASQ | `dataset/get_bioasq.sh` | 58,012 / 4,366 | Manually downloaded training14b ZIP passed with `--archive` |
| NewsQA | `dataset/get_newsqa.sh` | 12,744 / 4,293 | None; both original archives are downloaded and verified |
| COVID-QA | `dataset/get_covid_qa.sh` | 147 / 2,019 | None; the preserved official JSON version is downloaded |

See [dataset preparation](dataset/README.md#four-baseline-datasets) for all
commands, BioASQ manual download steps, and exact construction scope. These
entrypoints preserve the existing question/document rules; they do not evaluate
models. Existing dataset directories are not overwritten.

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

The common launcher runs the existing three construction steps: build a BGE-small
512-Llama-token retrieval DB, copy it and rebuild sentence/region metadata with
LateOn limits 100/180, then encode the sentences and store 2-bit residual vectors.
The source DB is retained. The metadata step reuses its indexed texts, vectors,
and retrieval graph; it does not generate new retrieval embeddings.

For example, after constructing NQ:

```bash
bash run/preprocess_baseline.sh \
    --dataset nq \
    --dataset-root ./dapr-nq-open
```

The other dataset roots use the same launcher:

```bash
bash run/preprocess_baseline.sh --dataset bioasq --dataset-root ./bioasq14b-linked-documents
bash run/preprocess_baseline.sh --dataset newsqa --dataset-root ./newsqa
bash run/preprocess_baseline.sh --dataset covid-qa --dataset-root ./covid-qa-full-test
```

Inspect the exact commands without loading models or creating dataset artifacts:

```bash
bash run/preprocess_baseline.sh \
    --dataset nq --dataset-root ./dapr-nq-open --dry-run
```

The final DB and LateOn artifact are written beneath each dataset root:

```text
sent-bge-small-v1.5-512-splitlong100-lateon/db
sent-bge-small-v1.5-512-splitlong100-lateon/colbert_window
```

Logs and the command plan are saved under `outputs/temp/`. Existing outputs are
refused. Use `--stage retrieval|regions|embeddings` to run one stage, or supply
`--source-db /path/to/an/existing/idle/db` to reuse a retrieval DB explicitly.
The default materialization encoder is `lightonai/LateOn`, with sentence-only
encoding, region window 180 and compression seed 123. The `COLBERT_*` variable
names and `colbert` backend name remain the shared interface names.

## Representative Experiment

Run the cache-off Subchunk-RAG experiment over all 2,390 questions:

```bash
DATASET_PREFIX=. \
DATASET=dapr-nq-open \
DATA_SUBDIR=sent-bge-small-v1.5-512-splitlong100-lateon \
COMPRESS_METHOD=colbert_sliding_region \
COLBERT_MODEL_NAME=lightonai/LateOn \
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
./dapr-nq-open/sent-bge-small-v1.5-512-splitlong100-lateon/db
./dapr-nq-open/sent-bge-small-v1.5-512-splitlong100-lateon/colbert_window
```

To reuse an existing artifact stored under another name, set
`COLBERT_WINDOW_DIR` to its directory. For example, the recent local execution
test used the same database family with an existing artifact named
`colbert_lateon_sentence_2bit`, whose recorded selection window is 180 tokens.

All remaining evaluation settings use the defaults in `run/eval.sh`.
Evaluation outputs are written to `outputs/` unless `OUTPUT_FILE` is set.
