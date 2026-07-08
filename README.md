# Subchunk-RAG

Subchunk-RAG is research code for RAG context compression. The main method
uses a vector DB for **wide retrieval**, then applies lightweight **fine
selection** to sentence-level regions before LLM inference.

The current main configuration is:

- retrieval DB: `default` Chroma embedding backend
- chunking: sentence-based retrievable chunks
- compression method: `colbert_sliding_region`
- ColBERT source: `third_party/ColBERT` submodule

For method and evaluation details, see `docs/method.md` and
`docs/eval_protocol.md`.

## Setup

After cloning, initialize the ColBERT submodule:

```bash
git submodule update --init --recursive
```

Install Python dependencies in your experiment environment:

```bash
pip install -r requirements.txt
```

The run scripts expect datasets in this layout:

```text
/mnt/nvme1/datasets/<dataset>/
  documents/
  questions/query.jsonl
  answers/answer.jsonl
```

Set `DATASET_PREFIX` if the datasets live somewhere else.

## Preprocess

Build a sentence-based default Chroma DB and ColBERT window artifact:

```bash
DATASET=longbench-hotpotqa \
PREPROCESS_SUBDIR=sent-default-512 \
SPLITTER=sentence \
RETRIEVABLE_CHUNK_SIZE=512 \
CACHEABLE_CHUNK_SIZE=None \
CHROMA_EMBED_BACKEND=default \
MATERIALIZE_CACHE=False \
MATERIALIZE_DB=True \
MATERIALIZE_COMPARE_EMBEDS=False \
MATERIALIZE_COLBERT_WINDOW=True \
./run/preprocess.sh
```

Outputs are written under:

```text
$DATASET_PATH/$PREPROCESS_SUBDIR/db
$DATASET_PATH/$PREPROCESS_SUBDIR/colbert_window
```

If the DB already exists and only the ColBERT window artifact is needed, use:

```bash
DATASET=longbench-hotpotqa \
PREPROCESS_SUBDIR=sent-default-512 \
./run/preprocess_colbert_window.sh
```

## Eval

Run the main cache-off Subchunk-RAG evaluation:

```bash
DATASET=longbench-hotpotqa \
DATA_SUBDIR=sent-default-512 \
COMPRESS_METHOD=colbert_sliding_region \
EVAL_USE_PAST_CACHE=False \
TOP_K=20 \
RETAIN_TOKEN_RATIO=0.4 \
./run/eval.sh
```

Useful budget knobs:

- `RETAIN_TOKEN_RATIO`: final prompt budget as a ratio of retrieved tokens
- `COLBERT_FINAL_TOKEN_BUDGET`: absolute final prompt token budget
- `GLOBAL_TOP_R`: fallback global sentence ratio when no final budget is active
- `COLBERT_SLIDING_WINDOW_TOKEN_BUDGET`: candidate region/window size

Evaluation outputs are written to `outputs/` unless `OUTPUT_FILE` is set.

## Grid Search

Run one grid YAML directly:

```bash
python run/grid_search/eval.py run/grid_search/grid.yaml
```

Run one or more grid YAMLs through the GPU wrapper:

```bash
run/run_grid.sh --wait-gpu --interval 15 \
  run/grid_search/grid.yaml \
  run/grid_search/grid_matkv.yaml
```

`run/run_grid.sh` defaults to GPU `0`, enables NVIDIA
`EXCLUSIVE_PROCESS` mode, and restores the previous compute mode on exit.
Use `--no-exclusive` to skip sudo and compute-mode changes.  If one grid
fails, later grid YAMLs still run; the wrapper exits with code `1` if any grid
failed.

Use `--help` for all wrapper options:

```bash
run/run_grid.sh --help
```
