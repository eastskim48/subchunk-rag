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

DB preprocessing and candidate-store construction are separate steps. First,
build a sentence-based Chroma DB:

```bash
DATASET=longbench-hotpotqa \
PREPROCESS_SUBDIR=sent-bge-small-v1.5-512 \
SPLITTER=sentence \
RETRIEVABLE_CHUNK_SIZE=512 \
CACHEABLE_CHUNK_SIZE=None \
CHROMA_EMBED_BACKEND=bge_small_v1_5 \
CHROMA_EMBED_DEVICE=cuda \
CHROMA_EMBED_BATCH_SIZE=128 \
MATERIALIZE_CACHE=False \
MATERIALIZE_DB=True \
DB_BATCH_SIZE=256 \
./run/preprocess.sh
```

`run/preprocess.sh` defaults to `SPLITTER=sentence` and
`MAX_SUBCHUNK_TOKENS=180`, so sentence units longer than 180 source-tokenizer
tokens are split before coarse chunks are built. Set
`MAX_SUBCHUNK_TOKENS=None` to disable this limit. Semantic preprocessing is
still available, but must explicitly set both `SPLITTER=semantic` and a
supported `MERGER`. Preprocessing creates cache and DB directories only when
their corresponding `MATERIALIZE_*` flag is enabled. Dense and ColBERT outputs
are created only by the candidate-store workflow below.

`DB_BATCH_SIZE` controls how many retrievable chunks are sent to Chroma in one
upsert and is recorded in `build_manifest.json`. It does not change chunk text
or IDs, but changing the insertion batching can change the approximate HNSW
graph. Comparisons must therefore share the same built DB rather than rebuild
one method's DB with a different batch size.

The implicit retrieval-backend default is `bge_small_v1_5`. Preprocessing
records `CHROMA_EMBED_BACKEND`, `CHROMA_EMBED_DEVICE`, and
`CHROMA_EMBED_BATCH_SIZE`. Evaluation does not inherit the build device:
ordinary `ChromaDB(db_dir)` construction always pins runtime query embedding
to CPU and exposes no device option. CUDA is reachable only through the
preprocessing-only `ChromaDB.for_build(...)` constructor.

This writes the DB and its preprocessing manifest under:

```text
$DATASET_PATH/$PREPROCESS_SUBDIR/db
```

After the DB manifest exists, build and validate the ColBERT materialized
candidate store:

```bash
DATASET=longbench-hotpotqa \
PREPROCESS_SUBDIR=sent-default-512 \
CANDIDATE_STORE_BACKEND=colbert \
./run/materialize_candidate_store.sh
```

`colbert` is the default backend. Its artifact construction always runs on CUDA
and disables the optional ColBERT tensorization cross-check. Set
`COLBERT_MODEL_NAME` to use a checkpoint other than
`colbert-ir/colbertv2.0`; the same variable is used at runtime.
The candidate IDs, texts, and source order are loaded directly from the
cacheable metadata persisted in the DB; ColBERT materialization does not parse
the source documents again. The requested ColBERT center unit must match the DB
splitter recorded in `build_manifest.json`.

The ColBERT candidate store is written under:

```text
$DATASET_PATH/$PREPROCESS_SUBDIR/colbert_window
```

Its JSON indexes contain scalar configuration and counts. Corpus-wide mappings
are split across `cacheable_rows.json`, build-only `window_ids.json`, and
`region_payloads.json`. Runtime initialization eagerly loads only the
cacheable-row and retrieval-region mappings into Python dicts; window
membership is not loaded. Token vectors remain memory-mapped in
`vectors.fp16.bin` with `offsets.npy`.

The same DB cacheable reader is used to build the dense baseline candidate
store:

```bash
DATASET=longbench-hotpotqa \
PREPROCESS_SUBDIR=sent-default-512 \
CANDIDATE_STORE_BACKEND=dense \
./run/materialize_candidate_store.sh
```

This writes `dense_embed/dense_embed.pt`. The dense cache unit is derived from
the DB build manifest rather than repeated in the command.

`run/preprocess.sh` does not build candidate artifacts, and
`run/materialize_candidate_store.sh` does not rebuild the DB.

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
- `FINAL_TOKEN_BUDGET`: absolute final prompt token budget

Exactly one budget control is required for `dense`, `colbert_subchunk`,
`colbert_sliding_region`, and `rerank_and_region`.

Candidate region size is fixed by the ColBERT artifact's stored window budget.

Evaluation outputs are written to `outputs/` unless `OUTPUT_FILE` is set.

### Retrieval and evidence-coverage evaluation

Use `run/eval_retrieval_only.sh` to measure whether retrieval or
context compression retains labeled evidence text. This path does not generate
LLM answers or compute answer EM/F1. It retrieves documents, optionally applies
the selected compression method, and compares the resulting context text with
the gold evidence text.

For example:

```bash
DATASET=dapr-nq-open \
DATA_SUBDIR=sent-bge-small-v1.5-512-splitlong180 \
EVIDENCE_FILE=/mnt/nvme1/datasets/dapr-nq-open/dataset_info/evidence_labels.jsonl \
COMPRESS_METHOD=colbert_sliding_region \
TOP_K=20 \
RETAIN_TOKEN_RATIO=0.25 \
EVAL_BSZ=32 \
./run/eval_retrieval_only.sh
```

This retrieval-only path uses the project's `text_evidence_exact` metric and a
project-specific custom evidence-label schema. It is not an official
dataset-wide evidence format. The label file may be JSONL, a JSON array, or a
JSON object with a `records` array. Each record must have this shape:

```json
{
  "query": "Who wrote Hamlet?",
  "evidence_passage_ids": ["passage-0"],
  "evidence_texts": ["Hamlet was written by William Shakespeare."]
}
```

The required invariants are:

- `query` is a non-empty string, is unique within the label file, and exactly
  matches the corresponding string in `questions/query.jsonl`.
- `evidence_passage_ids` and `evidence_texts` are lists of equal length with at
  least one entry.
- Every passage ID and evidence text is a non-empty string.
- Passage IDs are stable labels for result reporting; they do not need to
  match vector-DB document or chunk IDs.
- Additional fields such as `source_id`, `answers`, `document_file`, or source
  spans are allowed but are not required by this metric.

Multiple evidence passages for one query are represented by aligned list
positions:

```json
{
  "query": "Which two facts are needed?",
  "evidence_passage_ids": ["doc-a-0", "doc-b-2"],
  "evidence_texts": ["The first evidence passage.", "The second evidence passage."]
}
```

The primary metric is exact containment of each complete evidence passage
after whitespace normalization. Partial diagnostics use the longest exact
contiguous character or token substring; disconnected matches are not
combined. `MODEL_NAME` selects the tokenizer used for token-level diagnostics.
The default summary and detail outputs are written under
`outputs/retrieval_eval/$DATASET/`; set `OUTPUT_FILE` and `DETAILS_FILE` to
override them.

### Gold-evidence oracle evaluation

Use `run/eval_oracle.sh` to measure answer quality when approximate retrieval is
bypassed and the labeled evidence text is supplied directly to the LLM. This
path generates answers and computes the dataset's answer metrics, such as exact
match (EM) and token F1. It does not measure retrieval or compression evidence
coverage.

```bash
DATASET=dapr-nq-open \
GOLD_EVIDENCE_FILE=/mnt/nvme1/datasets/dapr-nq-open/dataset_info/evidence_labels.jsonl \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
TOTAL_NUM=2390 \
EVAL_BSZ=64 \
./run/eval_oracle.sh
```

`GOLD_EVIDENCE_FILE` uses the same custom `query`,
`evidence_passage_ids`, and `evidence_texts` fields described above. For each
query, the oracle reader preserves evidence order, removes exact duplicate
evidence texts, and exposes the remaining passages as the model context.
Questions and answers still come from `questions/query.jsonl` and
`answers/answer.jsonl`. The default prediction file is
`outputs/eval-$DATASET-gold-evidence.jsonl`; set `OUTPUT_FILE` to override it.

## Grid Search

Run one grid YAML directly:

```bash
python run/grid_search/eval.py run/grid_search/grid.yaml
```

Use `eval_cases` when each case needs its own environment values, such as a
fixed batch size for a specific `TOP_K` and `RETAIN_TOKEN_RATIO`:

```yaml
eval_cases:
  - TOP_K: "10"
    RETAIN_TOKEN_RATIO: "0.25"
    EVAL_BSZ: "4"
  - TOP_K: "20"
    RETAIN_TOKEN_RATIO: "0.25"
    EVAL_BSZ: "2"
```

`eval_cases` and `eval_axes` are mutually exclusive.

Run one or more grid YAMLs through the GPU wrapper:

```bash
run/run_grid.sh --wait-gpu --interval 15 \
  run/grid_search/grid.yaml \
  run/grid_search/grid_vanilla.yaml
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
