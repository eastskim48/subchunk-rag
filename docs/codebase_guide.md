# Codebase Guide

## Key Paths

### `run/`
- Store experiment scripts here.
- Use this directory for reproducible execution entry points.

### `test/`
- Store test code here.
- Use this directory for:
  - small feasibility checks
  - debugging with minimal examples
  - unit tests for methods

### `compressor/`

- Keep framework-level code such as the `Compressor` interface, factory,
  shared token-budget policy, and output helpers at the package root.
- Keep concrete implementations under `compressor/methods/`:
  - `dense.py`
  - `colbert/`
  - `ml_selector.py`
  - `summarization.py`

## Model

### Encoders

- `encoder/dense.py` owns dense model loading, pooling, normalization, and
  query-prefix policy shared by dense artifact materialization and compression.
- `materialize/dense_materializer.py` writes the dense embedding artifact
  consumed by the dense compression method. Its DB-based builder deduplicates
  overlapping retrieval-window cacheables by stable ID, rejects conflicting
  payloads, and orders document candidates deterministically. The
  candidate-store entrypoint selects this builder with `backend=dense`.
- `encoder/colbert.py` owns ColBERT query and document/span encoding shared by
  ColBERT artifact materialization and compression.
- `materialize/colbert_materializer.py` constructs the window-contextualized
  ColBERT candidate store and its query-independent region metadata from the
  exact cacheable IDs, texts, and source order persisted in the DB. It does not
  rerun the source parser.
- `vectordb.py` owns the Chroma embedding-function adapter and reuses the dense
  encoder rather than implementing model forward or pooling again.

### Chroma retrieval backend terminology

All `CHROMA_EMBED_BACKEND` choices are dense vector-retrieval backends. Do not
refer to only the non-MiniLM choices as "dense backends," and do not confuse
them with the `dense` context-compression baseline:

- The implicit environment default is `bge_small_v1_5`.
- The explicitly named `default` backend (alias `chroma_default`) is the main
  paper MiniLM retrieval setting. It uses Chroma's ONNX
  `all-MiniLM-L6-v2` embedding implementation. The project does not pass an
  HNSW configuration for this path, so Chroma 1.5.7 supplies
  `ef_construction=100`, `ef_search=100`, and `M=16` (`max_neighbors=16`).
- `bge_m3`, `bge_small_v1_5`, and `e5_small_v2` use the project's
  `DenseTextEmbedder`/SentenceTransformers path. The project explicitly creates
  these collections with `ef_construction=200`, `ef_search=200`, and `M=32`
  (`max_neighbors=32`).

`DenseTextEmbedder` applies model-specific pooling and prefixes:
`bge_small_v1_5` uses CLS pooling and the BGE retrieval query instruction;
`e5_small_v2` uses masked mean pooling and E5's `query: ` / `passage: `
prefixes. Runtime `ChromaDB` construction always places these query encoders on
CPU; `ChromaDB.for_build(...)` is the only path that accepts CUDA for corpus
embedding.

The retrieval backend determines the embeddings stored in Chroma and the
embedding function used for retrieval queries. It is independent of
`COMPRESS_METHOD=dense`, which selects candidate subchunks after coarse Chroma
retrieval using the separately materialized dense candidate store.

### Prompting
- Write prompts in a format that the current Llama-3.1-Instruct model can easily follow.
- Preserve the current cache reuse pipeline.
- Do not break the current prompt/cache structure:

`[Chunk Cache][Chunk Cache][System Prompt][Query]`

- Prefer prompt changes that are compatible with KV reuse.
- Avoid prompt formats that require rebuilding the whole prefix.

### Model choice
- Keep using the current model unless explicitly requested otherwise.
- Do not switch to another model without user approval.

## Dataset

- Default dataset path:

`/mnt/nvme1/datasets/longbench-hotpotqa`

Given `base_path = /mnt/nvme1/datasets/longbench-hotpotqa`:

- Questions: `base_path + /questions/query.jsonl`
- Answers: `base_path + /answers/answer.jsonl`
- Documents: `base_path + /documents/`
- Vector database: `base_path + /db/`
- KV caches: `base_path + /$chunk_size/`

Each newly materialized vector DB stores its preprocessing semantics in
`db/build_manifest.json`. Read fields such as `retrievable_chunk_size` and
`max_subchunk_tokens` from this manifest instead of inferring them from the
parent directory name. The DB embedding provider and embedding batch size are
recorded because CPU and CUDA execution are not bit-identical.
`db_batch_size` is also recorded: changing an upsert boundary does not change
chunk generation, but it can change the approximate HNSW graph. Methods in one
comparison must share the same physical DB.

General DB manifest construction, persistence, and hashing live in
`materialize/db_manifest.py`. ColBERT-specific relative manifest references and
their SHA-256 validation live in `colbert_artifact.py`. That module also owns
the runtime readers for both the candidate-store artifact and the separate
fixed-retrieval-chunk ColBERT reranking artifact.

Read-only iteration over cacheable subchunks persisted in Chroma metadata lives
in `materialize/db_cacheables.py`. It also owns stable-ID deduplication,
duplicate-payload validation, and deterministic parent-document/source ordering.
Dense and ColBERT candidate-store materializers share this DB-facing reader
instead of owning backend-specific SQLite queries.

ColBERT materialization validates the requested center unit against the DB
manifest before creating or overwriting artifact files. Sentence DBs accept
only `subchunk`/`subchunk_only`; fixed-size DBs accept only
`fixed_chunk`/`fixed_chunk_window` and require the same cacheable chunk size.
Semantic DBs are rejected until a distinct semantic-cacheable center mode is
defined.

ColBERT window artifact format `colbert_window_artifact_v2` stores a relative
`db_manifest.path` and its SHA-256 in `colbert_window/index.json`. It does not
duplicate the DB tokenizer or maximum subchunk length. Artifact build and
runtime both require the referenced manifest path and hash to match the active
DB. The only embedding layout is `colbert_window/data/`; per-document and
`compact/` layouts are intentionally unsupported. The two JSON indexes contain
only scalar artifact metadata. `data/cacheable_rows.json` maps cacheable IDs to
memory-mapped vector rows, and `data/region_payloads.json` stores
per-retrieval-chunk region spans. Runtime initialization eagerly loads these two
files into Python dicts. `data/window_ids.json` is build-only metadata used to
materialize regions and is not loaded by production compression. The stored
artifact window budget and region spans are required; runtime does not
reconstruct regions from token counts. The v3 data reader intentionally has no legacy
compatibility path.

Build the DB and candidate artifacts in separate steps:

- `run/preprocess.sh` builds DB and optional KV-cache outputs only.
- `run/materialize_candidate_store.sh` builds and validates the materialized
  candidate store against an existing DB. `CANDIDATE_STORE_BACKEND=colbert`
  remains the default; `dense` builds the baseline embedding artifact directly
  from persisted DB cacheables. The script invokes the
  `entrypoint/materialize_candidate_store.py` Python entrypoint.

The DB preprocessor defaults to `splitter=sentence`. It creates and opens a
Chroma DB only when `MATERIALIZE_DB=True`, so cache-only runs have no DB side
effect. Semantic splitting requires an explicit grouper, currently selected
through the legacy `MERGER` CLI option. The old PN-mapping and
resolved-sentence preprocessing paths are intentionally unsupported.

The explicitly named `default` Chroma MiniLM backend uses a provider-pinned
ONNX wrapper. `run/preprocess.sh` builds embeddings on CUDA by default and
records the build device and internal embedding batch in the DB manifest.
Ordinary `ChromaDB(db_dir)` runtime construction always pins query encoding to
CPU and has no device argument. CUDA is exposed only by the preprocessing-only
`ChromaDB.for_build(...)` constructor. Do not add runtime GPU query encoding.

Sentence segmentation and source-span alignment live in
`materialize/splitter/parser/sentence.py`. Parsers return `ParsedUnit` values
with text plus character and token spans. `ParsedUnitSplitter` owns the common
parse -> group -> cacheable-materialization pipeline. `SentenceWiseSplitter`
uses `IdentityGrouper`, additionally splits oversized parsed units according to
`max_subchunk_tokens`, and preserves `::sent_{idx}` cacheable IDs.
`SemanticSplitter` intentionally passes the original parsed units directly to
its configured grouper and preserves the existing grouper-based subchunk IDs.

ColBERT artifact construction uses CUDA and fixed disabled tensorization
verification. Its checkpoint is configured with `COLBERT_MODEL_NAME`, matching
the runtime variable.

The general DB preprocessing entry point intentionally has no integrated
candidate-artifact build option.

### ConditionalQA RAG Dataset

`test/prepare_conditionalqa_dataset.py` reconstructs the DAPR ConditionalQA
corpus in the same local directory shape used by the LongBench experiments:

- `questions/query.jsonl`
- `answers/answer.jsonl`
- `documents/doc_{dapr_doc_id}.txt`
- `dataset_info/evidence_labels.jsonl`

The converter writes each document title once, then joins its ordered DAPR
passages with blank lines. Answer records preserve every official answer text
but intentionally omit ConditionalQA condition annotations. Evidence labels
come only from DAPR test qrels and include the exact passage IDs, passage text,
and half-open character spans in the reconstructed document. The converter
does not chunk documents; use the normal preprocessing pipeline for DB and
ColBERT artifact construction.

### DAPR-NQ/NQ-open RAG Dataset

`test/prepare_dapr_nq_open_dataset.py` builds the exact-question intersection
between DAPR-NQ test and official NQ-open dev in the same local directory
shape. It writes all 108,626 DAPR parent documents, not only documents named by
qrels, so retrieval does not receive gold-document leakage. Each document is
its title followed by its ordered DAPR passages.

The 2,390 query records preserve the verbatim question. Answer records retain
the complete official NQ-open answer list as aliases. Evidence records retain
the DAPR qrel passage IDs, relevance scores, exact passage text, and half-open
character spans in the reconstructed parent document. This is an explicitly
named DAPR-NQ/NQ-open intersection, not the full standard NQ-open benchmark.
The converter does not chunk documents.

`run/eval_oracle.sh` provides the distinct gold-evidence oracle reader
condition. `src/gold_evidence_vectordb.py` supplies the labeled DAPR evidence
passages for each exact query through the `VectorDB` interface without opening
Chroma or running a compressor. This path is cache-off and must not be reported
as end-to-end retrieval latency.
The runner accepts an explicit `PROMPT_FORMAT`; the default remains
`raw_chunk_first`. `chat_system_user` uses the tokenizer's official chat
serialization and is rejected for cache-on evaluation because existing KV
artifacts were not materialized under that prefix.

### Official Dataset Fidelity
- Do not write custom local dataset builders that drop, collapse, rename, normalize, deduplicate, or reinterpret official dataset fields unless the user explicitly asks for that exact transformation.
- When using LongBench, preserve official fields such as `answers` exactly. Do not replace an answer-alias list with `answers[0]`.
- If local files must be generated from an official dataset, copy the required official fields verbatim and document any unavoidable structural change before running experiments.
- Do not silently create convenience formats that differ from the official benchmark format.

## Working Style
- Start with a very small sample for feasibility and debugging.
- Prefer minimal, local changes.
- Reuse existing interfaces when possible.
- Ask the user before modifying external code or reference library code.
- Prefer changes in `src/`, `run/`, and `test/` first.
- Do not create large temporary files or persistent cache artifacts under the project root by default.
- Prefer `/mnt/nvme1/dongseob/tmp/` for large intermediate outputs, reusable test caches, and other non-source artifacts unless reproducibility requires a documented project-local path.
- Do not use `/tmp` for project dependencies, repositories, caches, smoke artifacts, or experiment outputs. The official ColBERT source is managed as the `third_party/ColBERT` git submodule by default. ColBERT embedding/window artifacts remain dataset-local under paths such as `$DATASET_PATH/sent/colbert_window`.

## Explicit-Scope Discipline
- Do not implement behavior the user did not explicitly request.
- Do not add fallback logic, heuristic recovery, fuzzy matching, extra metrics, altered scoring, prompt changes, dataset handling changes, or evaluation-flow changes unless the user explicitly asks for them.
- If a change seems useful but was not requested, explain the option and ask before implementing it.
- Evaluation code is especially strict: preserve the requested metric definition exactly, even if another variant looks more robust or convenient.
- If the user asks to add a field, metric, option, or behavior, only add it. Do not remove, rename, replace, or reinterpret existing fields, metrics, options, or behavior unless the user explicitly asks for that removal or rename.

## Evaluation Invariants
- Evaluation must fail immediately when prediction and ground-truth record counts differ. Raise an explicit exception; never truncate either list, skip unmatched rows, change the denominator, or continue with a partial score.
- Prediction logs and ground-truth records must both contain stable IDs. Compare IDs at every row before scoring and fail immediately if an ID is missing or differs; equal record counts alone do not establish alignment.
- Retrieval-only evidence evaluation must persist each query's complete retrieved and compressed context strings in a run-local detail file so corrected labels or scoring rules can be replayed without rerunning retrieval or compression.
- More generally, when an evaluation invariant is violated, stop with an assertion or exception instead of inventing a fallback or silently fixing the data. Do not convert an observed mismatch into evaluation policy unless the user explicitly instructs the exact handling.

## Refactoring Discipline
- When refactoring, do not guess and make changes beyond what was explicitly requested.
- Do only what was asked.
- If a requested change requires an additional structural decision or scope expansion, ask before doing it.
