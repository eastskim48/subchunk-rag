# Evaluation Protocol

This document is the current canonical protocol for running and comparing
evaluation results.

## Entry Points

- Default single-run entry point: `run/eval.sh`.
- Grid-search entry point: `python run/grid_search/eval.py <grid.yaml>`.
- GPU-locked grid wrapper: `run/run_grid.sh <grid.yaml> [more-grid.yaml ...]`.
- Python evaluation entry point used by `run/eval.sh`:
  `src/entrypoint/eval.py`.
- Answer scoring implementation: `src/acc_metric.py`.
- Retrieval/evidence-only entry point:
  `run/eval_retrieval_only.sh`.
- Python implementation used by the retrieval/evidence-only shell:
  `src/entrypoint/eval_retrieval_only.py`.
- Gold-evidence oracle reader entry point:
  `run/eval_oracle.sh`.

`run/eval.sh` resolves datasets in this order:

1. `$DATASET_PREFIX/$DATASET` when `DATASET_PREFIX` is set.
2. `/mnt/nvme1/datasets/$DATASET` when that directory exists.
3. `$DATASET` as a direct path/name fallback.

## Default Runtime Settings

The current `run/eval.sh` defaults are:

- `DATASET=longbench-hotpotqa`
- `MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct`
- `DATA_SUBDIR=sent`
- `CACHE_SUBDIR=cache`
- `DB_DIR=$DATASET_PATH/$DATA_SUBDIR/db`
- `CACHE_DIR=$DATASET_PATH/$DATA_SUBDIR/$CACHE_SUBDIR`
- `DENSE_EMBED_DIR=$DATASET_PATH/$DATA_SUBDIR/dense_embed`
- `COLBERT_WINDOW_DIR=$DATASET_PATH/$DATA_SUBDIR/colbert_window`
- `EVAL_USE_PAST_CACHE=False`
- `MODEL_LOAD_IN_4BIT=False`
- `MEASURE_PROMPT_STATS=True`
- `RETRIEVAL_INCLUDE_DOCUMENTS=True`
- `USE_CLEANER=False`
- `EVAL_BSZ=4`
- `TOP_K=20`
- `TOTAL_NUM=200`
- `MAX_NEW_TOKENS=20`
- `COMPRESS_METHOD=dense`
- `RETAIN_TOKEN_RATIO=0.1`

Changing any of these can change the evaluated workload, model behavior, or
timing condition. Report changed values with every result table.

`COMPRESS_METHOD=dense` is the materialized dense-embedding selector. It uses
the same query embedding, stored sentence embeddings, and deduplication as the
former `compare_all_materialized` implementation, but now selects under the
shared token-budget policy. Earlier candidate-count-ratio dense results are not
directly comparable. The online `compare` and `compare_all` baselines and all
three old factory names are intentionally unsupported.

## Dataset And Scoring

Predictions are scored by `src/acc_metric.py` using dataset-aware evaluators.
The supported dataset names and scoring families are:

- `longbench-hotpotqa`: HotpotQA-style official EM/F1 with yes/no/noanswer
  special handling.
- `longbench-2wiki`: Hotpot-style EM/F1.
- `longbench-musique`: Hotpot-style EM/F1.
- `longbench-triviaqa`: TriviaQA-style alias-aware EM/F1.
- `triviaqa-unfiltered-wikipedia`: the same official TriviaQA alias-aware
  EM/F1 over the local Wikipedia-only unfiltered-dev condition.
- `longbench-qasper`: LongBench QA-style alias-aware EM/F1.
- `longbench-narrativeqa`: LongBench QA-style alias-aware EM/F1.
- `conditionalqa`: official ConditionalQA answer-only permutation EM/F1.
- `dapr-nq-open` (alias `nq-open`): official NQ-open normalized exact match
  over answer aliases, with auxiliary token F1.
- `newsqa`: NewsQA's official SQuAD v1.1 normalized EM and token-overlap F1.

For LongBench-family evaluators, ground-truth `answers` lists are treated as
valid aliases. If `answers` is absent, scoring falls back to `answer`.

ConditionalQA instead uses the answer-only portion of the official evaluator
from upstream commit `77bd295952daf415548b3244db10880d3d55cfe0`. Its
`answers` entries are distinct answers that must all be produced, not aliases.
Scoring pads missing predicted answers, selects the best permutation between
predictions and references, and applies the official exponential penalty for
extra predictions. The local RAG conversion intentionally omits condition
annotations, so conditional EM/F1 is not reported.

The current generation interface returns one free-form answer string per
query. It is represented as one official predicted answer with an empty
condition list; an empty generation is represented as no predicted answer. It
is not split heuristically on commas, newlines, or conjunctions. Consequently,
a single correct prediction for a question with `N` distinct gold answers
receives at most `1/N` answer EM/F1. Structured multi-answer generation would
change the prompt/output protocol and must be reported as a separate evaluation
condition.

The local NewsQA QA set uses only official test questions for which
`NewsQaDataset.get_consensus_answer` returns a character span after the
official loader's out-of-range endpoint clipping. This produces 4,293
question/answer pairs from the preceding 5,126-row raw-non-`None` test
conversion. Each `answers` list contains exactly the single consensus span.
The active files are a seed-42 sample of 400 pairs from those 4,293 records;
source IDs are preserved and are therefore not contiguous. NewsQA answer
scoring ports the SQuAD v1.1 evaluator specified by the NewsQA paper:
lowercase, remove ASCII punctuation and the English articles `a`/`an`/`the`,
normalize whitespace, then compute exact match and token-overlap F1. Local
scores are represented in `[0, 1]`; the upstream script multiplies the same
averages by 100 for display.

The earlier raw-answer files are preserved under
`query_raw_non_none_{full,seed42_400}.jsonl` and
`answer_raw_annotations_{full,seed42_400}.jsonl`. Results using those raw
annotator answers are not directly comparable to the consensus-set QA
results because both the evaluated question population and reference-answer
policy differ. Existing non-NewsQA dataset evaluators are unchanged.

The local `triviaqa-unfiltered-wikipedia` data preserves all 11,313 official
TriviaQA unfiltered dev question/answer pairs in `query_full.jsonl` and
`answer_full.jsonl`. Its active `query.jsonl` and `answer.jsonl` are the same
1,000 row indices sampled uniformly without replacement with Python
`random.Random(42)`, then restored to source order without renumbering IDs.
Every method compared on this condition must use this identical active sample.
The retrieval corpus is the union of the 16,078 packaged Wikipedia
`EntityPages` documents; Web `SearchResults` documents are excluded. Therefore,
results on the active sample are directly comparable across methods using this
same corpus, but not directly comparable to results over all 11,313 questions
or to the complete Wikipedia-plus-Web unfiltered condition. No evidence labels
are defined for this local dataset. Dataset name
`triviaqa-unfiltered-wikipedia` selects the existing official TriviaQA
alias-aware EM/F1 evaluator; this registration changes only name resolution,
not normalization or scoring.

The local `dapr-nq-open` evaluation set is the 2,390-query exact-question
intersection of DAPR-NQ test and official NQ-open dev; it is not the full
NQ-open dev set. Its primary answer metric ports the official FiD/DPR evaluator
at commit `fe769f30e3714e22476910ee39ea0054dd7921de`: lowercase the prediction
and each answer alias, remove ASCII punctuation and the English articles
`a`/`an`/`the`, normalize whitespace, then report exact match if any alias
matches. The upstream evaluator reports exact match only. The local `f1` field
is an auxiliary maximum token-overlap F1 over aliases using the same
normalization and must not be described as an official NQ-open metric. This
registration adds a previously unsupported dataset and therefore does not
change or invalidate results from existing dataset evaluators.

The DAPR-NQ gold-evidence oracle reader condition is a separate diagnostic
protocol. `run/eval_oracle.sh` bypasses approximate retrieval and
compression, looks up each query's labeled DAPR passages by exact query text,
removes exact duplicate passage texts while preserving label order, and places
only those passages in the existing QA prompt. It does not add titles,
neighboring sentences, or parent-document text. The prompt, generation, and
NQ-open answer evaluator are otherwise unchanged. These EM/F1 values estimate
reader performance given labeled evidence; their retrieval/compression timing
is not comparable to end-to-end RAG timing. `TOP_K=1` in this runner is only a
required engine placeholder and does not limit the number of labeled evidence
passages returned.

The oracle context source is implemented as
`GoldEvidenceVectorDB` in `src/gold_evidence_vectordb.py`. This name records
that it substitutes for the engine's `VectorDB` interface; it does not perform
vector search. Renaming the former `GoldEvidenceDB` implementation changes no
label loading, context construction, prompt, metric, or output, so results
before and after the rename remain directly comparable when all experiment
inputs match.

The separately named
`dapr-nq-gold-evidence-oracle-chat-0726` condition uses
`PROMPT_FORMAT=chat_system_user`. It serializes the same evidence and question
with the evaluated model tokenizer's official system/user chat template and an
assistant-generation header. It remains cache-off. Its EM/F1 values are not
directly comparable to raw-prompt Vanilla, subchunk, or oracle results because
prompt serialization changes model behavior. The default
`PROMPT_FORMAT=raw_chunk_first` remains unchanged for all other runs.

The separately named `dapr-nq-bge-rag-sllm-chat-0726` grid applies the same
`chat_system_user` serialization to all 14 Llama-3.2-1B-Instruct Vanilla and
subchunk cases. It is cache-off and is the matched standard-chat comparison for
the 1B oracle. Its answer-quality results are not directly comparable to the
earlier raw-prompt sLLM grid.

`USE_CLEANER=False` is the current default in `run/eval.sh`. Results with
different cleaner settings are not directly comparable unless the prediction
post-processing equivalence is explicitly checked.

## Retrieval/Evidence-Only Metrics

`run/eval_retrieval_only.sh` evaluates retrieval and optional
compression against evidence labels without running large language model
(LLM) generation. It always uses the project's custom `text_evidence_exact`
metric.

The evidence file is passed as `SAMPLE_FILE` and defaults to
`$DATASET_PATH/evidence_labels.json`. For the current original-context
benchmarks, the 200 queries are the same LongBench HotpotQA/2Wiki queries, exact
matched to original validation records and paired with original supporting
document/fact labels.

The metric compares each gold evidence passage directly with the final
retrieved or compressed context text. It does not use cacheable identifiers,
retrieved chunk identifiers, source token spans, source character spans, or
reconstructed source text. It reports:

- `evidence_char_exact_recall`: fraction of gold passages whose complete
  whitespace-normalized character sequence occurs contiguously in the context.
  This is the primary passage-level metric.
- `evidence_token_exact_recall`: fraction of gold passages whose complete
  `MODEL_NAME` token sequence occurs contiguously in the context.
- `all_evidence_char_exact` and `all_evidence_token_exact`: fraction of queries
  for which every gold evidence passage is exactly contained.
- `evidence_char_partial_recall` and `evidence_token_partial_recall`: auxiliary
  recall based only on the single longest exact contiguous substring shared by
  the gold passage and context. Disconnected matches are never combined.
- `*_partial_passage_recall_at_threshold`: auxiliary fraction of passages whose
  longest-contiguous-substring recall reaches `PASSAGE_RECALL_THRESHOLD`.
- conditional exact retention: among passages exactly contained after
  retrieval, the fraction still exactly contained after compression.
- conditional partial retention: compressed longest-contiguous overlap divided
  by retrieved longest-contiguous overlap, capped at the retrieved overlap.

Character matching collapses each whitespace run to one ASCII space before
matching, so newline/space formatting differences do not cause a false
negative, but inserting or deleting a separator between words still changes
the sequence. Token matching applies the same whitespace normalization and
prepends one space before independently tokenizing the complete gold passage
and context. The common leading space makes the gold passage's first token use
the same word-boundary convention as an occurrence inside a larger context.
The complete gold token sequence must then occur as consecutive token IDs.
This metric requires only
`evidence_texts` and `evidence_passage_ids`; source-document files and
source-position metadata are not evaluation inputs.

Results produced by either the former source-span reconstruction metric or the
discarded non-contiguous LCS implementation are not directly comparable with
`text_evidence_exact`. The current metric measures exact full-passage
containment as its primary outcome and uses only contiguous overlap for its
partial-match diagnostics.

The former `legacy_text` path, including Rouge-L, supporting-document, and
supporting-subchunk overlap metrics, has been removed. Existing
`text_evidence_exact` results remain directly comparable before and after this
removal when all experiment inputs match because its label loading, context
construction, scoring, and output fields did not change. Former `legacy_text`
results are not directly comparable with `text_evidence_exact` results.

ColBERT compression queries optionally use `COLBERT_QUERY_MAXLEN` and
`COLBERT_QUERY_TRUNCATION_SIDE`. `right` preserves the query head and `left`
preserves the query tail when truncation is necessary. Unless explicitly
overridden, the artifact's `query_maxlen` and right truncation remain the
default.

`COLBERT_QUERY_MINLEN` enables adaptive query length. The query encoder still
tensorizes up to `COLBERT_QUERY_MAXLEN`, but only
`max(MINLEN, min(MAXLEN, content_WordPieces + 3))` query-vector rows
participate in MaxSim. Thus batch padding or `[MASK]` expansion beyond the
per-query effective length is excluded. The intended ColBERTv2-compatible
adaptive setting uses `COLBERT_QUERY_MINLEN=32`; it preserves the checkpoint's
q32 expansion floor while treating `COLBERT_QUERY_MAXLEN` as a content-driven
cap. If `COLBERT_QUERY_MINLEN` is unset, the official fixed-length behavior is
preserved and all `MAXLEN` query-vector rows participate.

Changing max length, min length, or truncation side is a query-encoding
ablation. It does not change dense retrieval but can change compression
selection, so the output records all three values and results are not directly
comparable as identical configurations.

## Prompt And Cache Modes

`EVAL_USE_PAST_CACHE=False` is cache-off mode. It constructs the full visible
prompt and runs regular HuggingFace generation.

`EVAL_USE_PAST_CACHE=True` is cache-on mode. It loads reusable KV cache
artifacts from `CACHE_DIR` and decodes with prefilled cache state.

Cache-on and cache-off results are directly comparable only when all visible
prompt content, retrieval settings, compression settings, model, generation
settings, and scoring settings match. Cache-on also requires cache artifacts to
match the evaluated dataset, `DATA_SUBDIR`, cacheable units, model/tokenizer,
and cache construction settings.

`provence` and `exit` currently emit compressed prompt text directly and are
cache-off only. They must not be run with `EVAL_USE_PAST_CACHE=True`.

`TOP_K=0` is a context-free cache-off baseline and must not be used with
`EVAL_USE_PAST_CACHE=True`.

## Retrieval And Artifacts

`DB_DIR` determines the vector retrieval database. `DATA_SUBDIR` is only a
convention for selecting dataset-local subdirectories; the actual DB path is
`DB_DIR`.

DB preprocessing buffers retrievable chunks and upserts them in batches.
`DB_BATCH_SIZE` defaults to 256 and is recorded in `build_manifest.json`.
Changing it leaves the generated chunk texts, metadata, IDs, and insertion
order unchanged, but Chroma's approximate HNSW graph is not guaranteed to be
identical across different insertion batch boundaries. Vanilla and compression
methods in one comparison must use the same physical DB. Results from DBs built
with different batch sizes are not guaranteed to have identical retrieval
rankings, especially for near-tied candidates.

For `CHROMA_EMBED_BACKEND=default`, DB preprocessing explicitly records
`CHROMA_EMBED_DEVICE` and `CHROMA_EMBED_BATCH_SIZE`. CPU and CUDA use the same
MiniLM ONNX graph, tokenizer, 256-WordPiece truncation, mean pooling, and
normalization, but floating-point embeddings are not bit-identical. DBs built
on different providers are therefore not retrieval-identical conditions.
Runtime `ChromaDB(db_dir)` pins query encoding to CPU; build-time CUDA settings
do not make online query encoding use the GPU. The runtime constructor has no
device option; CUDA is available only through the preprocessing-only
`ChromaDB.for_build(...)` path. Runtime GPU query encoding is outside the
evaluation protocol and must not be enabled.

When `DEDUPLICATE_DOCUMENTS_BY_HASH=True` is used during preprocessing, exact
duplicate source documents are removed from the vector DB by normalized full
text hash while the local `documents/` directory remains unchanged. This changes
the retrieval corpus and can change retrieved chunk IDs, so results from a
deduplicated DB are not directly comparable to results from the original DB
unless the corpus condition is reported. ColBERT-window artifacts must be
rebuilt against the deduplicated DB before running ColBERT-based compression.

`CHROMA_EMBED_BACKEND` controls how Chroma collections are opened:

- `default` is the current default and main paper setting. It opens the
  collection with Chroma's ONNX `all-MiniLM-L6-v2` embedding implementation.
  Because this path does not pass an explicit HNSW configuration, Chroma 1.5.7
  uses `ef_construction=100`, `ef_search=100`, and `M=16`
  (`max_neighbors=16`). `chroma_default` is an alias for this same path.
- `bge_m3`, `bge_small_v1_5`, and `e5_small_v2` use the project's
  `DenseTextEmbedder`/SentenceTransformers implementation and an explicit HNSW
  configuration: `ef_construction=200`, `ef_search=200`, and `M=32`
  (`max_neighbors=32`).

The small-model backends follow their published model-specific encoding rules:

- `bge_small_v1_5` uses CLS pooling, no passage prefix, and
  `Represent this sentence for searching relevant passages: ` for queries.
- `e5_small_v2` uses masked mean pooling and the `passage: ` / `query: `
  prefixes.

Both use normalized 384-dimensional embeddings and a maximum sequence length
of 512. Results from the older July 7 BGE-small analysis script are not directly
comparable: that script passed BGE-small through the former generic mean-pooling
path. The ConditionalQA comparison and the July 24 HotpotQA DBs use the correct
model-specific pooling.

Every backend above performs dense vector retrieval. In this documentation,
"Chroma default MiniLM backend" means `default`; "SentenceTransformers
retrieval backends" means `bge_m3`, `bge_small_v1_5`, and `e5_small_v2`.
Neither term means the `dense` context-compression method. Changing the
retrieval backend changes the retrieval condition and must be reported as a
separate retriever setting.

Dense selectors use `DENSE_EMBED_DIR`. ColBERT-window selectors use
`COLBERT_WINDOW_DIR` and `COLBERT_MODEL_NAME`. Runtime ColBERT query encoding is
CPU-only, and initialization asserts that the checkpoint parameters are
actually on CPU. The artifact query length and right truncation are the
defaults; the query-shape ablation overrides documented above may change them.
Results produced before the 2026-07-23 runtime-device enforcement must not be
reported as CPU-query-encoding results merely because the wrapper requested
CPU: when CUDA was visible, upstream ColBERT could still place the checkpoint
and query tensors on CUDA. Those historical results are not directly comparable
for compression latency, and their selections can also differ because the CUDA
path used mixed precision while the enforced CPU path uses full precision.

ColBERT-window artifact construction is CUDA-only and uses the same
`COLBERT_MODEL_NAME` checkpoint variable as runtime. The optional tensorization
cross-check is disabled by fixed build policy rather than exposed as an
experiment environment variable; this does not change stored tensor values or
runtime scoring.

`COMPRESS_METHOD=colbert_subchunk` selects the global ColBERT subchunk
selector. This factory key replaces the former `colbert_window` key; the old
alias is intentionally unsupported. It walks globally ranked subchunks under
the budget configured by exactly one of `RETAIN_TOKEN_RATIO` or
`FINAL_TOKEN_BUDGET`.

Earlier `colbert_subchunk` candidate-count-ratio results and strict-fit token
budget results are not directly comparable to the current add-then-stop token
budget policy.

For `colbert_sliding_region`, v3 data artifacts store per-retrieval-chunk region
specs in `colbert_window/data/region_payloads.json`. They must be rebuilt after any
change to region-spec construction. In particular, results produced before the
skip-over packing fix in commit `6713d96` are not directly comparable to
results produced after rebuilding those sidecar specs. The split JSON sidecars are
mandatory. A missing chunk, mismatched cacheable-id list, or region budget
different from the artifact window budget is an error; runtime region
recomputation is intentionally unsupported.

Splitting the corpus-wide ID, window-membership, and region mappings into
`cacheable_rows.json`, `window_ids.json`, and `region_payloads.json` changes only
artifact storage and initialization. Production runtime eagerly loads only the
cacheable-row and region mappings into dicts. It does not load build-only window
membership. This does not change candidate text, ColBERT
vectors, region construction, scoring, selection, prompt construction, or
metrics, so pre-migration and v3 results are directly comparable when all experiment
inputs and stored values are otherwise identical. The v3 runtime intentionally
has no legacy reader; legacy artifacts must be rebuilt.

The coarse retrieval chunk size must be strictly larger than the stored
ColBERT region/window budget. The former additional 100-token heuristic buffer
was removed on 2026-07-23 because it had no artifact or scoring invariant
behind it and prevented the valid coarse-256/window-180 condition. This change
only enables configurations in that former buffer range; it does not change
region construction or selection for existing coarse-512 results.

### ConditionalQA Dense-Embedder Retrieval Comparison

`test/analyze_conditionalqa_embedder_retrieval.py` compares strict fixed chunks
from the existing ConditionalQA vanilla databases. It validates source-token
span continuity and source-text equality before building a separate Chroma
index for each embedder and chunk size. Passage embeddings are built on CUDA;
runtime query embeddings are deliberately computed on CPU.

The evaluated configurations are:

- `sentence-transformers/all-MiniLM-L6-v2`, no text prefixes, maximum sequence
  length 256.
- `intfloat/e5-small-v2`, `query: ` and `passage: ` prefixes, maximum sequence
  length 512.
- `BAAI/bge-small-en-v1.5`, its retrieval query instruction and no passage
  prefix, maximum sequence length 512.

All models use their SentenceTransformers pooling configuration, normalized
embeddings, the same Chroma HNSW configuration, and the same insertion order.
The primary metric is non-whitespace character coverage of DAPR source-span
evidence; Llama-3.1-8B token coverage is reported as a secondary metric.
Changing the embedder creates a new retrieval baseline, so these results are
not directly comparable to indexes built with a different pooling rule,
prefix, maximum length, or insertion order.

`RETRIEVAL_INCLUDE_DOCUMENTS=False` is allowed only for methods whose prompt
text can be reconstructed from cacheable metadata. Do not use it for methods
that require original retrieved document text unless that path has been
explicitly checked.

## Compression And Budget Controls

`COMPRESS_METHOD` selects the retrieval/compression variant. Empty
`COMPRESS_METHOD` means no downstream compressor; retrieved units are used
directly.

The unused `COMPRESS_METHOD=front` baseline has been removed. Existing results
from that method remain historical baseline results, but the configuration is
no longer accepted. The OpenAI-backed `COMPRESS_METHOD=summ` baseline remains
available and its behavior is unchanged.

`FINAL_TOKEN_BUDGET` controls the absolute prompt-token target for `dense`,
`colbert_subchunk`, `colbert_sliding_region`, and `rerank_and_region`.

`RETAIN_TOKEN_RATIO` computes the final token budget per query as a ratio of
retrieved context tokens. Exactly one of `RETAIN_TOKEN_RATIO` and
`FINAL_TOKEN_BUDGET` must be set for those four methods. Setting both or neither
is a configuration error.

For all fixed-budget and retained-ratio paths, final budget accounting uses
the prompt-visible passage text format, `text.strip() + "\n\n"`, tokenized with
the evaluated `MODEL_NAME` tokenizer. It must not use ColBERT artifact vector
row counts or a separate ColBERT/window-artifact
tokenizer, because those can differ from the LLM prompt tokenizer and can be
truncated by the ColBERT document max-length. Results produced before this
budget-accounting fix are not directly comparable to results produced after the
fix.

New DB/materialization outputs store this per-cacheable prompt-visible length as
`prompt_token_count` plus its `prompt_tokenizer_name` inside each
`cacheables_json` payload. Query-time selection uses that stored integer
only when `prompt_tokenizer_name` exactly matches the runtime `MODEL_NAME`; it
falls back to runtime `MODEL_NAME` tokenization when the count is missing or the
tokenizer does not match.

The final token budget is a soft target. Each selector adds the next complete
ranked candidate and then stops when accumulated prompt-visible tokens reach or
exceed the budget. Dense and `colbert_subchunk` can exceed the target by the
last subchunk; region methods can exceed it by the novel subchunks in the last
region. No method splits or skips a higher-scoring candidate merely because it
crosses the target.

Fixed-token-budget runs and retained-ratio runs are not directly comparable
unless the resulting input lengths are reported and intentionally matched.

## Generation And Timing

The generation workload is controlled by:

- `MODEL_NAME`
- `MODEL_LOAD_IN_4BIT`
- `EVAL_BSZ`
- `MAX_NEW_TOKENS`
- `TOTAL_NUM`
- `EVAL_USE_PAST_CACHE`
- cache and prompt construction settings

`MODEL_LOAD_IN_4BIT=False` is the current latency-oriented default. 4-bit and
fp16 runs are different inference backends and should be reported separately.

`EVAL_BSZ` affects both throughput and model behavior. Quality metrics from
different batch sizes should not be treated as identical unless equivalence has
been checked.

The timing fields printed by `src/engine.py` are batch averages unless otherwise
stated:

- retrieval time
- compression time
- prompt build time
- prompt stats time
- generate extra time
- prefill time
- decode time
- total time per batch

CUDA synchronization is applied around model prefill/decode timing windows in
`src/model.py`. Latency comparisons should be made on an otherwise idle GPU, or
under an explicitly documented GPU lock/exclusive-process setup.

ColBERT-based compression always clears process-local retrieved-vector and
sliding-region-spec caches before every measured compression batch. This
preserves intra-batch deduplication while preventing reuse across measured
batches. It also always runs one dummy query encoding after compressor
initialization, moving the first-forward cost into setup rather than the first
measured compression batch. These are fixed evaluation policies rather than
runtime options.

`COMPRESS_METHOD=colbert_chunk_rerank` is a fixed-retrieval-chunk reranking
baseline. It uses the same vector DB retrieval result as vanilla RAG, scores the
retrieved fixed chunks with materialized ColBERT chunk representations, and
keeps the top `COLBERT_CHUNK_RERANK_KEEP` chunks.

For this baseline:

- offline ColBERT chunk materialization time is not counted in eval latency.
- online query encoding, artifact lookup, ColBERT MaxSim scoring, reranking,
  prompt construction, and LLM inference are counted.
- use `colbert_fixed_chunk_docmax512` artifacts, not the older
  `colbert_fixed_chunk` artifacts built with ColBERT default `doc_maxlen=180`.
- kept chunks are prompted in ColBERT reranked order.

`COLBERT_RERANK_KEEP` controls the coarse-chunk keep count for both reranking
methods below. Both score a retrieved coarse chunk by loading all of its
subchunk representations from the matching DB's `colbert_window` artifact,
concatenating their token-vector pools, and applying one query-to-pool ColBERT
MaxSim score. Neither method uses the separate
`colbert_fixed_chunk_docmax512` artifact.

- `COMPRESS_METHOD=colbert_rerank` keeps the top-K coarse chunks and prompts
  their complete contents in reranked order.
- `COMPRESS_METHOD=rerank_and_region` keeps the top-K coarse chunks as an
  eligibility set, then constructs, scores, and selects sliding regions only
  inside those chunks. For retained-ratio runs, the token budget is computed
  from the kept coarse chunks rather than the original retrieval top-k set.

The removed `colbert_window_chunk_rerank` name has the same scoring/output
semantics as `colbert_rerank`, so those results are directly comparable after
renaming the configuration field. The former fixed-artifact pre-filter changed
its coarse scoring artifact and chunk alignment, so its results are not
directly comparable to the new `rerank_and_region`. The post-filter method was
removed and has no replacement.

`ttft_per_batch_avg_sec` in grid summaries is derived as:

```text
time_per_batch_avg_sec - decode_per_batch_avg_sec - generate_extra_per_batch_avg_sec
```

It is a reconstructed batch-average latency component, not an independently
instrumented server-side TTFT.

## Output Files And Grid Summaries

`run/eval.sh` writes prediction JSONL to `OUTPUT_FILE` when set, otherwise to
`outputs/eval-...jsonl` with suffixes derived from method, `TOP_K`, cache mode,
LLM precision, model tag, and only the selection controls effective for that
method. Retained-ratio runs use an `rtr` tag and absolute-budget runs use an
`ftb` tag. Rerank-only methods have no token-budget tag.

Prediction and answer JSONL parsing treats only LF (`U+000A`) as the record
delimiter. Other Unicode line-separator characters, including `U+0085` NEXT
LINE, remain part of JSON string values such as retrieved source text. This
delimiter rule changes no prediction, answer normalization, metric, or scoring
formula; it prevents embedded source characters from being misclassified as
record boundaries.

`run/grid_search/eval.py` writes:

- `manifest.json`
- `events.log`
- per-run logs under `logs/`
- `results.jsonl`
- `failures.jsonl`
- `summary-<dataset>.csv`
- `summary-<dataset>-summary2.csv`

`summary2` keeps a compact table with method, budget controls, input length,
EM/F1, prefill/decode latency, elapsed time, total batch time, compression time,
derived TTFT, and retrieval time.

## Comparability Rules

Treat two results as directly comparable only when all of the following match:

- dataset and dataset files
- answer scoring path and `USE_CLEANER`
- model, tokenizer, precision/backend, and generation settings
- `EVAL_BSZ`, `TOTAL_NUM`, and `MAX_NEW_TOKENS`
- cache-on/cache-off mode
- prompt template and visible prompt serialization
- retrieval DB, embedding backend, `TOP_K`, and retrieval artifacts
- compression method and all method-specific environment variables
- fixed ColBERT inter-batch cache clearing and query-encoder warmup policy
- final budget controls such as `FINAL_TOKEN_BUDGET`,
  `COLBERT_CHUNK_RERANK_KEEP`, and `RETAIN_TOKEN_RATIO`
- GPU/timing environment when reporting latency or throughput

If any of these differ, report the changed condition explicitly and treat the
result as a separate ablation or baseline.

### Experimental Sliding-Region Group Ordering

`COLBERT_REGION_GROUP_ORDER` is a temporary experiment control for
`colbert_sliding_region` and `rerank_and_region`. Its supported values are:

- `retrieval`: preserve the original coarse-retrieval chunk order.
- `max`: order selected coarse chunks by the maximum score among selected
  regions that contributed at least one previously unselected subchunk.
- `sum`: order selected coarse chunks by the sum of those contributing region
  scores.

All three modes preserve region selection, overlap deduplication, token budget,
selected subchunks, and source order within each coarse chunk. They change only
the order of coarse-chunk groups in the LLM prompt. Results from different
values are therefore prompt-order ablations and are not directly comparable as
identical evaluation configurations. Non-retrieval modes add an `rgo<mode>` tag
to the default output filename. This control is experimental and should be
removed after the ordering ablation is concluded.

## When This Document Must Change

Update this document whenever evaluation behavior changes in a way that affects:

- dataset handling
- retrieval flow
- prompt format
- cache validity
- generation settings
- timing definitions
- metrics or scoring
- output schema used by grid summaries
- result comparability

Historical one-off notes and old bug-fix logs should go to
`docs/eval_protocol_history.md` or `docs/handoff.md`, not into this canonical
protocol.
