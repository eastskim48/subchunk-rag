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
- `COMPARE_EMBED_DIR=$DATASET_PATH/$DATA_SUBDIR/compare_embed`
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
- `COMPRESS_METHOD=compare_all_materialized`
- `GLOBAL_TOP_R=0.1`

Changing any of these can change the evaluated workload, model behavior, or
timing condition. Report changed values with every result table.

## Dataset And Scoring

Predictions are scored by `src/acc_metric.py` using dataset-aware evaluators.
The supported dataset names and scoring families are:

- `longbench-hotpotqa`: HotpotQA-style official EM/F1 with yes/no/noanswer
  special handling.
- `longbench-2wiki`: Hotpot-style EM/F1.
- `longbench-musique`: Hotpot-style EM/F1.
- `longbench-triviaqa`: TriviaQA-style alias-aware EM/F1.
- `longbench-qasper`: LongBench QA-style alias-aware EM/F1.
- `longbench-narrativeqa`: LongBench QA-style alias-aware EM/F1.

Ground-truth `answers` lists are treated as valid aliases. If `answers` is
absent, scoring falls back to `answer`.

`USE_CLEANER=False` is the current default in `run/eval.sh`. Results with
different cleaner settings are not directly comparable unless the prediction
post-processing equivalence is explicitly checked.

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

`CHROMA_EMBED_BACKEND` controls how Chroma collections are opened:

- `default` is the current default and main paper setting. It opens the
  collection with Chroma defaults, i.e. the lightweight default embedding/index
  path such as MiniLM-backed Chroma collections.
- `bge_m3` is the stronger BGE-M3-oriented backend with the project's
  recall-oriented HNSW configuration.
- `matkv_bge_m3` is accepted only as a legacy alias for older DBs persisted
  with the previous embedding-function name.

Changing between `bge_m3` and `default` changes the retrieval condition and
should be reported as a separate retriever setting. `matkv_bge_m3` is only the
legacy persisted-function name for the same BGE-M3 path.

Dense compare selectors use `COMPARE_EMBED_DIR`. ColBERT-window selectors use
`COLBERT_WINDOW_DIR`, `COLBERT_MODEL_NAME`, `COLBERT_DEVICE`, and related
ColBERT environment variables. A result is valid only when these runtime
settings match the artifact that was built.

`RETRIEVAL_INCLUDE_DOCUMENTS=False` is allowed only for methods whose prompt
text can be reconstructed from cacheable metadata. Do not use it for methods
that require original retrieved document text unless that path has been
explicitly checked.

## Compression And Budget Controls

`COMPRESS_METHOD` selects the retrieval/compression variant. Empty
`COMPRESS_METHOD` means no downstream compressor; retrieved units are used
directly.

`GLOBAL_TOP_R` controls ratio-based candidate selection for methods that use a
global keep ratio. Results with different `GLOBAL_TOP_R` values are separate
compression settings.

`COLBERT_FINAL_TOKEN_BUDGET` controls fixed-token-budget ColBERT paths and is
included in output naming and `summary2` CSV rows when set.

`RETAIN_TOKEN_RATIO` computes the final token budget per query as a ratio of
retrieved context tokens. If both `RETAIN_TOKEN_RATIO` and
`COLBERT_FINAL_TOKEN_BUDGET` are set, the retained-ratio budget takes
precedence.

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

`ttft_per_batch_avg_sec` in grid summaries is derived as:

```text
time_per_batch_avg_sec - decode_per_batch_avg_sec - generate_extra_per_batch_avg_sec
```

It is a reconstructed batch-average latency component, not an independently
instrumented server-side TTFT.

## Output Files And Grid Summaries

`run/eval.sh` writes prediction JSONL to `OUTPUT_FILE` when set, otherwise to
`outputs/eval-...jsonl` with suffixes derived from method, `TOP_K`,
`GLOBAL_TOP_R`, cache mode, LLM precision, model tag, and optional budget tags.

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
- final budget controls such as `GLOBAL_TOP_R`, `COLBERT_FINAL_TOKEN_BUDGET`,
  and `RETAIN_TOKEN_RATIO`
- GPU/timing environment when reporting latency or throughput

If any of these differ, report the changed condition explicitly and treat the
result as a separate ablation or baseline.

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
