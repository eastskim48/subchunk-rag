# Project Context

## Background

This project studies efficient Retrieval-Augmented Generation (RAG) context
construction.

In a typical RAG pipeline, documents are split into fixed-size chunks, each
chunk is indexed in a vector database, and the top-k retrieved chunks are
inserted into the LLM prompt. This is simple and operationally useful, but the
unit that works well for vector search is often larger than the unit actually
needed for answer generation.

The current research direction is based on the following mismatch:

- Retrieval unit: the text unit indexed and retrieved by the vector database.
- Evidence unit: the smaller text unit the LLM actually needs to answer the
  query.

Fixed-size chunks such as 512 or 1024 tokens can be reasonable retrieval units,
but they often contain much more text than the answer requires. Passing the full
retrieved chunks to the LLM increases input length, prefill latency, memory
usage, and exposure to distractors.

## Core Problem

Existing fixed-chunk RAG treats retrieval units and evidence units as the same
thing. This creates redundant prompts:

- Top-k retrieval may fetch chunks that contain useful evidence.
- The useful evidence may be only a sentence, span, or small semantic unit.
- The prompt still includes the entire retrieved chunk.

This redundancy hurts serving efficiency and can also hurt answer quality,
especially for smaller LLMs that are more sensitive to long prompts and
irrelevant context.

## Retriever Strength And System Cost

The measured benefit of subchunk selection depends on how strong and expensive
the vector DB retrieval setup is.

If the vector DB uses a very strong embedding model, high-dimensional vectors,
and carefully tuned index/search configuration, retrieval recall can become high
for both coarse chunks and smaller retrieval units. In that regime, different
retrieval granularities may show little recall difference because the retriever
already finds the relevant documents or chunks reliably.

However, that is not a free baseline. Stronger vector DB retrieval usually
increases system cost:

- larger embedding vectors increase index memory and storage footprint.
- heavier embedding models increase offline indexing cost and query embedding
  latency.
- more recall-oriented HNSW/search settings can increase memory use and search
  time.
- maintaining multiple high-quality indexes for different chunk sizes can be
  operationally expensive.

This matters for positioning the project. Subchunk selection should not be
evaluated only as "does it improve initial retrieval recall over the strongest
possible vector DB." A sufficiently expensive retriever can hide recall
differences. The more relevant system question is whether we can keep retrieval
coarse and operationally practical, then use query-time evidence selection to
reduce prompt length and LLM serving cost.

Therefore, the project should interpret results under two regimes:

- Strong-retriever regime: recall differences may be small, so the main benefit
  is shorter prompts, lower latency, lower memory use, and reduced distractor
  exposure.
- Cost-constrained retriever regime: vector DB memory/search cost is limited,
  so retrieval granularity and evidence selection can have a clearer effect on
  recall-quality-efficiency trade-offs.

## Research Direction

The project investigates Contextualized Subchunk Selection:

1. Keep coarse fixed-size chunks as vector-DB retrieval units.
2. Split each retrieved chunk into smaller evidence candidates.
3. Contextualize each evidence candidate with neighboring text offline.
4. At query time, score only the candidates inside retrieved chunks.
5. Assemble the final prompt using selected evidence units instead of whole
   chunks.

The key idea is to separate the unit used for retrieval from the unit used for
generation. This preserves the practical advantages of fixed-size vector
retrieval while reducing the LLM input to query-relevant evidence.

For the current paper experiments, the intended main configuration is:

- method: `colbert_sliding_region`
- vector DB backend: Chroma `default`
- retrieval DB family: lightweight default/MiniLM-style DBs, such as
  `sent-default-512`

The stronger `bge_m3` backend is useful as a strong-retriever comparison, but it
is not the default main paper setting.

## Why Naive Subchunking Is Not Enough

Simply indexing smaller subchunks is not sufficient.

Small subchunks can lose local context. Pronouns, entity references, event
descriptions, table structure, or code dependencies may only make sense when
neighboring text is visible. A subchunk representation computed from the
subchunk alone can therefore miss relevant evidence or over-score misleading
surface matches.

The project addresses this by contextualizing each candidate evidence unit in a
larger surrounding window, while still selecting and prompting only the center
evidence unit.

## Current Method Shape

The main implementation direction uses ColBERT-style late interaction:

- Offline:
  - split documents into retrieval chunks and smaller evidence units.
  - build a context window around each evidence unit.
  - encode the window.
  - store only the token representations corresponding to the center evidence
    unit.
  - save mappings among document, retrieval chunk, evidence unit, prompt text,
    and representation.
- Online:
  - retrieve top-k chunks from the vector database.
  - collect evidence candidates attached to the retrieved chunks.
  - encode the query.
  - score candidates with late interaction between query tokens and stored
    evidence-token representations.
  - select evidence under a ratio or token budget.
  - assemble the final prompt from selected evidence text.

This moves most contextualization cost offline and keeps query-time compression
overhead lightweight.

## Evaluation Focus

The project evaluates whether contextualized subchunk selection can:

- preserve or improve answer quality.
- reduce input token length.
- reduce end-to-end latency and TTFT.
- reduce memory footprint.
- improve feasible batch size or throughput.
- keep online compression overhead small.
- mitigate long-context degradation in smaller LLMs.

Primary workloads are text-based multi-hop QA datasets such as LongBench
HotpotQA and 2Wiki-style settings. Other document types such as code and tables
are future extensions that may require structure-aware evidence units.

## Relation To KV Cache Reuse

Fine-grained evidence selection is also relevant to future subchunk-level KV
reuse, but the current paper direction should treat KV reuse as a longer-term
serving extension rather than the primary contribution.

Subchunk-level KV reuse introduces additional issues:

- KV caches are prefix- and position-dependent.
- Selecting only some subchunks changes attention context.
- Reordering or dropping surrounding text can create boundary effects.
- Position handling and selective recomputation may be needed.

The current project therefore separates two layers:

- Current main contribution: context-aware fine-grained evidence selection for
  efficient RAG prompts.
- Future extension: make selected evidence units cache-friendly for
  subchunk-level KV reuse.

## Documentation Notes

- `docs/research_goal.md` states the current research objectives.
- `docs/eval_protocol.md` defines how evaluation results should be run and
  compared.
- `docs/handoff.md` tracks session-level project state.
- `docs_backup/paper_draft_ko.md` is the current Korean paper draft used to
  rewrite this context.
- Older versions of this file are kept under `docs_backup/`.
