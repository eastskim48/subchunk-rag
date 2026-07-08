# Research Goal

## Main Goal

Develop and evaluate Contextualized Subchunk Selection for efficient RAG
serving.

The main claim is that RAG systems should not force the same text unit to serve
both retrieval and generation. Fixed-size chunks can remain useful retrieval
units, but the LLM prompt should be built from smaller query-relevant evidence
units.

## Problem Statement

Fixed-size chunk RAG has a unit mismatch:

- Vector search prefers retrieval units large enough to preserve context and
  stable enough for ANN retrieval.
- Answer generation often needs only a much smaller evidence unit.

When the whole retrieved chunk is inserted into the prompt, the system pays for
many irrelevant tokens. This increases prefill cost, latency, memory use, and
distractor exposure.

## Hypothesis

If evidence units are contextualized offline and selected at query time only
inside retrieved chunks, then RAG can reduce prompt length while preserving or
improving answer quality.

More specifically:

- Contextualized evidence representations should outperform naive standalone
  subchunk embeddings at low token budgets.
- Late interaction scoring should provide better fine-grained evidence matching
  than single-vector dense similarity.
- Moving contextualization offline should keep online compression overhead small.
- Shorter selected prompts should improve latency, TTFT, memory use, and
  throughput.
- Smaller LLMs may gain quality because distractor context is removed.

## Primary Objectives

1. Formalize the mismatch between retrieval units and evidence units in RAG.
2. Implement a retrieval-bounded evidence selection pipeline:
   - retrieve fixed-size chunks with the existing vector DB.
   - collect candidate evidence units from retrieved chunks.
   - score candidates with contextualized representations.
   - assemble prompts from selected evidence units.
3. Compare contextualized subchunk selection against:
   - full retrieved chunk prompting.
   - direct subchunk retrieval where applicable.
   - dense embedding based subchunk selection.
   - other lightweight compression baselines.
4. Measure quality and efficiency together:
   - EM/F1.
   - input length.
   - end-to-end latency.
   - TTFT.
   - prefill/decode breakdown.
   - compression overhead.
   - memory footprint and throughput when available.

## Method Direction

Use offline contextualization plus online late interaction.

Offline materialization:

- split documents into retrieval chunks and evidence units.
- construct a context window around each evidence unit.
- encode the context window.
- store representation for the center evidence unit.
- store metadata linking evidence units to retrieval chunks.

Online evaluation:

- retrieve top-k chunks.
- restrict candidate evidence units to those attached to retrieved chunks.
- encode the query.
- score candidates with late interaction.
- select evidence by ratio or token budget.
- preserve source order when assembling the prompt.
- run the LLM on the selected-context prompt.

## Evaluation Questions

The experiments should answer:

1. Can selected evidence preserve EM/F1 while reducing input length?
2. Does contextualized selection beat dense embedding selection under the same
   budget?
3. How much latency, TTFT, and prefill time are reduced?
4. Is online scoring/pruning overhead small enough to avoid canceling the
   inference savings?
5. Does reducing distractor context help small LLMs at high top-k?
6. How do fixed token budgets and retained-token-ratio budgets affect the
   quality-efficiency trade-off?

## Current Non-Goals

- Do not make KV cache reuse the primary paper contribution yet.
- Do not claim broad support for code, tables, or structured documents without
  dedicated evidence-unit parsers and experiments.
- Do not silently change evaluation protocol, dataset handling, prompt format,
  or scoring to make results look better.
- Do not report latency or throughput without documenting batch size, backend,
  GPU condition, and cache mode.

## Future Direction

The longer-term serving direction is to combine fine-grained evidence selection
with subchunk-level KV reuse.

That requires solving additional cache-specific issues:

- position dependence.
- boundary effects after dropping context.
- cache validity when prompt units are selected non-contiguously.
- selective recomputation or repair near evidence boundaries.

These are important follow-up problems, but the current research goal is to
first establish the evidence-selection contribution cleanly and reproducibly.
