# Glossary

## chunk
Text unit produced from an object.
Main retrieval unit in SubChunkKV.
Main retrieval and prompt unit in the vanilla baseline.

## subchunk
Smaller text unit produced from a chunk.
Main cacheable retrieval unit in this project.

## vanilla
Baseline that performs no subchunking or context compression.
Use retrieved chunks directly to construct the prompt.
Cache-on and cache-off are independent evaluation settings.

## subchunkKV
Proposed method in this project.
Extend KV materialization and reuse from chunk level to subchunk level.
Select only necessary subchunks at runtime.
Load and assemble only the corresponding subchunk KV.
Apply repair or recomputation near boundaries when needed.

## materialization
Offline process that runs prefill on a text unit and stores its KV cache for later reuse.

## prefill
Forward pass over the prefix context to build KV cache.

## KV cache
Transformer key-value cache used to avoid recomputing previous tokens.

## KV reuse
Reuse precomputed KV instead of recomputing full prefill at runtime.

## chunk-level KV reuse
Reuse precomputed KV where the cacheable unit is a chunk.
Cache-on setting for the vanilla fixed-chunk baseline.

## subchunk-level KV reuse
Reuse precomputed KV where the cacheable unit is a subchunk.
Core setting in subchunkKV.

## context compression
Reduce runtime context by keeping only text needed to answer the query.

## cache-friendly context compression
Context compression designed to preserve or improve KV reuse.
Main design goal of this project.

## boundary
Connection point between adjacent KV segments after assembly.

## inter-doc boundary
Boundary between subchunks from different original documents.

## intra-doc boundary
Boundary between subchunks from the same original chunk or document.

## boundary repair
Extra computation near boundaries to reduce quality loss after KV assembly.

## RoPE modification
Position-handling technique used in this project to improve KV reuse under changed context layout.

## LegoLink-0
Technique used in this project as a starting point for more stable KV reuse.
Use BOS-token-based handling to reduce attention sink effects.

## vanilla
Reference baseline without the proposed subchunkKV pipeline.

## answer quality
Task accuracy after retrieval, KV reuse, and decoding.
Must be preserved while reducing latency and context length.

## shared retrieval corpus
A single common document collection searched for every query.
For example, converting NewsQA into a RAG benchmark can place all 12,744 CNN
articles into one vector DB. Every question then retrieves from that same DB
instead of receiving its originally associated article directly.
Always explain this term when first used in a user-facing discussion.

## pooled retrieval corpus
A derived shared retrieval corpus built by taking the union of documents
associated with all queries in an evaluation split.
For one query, its associated document is relevant and documents associated
with other queries act as distractors. This is not automatically equivalent to
the dataset's original or official full-corpus retrieval setting. Always state
which documents were pooled and do not shorten this to "pooled" without an
explanation.
