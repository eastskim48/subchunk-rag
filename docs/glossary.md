# Glossary

## chunk
Text unit produced from an object.
Main retrieval unit in SubChunkKV.
Main cacheable unit in MatKV.

## subchunk
Smaller text unit produced from a chunk.
Main cacheable retrieval unit in this project.

## MatKV
Baseline framework for RAG serving.
Materialize KV cache offline for document- or chunk-level text.
Store precomputed KV in SSD or external storage.
Reuse stored KV at runtime to reduce prefill latency.

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
Typical setting in MatKV.

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