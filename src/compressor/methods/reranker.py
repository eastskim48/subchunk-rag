"""Official listwise reranking followed by whole-chunk prompting or native CASS."""

import json
import math
from numbers import Integral
import os
from pathlib import Path
import time

import torch
from transformers import AutoModel

from chunk import CacheableChunk
from compressor.base import Compressor
from compressor.methods.colbert.region import ColBERTSlidingRegionCompressor


class JinaListwiseReranker:
    def __init__(self):
        self.top_n = int(os.environ["RERANKER_TOP_N"])
        if self.top_n <= 0:
            raise ValueError("RERANKER_TOP_N must be positive")
        self.trace_file = os.getenv("RERANKER_TRACE_FILE")
        self.call_index = 0
        dtype = os.environ["RERANKER_DTYPE"]
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported reranker dtype: {dtype}")
        self.model = AutoModel.from_pretrained(
            os.environ["RERANKER_MODEL_NAME"],
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
            attn_implementation=os.environ["RERANKER_ATTN_IMPLEMENTATION"],
        ).to(os.environ["RERANKER_DEVICE"])
        self.model.eval()
        if not callable(getattr(self.model, "rerank", None)):
            raise ValueError("Reranker model must expose the official rerank() API")

    def rank_indices(self, docs, query):
        if not docs:
            return []
        texts = [doc.text for doc in docs]
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Reranking requires stored retrieval chunk texts")
        # The official API scores the candidate list jointly; do not split it
        # into independent query/document pairs or reimplement its ordering.
        with torch.inference_mode():
            ranked = self.model.rerank(query, texts, top_n=None)
        if len(ranked) != len(docs):
            raise ValueError("Official reranker returned an incomplete candidate list")
        indices, scores = [], []
        for result in ranked:
            index = result["index"]
            score = float(result["relevance_score"])
            if not isinstance(index, Integral) or not 0 <= index < len(docs):
                raise ValueError("Reranker returned an invalid document index")
            if not math.isfinite(score) or result["document"] != texts[index]:
                raise ValueError("Reranker score/text does not match its input")
            indices.append(int(index))
            scores.append(score)
        if len(set(indices)) != len(docs) or scores != sorted(scores, reverse=True):
            raise ValueError("Reranker returned duplicate indices or unsorted scores")
        selected = indices[: self.top_n]
        self.call_index += 1
        if self.trace_file:
            path = Path(self.trace_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(
                            call_index=self.call_index,
                            query=query,
                            input_chunk_ids=[str(doc.id) for doc in docs],
                            ranked_indices=indices,
                            ranked_scores=scores,
                            selected_indices=selected,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return selected


def rerank_batches(reranker, batches, queries):
    if len(batches) != len(queries):
        raise ValueError("Reranking requires one candidate list per query")
    return [
        [docs[i] for i in reranker.rank_indices(docs, query)]
        for docs, query in zip(batches, queries, strict=True)
    ]


class RerankerCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.reranker = JinaListwiseReranker()
        self.last_profile = {}

    def compress_batch_top_k_docs(self, batch_top_k_docs, batch_queries):
        started = time.perf_counter()
        batches = rerank_batches(self.reranker, batch_top_k_docs, batch_queries)
        result = []
        for docs in batches:
            selected = []
            for doc in docs:
                cloned = doc.clone()
                cloned.cacheables = [
                    CacheableChunk(
                        id=f"{doc.id}::reranked_whole_chunk",
                        text=doc.text,
                    )
                ]
                selected.append(cloned)
            result.append(selected)
        self.last_profile = {"rerank_time": time.perf_counter() - started}
        return result


class RerankerCASSCompressor(ColBERTSlidingRegionCompressor):
    def __init__(self):
        super().__init__()
        self.reranker = JinaListwiseReranker()

    def compress_batch_top_k_docs(self, batch_top_k_docs, batch_queries):
        started = time.perf_counter()
        selected = rerank_batches(self.reranker, batch_top_k_docs, batch_queries)
        rerank_seconds = time.perf_counter() - started
        # Keep native chunk metadata, sentence IDs and order for artifact lookup.
        result = super().compress_batch_top_k_docs(selected, batch_queries)
        self.last_profile["rerank_time"] = rerank_seconds
        return result
