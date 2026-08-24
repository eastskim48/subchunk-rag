"""XRAG Jina reranking and XRAG-to-CASS composition."""

from __future__ import annotations

import math
import os
import time
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from chunk import CacheableChunk, RetrievableChunk
from compressor.base import Compressor
from compressor.methods.colbert.region import ColBERTSlidingRegionCompressor


class _XRAGJinaReranker:
    """Reproduce XRAG's pinned FlagReranker path with Jina loading enabled."""

    def __init__(self) -> None:
        self.model_name = os.getenv(
            "XRAG_MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual"
        )
        self.top_n = int(os.getenv("XRAG_TOP_N", "10"))
        if self.top_n <= 0:
            raise ValueError("XRAG_TOP_N must be positive")

        self.batch_size = 256
        self.max_length = int(os.getenv("XRAG_MAX_LENGTH", "1024"))
        if self.max_length <= 0:
            raise ValueError("XRAG_MAX_LENGTH must be positive")
        self.device = os.getenv(
            "XRAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(
            "XRAG Jina reranking enabled. "
            f"Initializing model: {self.model_name} on {self.device}; "
            f"top_n={self.top_n}, max_length={self.max_length}, use_fp16=False"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # XRAG pins FlagEmbedding 1.2.10, whose FlagReranker omits this required
        # Jina loader flag. The model and scoring path are otherwise unchanged.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _document_text(doc: RetrievableChunk) -> str:
        text = getattr(doc, "text", None)
        return text if isinstance(text, str) else ""

    @staticmethod
    def _normalize_scores(raw_scores, expected: int) -> list[float]:
        if isinstance(raw_scores, torch.Tensor):
            scores = raw_scores.detach().float().cpu().reshape(-1).tolist()
        elif isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        elif hasattr(raw_scores, "tolist"):
            scores = raw_scores.tolist()
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
        else:
            scores = list(raw_scores)
        scores = [float(score) for score in scores]
        if len(scores) != expected:
            raise ValueError(
                "XRAG reranker returned an unexpected number of scores: "
                f"expected {expected}, got {len(scores)}"
            )
        return scores

    @torch.no_grad()
    def compute_score(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for start in range(0, len(sentence_pairs), self.batch_size):
            batch = sentence_pairs[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            batch_scores = (
                self.model(
                    **inputs,
                    return_dict=True,
                )
                .logits.view(-1)
                .float()
            )
            scores.extend(batch_scores.cpu().tolist())
        return scores

    def rank_indices(
        self,
        docs: List[RetrievableChunk],
        query: str,
        profile: dict[str, float | int],
    ) -> list[int]:
        if not docs:
            return []
        pairs = [(query, self._document_text(doc)) for doc in docs]
        score_start = time.perf_counter()
        scores = self._normalize_scores(self.compute_score(pairs), len(docs))
        profile["xrag_score_time"] += time.perf_counter() - score_start
        profile["xrag_score_pair_count"] += len(pairs)
        profile["xrag_score_batch_count"] += math.ceil(len(pairs) / self.batch_size)

        sort_start = time.perf_counter()
        scored_indices = list(zip(scores, range(len(docs))))
        # Match llama-index-postprocessor-flag-embedding-reranker 0.2.0.
        scored_indices.sort(key=lambda item: -item[0] if item[0] else 0)
        profile["xrag_sort_time"] += time.perf_counter() - sort_start
        return [idx for _, idx in scored_indices[: self.top_n]]


def _empty_xrag_profile(
    batch_top_k_docs: List[List[RetrievableChunk]],
) -> dict[str, float | int]:
    return {
        "xrag_score_time": 0.0,
        "xrag_sort_time": 0.0,
        "xrag_build_output_time": 0.0,
        "xrag_query_count": len(batch_top_k_docs),
        "xrag_retrieved_doc_count": sum(len(docs) for docs in batch_top_k_docs),
        "xrag_selected_doc_count": 0,
        "xrag_score_pair_count": 0,
        "xrag_score_batch_count": 0,
    }


def _whole_chunk_document(doc: RetrievableChunk) -> RetrievableChunk:
    cloned = doc.clone()
    text = _XRAGJinaReranker._document_text(doc)
    cloned.cacheables = (
        [CacheableChunk(id=f"{doc.id}::xrag_jina", text=text)] if text else []
    )
    return cloned


class XRAGJinaRerankerCompressor(Compressor):
    """Return XRAG's top-N Jina-reranked whole chunks."""

    def __init__(self) -> None:
        super().__init__()
        self.reranker = _XRAGJinaReranker()
        self.last_profile: dict[str, float | int] = {}

    def clear_inter_batch_cache(self) -> None:
        return None

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        if len(batch_top_k_docs) != len(batch_queries):
            raise ValueError(
                "XRAG requires one retrieved-document batch per query: "
                f"got {len(batch_top_k_docs)} document batches and "
                f"{len(batch_queries)} queries"
            )
        profile = _empty_xrag_profile(batch_top_k_docs)
        selected_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            selected_indices = self.reranker.rank_indices(docs, query, profile)
            build_start = time.perf_counter()
            selected_batches.append(
                [_whole_chunk_document(docs[idx]) for idx in selected_indices]
            )
            profile["xrag_selected_doc_count"] += len(selected_indices)
            profile["xrag_build_output_time"] += time.perf_counter() - build_start
        self.last_profile = profile
        return selected_batches


class XRAGJinaCASSCompressor(ColBERTSlidingRegionCompressor):
    """Apply CASS only to the top-N coarse chunks returned by XRAG Jina."""

    def __init__(self) -> None:
        super().__init__()
        self.reranker = _XRAGJinaReranker()

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        if len(batch_top_k_docs) != len(batch_queries):
            raise ValueError(
                "XRAG-to-CASS requires one retrieved-document batch per query: "
                f"got {len(batch_top_k_docs)} document batches and "
                f"{len(batch_queries)} queries"
            )
        xrag_profile = _empty_xrag_profile(batch_top_k_docs)
        reranked_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            selected_indices = self.reranker.rank_indices(docs, query, xrag_profile)
            reranked_batches.append([docs[idx].clone() for idx in selected_indices])
            xrag_profile["xrag_selected_doc_count"] += len(selected_indices)

        compressed_batches = super().compress_batch_top_k_docs(
            reranked_batches, batch_queries
        )
        self.last_profile.update(xrag_profile)
        return compressed_batches
