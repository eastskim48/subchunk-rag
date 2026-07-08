import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch

from chunk import RetrievableChunk
from compressor.base import Compressor
from compressor.comparison_compressor import MaterializedGlobalComparisonSummarizer

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SUPPORTED_BM25_FORMAT = "rank_bm25_parent_doc_idf_artifact_v1"


def tokenize_for_bm25(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def parse_positive_rate(name: str, raw_value: str) -> float:
    value = raw_value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
        rate = float(value) / 100.0
    else:
        rate = float(value)
        if rate > 1.0:
            rate /= 100.0
    if rate <= 0:
        raise ValueError(f"{name} must be positive, got {raw_value!r}")
    return rate


def global_top_count(candidate_count: int, rate: float) -> int:
    if candidate_count <= 0:
        return 0
    return min(candidate_count, max(1, math.ceil(candidate_count * rate)))


class BM25Scorer:
    def __init__(self, artifact_path: str):
        with open(artifact_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        artifact_format = artifact.get("format")
        if artifact_format != SUPPORTED_BM25_FORMAT:
            raise ValueError(
                f"unsupported BM25 artifact format: {artifact_format!r}; "
                f"expected {SUPPORTED_BM25_FORMAT!r}"
            )

        parameters = artifact.get("bm25_parameters", {})
        self.artifact_path = artifact_path
        self.idf: Dict[str, float] = {
            str(token): float(value) for token, value in artifact.get("idf", {}).items()
        }
        self.k1 = float(parameters.get("k1", 1.5))
        self.b = float(parameters.get("b", 0.75))
        self.avg_doc_len = float(artifact.get("avg_cacheable_doc_len") or 0.0)
        if self.avg_doc_len <= 0:
            raise ValueError(
                f"BM25 artifact has invalid avg_cacheable_doc_len: {self.avg_doc_len}"
            )

    def score_text(self, query_tokens: Sequence[str], text: str) -> float:
        doc_tokens = tokenize_for_bm25(text)
        if not query_tokens or not doc_tokens:
            return 0.0

        term_counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
        score = 0.0
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(token)
            if idf is None:
                continue
            score += idf * (tf * (self.k1 + 1.0) / (tf + length_norm))
        return score

    def score_texts(self, query: str, texts: Sequence[str]) -> List[float]:
        query_tokens = tokenize_for_bm25(query)
        return [self.score_text(query_tokens, text) for text in texts]


def resolve_bm25_artifact_path() -> str:
    explicit_path = os.getenv("BM25_IDF_PATH")
    if explicit_path:
        return explicit_path

    dataset_path = os.getenv("DATASET_PATH")
    data_subdir = os.getenv("DATA_SUBDIR")
    if dataset_path and data_subdir:
        return os.path.join(dataset_path, data_subdir, "bm25", "bm25_idf.json")

    raise ValueError(
        "BM25_IDF_PATH must be set, or DATASET_PATH and DATA_SUBDIR must point to a "
        "dataset subdir containing bm25/bm25_idf.json"
    )


def resolve_bm25_async_artifact_path() -> str:
    explicit_path = os.getenv("BM25_ASYNC_IDF_PATH") or os.getenv("BM25_IDF_PATH")
    if explicit_path:
        return explicit_path

    dataset_path = os.getenv("DATASET_PATH")
    bm25_data_subdir = os.getenv("BM25_ASYNC_DATA_SUBDIR", "sent")
    if dataset_path:
        return os.path.join(dataset_path, bm25_data_subdir, "bm25", "bm25_idf.json")

    raise ValueError(
        "BM25_ASYNC_IDF_PATH or BM25_IDF_PATH must be set, or DATASET_PATH must point to "
        "a dataset with the BM25 async subdir artifact"
    )


def normalize_by_owner(
    values: torch.Tensor, owner_indices: Sequence[int]
) -> torch.Tensor:
    if values.numel() == 0:
        return values
    normalized = torch.zeros_like(values, dtype=torch.float32)
    owner_tensor = torch.tensor(owner_indices, dtype=torch.long)
    for owner_idx in owner_tensor.unique().tolist():
        mask = owner_tensor == owner_idx
        owner_values = values[mask].to(torch.float32)
        std = owner_values.std(unbiased=False)
        if float(std) <= 1e-12:
            continue
        normalized[mask] = (owner_values - owner_values.mean()) / std
    return normalized


def reciprocal_rank_fusion_by_owner(
    dense_scores: torch.Tensor,
    sparse_scores: torch.Tensor,
    owner_indices: Sequence[int],
    rrf_k: float = 60.0,
    dense_weight: float = 0.5,
) -> torch.Tensor:
    if dense_scores.shape != sparse_scores.shape:
        raise ValueError(
            f"RRF score shape mismatch: dense={tuple(dense_scores.shape)} sparse={tuple(sparse_scores.shape)}"
        )
    if dense_scores.numel() == 0:
        return dense_scores.to(torch.float32)
    if rrf_k < 0:
        raise ValueError(f"RRF k must be non-negative, got {rrf_k}")
    if dense_weight < 0.0 or dense_weight > 1.0:
        raise ValueError(f"dense_weight must be between 0 and 1, got {dense_weight}")

    owner_tensor = torch.tensor(owner_indices, dtype=torch.long)
    if owner_tensor.numel() != dense_scores.numel():
        raise ValueError(
            f"owner_indices length mismatch: owners={owner_tensor.numel()} scores={dense_scores.numel()}"
        )

    fused = torch.zeros_like(dense_scores, dtype=torch.float32)
    sparse_weight = 1.0 - dense_weight
    for owner_idx in owner_tensor.unique().tolist():
        mask = owner_tensor == owner_idx
        local_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if local_indices.numel() == 0:
            continue

        dense_order = torch.argsort(
            dense_scores[mask].to(torch.float32), descending=True, stable=True
        )
        sparse_order = torch.argsort(
            sparse_scores[mask].to(torch.float32), descending=True, stable=True
        )
        dense_ranks = torch.empty(local_indices.numel(), dtype=torch.float32)
        sparse_ranks = torch.empty(local_indices.numel(), dtype=torch.float32)
        dense_ranks[dense_order] = torch.arange(
            1, local_indices.numel() + 1, dtype=torch.float32
        )
        sparse_ranks[sparse_order] = torch.arange(
            1, local_indices.numel() + 1, dtype=torch.float32
        )

        fused[local_indices] = dense_weight / (rrf_k + dense_ranks) + sparse_weight / (
            rrf_k + sparse_ranks
        )
    return fused


class BM25GlobalSummarizer(Compressor):
    def __init__(self):
        super().__init__()
        self.global_top_r = parse_positive_rate(
            "GLOBAL_TOP_R", os.getenv("GLOBAL_TOP_R", "0.1")
        )
        self.bm25_path = resolve_bm25_artifact_path()
        self.bm25 = BM25Scorer(self.bm25_path)
        print(f"BM25 global subchunk selection enabled. artifact={self.bm25_path}")

    def _build_unselected_document(self, doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = []
        return cloned

    def _score_candidates(
        self,
        docs: List[RetrievableChunk],
        query: str,
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        refs = []
        texts = []
        for doc_idx, doc in enumerate(docs):
            for cacheable_idx, cacheable in enumerate(
                getattr(doc, "cacheables", []) or []
            ):
                if not cacheable.text:
                    continue
                refs.append((doc_idx, cacheable_idx))
                texts.append(cacheable.text)

        if not refs:
            return refs, torch.empty(0, dtype=torch.float32)

        scores = self.bm25.score_texts(query, texts)
        return refs, torch.tensor(scores, dtype=torch.float32)

    @staticmethod
    def _deduplicate_refs(
        docs: List[RetrievableChunk],
        refs: List[Tuple[int, int]],
        scores: torch.Tensor,
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        dedup_refs: List[Tuple[int, int]] = []
        dedup_scores: List[torch.Tensor] = []
        index_by_cacheable_id: Dict[str, int] = {}

        for flat_idx, (doc_idx, cacheable_idx) in enumerate(refs):
            cacheables = getattr(docs[doc_idx], "cacheables", []) or []
            cacheable = (
                cacheables[cacheable_idx]
                if 0 <= cacheable_idx < len(cacheables)
                else None
            )
            cacheable_id = getattr(cacheable, "id", None)
            score = scores[flat_idx]

            if cacheable_id:
                existing_idx = index_by_cacheable_id.get(cacheable_id)
                if existing_idx is not None:
                    if score > dedup_scores[existing_idx]:
                        dedup_refs[existing_idx] = (doc_idx, cacheable_idx)
                        dedup_scores[existing_idx] = score
                    continue
                index_by_cacheable_id[cacheable_id] = len(dedup_refs)

            dedup_refs.append((doc_idx, cacheable_idx))
            dedup_scores.append(score)

        if not dedup_scores:
            return dedup_refs, torch.empty(0, dtype=scores.dtype)
        return dedup_refs, torch.stack(dedup_scores).to(scores.dtype)

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        summarized_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            summarized_chunks = [self._build_unselected_document(doc) for doc in docs]
            sentence_refs, bm25_scores = self._score_candidates(docs, query)
            sentence_refs, bm25_scores = self._deduplicate_refs(
                docs, sentence_refs, bm25_scores
            )
            keep_count = global_top_count(len(sentence_refs), self.global_top_r)
            if keep_count <= 0:
                summarized_batches.append(summarized_chunks)
                continue

            top_indices = torch.topk(bm25_scores, k=keep_count).indices.tolist()
            selected_by_doc = {}
            for flat_idx in top_indices:
                doc_idx, cacheable_idx = sentence_refs[flat_idx]
                selected_by_doc.setdefault(doc_idx, []).append(cacheable_idx)

            for doc_idx, selected_indices in selected_by_doc.items():
                summarized_chunks[doc_idx] = self._build_selected_document(
                    docs[doc_idx], selected_indices
                )
            summarized_batches.append(summarized_chunks)
        return summarized_batches

    def compress(self, document_text: str, query: str) -> str:
        del query
        return document_text


class PNRawPromptReplacementMixin:
    @staticmethod
    def _reject_cache_on(method_name: str) -> None:
        use_past_cache = os.getenv("EVAL_USE_PAST_CACHE", "False").strip().lower()
        if use_past_cache in {"1", "true", "yes"}:
            raise ValueError(
                f"compress_method='{method_name}' is cache-off only: "
                "selection uses resolved text but prompt text is replaced with raw original text"
            )

    def _init_raw_prompt_replacement(self) -> None:
        self.pn_mapping_dir = self._resolve_pn_mapping_dir()
        self._original_text_cache: Dict[str, Dict[str, str]] = {}
        print(
            f"Async raw-prompt replacement enabled. pn_mapping_dir={self.pn_mapping_dir}"
        )

    @staticmethod
    def _resolve_pn_mapping_dir() -> str:
        explicit_path = os.getenv("PN_MAPPING_DIR")
        if explicit_path:
            return explicit_path

        dataset_path = os.getenv("DATASET_PATH")
        if dataset_path:
            return os.path.join(dataset_path, "pn_mapping")

        raise ValueError(
            "PN_MAPPING_DIR must be set, or DATASET_PATH must point to a dataset with pn_mapping/"
        )

    def _load_original_texts(self, filename: str) -> Dict[str, str]:
        if filename in self._original_text_cache:
            return self._original_text_cache[filename]

        mapping_path = os.path.join(self.pn_mapping_dir, f"{filename}.json")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"pn mapping missing for raw prompt replacement: {mapping_path}"
            )

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if mapping.get("format") != "pn_mapping_v1":
            raise ValueError(
                f"unsupported pn mapping format in {mapping_path}: {mapping.get('format')!r}"
            )

        original_by_sentence_id = {}
        for sentence_view in mapping.get("sentence_views", []):
            sentence_id = sentence_view.get("sentence_id")
            original_text = sentence_view.get("original_text")
            if sentence_id and original_text:
                original_by_sentence_id[str(sentence_id)] = str(original_text)

        self._original_text_cache[filename] = original_by_sentence_id
        return original_by_sentence_id

    @staticmethod
    def _filename_for_cacheable(cacheable) -> str:
        parent_doc_id = getattr(cacheable, "parent_doc_id", None)
        if parent_doc_id:
            return str(parent_doc_id)
        cacheable_id = getattr(cacheable, "id", "")
        if "::" in cacheable_id:
            return cacheable_id.split("::", 1)[0]
        raise ValueError(
            f"cannot infer pn mapping filename from cacheable id: {cacheable_id!r}"
        )

    def _replace_selected_with_original_text(
        self, summarized_batches: List[List[RetrievableChunk]]
    ):
        for docs in summarized_batches:
            for doc in docs:
                for cacheable in getattr(doc, "cacheables", []) or []:
                    filename = self._filename_for_cacheable(cacheable)
                    original_by_sentence_id = self._load_original_texts(filename)
                    original_text = original_by_sentence_id.get(cacheable.id)
                    if original_text is None:
                        raise KeyError(
                            f"pn mapping original_text missing for cacheable id {cacheable.id!r} "
                            f"in filename {filename!r}"
                        )
                    cacheable.text = original_text
                    cacheable.sentence_texts = [original_text]
        return summarized_batches


class BM25GlobalAsyncSummarizer(PNRawPromptReplacementMixin, BM25GlobalSummarizer):
    def __init__(self):
        self._reject_cache_on("bm25_global_async")
        Compressor.__init__(self)
        self.global_top_r = parse_positive_rate(
            "GLOBAL_TOP_R", os.getenv("GLOBAL_TOP_R", "0.1")
        )
        self.bm25_path = resolve_bm25_async_artifact_path()
        self.bm25 = BM25Scorer(self.bm25_path)
        print(
            f"BM25 global async subchunk selection enabled. artifact={self.bm25_path}"
        )
        self._init_raw_prompt_replacement()

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        selected_batches = super().compress_batch_top_k_docs(
            batch_top_k_docs, batch_queries
        )
        return self._replace_selected_with_original_text(selected_batches)


class MaterializedGlobalComparisonAsyncSummarizer(
    PNRawPromptReplacementMixin,
    MaterializedGlobalComparisonSummarizer,
):
    def __init__(self):
        self._reject_cache_on("compare_all_materialized_async")
        super().__init__()
        self._init_raw_prompt_replacement()

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        selected_batches = super().compress_batch_top_k_docs(
            batch_top_k_docs, batch_queries
        )
        return self._replace_selected_with_original_text(selected_batches)


class HybridBgeBM25Summarizer(MaterializedGlobalComparisonSummarizer):
    def __init__(self):
        super().__init__()
        self.bge_weight = self._parse_weight(os.getenv("HYBRID_BGE_WEIGHT", "0.7"))
        self.bm25_path = resolve_bm25_artifact_path()
        self.bm25 = BM25Scorer(self.bm25_path)
        print(
            "Hybrid BGE+BM25 subchunk selection enabled. "
            f"HYBRID_BGE_WEIGHT={self.bge_weight} artifact={self.bm25_path}"
        )

    @staticmethod
    def _parse_weight(raw_value: str) -> float:
        value = float(raw_value)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"HYBRID_BGE_WEIGHT must be between 0 and 1, got {raw_value!r}"
            )
        return value

    @staticmethod
    def _owner_indices_from_batch_chunk_texts(
        batch_chunk_texts_per_doc: List[List[List[str]]],
    ) -> List[int]:
        owner_indices = []
        for query_idx, chunk_texts_per_doc in enumerate(batch_chunk_texts_per_doc):
            for chunk_texts in chunk_texts_per_doc:
                owner_indices.extend([query_idx] * len(chunk_texts))
        return owner_indices

    def _bm25_scores_for_batch(
        self,
        batch_chunk_texts_per_doc: List[List[List[str]]],
        batch_queries: List[str],
    ) -> torch.Tensor:
        scores = []
        for query, chunk_texts_per_doc in zip(batch_queries, batch_chunk_texts_per_doc):
            texts = [
                text for chunk_texts in chunk_texts_per_doc for text in chunk_texts
            ]
            scores.extend(self.bm25.score_texts(query, texts))
        return torch.tensor(scores, dtype=torch.float32)

    def _bm25_scores_for_selection(
        self,
        batch_chunk_texts_per_doc: List[List[List[str]]],
        batch_queries: List[str],
        batch_top_k_docs: List[List[RetrievableChunk]],
    ) -> torch.Tensor:
        del batch_top_k_docs
        return self._bm25_scores_for_batch(batch_chunk_texts_per_doc, batch_queries)

    def _fuse_local_scores(
        self,
        bge_scores: torch.Tensor,
        bm25_scores: torch.Tensor,
    ) -> torch.Tensor:
        owner_indices = [0] * int(bge_scores.numel())
        normalized_bge = normalize_by_owner(bge_scores.to(torch.float32), owner_indices)
        normalized_bm25 = normalize_by_owner(
            bm25_scores.to(torch.float32), owner_indices
        )
        return (
            self.bge_weight * normalized_bge + (1.0 - self.bge_weight) * normalized_bm25
        )

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, bge_scores = super()._score_chunk_texts_for_batch(
            batch_top_k_docs=batch_top_k_docs,
            batch_queries=batch_queries,
        )
        if bge_scores is None:
            return batch_chunk_texts_per_doc, None

        bm25_scores = self._bm25_scores_for_batch(
            batch_chunk_texts_per_doc, batch_queries
        )
        if bm25_scores.shape != bge_scores.shape:
            raise ValueError(
                f"hybrid score shape mismatch: bge={tuple(bge_scores.shape)} bm25={tuple(bm25_scores.shape)}"
            )

        owner_indices = self._owner_indices_from_batch_chunk_texts(
            batch_chunk_texts_per_doc
        )
        normalized_bge = normalize_by_owner(bge_scores.to(torch.float32), owner_indices)
        normalized_bm25 = normalize_by_owner(bm25_scores, owner_indices)
        hybrid_scores = (
            self.bge_weight * normalized_bge + (1.0 - self.bge_weight) * normalized_bm25
        )
        return batch_chunk_texts_per_doc, hybrid_scores

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, bge_scores = (
            MaterializedGlobalComparisonSummarizer._score_chunk_texts_for_batch(
                self,
                batch_top_k_docs=batch_top_k_docs,
                batch_queries=batch_queries,
            )
        )
        summarized_batches = []
        cursor = 0
        if bge_scores is None:
            return [
                [self._build_unselected_document(chunk) for chunk in chunks]
                for chunks in batch_top_k_docs
            ]

        bm25_scores = self._bm25_scores_for_selection(
            batch_chunk_texts_per_doc=batch_chunk_texts_per_doc,
            batch_queries=batch_queries,
            batch_top_k_docs=batch_top_k_docs,
        )
        if bm25_scores.shape != bge_scores.shape:
            raise ValueError(
                f"hybrid score shape mismatch: bge={tuple(bge_scores.shape)} bm25={tuple(bm25_scores.shape)}"
            )

        for chunks, chunk_texts_per_doc in zip(
            batch_top_k_docs, batch_chunk_texts_per_doc
        ):
            summarized_chunks = [
                self._build_unselected_document(chunk) for chunk in chunks
            ]

            sentence_refs = []
            for chunk_idx, chunk_texts in enumerate(chunk_texts_per_doc):
                for sentence_idx in range(len(chunk_texts)):
                    sentence_refs.append((chunk_idx, sentence_idx))

            local_count = len(sentence_refs)
            local_bge_scores = bge_scores[cursor : cursor + local_count]
            local_bm25_scores = bm25_scores[cursor : cursor + local_count]
            cursor += local_count

            sentence_refs, dedup_scores = self._deduplicate_sentence_refs_multi_scores(
                chunks=chunks,
                sentence_refs=sentence_refs,
                score_tensors=[local_bge_scores, local_bm25_scores],
            )
            if not sentence_refs:
                summarized_batches.append(summarized_chunks)
                continue

            local_bge_scores, local_bm25_scores = dedup_scores
            fused_scores = self._fuse_local_scores(local_bge_scores, local_bm25_scores)

            keep_count = self._global_top_count(len(sentence_refs))
            if keep_count <= 0:
                summarized_batches.append(summarized_chunks)
                continue

            top_indices = torch.topk(fused_scores, k=keep_count).indices.tolist()
            selected_by_doc = {}
            for flat_idx in top_indices:
                chunk_idx, sentence_idx = sentence_refs[flat_idx]
                selected_by_doc.setdefault(chunk_idx, []).append(sentence_idx)

            for chunk_idx, selected_indices in selected_by_doc.items():
                summarized_chunks[chunk_idx] = self._build_selected_document(
                    chunks[chunk_idx], selected_indices
                )
            summarized_batches.append(summarized_chunks)

        return summarized_batches


class HybridBgeBM25RRFSummarizer(HybridBgeBM25Summarizer):
    def __init__(self):
        MaterializedGlobalComparisonSummarizer.__init__(self)
        self.bge_weight = self._parse_weight(os.getenv("HYBRID_BGE_WEIGHT", "0.5"))
        self.rrf_k = float(os.getenv("HYBRID_RRF_K", "60"))
        if self.rrf_k < 0:
            raise ValueError(f"HYBRID_RRF_K must be non-negative, got {self.rrf_k}")
        self.bm25_path = resolve_bm25_artifact_path()
        self.bm25 = BM25Scorer(self.bm25_path)
        print(
            "Hybrid BGE+BM25 RRF subchunk selection enabled. "
            f"HYBRID_BGE_WEIGHT={self.bge_weight} HYBRID_RRF_K={self.rrf_k} artifact={self.bm25_path}"
        )

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, bge_scores = (
            MaterializedGlobalComparisonSummarizer._score_chunk_texts_for_batch(
                self,
                batch_top_k_docs=batch_top_k_docs,
                batch_queries=batch_queries,
            )
        )
        if bge_scores is None:
            return batch_chunk_texts_per_doc, None

        bm25_scores = self._bm25_scores_for_batch(
            batch_chunk_texts_per_doc, batch_queries
        )
        if bm25_scores.shape != bge_scores.shape:
            raise ValueError(
                f"hybrid RRF score shape mismatch: bge={tuple(bge_scores.shape)} bm25={tuple(bm25_scores.shape)}"
            )

        owner_indices = self._owner_indices_from_batch_chunk_texts(
            batch_chunk_texts_per_doc
        )
        rrf_scores = reciprocal_rank_fusion_by_owner(
            dense_scores=bge_scores.to(torch.float32),
            sparse_scores=bm25_scores,
            owner_indices=owner_indices,
            rrf_k=self.rrf_k,
            dense_weight=self.bge_weight,
        )
        return batch_chunk_texts_per_doc, rrf_scores

    def _fuse_local_scores(
        self,
        bge_scores: torch.Tensor,
        bm25_scores: torch.Tensor,
    ) -> torch.Tensor:
        return reciprocal_rank_fusion_by_owner(
            dense_scores=bge_scores.to(torch.float32),
            sparse_scores=bm25_scores.to(torch.float32),
            owner_indices=[0] * int(bge_scores.numel()),
            rrf_k=self.rrf_k,
            dense_weight=self.bge_weight,
        )


class HybridBgeBM25AsyncSummarizer(
    PNRawPromptReplacementMixin, HybridBgeBM25Summarizer
):
    def __init__(self):
        MaterializedGlobalComparisonSummarizer.__init__(self)
        self.bge_weight = self._parse_weight(os.getenv("HYBRID_BGE_WEIGHT", "0.7"))
        self.bm25_path = resolve_bm25_async_artifact_path()
        self.bm25 = BM25Scorer(self.bm25_path)
        self.pn_mapping_dir = self._resolve_pn_mapping_dir()
        self._resolved_text_cache: Dict[str, Dict[str, str]] = {}
        print(
            "Hybrid BGE+BM25 with resolved BM25 view enabled. "
            f"HYBRID_BGE_WEIGHT={self.bge_weight} artifact={self.bm25_path} "
            f"pn_mapping_dir={self.pn_mapping_dir}"
        )

    def _load_resolved_texts(self, filename: str) -> Dict[str, str]:
        if filename in self._resolved_text_cache:
            return self._resolved_text_cache[filename]

        mapping_path = os.path.join(self.pn_mapping_dir, f"{filename}.json")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"pn mapping missing for resolved BM25 scoring: {mapping_path}"
            )

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if mapping.get("format") != "pn_mapping_v1":
            raise ValueError(
                f"unsupported pn mapping format in {mapping_path}: {mapping.get('format')!r}"
            )

        resolved_by_sentence_id = {}
        for sentence_view in mapping.get("sentence_views", []):
            sentence_id = sentence_view.get("sentence_id")
            resolved_text = sentence_view.get("resolved_text")
            if sentence_id and resolved_text:
                resolved_by_sentence_id[str(sentence_id)] = str(resolved_text)

        self._resolved_text_cache[filename] = resolved_by_sentence_id
        return resolved_by_sentence_id

    def _resolved_text_for_cacheable(self, cacheable) -> str:
        filename = self._filename_for_cacheable(cacheable)
        resolved_by_sentence_id = self._load_resolved_texts(filename)
        resolved_text = resolved_by_sentence_id.get(cacheable.id)
        if resolved_text is None:
            raise KeyError(
                f"pn mapping resolved_text missing for cacheable id {cacheable.id!r} "
                f"in filename {filename!r}"
            )
        return resolved_text

    def _bm25_scores_for_batch(
        self,
        batch_chunk_texts_per_doc: List[List[List[str]]],
        batch_queries: List[str],
        batch_top_k_docs: List[List[RetrievableChunk]],
    ) -> torch.Tensor:
        del batch_chunk_texts_per_doc
        scores = []
        for query, docs in zip(batch_queries, batch_top_k_docs):
            resolved_texts = []
            for doc in docs:
                for cacheable in getattr(doc, "cacheables", []) or []:
                    if cacheable.text:
                        resolved_texts.append(
                            self._resolved_text_for_cacheable(cacheable)
                        )
            scores.extend(self.bm25.score_texts(query, resolved_texts))
        return torch.tensor(scores, dtype=torch.float32)

    def _bm25_scores_for_selection(
        self,
        batch_chunk_texts_per_doc: List[List[List[str]]],
        batch_queries: List[str],
        batch_top_k_docs: List[List[RetrievableChunk]],
    ) -> torch.Tensor:
        return self._bm25_scores_for_batch(
            batch_chunk_texts_per_doc=batch_chunk_texts_per_doc,
            batch_queries=batch_queries,
            batch_top_k_docs=batch_top_k_docs,
        )

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, bge_scores = (
            MaterializedGlobalComparisonSummarizer._score_chunk_texts_for_batch(
                self,
                batch_top_k_docs=batch_top_k_docs,
                batch_queries=batch_queries,
            )
        )
        if bge_scores is None:
            return batch_chunk_texts_per_doc, None

        bm25_scores = self._bm25_scores_for_batch(
            batch_chunk_texts_per_doc=batch_chunk_texts_per_doc,
            batch_queries=batch_queries,
            batch_top_k_docs=batch_top_k_docs,
        )
        if bm25_scores.shape != bge_scores.shape:
            raise ValueError(
                f"hybrid BM25-async score shape mismatch: "
                f"bge={tuple(bge_scores.shape)} bm25={tuple(bm25_scores.shape)}"
            )

        owner_indices = self._owner_indices_from_batch_chunk_texts(
            batch_chunk_texts_per_doc
        )
        normalized_bge = normalize_by_owner(bge_scores.to(torch.float32), owner_indices)
        normalized_bm25 = normalize_by_owner(bm25_scores, owner_indices)
        hybrid_scores = (
            self.bge_weight * normalized_bge + (1.0 - self.bge_weight) * normalized_bm25
        )
        return batch_chunk_texts_per_doc, hybrid_scores


class HybridBgeBM25RRFAsyncSummarizer(HybridBgeBM25AsyncSummarizer):
    def __init__(self):
        MaterializedGlobalComparisonSummarizer.__init__(self)
        self.bge_weight = self._parse_weight(os.getenv("HYBRID_BGE_WEIGHT", "0.5"))
        self.rrf_k = float(os.getenv("HYBRID_RRF_K", "60"))
        if self.rrf_k < 0:
            raise ValueError(f"HYBRID_RRF_K must be non-negative, got {self.rrf_k}")
        self.bm25_path = resolve_bm25_async_artifact_path()
        self.bm25 = BM25Scorer(self.bm25_path)
        self.pn_mapping_dir = self._resolve_pn_mapping_dir()
        self._resolved_text_cache: Dict[str, Dict[str, str]] = {}
        print(
            "Hybrid BGE+BM25 RRF with resolved BM25 view enabled. "
            f"HYBRID_BGE_WEIGHT={self.bge_weight} HYBRID_RRF_K={self.rrf_k} "
            f"artifact={self.bm25_path} pn_mapping_dir={self.pn_mapping_dir}"
        )

    def _score_chunk_texts_for_batch(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        batch_chunk_texts_per_doc, bge_scores = (
            MaterializedGlobalComparisonSummarizer._score_chunk_texts_for_batch(
                self,
                batch_top_k_docs=batch_top_k_docs,
                batch_queries=batch_queries,
            )
        )
        if bge_scores is None:
            return batch_chunk_texts_per_doc, None

        bm25_scores = self._bm25_scores_for_batch(
            batch_chunk_texts_per_doc=batch_chunk_texts_per_doc,
            batch_queries=batch_queries,
            batch_top_k_docs=batch_top_k_docs,
        )
        if bm25_scores.shape != bge_scores.shape:
            raise ValueError(
                f"hybrid RRF BM25-async score shape mismatch: "
                f"bge={tuple(bge_scores.shape)} bm25={tuple(bm25_scores.shape)}"
            )

        owner_indices = self._owner_indices_from_batch_chunk_texts(
            batch_chunk_texts_per_doc
        )
        rrf_scores = reciprocal_rank_fusion_by_owner(
            dense_scores=bge_scores.to(torch.float32),
            sparse_scores=bm25_scores,
            owner_indices=owner_indices,
            rrf_k=self.rrf_k,
            dense_weight=self.bge_weight,
        )
        return batch_chunk_texts_per_doc, rrf_scores

    def _fuse_local_scores(
        self,
        bge_scores: torch.Tensor,
        bm25_scores: torch.Tensor,
    ) -> torch.Tensor:
        return reciprocal_rank_fusion_by_owner(
            dense_scores=bge_scores.to(torch.float32),
            sparse_scores=bm25_scores.to(torch.float32),
            owner_indices=[0] * int(bge_scores.numel()),
            rrf_k=self.rrf_k,
            dense_weight=self.bge_weight,
        )
