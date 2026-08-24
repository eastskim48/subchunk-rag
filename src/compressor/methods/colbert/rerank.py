"""Document-level ColBERT reranking over fixed or contextualized artifacts."""

import os
import time
from typing import List

import torch

from chunk import RetrievableChunk
from colbert_artifact import FixedChunkColBERTArtifact
from compressor.base import Compressor
from compressor.methods.colbert.base import ColBERTWindowCompressorBase
from compressor.methods.colbert.scoring import (
    aggregate_sentence_maxsim,
    score_maxsim,
    sentence_token_maxsim,
)
from encoder.colbert import ColBERTEncoder, default_colbert_repo_path


def _configured_colbert_rerank_keep() -> int:
    keep = int(os.getenv("COLBERT_RERANK_KEEP", "0"))
    if keep <= 0:
        raise ValueError("COLBERT_RERANK_KEEP must be positive")
    return keep


class _ColBERTRerankMixin:
    """Add coarse chunk reranking to compressors that explicitly request it."""

    @staticmethod
    def _score_coarse_chunk(
        query_vector: torch.Tensor, vectors: list[torch.Tensor]
    ) -> float:
        sentence_scores = sentence_token_maxsim(query_vector, vectors)
        return aggregate_sentence_maxsim(sentence_scores, [list(range(len(vectors)))])[
            0
        ]

    def _rerank_chunk_indices(
        self,
        docs: List[RetrievableChunk],
        query_vector: torch.Tensor,
        profile: dict[str, float | int] | None = None,
    ) -> list[int]:
        self._coarse_sentence_scores_by_source_id = {}
        vectors_by_doc = []
        for doc in docs:
            lookup_start = time.perf_counter()
            vectors = self.artifact.vectors_for_doc(doc)
            self._profile_add(
                profile, "artifact_lookup_time", time.perf_counter() - lookup_start
            )
            vectors_by_doc.append(vectors)

        score_start = time.perf_counter()
        all_sentence_vectors = [
            vector for vectors in vectors_by_doc for vector in vectors
        ]
        sentence_scores = sentence_token_maxsim(query_vector, all_sentence_vectors)
        sentence_groups = []
        offset = 0
        for vectors in vectors_by_doc:
            next_offset = offset + len(vectors)
            sentence_groups.append(list(range(offset, next_offset)))
            self._coarse_sentence_scores_by_source_id[id(vectors)] = sentence_scores[
                offset:next_offset
            ]
            offset = next_offset
        chunk_scores = aggregate_sentence_maxsim(sentence_scores, sentence_groups)
        self._profile_add(
            profile,
            "coarse_rerank_score_time",
            time.perf_counter() - score_start,
        )
        scored_chunks = list(zip(chunk_scores, range(len(docs))))

        sort_start = time.perf_counter()
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        self._profile_add(
            profile, "coarse_rerank_sort_time", time.perf_counter() - sort_start
        )
        return [chunk_idx for _, chunk_idx in scored_chunks]


class FixedChunkColBERTRerankCompressor(Compressor):
    """Rerank retrieved chunks with independently encoded fixed-chunk vectors."""

    def __init__(self):
        super().__init__()
        dataset_path = os.getenv("DATASET_PATH")
        data_subdir = os.getenv("DATA_SUBDIR")
        if not dataset_path or not data_subdir:
            raise ValueError("DATASET_PATH and DATA_SUBDIR are required")
        artifact_dir = os.path.join(
            dataset_path, data_subdir, "colbert_fixed_chunk_docmax512"
        )
        self.keep = int(os.getenv("COLBERT_CHUNK_RERANK_KEEP", "0"))
        if self.keep <= 0:
            raise ValueError("COLBERT_CHUNK_RERANK_KEEP must be positive")
        model_name = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
        repo_path = os.getenv("COLBERT_REPO_PATH") or default_colbert_repo_path()
        batch_size = int(os.getenv("COLBERT_BATCH_SIZE", "32"))

        print(f"Fixed-chunk ColBERT rerank enabled. Loading artifact: {artifact_dir}")
        self.artifact = FixedChunkColBERTArtifact(artifact_dir)
        artifact_model_name = self.artifact.index.get("model_name")
        if artifact_model_name != model_name:
            raise ValueError(
                "COLBERT_MODEL_NAME does not match the fixed-chunk ColBERT artifact: "
                f"runtime={model_name!r}, artifact={artifact_model_name!r}"
            )
        self.encoder = ColBERTEncoder(
            model_name=model_name,
            repo_path=repo_path,
            device="cpu",
            batch_size=batch_size,
            doc_maxlen=int(self.artifact.index.get("official_doc_maxlen", 0)),
            query_maxlen=None,
            disable_cpu_extension=True,
            verify_tensorization=False,
        )
        self.query_encoder_warmup_time = 0.0
        self.last_profile: dict[str, float | int] = {}

    def warmup_query_encoder(self) -> float:
        start = time.perf_counter()
        self.encoder.encode_queries(["warmup query"])
        self.query_encoder_warmup_time = time.perf_counter() - start
        return self.query_encoder_warmup_time

    def clear_inter_batch_cache(self) -> None:
        return None

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = {
            "query_encode_time": 0.0,
            "artifact_lookup_time": 0.0,
            "score_time": 0.0,
            "sort_time": 0.0,
            "build_output_time": 0.0,
            "query_count": len(batch_queries),
            "retrieved_doc_count": sum(len(docs) for docs in batch_top_k_docs),
            "selected_doc_count": 0,
        }
        start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - start

        summarized_batches = []
        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            scored_docs = []
            for doc in docs:
                lookup_start = time.perf_counter()
                vectors = self.artifact.vectors_for_chunk_id(str(doc.id))
                profile["artifact_lookup_time"] += time.perf_counter() - lookup_start
                score_start = time.perf_counter()
                score = score_maxsim(query_vector, vectors)
                profile["score_time"] += time.perf_counter() - score_start
                scored_docs.append((score, doc))

            sort_start = time.perf_counter()
            scored_docs.sort(key=lambda item: item[0], reverse=True)
            selected = scored_docs[: min(self.keep, len(scored_docs))]
            profile["sort_time"] += time.perf_counter() - sort_start

            build_start = time.perf_counter()
            summarized_batches.append([doc.clone() for _, doc in selected])
            profile["selected_doc_count"] += len(selected)
            profile["build_output_time"] += time.perf_counter() - build_start

        self.last_profile = profile
        return summarized_batches


class ColBERTRerankCompressor(_ColBERTRerankMixin, ColBERTWindowCompressorBase):
    """Rerank chunks by aggregating their contextualized subchunk vectors."""

    def __init__(self):
        keep = _configured_colbert_rerank_keep()
        super().__init__(
            initialize_token_budget=False,
        )
        self.keep = keep

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = {
            "query_encode_time": 0.0,
            "artifact_lookup_time": 0.0,
            "coarse_rerank_score_time": 0.0,
            "coarse_rerank_sort_time": 0.0,
            "build_output_time": 0.0,
            "query_count": len(batch_queries),
            "retrieved_doc_count": sum(len(docs) for docs in batch_top_k_docs),
            "selected_doc_count": 0,
        }
        start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - start

        summarized_batches = []
        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            ranked_indices = self._rerank_chunk_indices(docs, query_vector, profile)
            selected_indices = ranked_indices[: min(self.keep, len(ranked_indices))]

            build_start = time.perf_counter()
            summarized_batches.append([docs[idx].clone() for idx in selected_indices])
            profile["selected_doc_count"] += len(selected_indices)
            profile["build_output_time"] += time.perf_counter() - build_start

        self.last_profile = profile
        return summarized_batches
