"""ColBERT region-selection compressor implementations."""

import os
import time
from typing import List

import torch

from chunk import CacheableChunk, RetrievableChunk
from compressor.methods.colbert.base import (
    ColBERTWindowCompressorBase,
    _resolve_configured_retrieval_chunk_size,
)
from compressor.methods.colbert.rerank import (
    _ColBERTRerankMixin,
    _configured_colbert_rerank_keep,
)
from compressor.methods.colbert.scoring import (
    aggregate_sentence_maxsim,
    score_maxsim,
    sentence_token_maxsim,
)


def _configured_region_group_order() -> str:
    order = os.getenv("COLBERT_REGION_GROUP_ORDER", "retrieval").strip().lower()
    if order not in {"retrieval", "max", "sum"}:
        raise ValueError(
            "COLBERT_REGION_GROUP_ORDER must be one of retrieval, max, or sum, "
            f"got {order!r}"
        )
    return order


def _validate_retrieval_chunk_larger_than_region(
    retrieval_chunk_size: int, region_token_budget: int
) -> None:
    if retrieval_chunk_size <= region_token_budget:
        raise ValueError(
            "retrieval chunk size must be larger than the ColBERT region budget: "
            f"retrieval_chunk_size={retrieval_chunk_size}, "
            f"region_token_budget={region_token_budget}"
        )


class ColBERTSlidingRegionCompressor(ColBERTWindowCompressorBase):
    """Score overlapping sentence regions and retain them under a final budget."""

    def __init__(self):
        super().__init__()
        self.region_group_order = _configured_region_group_order()
        self._sliding_region_spec_cache = {}
        self.region_token_budget = int(self.artifact.index["window_token_budget"])
        if self.region_token_budget <= self.encoder.doc_token_overhead:
            raise ValueError(
                "sliding region token budget must be larger than ColBERT document token overhead, "
                f"got {self.region_token_budget}"
            )
        retrieval_chunk_size = _resolve_configured_retrieval_chunk_size()
        _validate_retrieval_chunk_larger_than_region(
            retrieval_chunk_size, self.region_token_budget
        )

    def clear_inter_batch_cache(self) -> None:
        super().clear_inter_batch_cache()
        self._sliding_region_spec_cache.clear()

    @staticmethod
    def _source_parent_document_id(
        doc: RetrievableChunk, cacheables: list[CacheableChunk]
    ) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        parent_doc_id = metadata.get("parent_doc_id")
        if isinstance(parent_doc_id, str) and parent_doc_id:
            return parent_doc_id
        for cacheable in cacheables:
            parent_doc_id = getattr(cacheable, "parent_doc_id", None)
            if isinstance(parent_doc_id, str) and parent_doc_id:
                return parent_doc_id
        raise ValueError(
            f"retrieval chunk {getattr(doc, 'id', '')!r} has no source parent document ID"
        )

    @staticmethod
    def _build_region_document(
        doc: RetrievableChunk, selected_cacheables: list[CacheableChunk]
    ) -> RetrievableChunk:
        source_order_by_id = {
            str(cacheable.id): source_idx
            for source_idx, cacheable in enumerate(doc.cacheables)
        }

        def source_order(cacheable: CacheableChunk) -> int:
            if not cacheable.sentence_ids:
                raise ValueError(
                    f"selected ColBERT region {cacheable.id!r} has no source IDs"
                )
            first_source_id = str(cacheable.sentence_ids[0])
            if first_source_id not in source_order_by_id:
                raise ValueError(
                    "selected ColBERT region references a source ID outside its "
                    f"retrieval chunk: region={cacheable.id!r}, "
                    f"source_id={first_source_id!r}, chunk={doc.id!r}"
                )
            return source_order_by_id[first_source_id]

        cloned = doc.clone()
        cloned.cacheables = [
            cacheable.clone()
            for cacheable in sorted(selected_cacheables, key=source_order)
        ]
        return cloned

    def _sliding_region_cache_key(
        self, doc: RetrievableChunk, cacheables: list[CacheableChunk]
    ):
        return (
            str(doc.id),
            self.region_token_budget,
            tuple(str(getattr(cacheable, "id", "")) for cacheable in cacheables),
        )

    def _cached_sliding_region_specs(
        self, doc: RetrievableChunk, cacheables: list[CacheableChunk]
    ):
        """Load query-independent region membership from the artifact sidecar."""

        cache_key = self._sliding_region_cache_key(doc, cacheables)
        cached = self._sliding_region_spec_cache.get(cache_key)
        if cached is not None:
            return cached

        region_specs = []
        sidecar_specs = self.artifact.region_specs_for_doc(
            doc, self.region_token_budget
        )
        for center_idx, selected_indices in sidecar_specs:
            if not selected_indices:
                continue
            region_specs.append((center_idx, selected_indices))

        self._sliding_region_spec_cache[cache_key] = region_specs
        return region_specs

    def _sliding_regions_for_doc(
        self,
        doc: RetrievableChunk,
        chunk_idx: int,
        profile: dict[str, float | int] | None = None,
    ):
        """Attach one retrieved chunk's runtime data to its materialized regions."""

        cacheables = [
            cacheable
            for cacheable in getattr(doc, "cacheables", []) or []
            if cacheable.text
        ]
        if not cacheables:
            return []
        parent_doc_id = self._source_parent_document_id(doc, cacheables)
        lookup_start = time.perf_counter()
        vectors = self.artifact.vectors_for_doc(doc)
        self._profile_add(
            profile, "artifact_lookup_time", time.perf_counter() - lookup_start
        )

        region_specs = self._cached_sliding_region_specs(doc, cacheables)

        regions = []
        for center_idx, selected_indices in region_specs:
            regions.append(
                {
                    "chunk_idx": chunk_idx,
                    "center_idx": center_idx,
                    "cacheable": None,
                    "region_id": f"{doc.id}::sliding_region_{center_idx}",
                    "parent_doc_id": parent_doc_id,
                    "source_doc": doc,
                    "selected_indices": selected_indices,
                    "source_cacheables": cacheables,
                    "source_vectors": vectors,
                }
            )
        self._profile_add(profile, "region_count", len(regions))
        return regions

    @staticmethod
    def _sentence_score_cache_key(region, idx: int):
        cacheables = region["source_cacheables"]
        raw_cacheable_id = getattr(cacheables[idx], "id", None)
        source_vectors = region.get("source_vectors")
        source_key = id(source_vectors) if source_vectors is not None else "no_source"
        if raw_cacheable_id:
            return (source_key, idx, str(raw_cacheable_id))
        return (source_key, region["chunk_idx"], idx)

    def _score_sliding_regions_vectorized(
        self,
        query_vector: torch.Tensor,
        regions,
        profile: dict[str, float | int] | None = None,
    ) -> list[float]:
        """Score overlapping regions while computing each unique sentence once."""

        if not regions:
            return []

        sentence_index_by_key: dict[tuple, int] = {}
        sentence_vectors = []
        region_sentence_indices = []
        fallback_regions = []

        for region_idx, region in enumerate(regions):
            if region.get("source_vectors") is None:
                fallback_regions.append(region_idx)
                region_sentence_indices.append([])
                continue
            indices = []
            source_vectors = region["source_vectors"]
            cacheables = region["source_cacheables"]
            for idx in region["selected_indices"]:
                if not (0 <= idx < len(cacheables)):
                    continue
                cache_key = self._sentence_score_cache_key(region, idx)
                sentence_idx = sentence_index_by_key.get(cache_key)
                if sentence_idx is None:
                    vectors = (
                        source_vectors[idx]
                        if idx < len(source_vectors)
                        else torch.empty((0, 0))
                    )
                    sentence_idx = len(sentence_vectors)
                    sentence_index_by_key[cache_key] = sentence_idx
                    sentence_vectors.append(vectors)
                indices.append(sentence_idx)
            region_sentence_indices.append(indices)

        if not sentence_vectors:
            scores = [float("-inf")] * len(regions)
            for region_idx in fallback_regions:
                region = regions[region_idx]
                scores[region_idx] = score_maxsim(
                    query_vector, region.get("vectors", torch.empty((0, 0)))
                )
            return scores

        precomputed_by_source_id = getattr(
            self, "_coarse_sentence_scores_by_source_id", {}
        )
        precomputed_rows = []
        for cache_key, sentence_idx in sentence_index_by_key.items():
            source_id, source_idx, _ = cache_key
            source_scores = precomputed_by_source_id.get(source_id)
            if source_scores is None or source_idx >= len(source_scores):
                precomputed_rows = []
                break
            if sentence_idx != len(precomputed_rows):
                precomputed_rows = []
                break
            precomputed_rows.append(source_scores[source_idx])

        if len(precomputed_rows) == len(sentence_vectors):
            sentence_scores = torch.stack(precomputed_rows)
        else:
            sentence_scores = sentence_token_maxsim(query_vector, sentence_vectors)
        scores = aggregate_sentence_maxsim(sentence_scores, region_sentence_indices)

        for region_idx in fallback_regions:
            region = regions[region_idx]
            scores[region_idx] = score_maxsim(
                query_vector, region.get("vectors", torch.empty((0, 0)))
            )
        return scores

    def _region_sentence_indices_to_keep(
        self, query_vector: torch.Tensor | None, region
    ) -> set[int]:
        return set(region["selected_indices"])

    def _make_region_run_cacheables(self, region, selected_indices):
        runs = []
        current_run = []
        previous_idx = None
        for idx in selected_indices:
            if previous_idx is not None and idx != previous_idx + 1:
                if current_run:
                    runs.append(current_run)
                current_run = []
            current_run.append(idx)
            previous_idx = idx
        if current_run:
            runs.append(current_run)

        cacheables = []
        source_cacheables = region["source_cacheables"]
        for run in runs:
            run_cacheables = [source_cacheables[idx] for idx in run]
            cacheables.append(
                CacheableChunk(
                    id=f"{region['region_id']}::dedup_{run[0]}_{run[-1]}",
                    text=" ".join(cacheable.text for cacheable in run_cacheables),
                    parent_doc_id=region["parent_doc_id"],
                    chunk_size=self.region_token_budget,
                    sentence_ids=[cacheable.id for cacheable in run_cacheables],
                    sentence_texts=[cacheable.text for cacheable in run_cacheables],
                )
            )
        return cacheables

    def _select_sliding_regions_with_scores(
        self,
        scored_regions,
        query_vector: torch.Tensor | None = None,
        final_token_budget: int | None = None,
    ):
        """Select scored regions, deduplicating sentences shared by overlaps."""

        if final_token_budget is None:
            raise ValueError("final_token_budget is required for region selection")
        selected_region_scores_by_chunk: dict[int, list[float]] = {}
        self._cacheable_token_lens(
            self._unique_cacheables(
                [
                    source
                    for _, region in scored_regions
                    for source in region["source_cacheables"]
                ]
            )
        )
        selected_cacheables = []
        selected_sentence_ids = set()
        used_tokens = 0
        for score, region in scored_regions:
            novel_indices = []
            novel_token_count = 0
            allowed_indices = self._region_sentence_indices_to_keep(
                query_vector, region
            )
            for idx in region["selected_indices"]:
                if idx not in allowed_indices:
                    continue
                source = region["source_cacheables"][idx]
                if source.id in selected_sentence_ids:
                    continue
                token_len = self._cacheable_token_len(source)
                novel_indices.append(idx)
                novel_token_count += token_len

            if not novel_indices:
                continue

            for idx in novel_indices:
                selected_sentence_ids.add(region["source_cacheables"][idx].id)
            used_tokens += novel_token_count
            selected_region_scores_by_chunk.setdefault(region["chunk_idx"], []).append(
                float(score)
            )

            selected_cacheables.extend(
                (region["chunk_idx"], cacheable)
                for cacheable in self._make_region_run_cacheables(region, novel_indices)
            )

            # Match the project-wide add-then-check budget policy.
            if used_tokens >= final_token_budget:
                break

        return selected_cacheables, selected_region_scores_by_chunk

    def _select_sliding_regions(
        self,
        scored_regions,
        query_vector: torch.Tensor | None = None,
        final_token_budget: int | None = None,
    ):
        selected_cacheables, _ = self._select_sliding_regions_with_scores(
            scored_regions,
            query_vector=query_vector,
            final_token_budget=final_token_budget,
        )
        return selected_cacheables

    def _order_document_groups(self, docs, region_scores_by_chunk):
        region_group_order = getattr(self, "region_group_order", "retrieval")
        if region_group_order == "retrieval":
            return docs

        selected_indices = [
            chunk_idx
            for chunk_idx in range(len(docs))
            if region_scores_by_chunk.get(chunk_idx)
        ]
        if region_group_order == "max":
            group_score = lambda chunk_idx: max(region_scores_by_chunk[chunk_idx])
        else:
            group_score = lambda chunk_idx: sum(region_scores_by_chunk[chunk_idx])
        selected_indices.sort(
            key=lambda chunk_idx: (-group_score(chunk_idx), chunk_idx)
        )
        selected_index_set = set(selected_indices)
        return [docs[chunk_idx] for chunk_idx in selected_indices] + [
            doc
            for chunk_idx, doc in enumerate(docs)
            if chunk_idx not in selected_index_set
        ]

    def _region_chunk_indices(self, docs, query_vector, profile) -> set[int]:
        """Return retrieval-chunk indices eligible for region construction."""

        return set(range(len(docs)))

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        profile = self._empty_profile()
        profile["query_count"] = len(batch_queries)
        profile["retrieved_doc_count"] = sum(len(docs) for docs in batch_top_k_docs)
        query_encode_start = time.perf_counter()
        query_vectors = self.encoder.encode_queries(batch_queries)
        profile["query_encode_time"] = time.perf_counter() - query_encode_start
        summarized_batches = []

        for docs, query_vector in zip(batch_top_k_docs, query_vectors):
            summarized_docs = [self._build_unselected_document(doc) for doc in docs]
            allowed_indices = self._region_chunk_indices(docs, query_vector, profile)
            gated_docs = [doc for idx, doc in enumerate(docs) if idx in allowed_indices]
            budget_start = time.perf_counter()
            final_token_budget = self._resolve_final_token_budget(gated_docs)
            self._profile_add(
                profile, "budget_time", time.perf_counter() - budget_start
            )
            regions = []
            for chunk_idx, doc in enumerate(docs):
                if chunk_idx not in allowed_indices:
                    continue
                regions.extend(self._sliding_regions_for_doc(doc, chunk_idx, profile))
            if not regions:
                summarized_batches.append(summarized_docs)
                continue

            region_scores = self._score_sliding_regions_vectorized(
                query_vector, regions, profile
            )
            scored_regions = list(zip(region_scores, regions))
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            selected_by_doc: dict[int, list[CacheableChunk]] = {}
            select_start = time.perf_counter()
            selected_cacheables, region_scores_by_chunk = (
                self._select_sliding_regions_with_scores(
                    scored_regions,
                    query_vector,
                    final_token_budget=final_token_budget,
                )
            )
            for chunk_idx, cacheable in selected_cacheables:
                selected_by_doc.setdefault(chunk_idx, []).append(cacheable)
            self._profile_add(
                profile, "select_time", time.perf_counter() - select_start
            )

            output_start = time.perf_counter()
            for chunk_idx, selected_cacheables in selected_by_doc.items():
                summarized_docs[chunk_idx] = self._build_region_document(
                    docs[chunk_idx], selected_cacheables
                )
            summarized_docs = self._order_document_groups(
                summarized_docs, region_scores_by_chunk
            )
            self._profile_add(
                profile, "build_output_time", time.perf_counter() - output_start
            )
            summarized_batches.append(summarized_docs)

        self.last_profile = profile
        return summarized_batches


class ColBERTRerankAndRegionCompressor(
    _ColBERTRerankMixin, ColBERTSlidingRegionCompressor
):
    """Gate region selection with coarse ColBERT chunk reranking."""

    def __init__(self):
        self.rerank_keep = _configured_colbert_rerank_keep()
        super().__init__()

    def _region_chunk_indices(self, docs, query_vector, profile) -> set[int]:
        ranked_indices = self._rerank_chunk_indices(docs, query_vector, profile)
        allowed_indices = set(ranked_indices[: self.rerank_keep])
        self._profile_add(profile, "rerank_kept_chunk_count", len(allowed_indices))
        return allowed_indices
